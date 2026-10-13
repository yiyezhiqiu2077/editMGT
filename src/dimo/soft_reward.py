"""Differentiable editing rewards. No changes to inference/evaluation metrics.

Generated images stay in torch autograd; only target features and diagnostics
are evaluated without gradients. The source codebook lookup is deliberately
constant. This is an editing extension, not a reproduction of text rewards.
"""
from __future__ import annotations

import copy
import math
from functools import lru_cache
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

NAMES = ("lpips", "dino", "clip")


@lru_cache(maxsize=64)
def _antialias_axis_weights(length, output_length, mode, device, dtype):
    # Construct the fixed linear resampling operator on CPU. Only constants
    # cross devices; generated images and their gradients stay on their device.
    basis = torch.eye(length, dtype=dtype).reshape(length, 1, 1, length)
    weights = F.interpolate(basis, size=(1, output_length), mode=mode,
                            align_corners=False, antialias=True)[:, 0, 0].T
    return weights.contiguous().to(device=device)


def deterministic_antialias_resize(image, size, *, mode):
    """Separable PyTorch antialias filter with deterministic matmul backward.

    CUDA interpolate(antialias=True) backward uses atomic accumulation and
    raises under strict deterministic algorithms. Applying its constant axis
    filters as matrix products preserves the antialias recipe and autograd.
    """
    if mode not in ("bilinear", "bicubic"):
        raise ValueError("unsupported differentiable antialias filter")
    if tuple(image.shape[-2:]) == tuple(size):
        return image
    h, w = image.shape[-2:]
    a = _antialias_axis_weights(h, size[0], mode, image.device, image.dtype)
    b = _antialias_axis_weights(w, size[1], mode, image.device, image.dtype)
    return torch.matmul(torch.matmul(a, image), b.T)


def freeze(module):
    module.requires_grad_(False).eval()
    return module


class SoftEmbeddingEditDecoder(nn.Module):
    def __init__(self, vqvae, *, codebook_size=8192, activation_checkpointing=True):
        super().__init__()
        self.vqvae = freeze(vqvae)
        self.codebook_size = codebook_size
        self.activation_checkpointing = activation_checkpointing
        if vqvae.quantize.embedding.weight.shape[0] != codebook_size:
            raise ValueError("VQ codebook must contain exactly the valid image classes")

    def embeddings(self, logits, source, region):
        if logits.shape[:-1] != source.shape or source.shape != region.shape:
            raise ValueError("expected logits [B,H,W,V] and matching source/ROI")
        if logits.shape[-1] < self.codebook_size or (source < 0).any() or (source >= self.codebook_size).any():
            raise ValueError("invalid source codes or vocabulary")
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=logits.device.type, enabled=False):
            book = self.vqvae.quantize.embedding.weight.to(dtype)
            # Compute expectations only inside ROI, excluding the mask class.
            selected = logits[..., :self.codebook_size][region.bool()].to(dtype)
            expected = selected.softmax(-1) @ book
            locked = F.embedding(source.long(), book)
            latent = locked.clone()
            latent[region.bool()] = expected
            return latent.permute(0, 3, 1, 2).contiguous()

    def decode_embeddings(self, latent):
        def operation(z):
            z2 = self.vqvae.post_quant_conv(z)
            return self.vqvae.decoder(z2, z if self.vqvae.config.norm_type == "spatial" else None)
        with torch.autocast(device_type=latent.device.type, enabled=False):
            value = (checkpoint(operation, latent, use_reentrant=False)
                     if self.activation_checkpointing and latent.requires_grad else operation(latent))
            # Match EditMGT native hard decode, including its range convention.
            return value.clamp(0, 1)

    def forward(self, logits, source, region):
        return self.decode_embeddings(self.embeddings(logits, source, region))

    @torch.no_grad()
    def hard(self, tokens):
        book = self.vqvae.quantize.embedding.weight
        return self.decode_embeddings(F.embedding(tokens.long(), book).permute(0, 3, 1, 2).contiguous())


class DifferentiableMaskedLPIPS(nn.Module):
    """Same learned feature aggregation as MaskedLPIPS, with input gradients."""
    def __init__(self, model):
        super().__init__()
        self.model = freeze(model)

    def forward(self, generated, target, mask):
        import lpips
        ga = self.model.net(self.model.scaling_layer(generated * 2 - 1))
        with torch.no_grad():
            ta = self.model.net(self.model.scaling_layer(target * 2 - 1))
        values = []
        for i, (a, b) in enumerate(zip(ga, ta)):
            distance = self.model.lins[i]((lpips.normalize_tensor(a) - lpips.normalize_tensor(b)).square())
            weight = F.interpolate(mask.float(), size=distance.shape[-2:], mode="area")
            denominator = weight.sum((1, 2, 3))
            if (denominator <= 0).any():
                raise ValueError("empty LPIPS reward region")
            values.append((distance * weight).sum((1, 2, 3)) / denominator)
        return torch.stack(values).sum(0).mean()


def roi_boxes(mask):
    if mask.ndim == 3:
        mask = mask[:, None]
    boxes = []
    for m in mask[:, 0]:
        y, x = torch.where(m > 0)
        if not len(y):
            raise ValueError("empty reward ROI")
        x0, x1, y0, y1 = int(x.min()), int(x.max())+1, int(y.min()), int(y.max())+1
        margin = max(8, round(.1 * max(x1-x0, y1-y0)))
        boxes.append((max(0, x0-margin), max(0, y0-margin), min(m.shape[1], x1+margin), min(m.shape[0], y1+margin)))
    return boxes


def crop_preprocess(image, boxes, *, shortest_edge, crop_size, mean, std):
    """Torch bicubic resize + center crop; exactly shared by both image paths.

    Input is RGB [0,1]; no rescale by 1/255. PIL/HF uint8 interpolation is
    intentionally not used in training. Frozen DEV keeps its native processor.
    """
    processed = []
    for row, (x0, y0, x1, y1) in zip(image, boxes):
        crop = row[:, y0:y1, x0:x1][None]
        h, w = crop.shape[-2:]
        size = (max(crop_size, int(h * shortest_edge / min(h, w))),
                max(crop_size, int(w * shortest_edge / min(h, w))))
        crop = deterministic_antialias_resize(crop.float(), size, mode="bicubic")
        h, w = crop.shape[-2:]
        crop = crop[:, :, (h-crop_size)//2:(h-crop_size)//2+crop_size,
                    (w-crop_size)//2:(w-crop_size)//2+crop_size]
        processed.append(crop)
    result = torch.cat(processed)
    return (result - result.new_tensor(mean)[None, :, None, None]) / result.new_tensor(std)[None, :, None, None]


class EditingRewards(nn.Module):
    def __init__(self, lpips_model, dino, clip, *, lpips_resolution=512, activation_checkpointing=True):
        super().__init__()
        self.lpips = DifferentiableMaskedLPIPS(lpips_model)
        self.dino, self.clip = freeze(dino), freeze(clip)
        self.lpips_resolution = lpips_resolution
        self.activation_checkpointing = activation_checkpointing

    def _features(self, image, name):
        def run(x):
            output = self.clip.get_image_features(pixel_values=x) if name == "clip" else self.dino(pixel_values=x).last_hidden_state[:, 0]
            return F.normalize(output.float(), dim=-1)
        return (checkpoint(run, image, use_reentrant=False) if self.activation_checkpointing and image.requires_grad else run(image))

    def forward(self, generated, target, pixel_mask):
        if generated.shape != target.shape:
            raise ValueError("reward source/target geometry must match")
        mask = pixel_mask[:, None] if pixel_mask.ndim == 3 else pixel_mask
        boxes = roi_boxes(mask)
        size = (min(generated.shape[-2], self.lpips_resolution), min(generated.shape[-1], self.lpips_resolution))
        ga = deterministic_antialias_resize(generated.float(), size, mode="bilinear")
        ta = deterministic_antialias_resize(target.float(), size, mode="bilinear")
        ma = F.interpolate(mask.float(), size=size, mode="area")
        losses = {"lpips": self.lpips(ga, ta, ma)}
        for name in ("dino", "clip"):
            spec = (dict(shortest_edge=256, crop_size=224, mean=[.485,.456,.406], std=[.229,.224,.225])
                    if name == "dino" else dict(shortest_edge=224, crop_size=224,
                    mean=[.48145466,.4578275,.40821073], std=[.26862954,.26130258,.27577711]))
            a = crop_preprocess(generated, boxes, **spec)
            b = crop_preprocess(target, boxes, **spec)
            feature_a = self._features(a, name)
            with torch.no_grad():
                feature_b = self._features(b, name)
            losses[name] = (1 - (feature_a * feature_b).sum(-1)).mean()
        return losses


def global_mean(values):
    values = values.clone()
    if dist.is_initialized():
        dist.all_reduce(values)
        values /= dist.get_world_size()
    return values


class RewardGradientCalibration:
    """Warm up on fixed count A batches, then freeze gradient-derived lambdas."""
    def __init__(self, batches=4, relative_scale=.1, max_weight=100.):
        self.batches, self.relative_scale, self.max_weight = batches, relative_scale, max_weight
        self.state = {"count": 0, "norm_sums": [0.] * 4, "weights": None}

    def load_state_dict(self, state):
        if set(state) != {"count", "norm_sums", "weights"} or not 0 <= state["count"] <= self.batches:
            raise ValueError("invalid reward calibration state")
        if len(state["norm_sums"]) != 4 or any(not math.isfinite(x) or x < 0 for x in state["norm_sums"]):
            raise ValueError("invalid reward norm accumulation")
        if (state["weights"] is None) != (state["count"] < self.batches):
            raise ValueError("incomplete reward calibration state")
        if state["weights"] is not None and (set(state["weights"]) != set(NAMES) or any(not 0 < x <= self.max_weight for x in state["weights"].values())):
            raise ValueError("invalid calibrated reward weights")
        self.state = copy.deepcopy(state)

    def weights_for_step(self, norms):
        # Every rank sees the same averaged norms and freezes the same weights.
        if self.state["weights"] is not None:
            return self.state["weights"], False
        if not torch.isfinite(norms).all() or (norms <= 1e-12).any():
            raise RuntimeError("REWARD_LOGITS_GRADIENT_MISSING_OR_NONFINITE")
        self.state["norm_sums"] = [a+b for a,b in zip(self.state["norm_sums"], norms.tolist())]
        self.state["count"] += 1
        if self.state["count"] == self.batches:
            sums = self.state["norm_sums"]
            weights = {name: self.relative_scale * sums[0] / sums[i+1] for i,name in enumerate(NAMES)}
            if any(not math.isfinite(x) or not 0 < x <= self.max_weight for x in weights.values()):
                raise RuntimeError("REWARD_CALIBRATION_UNSAFE_WEIGHT_REVIEW_REQUIRED")
            self.state["weights"] = weights
        # All warmup batches optimize DiMO only, including the final warmup.
        return dict.fromkeys(NAMES, 0.), True


class SoftRewardObjective:
    def __init__(self, decoder, rewards, config):
        self.decoder, self.rewards, self.config = decoder, rewards, config
        self.calibration = RewardGradientCalibration(**config["calibration"])

    def __call__(self, logits, linear_loss, source, region, target, pixel_mask, hard_tokens, step):
        with torch.autocast(device_type=logits.device.type, enabled=False):
            error = None
            try:
                image = self.decoder(logits, source, region)
                losses = self.rewards(image, target, pixel_mask)
                terms = [linear_loss] + [losses[k] for k in NAMES]
                # Stop at logits: no DDP parameter hooks or updates.
                gradients = [torch.autograd.grad(loss, logits, retain_graph=True)[0].float() for loss in terms]
                local_norms = torch.stack([g.norm() for g in gradients])
                if not torch.isfinite(local_norms).all() or (local_norms <= 1e-12).any():
                    raise RuntimeError("REWARD_LOGITS_GRADIENT_MISSING_OR_NONFINITE")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            # All ranks agree before entering norm reduction or DDP backward,
            # including an OOM/zero-gradient failure on a single rank.
            from .transaction import collective_status
            collective_status(error)
            norms = global_mean(local_norms)
            weights, warmup = self.calibration.weights_for_step(norms)
            total = linear_loss + sum(weights[k]*losses[k] for k in NAMES)
            total_gradient = gradients[0] + sum(weights[k]*g for k,g in zip(NAMES, gradients[1:]))
            metrics = {"reward_losses": {k:float(v.detach()) for k,v in losses.items()},
                       "reward_logits_grad_norms": dict(zip(("linear",)+NAMES, norms.tolist())),
                       "reward_weights": dict(weights), "reward_calibration_only": warmup,
                       "weighted_reward_logits_grad_norms": {k:weights[k]*float(norms[i+1]) for i,k in enumerate(NAMES)},
                       "total_logits_grad_norm": float(global_mean(total_gradient.norm())[()]),
                       "loss_total": float(total.detach())}
            if step == 0 or (step+1) % self.config["gap_interval"] == 0:
                # Hard branch is diagnostic only and uses the actual seeded
                # categorical sample from the DiMO step, never soft argmax.
                with torch.no_grad():
                    hard = self.decoder.hard(hard_tokens)
                    hard_losses = self.rewards(hard, target, pixel_mask)
                    metrics["soft_hard_gap"] = {k:float(hard_losses[k]-losses[k]) for k in NAMES}
                    metrics["hard_reward_losses"] = {k:float(v) for k,v in hard_losses.items()}
                    metrics["soft_hard_pixel_l1"] = float((image-hard).abs().mean())
            if not torch.isfinite(total):
                raise RuntimeError("NONFINITE_SOFT_REWARD_TOTAL")
        return total, metrics

    def assert_frozen(self):
        if any(p.requires_grad or p.grad is not None for m in (self.decoder, self.rewards) for p in m.parameters()):
            raise RuntimeError("REWARD_FROZEN_PARAMETER_VIOLATION")


def build_reward_objective(vqvae, config, device):
    import lpips
    from transformers import AutoModel, CLIPModel
    freeze(vqvae)
    models = config["models"]
    rewards = EditingRewards(lpips.LPIPS(net="alex"),
        AutoModel.from_pretrained(models["dino"], local_files_only=True),
        CLIPModel.from_pretrained(models["clip"], local_files_only=True),
        lpips_resolution=config["lpips_resolution"]).to(device=device, dtype=torch.float32).eval()
    decoder = SoftEmbeddingEditDecoder(vqvae).to(device=device, dtype=torch.float32).eval()
    return SoftRewardObjective(decoder, rewards, config)
