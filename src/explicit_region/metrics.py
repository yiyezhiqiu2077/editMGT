"""Formal image-editing metrics with spatially masked aggregation."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _mask(mask, image):
    if mask.ndim == 3:
        mask = mask[:, None]
    return mask.to(device=image.device, dtype=image.dtype)


def masked_l1(a, b, mask, eps=1e-12):
    weight = _mask(mask, a)
    denominator = weight.sum((1, 2, 3)) * a.shape[1]
    valid = denominator > 0
    value = ((a - b).abs() * weight).sum((1, 2, 3)) / denominator.clamp_min(eps)
    return value, valid


def masked_psnr(a, b, mask, eps=1e-12):
    weight = _mask(mask, a)
    denominator = weight.sum((1, 2, 3)) * a.shape[1]
    valid = denominator > 0
    mse = (((a - b) ** 2) * weight).sum((1, 2, 3)) / denominator.clamp_min(eps)
    return -10 * torch.log10(mse.clamp_min(eps)), valid


def _gaussian_window(device, dtype, channels=3, size=11, sigma=1.5):
    x = torch.arange(size, device=device, dtype=dtype) - size // 2
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel)
    return window.expand(channels, 1, size, size).contiguous()


def ssim_map(a, b):
    window = _gaussian_window(a.device, a.dtype, a.shape[1])
    kwargs = dict(padding=5, groups=a.shape[1])
    mu_a, mu_b = F.conv2d(a, window, **kwargs), F.conv2d(b, window, **kwargs)
    mu_a2, mu_b2, mu_ab = mu_a.square(), mu_b.square(), mu_a * mu_b
    var_a = F.conv2d(a.square(), window, **kwargs) - mu_a2
    var_b = F.conv2d(b.square(), window, **kwargs) - mu_b2
    cov = F.conv2d(a * b, window, **kwargs) - mu_ab
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mu_ab + c1) * (2 * cov + c2) / ((mu_a2 + mu_b2 + c1) * (var_a + var_b + c2))).mean(1, keepdim=True)


def masked_ssim(a, b, mask, eps=1e-12):
    spatial = ssim_map(a, b)
    weight = F.interpolate(_mask(mask, a), size=spatial.shape[-2:], mode="area")
    denominator = weight.sum((1, 2, 3))
    valid = denominator > 0
    value = (spatial * weight).sum((1, 2, 3)) / denominator.clamp_min(eps)
    return value, valid


class MaskedLPIPS:
    """LPIPS learned feature distances aggregated with an area-resized ROI."""

    def __init__(self, net="alex", device="cpu"):
        import lpips
        self.model = lpips.LPIPS(net=net).to(device).eval()

    @torch.no_grad()
    def __call__(self, a, b, mask):
        import lpips
        a, b = a * 2 - 1, b * 2 - 1
        feats_a = self.model.net.forward(self.model.scaling_layer(a))
        feats_b = self.model.net.forward(self.model.scaling_layer(b))
        values, valid_levels = [], []
        for index, (fa, fb) in enumerate(zip(feats_a, feats_b)):
            distance_map = self.model.lins[index]((lpips.normalize_tensor(fa) - lpips.normalize_tensor(fb)) ** 2)
            weight = F.interpolate(_mask(mask, a), size=distance_map.shape[-2:], mode="area")
            denominator = weight.sum((1, 2, 3))
            valid_levels.append(denominator > 0)
            values.append((distance_map * weight).sum((1, 2, 3)) / denominator.clamp_min(1e-12))
        return torch.stack(values).sum(0), torch.stack(valid_levels).all(0)

    @torch.no_grad()
    def full(self, a, b):
        return self.model(a * 2 - 1, b * 2 - 1).flatten()


def cosine_similarity(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def finite_mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(values) / len(values) if values else None
