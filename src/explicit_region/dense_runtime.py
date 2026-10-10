"""Dense-only safety, low-frequency diagnostics and read-only admission."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import torch
import torch.distributed as dist
from safetensors import safe_open

from .checkpoint import recipe_fingerprint
from .contracts import sha256_file
from .dense_checkpoint import write_json
from .quality import QualityGate


def validate_dense_config(config, *, formal=False):
    expected = {"training_mode": "full_transformer", "precision": "bf16", "resolution": 1024,
                "batch_per_gpu": 1, "gradient_accumulation": 4, "scheduler_horizon_steps": 31250,
                "warmup_steps": 200, "gradient_checkpointing": True, "attention_backend": "math_sdp"}
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"DENSE_CONFIG_MISMATCH: {key}")
    if config.get("lora", {}).get("enabled") is not False:
        raise RuntimeError("DENSE_LORA_MUST_BE_DISABLED")
    if config.get("component_dtypes") != {"transformer": "fp32", "text_encoders": "bf16", "vqvae": "fp32"}:
        raise RuntimeError("DENSE_DTYPE_CONFIG_MISMATCH")
    opt = config["optimizer"]
    for key, value in {"learning_rate": 2e-6, "betas": [0.9, 0.95], "weight_decay": .01,
                       "gradient_clip": 1.0, "scheduler": "constant_with_warmup", "name": "AdamW"}.items():
        if opt.get(key) != value:
            raise RuntimeError(f"DENSE_OPTIMIZER_MISMATCH: {key}")
    if config["corruption"] != {"mode": "roi_hardlock", "roi_probability": 1.0,
                                "full_roi_mask_probability": .15, "persistent_conditioning": True}:
        raise RuntimeError("DENSE_E3_CORRUPTION_MISMATCH")
    if config["condition_dropout"] != {"enabled": True, "probability": .1, "scope": "text_only"}:
        raise RuntimeError("DENSE_CONDITION_DROPOUT_MISMATCH")
    if config["token_mask"] != {"mode": "any_overlap", "coverage_threshold": .5,
                                "dilation_tokens": 0, "minimum_edit_tokens": 1}:
        raise RuntimeError("DENSE_TOKEN_MASK_MISMATCH")
    if formal and (config["data"]["name"] != "fixed_200k" or config["epochs"] != 5 or config["max_optimizer_steps"] != 31250):
        raise RuntimeError("DENSE_FORMAL_BUDGET_MISMATCH")
    return {"status": "PASS", "config_sha256": recipe_fingerprint(config)}


def audit_released_fp32(root):
    folder = Path(root) / "editmgt"
    index = folder / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        files = ["diffusion_pytorch_model.safetensors"]
    tensors, total = {}, 0
    for name in files:
        if Path(name).name != name:
            raise RuntimeError("RELEASED_UNSAFE_SHARD_NAME")
        with safe_open(folder / name, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_slice(key)
                if tensor.get_dtype() != "F32":
                    raise RuntimeError(f"RELEASED_DTYPE_NOT_FP32: {key}")
                if key in tensors or "lora_" in key:
                    raise RuntimeError("RELEASED_DUPLICATE_OR_LORA_KEY")
                tensors[key] = tensor.get_shape()
                import math
                total += math.prod(tensor.get_shape())
    return {"status": "PASS", "load_dtype": "torch.float32", "cast_via_bf16": False,
            "stored_numel": total, "stored_tensors": tensors,
            "files": {name: sha256_file(folder / name) for name in files}}


def check_direct_initialization(model, root, audit):
    named = model.state_dict()
    missing = set(named) - set(audit["stored_tensors"])
    if missing != {"edit_region_embedding"} or not torch.equal(model.edit_region_embedding.detach().cpu(), torch.zeros_like(model.edit_region_embedding.detach().cpu())):
        raise RuntimeError("DENSE_RELEASED_INITIALIZATION_KEYS_MISMATCH")
    for name in audit["files"]:
        with safe_open(Path(root) / "editmgt" / name, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key not in named or named[key].dtype != torch.float32 or not torch.equal(named[key].detach().cpu(), handle.get_tensor(key)):
                    raise RuntimeError(f"DENSE_DIRECT_LOAD_MISMATCH: {key}")
    return {"status": "PASS", "equivalence": "bitwise_exact", "region_initialization": "zeros"}


def trainable_report(components, optimizer):
    model = components.transformer
    rows = [{"name": n, "numel": p.numel(), "dtype": str(p.dtype), "requires_grad": p.requires_grad}
            for n, p in model.named_parameters()]
    if not rows or any(not r["requires_grad"] or r["dtype"] != "torch.float32" or "lora_" in r["name"] for r in rows):
        raise RuntimeError("DENSE_TRAINABLE_COVERAGE_FAILURE")
    expected = {id(p) for p in model.parameters()}
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    if len(actual) != len(expected) or {id(p) for p in actual} != expected:
        raise RuntimeError("DENSE_OPTIMIZER_COVERAGE_FAILURE")
    frozen = {}
    for name in ("text_encoder", "llm_encoder", "vqvae"):
        module = getattr(components, name)
        if any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise RuntimeError(f"DENSE_ENCODER_NOT_FROZEN: {name}")
        frozen[name] = sum(p.numel() for p in module.parameters())
    return {"status": "PASS", "training_mode": "full_transformer", "no_lora_parameters": True,
            "trainable_numel": sum(r["numel"] for r in rows),
            "optimizer_numel": sum(p.numel() for p in actual), "frozen": frozen, "parameters": rows}


def collective_require(condition, message, device):
    flag = torch.tensor(int(bool(condition)), device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise RuntimeError(message)


def model_digest(model):
    digest = hashlib.sha256()
    # CPU transfer per tensor only at explicit low-frequency verification points.
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_rank_parameters(model):
    digest = model_digest(model)
    world = dist.get_world_size() if dist.is_initialized() else 1
    values = [digest]
    if world > 1:
        values = [None] * world
        dist.all_gather_object(values, digest)
    if any(value != digest for value in values):
        raise RuntimeError("DENSE_DDP_PARAMETER_DESYNCHRONIZATION")
    return {"status": "PASS", "world_size": world, "sha256": digest}


def gradient_report(model, *, allowed_unused=()):
    missing, dtypes, finite, norms = [], {}, [], []
    for name, p in model.named_parameters():
        if p.grad is None:
            missing.append(name)
        else:
            dtypes[str(p.grad.dtype)] = dtypes.get(str(p.grad.dtype), 0) + p.numel()
            finite.append(torch.isfinite(p.grad).all())
            norms.append(p.grad.detach().norm())
    device = next(model.parameters()).device
    collective_require(bool(finite) and bool(torch.stack(finite).all()), "DENSE_NONFINITE_GRADIENT", device)
    collective_require(set(missing) <= set(allowed_unused), f"DENSE_UNEXPLAINED_UNUSED: {missing}", device)
    if any(dtype != "torch.float32" for dtype in dtypes):
        raise RuntimeError("DENSE_GRADIENT_NOT_FP32")
    region = model.edit_region_embedding.grad
    collective_require(region is not None and bool(region.detach().norm() > 0), "DENSE_REGION_GRADIENT_MISSING", device)
    return {"grad_none_parameters": missing, "gradient_dtypes": dtypes,
            "gradient_norm": float(torch.stack(norms).norm()),
            "region_gradient_norm": float(region.detach().norm())}


class UpdateSample:
    """Explicitly sampled representatives, not a fabricated full-model norm."""
    def __init__(self, model, sample_elements=4096):
        self.model = model
        self.before = {n: p.detach().reshape(-1)[:sample_elements].clone()
                       for n, p in model.named_parameters()}

    def finish(self, *, require_update=True):
        named = dict(self.model.named_parameters())
        deltas = [((named[n].detach().reshape(-1)[:v.numel()] - v).square().sum())
                  for n, v in self.before.items()]
        bases = [v.square().sum() for v in self.before.values()]
        norm = float(torch.stack(deltas).sum().sqrt())
        denominator = float(torch.stack(bases).sum().sqrt())
        if norm == 0 and require_update:
            raise RuntimeError("DENSE_NO_REPRESENTATIVE_UPDATE")
        return {"dense_update_norm": norm, "dense_update_ratio": norm / max(denominator, 1e-12),
                "update_diagnostic_scope": "first_4096_elements_per_parameter",
                "representative_module_updates": {n: float(d.sqrt()) for n, d in zip(self.before, deltas)}}


class DenseQualityGate:
    """Reuse the CE gate; explicitly distinguish sparse safety observations."""
    def __init__(self, config, state=None):
        self.gate = QualityGate.from_state(config, state.get("gate") if state else None)
        self.last_diagnostic_step = state.get("last_diagnostic_step") if state else None
        self.config = config

    def update(self, step, *, loss, max_abs_logit, clip_applied, update_ratio=None, finite_parameters=None):
        import math
        if finite_parameters is False or (update_ratio is not None and
                (not math.isfinite(update_ratio) or update_ratio > self.config.get("max_update_ratio", float("inf")))):
            self.gate.status, self.gate.reason = "QUALITY_FAILED", "dense_sampled_update_or_parameters"
            return False
        if finite_parameters is not None:
            self.last_diagnostic_step = step
        # The old gate's complete-update metric is not silently substituted by
        # the representative metric: disable that check in the reused CE gate.
        gate_config = self.gate.config
        self.gate.config = dict(gate_config, max_update_ratio=float("inf"))
        result = self.gate.update(step, loss=loss, max_abs_logit=max_abs_logit,
                                  clip_applied=clip_applied, update_ratio=None, finite_parameters=finite_parameters)
        self.gate.config = gate_config
        return result

    def state_dict(self):
        return {"gate": self.gate.state_dict(), "last_diagnostic_step": self.last_diagnostic_step,
                "full_model_update_ratio": "NOT_COLLECTED", "safety_coverage": "LOSS_EACH_STEP_SAMPLED_PARAMETERS"}


def memory_record(device, phase, step=0):
    if device.type != "cuda":
        return {"phase": phase, "step": step, "allocated": None, "reserved": None, "peak": None}
    torch.cuda.synchronize(device)
    return {"phase": phase, "step": step, "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device), "peak": torch.cuda.max_memory_allocated(device),
            "peak_reserved": torch.cuda.max_memory_reserved(device)}


def readonly_reattest(config, output, *, git_sha, full_loader=True):
    """Write only a Dense-owned evidence file; never publish shared READY files."""
    from .fixed_corpus import verify_corpus_ready, verify_mini_corpus_ready
    from .fixed_dataset import FixedCorpusDataset, CanonicalAlignedDataset
    from .reattestation import frozen_identity, read_rows
    data = config["data"]
    ready_path = Path(data["corpus_ready"])
    before = sha256_file(ready_path)
    if data["name"] == "fixed_200k":
        ready = verify_corpus_ready(ready_path)
        frozen, rows, validation = frozen_identity(ready)
    else:
        ready = verify_mini_corpus_ready(ready_path, expected_rows=data["expected_rows"])
        rows = read_rows(data["manifest"])
        frozen = {"train_sha256": sha256_file(data["manifest"]), "rows": len(rows)}
        validation = {"periodic": Path(config["validation"]["manifest"])} if config["validation"].get("enabled") else {}
    if frozen["train_sha256"] != sha256_file(data["manifest"]):
        raise RuntimeError("DENSE_READY_MANIFEST_MISMATCH")
    actual_count = len(rows)
    if actual_count != (200000 if data["name"] == "fixed_200k" else data["expected_rows"]):
        raise RuntimeError("DENSE_CORPUS_ROWS_MISMATCH")
    train_sources = {r["source_sha256"] for r in rows}
    train_groups = {(r["dataset_name"], r["group_id"]) for r in rows}
    for path in validation.values():
        for row in read_rows(path):
            if row["source_sha256"] in train_sources or (row["dataset_name"], row["group_id"]) in train_groups:
                raise RuntimeError("DENSE_TRAIN_DEV_LEAKAGE")
    if full_loader:
        dataset = FixedCorpusDataset(data["manifest"], data["roots"], resolution=config["resolution"],
                                      base_seed=config["seed"], verify_hashes=True,
                                      max_random_attempts=config["geometry"]["max_resample_attempts"],
                                      minimum_mask_retention=config["geometry"]["minimum_mask_retention"])
        for i in range(len(dataset)):
            dataset[i]
            if (i + 1) % 1000 == 0:
                print(f"DENSE_READONLY_LOADER_VERIFIED={i + 1}", flush=True)
        for path in validation.values():
            grouped = {}
            for row in read_rows(path):
                grouped.setdefault(row["dataset_name"], []).append(row)
            for name, samples in grouped.items():
                root = config["validation"]["dataset_root"] if name == "magicbrush" else data["roots"][name]
                dev = CanonicalAlignedDataset(samples, root, resolution=config["resolution"], base_seed=config["seed"], verify_hashes=True)
                for i in range(len(dev)):
                    dev[i]
    if sha256_file(ready_path) != before or sha256_file(data["manifest"]) != frozen["train_sha256"]:
        raise RuntimeError("DENSE_SHARED_DATA_CHANGED_DURING_REATTEST")
    result = {"schema": "dense-readonly-reattest-v1", "status": "PASS" if full_loader else "IDENTITY_ONLY",
              "git_sha": git_sha, "config_sha256": recipe_fingerprint(config), "frozen": frozen,
              "shared_ready_sha256": before, "shared_ready_unchanged": True,
              "loader": "PASS" if full_loader else "NOT_RUN",
              "validation": {str(p): sha256_file(p) for p in validation.values()}}
    write_json(output, result)
    return result
