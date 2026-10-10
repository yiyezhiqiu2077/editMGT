"""Streaming role weights, atomic publication and complete-step state."""
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from src.explicit_region.checkpoint import capture_rank_rng_state, restore_rank_rng_state
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dense_checkpoint import write_json, architecture_of
from .transaction import collective_status

SCHEMA = "full-dense-dimo-ddp-v2"


def write_weights(root, role, tensors, shard_bytes):
    shard, weight_map, size, count = {}, {}, 0, 0
    def flush():
        nonlocal shard, size, count
        if not shard:
            return
        count += 1
        filename = f"{role}-{count:05d}.safetensors"
        save_file(shard, root / filename)
        weight_map.update({name: filename for name in shard})
        shard, size = {}, 0
    for name, value in tensors.items():
        nbytes = value.numel() * value.element_size()
        if shard and size + nbytes > shard_bytes:
            flush()
        if value.is_floating_point() and value.dtype != torch.float32:
            raise RuntimeError("FULL_DIMO_CHECKPOINT_NOT_FP32")
        shard[name] = value.detach().cpu().contiguous().clone()
        if value.is_floating_point() and not bool(torch.isfinite(shard[name]).all()):
            raise RuntimeError("FULL_DIMO_CHECKPOINT_NONFINITE_WEIGHT")
        size += nbytes
    flush()
    write_json(root / f"{role}.index.json", {"weight_map": weight_map})


def verify(root, *, full=False, weights="student", expected_identity=None):
    root = Path(root)
    marker = json.loads((root / "COMPLETED.json").read_text())
    if marker.get("schema") != SCHEMA:
        raise RuntimeError("FULL_DIMO_CHECKPOINT_INCOMPLETE")
    ledger = marker["files"]
    if weights not in ("student", "ema", "auxiliary"):
        raise ValueError("unknown weights")
    names = list(ledger) if full else ["inference_identity.json", "architecture.json", f"{weights}.index.json"]
    for name in names:
        if Path(name).name != name or (root / name).is_symlink() or sha256_file(root / name) != ledger.get(name):
            raise RuntimeError("FULL_DIMO_CHECKPOINT_INTEGRITY_FAILURE")
    index = json.loads((root / f"{weights}.index.json").read_text())["weight_map"]
    if not index or "edit_region_embedding" not in index:
        raise RuntimeError("FULL_DIMO_WEIGHT_INDEX_INCOMPLETE")
    for filename in set(index.values()):
        if Path(filename).name != filename or sha256_file(root / filename) != ledger.get(filename):
            raise RuntimeError("FULL_DIMO_WEIGHT_SHARD_CORRUPTED")
    identity = json.loads((root / "inference_identity.json").read_text())
    if expected_identity is not None and identity != expected_identity:
        raise RuntimeError("FULL_DIMO_IDENTITY_MISMATCH")
    return identity, index


@torch.no_grad()
def load_weights(root, model, *, weights="student", expected_identity=None):
    _, index = verify(root, weights=weights, expected_identity=expected_identity)
    if json.loads((Path(root) / "architecture.json").read_text()) != architecture_of(model):
        raise RuntimeError("FULL_DIMO_ARCHITECTURE_MISMATCH")
    current = model.state_dict()
    if set(current) != set(index):
        raise RuntimeError("FULL_DIMO_PARAMETER_COVERAGE_MISMATCH")
    for filename in sorted(set(index.values())):
        shard = load_file(Path(root) / filename)
        if set(shard) != {n for n, f in index.items() if f == filename}:
            raise RuntimeError("FULL_DIMO_SHARD_KEYS_MISMATCH")
        for name, value in shard.items():
            target = current[name]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise RuntimeError(f"FULL_DIMO_TENSOR_IDENTITY_MISMATCH: {name}")
            target.copy_(value.to(target.device))


def save(root, *, roles, optimizers, schedulers, ema, step, cursor, identity, config_hash,
         commit, sampler_state=None, loader_rng=None, shard_bytes=512*1024**2, inference_only=False):
    commit.require_checkpointable(step, cursor)
    ema.synchronize()
    if ema.update_count != step:
        raise RuntimeError("FULL_DIMO_EMA_COMMIT_COUNT_MISMATCH")
    if shard_bytes <= 0:
        raise ValueError("invalid shard size")
    rng = None if inference_only else capture_rank_rng_state()
    rank = dist.get_rank() if dist.is_initialized() else 0
    error = None
    if rank == 0:
        try:
            root = Path(root)
            if root.exists():
                raise FileExistsError(root)
            root.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.partial-", dir=root.parent))
            for role in (("student",) if inference_only else ("student", "auxiliary")):
                write_weights(temporary, role, roles.model_for(role).state_dict(), shard_bytes)
            # Copy buffers from live student; EMA covers every trainable parameter.
            ema_values = dict(roles.model_for("student").state_dict())
            ema_values.update(ema.shadow)
            write_weights(temporary, "ema", ema_values, shard_bytes)
            write_json(temporary / "architecture.json", architecture_of(roles.model_for("student")))
            write_json(temporary / "inference_identity.json", identity)
            if not inference_only:
                torch.save({"schema": SCHEMA, "step": step, "cursor": cursor, "epoch": 0,
                    "identity": identity, "config_hash": config_hash, "rank_rng": rng,
                    "sampler_state": sampler_state, "loader_rng": loader_rng,
                    "optimizers": [o.state_dict() for o in optimizers],
                    "schedulers": [s.state_dict() for s in schedulers],
                    "ema_decay": ema.decay, "ema_update_count": ema.update_count}, temporary / "training_state.pt")
            for path in temporary.iterdir():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            ledger = {p.name: sha256_file(p) for p in temporary.iterdir()}
            write_json(temporary / "COMPLETED.json", {"schema": SCHEMA, "files": ledger, "committed_step": step})
            verify(temporary, full=True, expected_identity=identity)
            os.rename(temporary, root)
            fd = os.open(root.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    collective_status(error)


def load(root, *, roles, optimizers, schedulers, ema, expected_identity, expected_config_hash):
    verify(root, full=True, expected_identity=expected_identity)
    state = torch.load(str(Path(root) / "training_state.pt"), map_location="cpu", weights_only=False, mmap=True)
    if (state.get("schema") != SCHEMA or state["identity"] != expected_identity
            or state["config_hash"] != expected_config_hash or state["ema_update_count"] != state["step"]
            or state["ema_decay"] != ema.decay or len(state["optimizers"]) != 2 or len(state["schedulers"]) != 2):
        raise RuntimeError("FULL_DIMO_RESUME_STATE_MISMATCH")
    for role in ("student", "auxiliary"):
        load_weights(root, roles.model_for(role), weights=role, expected_identity=expected_identity)
    for optimizer, saved in zip(optimizers, state["optimizers"]):
        optimizer.load_state_dict(saved)
    # Do not retain a second CPU copy of all AdamW moments for the whole run.
    del state["optimizers"]
    for scheduler, saved in zip(schedulers, state["schedulers"]):
        scheduler.load_state_dict(saved)
    _, index = verify(root, weights="ema", expected_identity=expected_identity)
    if not set(ema.shadow) <= set(index):
        raise RuntimeError("FULL_DIMO_EMA_COVERAGE_MISMATCH")
    for filename in sorted(set(index.values())):
        for name, value in load_file(Path(root) / filename).items():
            if name in ema.shadow:
                if value.shape != ema.shadow[name].shape or value.dtype != torch.float32:
                    raise RuntimeError("FULL_DIMO_EMA_TENSOR_MISMATCH")
                ema.shadow[name].copy_(value.to(ema.device))
    ema.update_count = state["ema_update_count"]
    restore_rank_rng_state(state["rank_rng"])
    return state
