"""Versioned sharded FP32 checkpoints; inference never unpickles training state."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .checkpoint import (capture_rank_rng_state, restore_rank_rng_state,
                         distributed_rank_world, recipe_fingerprint, validate_rank_rng_state)
from .contracts import sha256_file

SCHEMA = "editmgt-dense-checkpoint-v1"


def write_json(path, value):
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def architecture_of(model):
    config = dict(model.config) if hasattr(model, "config") else {
        "class": type(model).__name__,
        "parameters": {n: list(v.shape) for n, v in model.state_dict().items()},
    }
    # Diffusers path/version metadata is not part of the architecture; tuples
    # must compare equally with their JSON list representation after reload.
    return json.loads(json.dumps({k: v for k, v in config.items() if not k.startswith("_")}))


def _safe_path(root, name):
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise RuntimeError("DENSE_UNSAFE_FILE_LEDGER")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"DENSE_MISSING_OR_UNSAFE_FILE: {name}")
    return path


def verify_dense_checkpoint(root, *, inference_only=True, expected_fingerprint=None):
    root = Path(root)
    try:
        completed = json.loads((root / "COMPLETED.json").read_text())
        identity = json.loads((root / "identity.json").read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("DENSE_CHECKPOINT_INCOMPLETE") from exc
    if completed.get("schema") != SCHEMA or identity.get("schema") != SCHEMA:
        raise RuntimeError("DENSE_CHECKPOINT_SCHEMA_MISMATCH")
    ledger = completed.get("files", {})
    required = identity.get("inference_files", {})
    if not required or not set(required) <= set(ledger) or "identity.json" not in ledger:
        raise RuntimeError("DENSE_CHECKPOINT_LEDGER_INCOMPLETE")
    to_check = dict(required) | {"identity.json": ledger["identity.json"]} if inference_only else ledger
    for name, digest in to_check.items():
        if sha256_file(_safe_path(root, name)) != digest or ledger.get(name) != digest:
            raise RuntimeError(f"DENSE_CHECKPOINT_HASH_MISMATCH: {name}")
    if identity["checkpoint_sha256"] != recipe_fingerprint(required):
        raise RuntimeError("DENSE_CHECKPOINT_IDENTITY_MISMATCH")
    payload = json.loads((root / "fingerprint.json").read_text())
    if payload.get("sha256") != recipe_fingerprint(payload["payload"]):
        raise RuntimeError("DENSE_RECIPE_HASH_MISMATCH")
    if expected_fingerprint is not None and payload["sha256"] != recipe_fingerprint(expected_fingerprint):
        raise RuntimeError("DENSE_RESUME_FINGERPRINT_MISMATCH")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    if not index.get("weight_map") or not set(index["weight_map"].values()) <= set(required):
        raise RuntimeError("DENSE_SHARD_LEDGER_MISMATCH")
    return identity


def save_dense_checkpoint(root, *, model, fingerprint_payload, optimizer=None, scheduler=None,
                          global_optimizer_step=0, committed_global_sample_count=0,
                          sampler_state=None, quality_state=None, loader_generator=None,
                          max_shard_bytes=512 * 1024**2):
    """Collective call at successful update boundaries; publish only rank0.

    Only one CPU weight shard is materialized at a time. No GPU clone of the
    full transformer, no optimizer duplication per rank. Existing outputs are
    never overwritten. Errors on rank0 are broadcast instead of deadlocking.
    """
    rank, world = distributed_rank_world()
    rng = capture_rank_rng_state()
    error = None
    identity = None
    if rank == 0:
        try:
            root = Path(root)
            if root.exists():
                raise FileExistsError(root)
            if max_shard_bytes <= 0:
                raise ValueError("max_shard_bytes must be positive")
            root.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.partial-", dir=root.parent))
            state = model.state_dict()
            weight_map, shard, size, number = {}, {}, 0, 0

            def flush():
                nonlocal shard, size, number
                if not shard:
                    return
                number += 1
                name = f"model-{number:05d}.safetensors"
                save_file(shard, temporary / name)
                weight_map.update({key: name for key in shard})
                shard, size = {}, 0

            for name, value in sorted(state.items()):
                if "lora_" in name or ".adapter" in name:
                    raise RuntimeError("DENSE_CHECKPOINT_CONTAINS_ADAPTER")
                if value.is_floating_point() and value.dtype != torch.float32:
                    raise RuntimeError(f"DENSE_WEIGHT_NOT_FP32: {name}")
                nbytes = value.numel() * value.element_size()
                if size and size + nbytes > max_shard_bytes:
                    flush()
                shard[name] = value.detach().cpu().contiguous().clone()
                size += nbytes
            flush()
            if "edit_region_embedding" not in weight_map:
                raise RuntimeError("DENSE_REGION_EMBEDDING_MISSING")
            write_json(temporary / "model.safetensors.index.json", {"weight_map": weight_map})
            write_json(temporary / "architecture.json", architecture_of(model))
            write_json(temporary / "fingerprint.json", {
                "payload": fingerprint_payload, "sha256": recipe_fingerprint(fingerprint_payload)})
            write_json(temporary / "metadata.json", {
                "global_optimizer_step": int(global_optimizer_step),
                "committed_global_sample_count": int(committed_global_sample_count),
                "world_size": world, "sampler_state": sampler_state,
                "training_mode": "full_transformer"})
            files = {p.name: sha256_file(p) for p in temporary.iterdir()}
            identity = {"schema": SCHEMA, "checkpoint_sha256": recipe_fingerprint(files),
                        "inference_files": files, "has_training_state": optimizer is not None}
            write_json(temporary / "identity.json", identity)
            if optimizer is not None:
                torch.save({"schema": SCHEMA, "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "rank_rng": rng,
                            "sampler_state": sampler_state, "quality_state": quality_state,
                            "global_optimizer_step": int(global_optimizer_step),
                            "committed_global_sample_count": int(committed_global_sample_count),
                            "loader_generator": loader_generator.get_state() if loader_generator is not None else None,
                            "fingerprint": recipe_fingerprint(fingerprint_payload)},
                           temporary / "training_state.pt")
            ledger = {p.name: sha256_file(p) for p in temporary.iterdir()}
            # Flush file contents before publishing the completion marker.
            for path in temporary.iterdir():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            write_json(temporary / "COMPLETED.json", {"schema": SCHEMA, "files": ledger})
            verify_dense_checkpoint(temporary, inference_only=False)
            os.rename(temporary, root)
            fd = os.open(root.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    result = [error, identity]
    if world > 1:
        dist.broadcast_object_list(result, src=0)
    if result[0]:
        raise RuntimeError(f"DENSE_CHECKPOINT_PUBLICATION_FAILED: {result[0]}")
    return result[1]


def load_dense_weights(model, root, *, expected_fingerprint=None):
    identity = verify_dense_checkpoint(root, expected_fingerprint=expected_fingerprint)
    root = Path(root)
    if architecture_of(model) != json.loads((root / "architecture.json").read_text()):
        raise RuntimeError("DENSE_ARCHITECTURE_MISMATCH")
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    current = model.state_dict()
    if set(index) != set(current):
        raise RuntimeError("DENSE_MODEL_KEYS_MISMATCH")
    # Prevalidate every key/shape/dtype before mutating the model.
    for filename in sorted(set(index.values())):
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            expected = {key for key, value in index.items() if value == filename}
            if set(handle.keys()) != expected:
                raise RuntimeError("DENSE_SHARD_KEYS_MISMATCH")
            for key in expected:
                tensor = handle.get_slice(key)
                if list(current[key].shape) != tensor.get_shape():
                    raise RuntimeError(f"DENSE_MODEL_SHAPE_MISMATCH: {key}")
                if current[key].is_floating_point() and (tensor.get_dtype() != "F32" or current[key].dtype != torch.float32):
                    raise RuntimeError(f"DENSE_MODEL_DTYPE_MISMATCH: {key}")
    with torch.no_grad():
        for filename in sorted(set(index.values())):
            tensors = load_file(root / filename)
            if any(not bool(torch.isfinite(v).all()) for v in tensors.values() if v.is_floating_point()):
                raise RuntimeError("DENSE_NONFINITE_WEIGHT")
            for key, value in tensors.items():
                current[key].copy_(value.to(current[key].device))
            del tensors
    return identity


def read_dense_resume(root, *, fingerprint_payload, world_size):
    identity = verify_dense_checkpoint(root, inference_only=False, expected_fingerprint=fingerprint_payload)
    if not identity["has_training_state"]:
        raise RuntimeError("DENSE_INFERENCE_ONLY_CHECKPOINT")
    # Only trusted, locally-produced training checkpoints may be unpickled.
    state = torch.load(Path(root) / "training_state.pt", map_location="cpu", weights_only=False)
    if state.get("schema") != SCHEMA or state.get("fingerprint") != recipe_fingerprint(fingerprint_payload):
        raise RuntimeError("DENSE_RESUME_STATE_MISMATCH")
    metadata = json.loads((Path(root) / "metadata.json").read_text())
    for key in ("global_optimizer_step", "committed_global_sample_count", "sampler_state"):
        if state[key] != metadata[key]:
            raise RuntimeError(f"DENSE_RESUME_METADATA_MISMATCH: {key}")
    validate_rank_rng_state(state["rank_rng"], world_size=world_size)
    return state


def restore_dense_resume(model, optimizer, scheduler, state, root, *, loader_generator=None):
    load_dense_weights(model, root)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if loader_generator is not None and state["loader_generator"] is not None:
        loader_generator.set_state(state["loader_generator"])
    restore_rank_rng_state(state["rank_rng"])
    return state
