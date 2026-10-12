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
V2_SCHEMA = "full-dense-dimo-ddp-v3"
SCHEMAS = (SCHEMA, V2_SCHEMA)


def checkpoint_schema(identity):
    return V2_SCHEMA if identity.get("recipe") == "dimo-v2" else SCHEMA


def validate_v2_state(identity, roles, ema, *, step, cursor, runtime_stats, sampler_state=None):
    from .recipe import role_parameter_contract
    contract = identity["recipe_contract"]
    if (role_parameter_contract(roles) != contract["optimizer_parameters"]
            or set(ema.shadow) != set(contract["ema_parameter_names"])
            or cursor != step * contract["budget"]["samples_per_optimizer_step"]
            or not 0 <= step <= contract["budget"]["optimizer_steps"]
            or type(step) is not int or type(cursor) is not int
            or runtime_stats is None
            or set(runtime_stats) != {"student_clip_count"}
            or type(runtime_stats["student_clip_count"]) is not int
            or not 0 <= runtime_stats["student_clip_count"] <= step):
        raise RuntimeError("DIMO_V2_CHECKPOINT_RECIPE_STATE_MISMATCH")
    if sampler_state is not None:
        rows = contract["budget"]["rows_per_epoch"]
        quantum = contract["budget"]["samples_per_optimizer_step"]
        if (sampler_state["epoch"] * rows + sampler_state["samples_consumed_in_epoch"] != cursor
                or sampler_state["samples_consumed_in_epoch"] % quantum):
            raise RuntimeError("DIMO_V2_CHECKPOINT_SAMPLER_CURSOR_MISMATCH")


def validate_optimizer_states(optimizers, saved, step):
    for optimizer, state in zip(optimizers, saved):
        groups = state["param_groups"]
        if len(groups) != len(optimizer.param_groups):
            raise RuntimeError("DIMO_V2_OPTIMIZER_GROUP_MISMATCH")
        ids, parameters = [], []
        for group, current in zip(groups, optimizer.param_groups):
            if ({k: v for k, v in group.items() if k != "params"}
                    != {k: v for k, v in current.items() if k != "params"}
                    or len(group["params"]) != len(current["params"])):
                raise RuntimeError("DIMO_V2_OPTIMIZER_MEMBERSHIP_MISMATCH")
            ids.extend(group["params"]); parameters.extend(current["params"])
        if len(set(ids)) != len(ids) or set(state["state"]) != (set(ids) if step else set()):
            raise RuntimeError("DIMO_V2_OPTIMIZER_STATE_COVERAGE_MISMATCH")
        for index, parameter in zip(ids, parameters):
            if not step:
                continue
            values = state["state"][index]
            if (set(values) != {"step", "exp_avg", "exp_avg_sq"}
                    or values["step"].numel() != 1 or values["step"].item() != step
                    or any(values[k].shape != parameter.shape or values[k].dtype != torch.float32
                           or not torch.isfinite(values[k]).all() for k in ("exp_avg", "exp_avg_sq"))):
                raise RuntimeError("DIMO_V2_OPTIMIZER_TENSOR_MISMATCH")


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
    if marker.get("schema") not in SCHEMAS:
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
    if marker["schema"] != checkpoint_schema(identity):
        raise RuntimeError("FULL_DIMO_CHECKPOINT_RECIPE_SCHEMA_MISMATCH")
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
         commit, sampler_state=None, loader_rng=None, runtime_stats=None, shard_bytes=512*1024**2, inference_only=False):
    commit.require_checkpointable(step, cursor)
    schema = checkpoint_schema(identity)
    if schema == V2_SCHEMA:
        error = None
        try:
            validate_v2_state(identity, roles, ema, step=step, cursor=cursor, runtime_stats=runtime_stats,
                              sampler_state=sampler_state)
            if not inference_only and sampler_state is None:
                raise RuntimeError("DIMO_V2_CHECKPOINT_SAMPLER_MISSING")
            roles.assert_frozen_unchanged(identity["frozen_parameter_hashes"])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        collective_status(error)
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
                torch.save({"schema": schema, "step": step, "cursor": cursor, "epoch": (sampler_state or {}).get('epoch',0),
                    "identity": identity, "config_hash": config_hash, "rank_rng": rng,
                    "sampler_state": sampler_state, "loader_rng": loader_rng,
                    **({"runtime_stats": runtime_stats} if schema == V2_SCHEMA else {}),
                    "optimizers": [o.state_dict() for o in optimizers],
                    "schedulers": [s.state_dict() for s in schedulers],
                    "ema_decay": ema.decay, "ema_update_count": ema.update_count}, temporary / "training_state.pt")
            for path in temporary.iterdir():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            ledger = {p.name: sha256_file(p) for p in temporary.iterdir()}
            write_json(temporary / "COMPLETED.json", {"schema": schema, "files": ledger, "committed_step": step})
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
    if (state.get("schema") != checkpoint_schema(expected_identity) or state["identity"] != expected_identity
            or state["config_hash"] != expected_config_hash or state["ema_update_count"] != state["step"]
            or state["ema_decay"] != ema.decay or len(state["optimizers"]) != 2 or len(state["schedulers"]) != 2):
        raise RuntimeError("FULL_DIMO_RESUME_STATE_MISMATCH")
    if state["schema"] == V2_SCHEMA:
        if state["sampler_state"] is None:
            raise RuntimeError("DIMO_V2_RESUME_SAMPLER_MISSING")
        validate_v2_state(expected_identity, roles, ema, step=state["step"], cursor=state["cursor"],
                          runtime_stats=state.get("runtime_stats"), sampler_state=state["sampler_state"])
        validate_optimizer_states(optimizers, state["optimizers"], state["step"])
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
    if state["schema"] == V2_SCHEMA:
        roles.assert_frozen_unchanged(expected_identity["frozen_parameter_hashes"])
    restore_rank_rng_state(state["rank_rng"])
    return state
