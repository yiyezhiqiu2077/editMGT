"""Collective complete-step DiMO resume and optimizer-free inference sidecars."""
import json
import os
from pathlib import Path
import tempfile
import torch
import torch.distributed as dist
from safetensors.torch import save_file, load_file
from src.explicit_region.checkpoint import capture_rank_rng_state, restore_rank_rng_state, recipe_fingerprint
from src.explicit_region.dense_checkpoint import write_json
from src.explicit_region.contracts import sha256_file


def save_checkpoint(root, *, roles, optimizers, schedulers, ema, step, cursor, identity, config_hash,
                    commit=None, sampler_state=None, loader_rng=None, inference_only=False):
    if getattr(roles, "backend", None) == "full_dense":
        from .full_checkpoint import save
        if commit is None:
            raise RuntimeError("FULL_DIMO_COMMIT_TOKEN_REQUIRED")
        return save(root, roles=roles, optimizers=optimizers, schedulers=schedulers, ema=ema,
            step=step, cursor=cursor, identity=identity, config_hash=config_hash, commit=commit,
            sampler_state=sampler_state, loader_rng=loader_rng, inference_only=inference_only)
    rng = capture_rank_rng_state()
    rank = dist.get_rank() if dist.is_initialized() else 0
    error = None
    if rank == 0:
        try:
            root = Path(root)
            if root.exists():
                raise FileExistsError(root)
            root.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.partial-", dir=root.parent))
            for role in ("student", "auxiliary"):
                save_file(roles.role_state_dict(role), temporary / f"{role}.safetensors")
            save_file(ema.state_dict()["shadow"], temporary / "ema.safetensors")
            write_json(temporary / "inference_identity.json", identity)
            torch.save({"schema": "dense-teacher-dimo-ddp-v1", "step": step, "cursor": cursor,
                "identity": identity, "config_hash": config_hash, "rank_rng": rng,
                "optimizers": [o.state_dict() for o in optimizers],
                "schedulers": [s.state_dict() for s in schedulers], "ema_decay": ema.decay}, temporary / "training_state.pt")
            ledger = {p.name: sha256_file(p) for p in temporary.iterdir()}
            write_json(temporary / "COMPLETED.json", {"schema": "dense-teacher-dimo-ddp-v1", "files": ledger})
            os.rename(temporary, root)
        except Exception as exc:
            error = str(exc)
    errors = [error]
    if dist.is_initialized():
        dist.broadcast_object_list(errors, src=0)
    if errors[0]:
        raise RuntimeError(f"DENSE_DIMO_CHECKPOINT_FAILED: {errors[0]}")


def load_inference(root, *, roles, expected_identity, weights="student", full=False):
    root = Path(root)
    marker = json.loads((root / "COMPLETED.json").read_text())
    if marker.get("schema") == "full-dense-dimo-ddp-v2":
        from .full_checkpoint import load_weights, verify
        verify(root, full=full, weights=weights, expected_identity=expected_identity)
        return load_weights(root, roles.model_for("student"), weights=weights, expected_identity=expected_identity)
    if marker.get("schema") != "dense-teacher-dimo-ddp-v1":
        raise RuntimeError("DENSE_DIMO_CHECKPOINT_INCOMPLETE")
    names = list(marker["files"]) if full else ["inference_identity.json", f"{weights}.safetensors"]
    for name in names:
        if Path(name).name != name or sha256_file(root / name) != marker["files"].get(name):
            raise RuntimeError("DENSE_DIMO_FILE_IDENTITY_MISMATCH")
    if json.loads((root / "inference_identity.json").read_text()) != expected_identity:
        raise RuntimeError("DENSE_DIMO_TEACHER_IDENTITY_MISMATCH")
    roles.load_role_state_dict("student", load_file(root / f"{weights}.safetensors"))


def load_checkpoint(root, *, roles, optimizers, schedulers, ema, expected_identity, expected_config_hash):
    if getattr(roles, "backend", None) == "full_dense":
        from .full_checkpoint import load
        return load(root, roles=roles, optimizers=optimizers, schedulers=schedulers, ema=ema,
            expected_identity=expected_identity, expected_config_hash=expected_config_hash)
    load_inference(root, roles=roles, expected_identity=expected_identity, full=True)
    state = torch.load(Path(root) / "training_state.pt", map_location="cpu", weights_only=False)
    if state["identity"] != expected_identity or state["config_hash"] != expected_config_hash:
        raise RuntimeError("DENSE_DIMO_RESUME_IDENTITY_MISMATCH")
    roles.load_role_state_dict("auxiliary", load_file(Path(root) / "auxiliary.safetensors"))
    for o, value in zip(optimizers, state["optimizers"]):
        o.load_state_dict(value)
    for s, value in zip(schedulers, state["schedulers"]):
        s.load_state_dict(value)
    ema.load_state_dict({"decay": state["ema_decay"], "shadow": load_file(Path(root) / "ema.safetensors")})
    restore_rank_rng_state(state["rank_rng"])
    return state
