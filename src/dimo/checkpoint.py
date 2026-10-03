"""Complete-step-only deterministic Region-DiMO checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .rng import stream_identity


SCHEMA_VERSION = "dimo-editing-checkpoint-v1"


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_dimo_checkpoint(
    output_dir: str | Path,
    *,
    roles,
    student_optimizer,
    auxiliary_optimizer,
    student_scheduler,
    auxiliary_scheduler,
    student_ema,
    committed_dimo_step: int,
    fixed_corpus_cursor: int,
    epoch: int,
    samples_consumed: int,
    config_sha256: str,
    teacher_checkpoint_hash: str,
    upstream_commit: str,
    step_phase: str = "complete",
) -> None:
    if step_phase != "complete":
        raise RuntimeError("DIMO_CHECKPOINT_REQUIRES_COMPLETE_STEP")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = {
        "schema": SCHEMA_VERSION,
        "step_phase": "complete",
        "global_dimo_step": int(committed_dimo_step),
        "student_optimizer": student_optimizer.state_dict(),
        "auxiliary_optimizer": auxiliary_optimizer.state_dict(),
        "student_scheduler": student_scheduler.state_dict(),
        "auxiliary_scheduler": auxiliary_scheduler.state_dict(),
        "student_state": roles.role_state_dict("student"),
        "auxiliary_state": roles.role_state_dict("auxiliary"),
        "student_ema": student_ema.state_dict(),
        "fixed_corpus_cursor": int(fixed_corpus_cursor),
        "epoch": int(epoch),
        "samples_consumed": int(samples_consumed),
        "rng_stream_identity": stream_identity(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        "config_hash": config_sha256,
        "teacher_checkpoint_hash": teacher_checkpoint_hash,
        "dimo_upstream_reference_commit": upstream_commit,
    }
    temporary = output / "training_state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(output / "training_state.pt")


def load_dimo_checkpoint(
    input_dir: str | Path,
    *,
    roles,
    student_optimizer,
    auxiliary_optimizer,
    student_scheduler,
    auxiliary_scheduler,
    student_ema,
    expected_config_hash: str,
    expected_teacher_hash: str,
    expected_upstream_commit: str,
) -> dict[str, Any]:
    state = torch.load(Path(input_dir) / "training_state.pt", map_location="cpu")
    if state.get("schema") != SCHEMA_VERSION or state.get("step_phase") != "complete":
        raise RuntimeError("invalid or partial DiMO checkpoint")
    checks = {
        "config_hash": expected_config_hash,
        "teacher_checkpoint_hash": expected_teacher_hash,
        "dimo_upstream_reference_commit": expected_upstream_commit,
        "rng_stream_identity": stream_identity(),
    }
    for key, expected in checks.items():
        if state.get(key) != expected:
            raise RuntimeError(f"DiMO checkpoint {key} mismatch")
    roles.load_role_state_dict("student", state["student_state"])
    roles.load_role_state_dict("auxiliary", state["auxiliary_state"])
    student_optimizer.load_state_dict(state["student_optimizer"])
    auxiliary_optimizer.load_state_dict(state["auxiliary_optimizer"])
    student_scheduler.load_state_dict(state["student_scheduler"])
    auxiliary_scheduler.load_state_dict(state["auxiliary_scheduler"])
    student_ema.load_state_dict(state["student_ema"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state["cuda_rng_state_all"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    return state
