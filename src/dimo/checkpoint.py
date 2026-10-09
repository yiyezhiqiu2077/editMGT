"""Complete-step-only deterministic Region-DiMO checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch
import torch.distributed as dist

from src.explicit_region.checkpoint import (
    capture_rank_rng_state, distributed_rank_world, restore_rank_rng_state, validate_rank_rng_state,
)

from .rng import stream_identity


SCHEMA_VERSION = "dimo-editing-checkpoint-v1.1"
DISTRIBUTED_SCHEMA_VERSION = 'dimo-editing-checkpoint-v1.2'


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
    teacher_bundle_fingerprint: dict[str, Any],
    inference_fingerprint: dict[str, Any],
    upstream_commit: str,
    step_phase: str = "complete",
    sampler_state: dict | None = None,
) -> None:
    if step_phase != "complete":
        raise RuntimeError("DIMO_CHECKPOINT_REQUIRES_COMPLETE_STEP")
    output = Path(output_dir)
    rank, world_size = distributed_rank_world()
    rank_rng = capture_rank_rng_state() if world_size > 1 else None
    status = [None]
    if rank != 0:
        dist.broadcast_object_list(status, src=0)
        if status[0] != 'PASS':
            raise RuntimeError(f'DIMO_CHECKPOINT_PUBLICATION_FAILED: {status[0]}')
        return
    state = {
        "schema": DISTRIBUTED_SCHEMA_VERSION if world_size > 1 else SCHEMA_VERSION,
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
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if world_size == 1 and torch.cuda.is_initialized() else None,
        "config_hash": config_sha256,
        "teacher_bundle_fingerprint": teacher_bundle_fingerprint,
        "teacher_bundle_sha256": teacher_bundle_fingerprint["bundle_sha256"],
        "inference_fingerprint": inference_fingerprint,
        "dimo_upstream_reference_commit": upstream_commit,
        'world_size': world_size, 'rank_rng': rank_rng, 'sampler_state': sampler_state,
        'dev_only': True, 'formal_teacher': False,
    }
    temporary = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and any(output.iterdir()):
            raise RuntimeError('IMMUTABLE_DIMO_CHECKPOINT_ALREADY_EXISTS')
        temporary = Path(tempfile.mkdtemp(prefix=f'.{output.name}.partial-', dir=output.parent))
        torch.save(state, temporary / 'training_state.pt')
        from .contracts import _sha256
        ready = {'schema': state['schema'], 'step': int(committed_dimo_step),
                 'world_size': world_size, 'training_state_sha256': _sha256(temporary / 'training_state.pt')}
        (temporary / 'CHECKPOINT_READY.json').write_text(json.dumps(ready, sort_keys=True) + '\n')
        os.replace(temporary, output)
        status[0] = 'PASS'
    except Exception as exc:
        status[0] = str(exc)
        raise
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        if world_size > 1:
            dist.broadcast_object_list(status, src=0)


def read_dimo_state(input_dir):
    root = Path(input_dir)
    marker = root / 'CHECKPOINT_READY.json'
    if marker.exists():
        from .contracts import _sha256
        if json.loads(marker.read_text()).get('training_state_sha256') != _sha256(root / 'training_state.pt'):
            raise RuntimeError('DIMO_CHECKPOINT_CONTENT_HASH_MISMATCH')
    state = torch.load(root / 'training_state.pt', map_location='cpu', weights_only=False)
    if state.get('schema') == DISTRIBUTED_SCHEMA_VERSION and not marker.is_file():
        raise RuntimeError('DIMO_CHECKPOINT_PUBLICATION_INCOMPLETE')
    if marker.is_file():
        publication = json.loads(marker.read_text())
        expected = {'schema': state.get('schema'), 'step': state.get('global_dimo_step'),
                    'world_size': state.get('world_size', 1)}
        if any(publication.get(k) != v for k, v in expected.items()):
            raise RuntimeError('DIMO_CHECKPOINT_PUBLICATION_IDENTITY_MISMATCH')
    return state


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
    expected_teacher_bundle_fingerprint: dict[str, Any],
    expected_inference_fingerprint: dict[str, Any],
    expected_upstream_commit: str,
    expected_sampler_identity: dict | None = None,
) -> dict[str, Any]:
    state = read_dimo_state(input_dir)
    if state.get("schema") not in (SCHEMA_VERSION, DISTRIBUTED_SCHEMA_VERSION) or state.get("step_phase") != "complete":
        raise RuntimeError("invalid or partial DiMO checkpoint")
    world_size = distributed_rank_world()[1]
    if state.get('world_size', 1) != world_size:
        raise RuntimeError('DiMO checkpoint world_size mismatch')
    if world_size > 1:
        if state.get('schema') != DISTRIBUTED_SCHEMA_VERSION:
            raise RuntimeError('distributed DiMO requires per-rank resume state')
        validate_rank_rng_state(state.get('rank_rng'), world_size=world_size)
        if state.get('dev_only') is not True or state.get('formal_teacher') is not False:
            raise RuntimeError('DiMO pilot checkpoint must remain dev-only')
    if expected_sampler_identity is not None:
        sampler = state.get('sampler_state') or {}
        if any(sampler.get(k) != v for k, v in expected_sampler_identity.items()):
            raise RuntimeError('DiMO checkpoint sampler identity mismatch')
    checks = {
        "config_hash": expected_config_hash,
        "teacher_bundle_fingerprint": expected_teacher_bundle_fingerprint,
        "teacher_bundle_sha256": expected_teacher_bundle_fingerprint["bundle_sha256"],
        "inference_fingerprint": expected_inference_fingerprint,
        "dimo_upstream_reference_commit": expected_upstream_commit,
        "rng_stream_identity": stream_identity(),
    }
    for key, expected in checks.items():
        if state.get(key) != expected:
            raise RuntimeError(f"DiMO checkpoint {key} mismatch")
    for role in ('student', 'auxiliary'):
        current = dict(roles.role_named_parameters(role))
        saved = state.get(role + '_state', {})
        if set(saved) != set(current) or any(saved[n].shape != p.shape or saved[n].dtype != p.dtype for n, p in current.items()):
            raise RuntimeError(f'DiMO checkpoint {role} tensor identity mismatch')
    roles.load_role_state_dict("student", state["student_state"])
    roles.load_role_state_dict("auxiliary", state["auxiliary_state"])
    student_optimizer.load_state_dict(state["student_optimizer"])
    auxiliary_optimizer.load_state_dict(state["auxiliary_optimizer"])
    student_scheduler.load_state_dict(state["student_scheduler"])
    auxiliary_scheduler.load_state_dict(state["auxiliary_scheduler"])
    student_ema.load_state_dict(state["student_ema"])
    if world_size > 1:
        restore_rank_rng_state(state['rank_rng'])
    else:
        torch.set_rng_state(state["torch_rng_state"])
    if world_size == 1 and torch.cuda.is_available() and state["cuda_rng_state_all"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    return state


def validate_inference_checkpoint(
    state: dict[str, Any],
    *,
    expected_teacher_bundle_fingerprint: dict[str, Any],
    expected_inference_fingerprint: dict[str, Any],
    expected_upstream_commit: str,
) -> None:
    checks = {
        "step_phase": "complete",
        "teacher_bundle_fingerprint": expected_teacher_bundle_fingerprint,
        "teacher_bundle_sha256": expected_teacher_bundle_fingerprint["bundle_sha256"],
        "inference_fingerprint": expected_inference_fingerprint,
        "dimo_upstream_reference_commit": expected_upstream_commit,
    }
    errors = [key for key, expected in checks.items() if state.get(key) != expected]
    if state.get('schema') not in (SCHEMA_VERSION, DISTRIBUTED_SCHEMA_VERSION):
        errors.append('schema')
    if state.get('schema') == DISTRIBUTED_SCHEMA_VERSION and (
            state.get('dev_only') is not True or state.get('formal_teacher') is not False):
        errors.append('pilot_status')
    if not isinstance(state.get("config_hash"), str) or not state.get("config_hash"):
        errors.append("config_hash")
    if errors:
        raise RuntimeError("DIMO_INFERENCE_CHECKPOINT_MISMATCH: " + ", ".join(errors))
