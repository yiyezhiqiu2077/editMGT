"""Complete-step commit only; a failed in-memory update is never rolled back."""
from dataclasses import dataclass
import hashlib
import json

import torch
import torch.distributed as dist

CONTROL_GROUP = None


def set_control_group(group):
    global CONTROL_GROUP
    CONTROL_GROUP = group


def collective_status(error=None):
    errors = [None] * (dist.get_world_size() if dist.is_initialized() else 1)
    if dist.is_initialized():
        dist.all_gather_object(errors, error, group=CONTROL_GROUP)
    else:
        errors[0] = error
    if any(errors):
        raise RuntimeError(f"DIMO_STEP_ABORTED_RELOAD_CHECKPOINT: {errors}")


def sampled_rank_state(roles, optimizers, schedulers, ema, step, cursor):
    """Small fixed-coordinate checks supplement DDP's full gradient reduction.

    This is explicitly a sampled check, not a full-state bitwise hash. No full
    state_dict, dense clone, or full optimizer traversal is made per step.
    """
    digest = hashlib.sha256()
    finite = True
    with torch.no_grad():
        for role in ("teacher", "student", "auxiliary"):
            for name, p in roles.role_named_parameters(role):
                finite = finite and p.dtype == torch.float32
                sample = p.detach().reshape(-1)[::max(1, p.numel() // 16)][:16].cpu().contiguous()
                finite = finite and bool(torch.isfinite(sample).all())
                digest.update(role.encode() + name.encode() + sample.view(torch.uint8).numpy().tobytes())
        for name, p in ema.shadow.items():
            finite = finite and p.dtype == torch.float32
            sample = p.reshape(-1)[::max(1, p.numel() // 16)][:16].cpu().contiguous()
            finite = finite and bool(torch.isfinite(sample).all())
            digest.update(name.encode() + sample.view(torch.uint8).numpy().tobytes())
    for optimizer in optimizers:
        expected_parameters = [p for group in optimizer.param_groups for p in group["params"]]
        finite = finite and set(optimizer.state) == set(expected_parameters)
        for p in optimizer.state:
            state = optimizer.state[p]
            finite = finite and set(state) == {"step", "exp_avg", "exp_avg_sq"}
            finite = finite and int(state["step"].item()) == step
            finite = finite and all(state[field].dtype == torch.float32 and state[field].shape == p.shape
                                       for field in ("exp_avg", "exp_avg_sq"))
            for field in ("step", "exp_avg", "exp_avg_sq"):
                value = state.get(field)
                if isinstance(value, torch.Tensor):
                    sample = value.detach().reshape(-1)[::max(1, value.numel() // 4)][:4].cpu().contiguous()
                    finite = finite and bool(torch.isfinite(sample).all())
                    digest.update(field.encode() + sample.view(torch.uint8).numpy().tobytes())
    row = {"step": step, "cursor": cursor, "ema_count": ema.update_count,
           "schedulers": [s.state_dict() for s in schedulers], "digest": digest.hexdigest()}
    collective_status(None if finite else "NONFINITE_SAMPLED_STATE")
    rows = [None] * (dist.get_world_size() if dist.is_initialized() else 1)
    if dist.is_initialized():
        dist.all_gather_object(rows, row)
    else:
        rows[0] = row
    if any(other != row for other in rows):
        raise RuntimeError("DIMO_RANK_STATE_MISMATCH")
    return {"status": "PASS", "scope": "fixed_coordinate_sample", "world_size": len(rows)}


@dataclass
class StepCommit:
    step: int = 0
    cursor: int = 0
    failed: bool = False
    in_flight: bool = False

    def execute(self, operation, *, samples, finalize=None, failure_injector=None):
        if self.failed or self.in_flight:
            raise RuntimeError("DIMO_TRANSACTION_REQUIRES_CHECKPOINT_RELOAD")
        self.in_flight = True
        try:
            result = operation()
            if finalize is not None:
                finalize(self.step + 1, self.cursor + samples)
            error = None
            try:
                if failure_injector is not None:
                    failure_injector("before_commit")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            collective_status(error)
            # This assignment is the only committed cursor/step mutation.
            self.step += 1
            self.cursor += samples
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            self.in_flight = False

    def require_checkpointable(self, step, cursor):
        if self.failed or self.in_flight or (self.step, self.cursor) != (step, cursor):
            raise RuntimeError("DIMO_CHECKPOINT_REQUIRES_COMPLETE_COMMITTED_STEP")
