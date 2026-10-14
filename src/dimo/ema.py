"""EMA state restricted to the student's trainable role state."""

from __future__ import annotations

from collections.abc import Iterable

import torch


class TrainableEMA:
    def __init__(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]], decay: float = 0.9999, device="cpu"):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0,1)")
        self.decay = float(decay)
        self.device = torch.device(device)
        self.update_count = 0
        self.shadow = {
            name: parameter.detach().to(device=self.device, dtype=torch.float32).clone()
            for name, parameter in named_parameters if parameter.requires_grad
        }
        if not self.shadow:
            raise ValueError("EMA requires at least one trainable student parameter")

    @torch.no_grad()
    def update(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]], *, failure_injector=None) -> None:
        current = {name: parameter for name, parameter in named_parameters if name in self.shadow}
        if set(current) != set(self.shadow):
            raise RuntimeError("EMA parameter identity changed")
        for name, parameter in current.items():
            self.shadow[name].mul_(self.decay).add_(parameter.detach().to(self.device, torch.float32), alpha=1 - self.decay)
            if failure_injector is not None:
                failure_injector("ema_during")
        self.update_count += 1

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def state_dict(self) -> dict[str, object]:
        self.synchronize()
        return {"decay": self.decay, "update_count": self.update_count,
                "shadow": {k: v.detach().cpu().clone() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if set(state["shadow"]) != set(self.shadow):
            raise RuntimeError("EMA state parameter identity mismatch")
        self.decay = float(state["decay"])
        self.update_count = int(state.get("update_count", 0))
        self.shadow = {name: value.detach().to(self.device, torch.float32).clone() for name, value in state["shadow"].items()}


@torch.no_grad()
def apply_ema_to_student_role(roles, ema_state: dict[str, object]) -> None:
    """Strictly apply an FP32 EMA shadow to the student's live parameters."""
    if not isinstance(ema_state, dict) or "shadow" not in ema_state:
        raise RuntimeError("invalid student EMA state")
    current = dict(roles.role_named_parameters("student"))
    shadow = ema_state["shadow"]
    if not isinstance(shadow, dict) or set(shadow) != set(current):
        raise RuntimeError("EMA state parameter identity mismatch")
    for name, parameter in current.items():
        value = shadow[name]
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"EMA tensor is invalid: {name}")
        if value.shape != parameter.shape:
            raise RuntimeError(f"EMA parameter shape mismatch: {name}")
        if value.dtype != torch.float32:
            raise RuntimeError(f"EMA parameter dtype must be float32: {name}")
        if not torch.is_floating_point(parameter):
            raise RuntimeError(f"student parameter dtype is not floating: {name}")
    for name, parameter in current.items():
        value = shadow[name]
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
