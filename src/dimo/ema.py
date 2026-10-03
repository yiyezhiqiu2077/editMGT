"""EMA state restricted to the student's trainable role state."""

from __future__ import annotations

from collections.abc import Iterable

import torch


class TrainableEMA:
    def __init__(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]], decay: float = 0.9999):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0,1)")
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().cpu().float().clone()
            for name, parameter in named_parameters if parameter.requires_grad
        }
        if not self.shadow:
            raise ValueError("EMA requires at least one trainable student parameter")

    @torch.no_grad()
    def update(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> None:
        current = {name: parameter for name, parameter in named_parameters if name in self.shadow}
        if set(current) != set(self.shadow):
            raise RuntimeError("EMA parameter identity changed")
        for name, parameter in current.items():
            self.shadow[name].mul_(self.decay).add_(parameter.detach().cpu().float(), alpha=1 - self.decay)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "shadow": {k: v.clone() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if set(state["shadow"]) != set(self.shadow):
            raise RuntimeError("EMA state parameter identity mismatch")
        self.decay = float(state["decay"])
        self.shadow = {name: value.detach().cpu().float().clone() for name, value in state["shadow"].items()}
