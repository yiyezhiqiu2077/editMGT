"""Analytical DiMO FKL/RKL/Jeffreys logit-gradient fields."""

from __future__ import annotations

import torch


def dimo_divergence_gradient(
    teacher_logits: torch.Tensor,
    auxiliary_logits: torch.Tensor,
    *,
    temperature_teacher: float = 1.0,
    temperature_auxiliary: float = 1.0,
    mode: str = "FKL",
    beta: float = 0.5,
    pseudo_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if teacher_logits.shape != auxiliary_logits.shape or teacher_logits.ndim < 2:
        raise ValueError("teacher and auxiliary logits must have the same [...,V] shape")
    if temperature_teacher <= 0 or temperature_auxiliary <= 0:
        raise ValueError("temperatures must be positive")
    if not 0 <= beta <= 1:
        raise ValueError("Jeffreys beta must be in [0,1]")
    log_teacher = torch.log_softmax(teacher_logits / temperature_teacher, dim=-1)
    log_auxiliary = torch.log_softmax(auxiliary_logits / temperature_auxiliary, dim=-1)
    teacher_prob = log_teacher.exp()
    auxiliary_prob = log_auxiliary.exp()
    fkl = auxiliary_prob - teacher_prob
    rkl_value = (auxiliary_prob * (log_auxiliary - log_teacher)).sum(dim=-1, keepdim=True)
    rkl = auxiliary_prob * (log_auxiliary - log_teacher - rkl_value)
    normalized = mode.lower()
    if normalized == "fkl":
        gradient = fkl
    elif normalized == "rkl":
        gradient = rkl
    elif normalized in {"jeffreys", "jeffrey"}:
        gradient = (1 - beta) * fkl + beta * rkl
    else:
        raise ValueError(f"unsupported DiMO divergence: {mode}")
    if pseudo_mask is not None:
        if pseudo_mask.shape != gradient.shape[:-1]:
            raise ValueError("pseudo_mask must match all non-vocabulary logit dimensions")
        gradient = gradient.masked_fill(~pseudo_mask.bool().unsqueeze(-1), 0)
    return torch.nan_to_num(gradient).detach()
