"""Surrogate student-logit objective with per-sample ROI normalization."""

from __future__ import annotations

import torch


def surrogate_logit_loss(
    student_logits: torch.Tensor,
    dimo_gradient: torch.Tensor,
    pseudo_mask: torch.Tensor,
) -> torch.Tensor:
    if student_logits.shape != dimo_gradient.shape:
        raise ValueError("student logits and DiMO gradient must match")
    if pseudo_mask.shape != student_logits.shape[:-1]:
        raise ValueError("pseudo mask must match non-vocabulary logit dimensions")
    flat_mask = pseudo_mask.bool().flatten(1)
    counts = flat_mask.sum(1)
    if (counts == 0).any():
        raise ValueError("each sample must contain at least one pseudo-masked token")
    target = (student_logits - dimo_gradient).detach()
    token_loss = 0.5 * (student_logits - target).square().sum(dim=-1)
    per_sample = (token_loss.flatten(1) * flat_mask).sum(1) / counts
    return per_sample.mean()
