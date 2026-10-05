"""Auxiliary-model objectives for detached student samples."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def auxiliary_loss(
    auxiliary_logits: torch.Tensor,
    student_tokens: torch.Tensor,
    pseudo_mask: torch.Tensor,
    *,
    student_logits: torch.Tensor | None = None,
    soft_target_weight: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-sample normalized hard CE, with a reserved optional soft target."""
    if auxiliary_logits.shape[:-1] != student_tokens.shape or pseudo_mask.shape != student_tokens.shape:
        raise ValueError("auxiliary logits, student tokens, and pseudo mask have incompatible shapes")
    if not 0 <= soft_target_weight <= 1 or temperature <= 0:
        raise ValueError("invalid soft-target weight or temperature")
    mask = pseudo_mask.bool().flatten(1)
    counts = mask.sum(1)
    if (counts == 0).any():
        raise ValueError("each sample must contain pseudo-masked auxiliary targets")
    vocab = auxiliary_logits.shape[-1]
    hard = F.cross_entropy(
        auxiliary_logits.reshape(-1, vocab), student_tokens.detach().reshape(-1), reduction="none"
    ).reshape(student_tokens.shape[0], -1)
    hard = (hard * mask).sum(1) / counts
    if soft_target_weight == 0:
        return hard.mean()
    if student_logits is None or student_logits.shape != auxiliary_logits.shape:
        raise ValueError("soft targets require matching detached student logits")
    target_prob = torch.softmax(student_logits.detach() / temperature, dim=-1)
    soft = F.kl_div(
        torch.log_softmax(auxiliary_logits / temperature, dim=-1),
        target_prob,
        reduction="none",
    ).sum(-1).flatten(1) * temperature**2
    soft = (soft * mask).sum(1) / counts
    return ((1 - soft_target_weight) * hard + soft_target_weight * soft).mean()
