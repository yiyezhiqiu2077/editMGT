"""Surrogate student-logit objective with per-sample ROI normalization."""

from __future__ import annotations

import torch


def _roi_mean(token_values, pseudo_mask):
    if pseudo_mask.shape != token_values.shape:
        raise ValueError("pseudo mask must match non-vocabulary logit dimensions")
    mask = pseudo_mask.bool().flatten(1)
    counts = mask.sum(1)
    if (counts == 0).any():
        raise ValueError("each sample must contain at least one pseudo-masked token")
    # masked_fill excludes even non-finite values outside the selected region.
    return (token_values.flatten(1).masked_fill(~mask, 0).sum(1) / counts).mean()


def linear_surrogate_logit_loss(student_logits, dimo_gradient, pseudo_mask):
    """Stable gradient injection; its signed scalar is not a divergence/CE.

    dL/dz[b,t,v] = detached g[b,t,v] / (B * selected_tokens[b]).
    Double inputs retain double precision for analytical reference tests.
    """
    if student_logits.shape != dimo_gradient.shape:
        raise ValueError("student logits and DiMO gradient must match")
    dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    z, g = student_logits.to(dtype), dimo_gradient.detach().to(dtype)
    return _roi_mean((z * g).sum(-1), pseudo_mask)


def distribution_energy(dimo_gradient, pseudo_mask):
    """0.5 * vocabulary-sum squared g, with the exact training ROI reduction."""
    return _roi_mean(0.5 * dimo_gradient.detach().float().square().sum(-1), pseudo_mask)


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
    # Keep the complete vocabulary-SUM surrogate in FP32. Autograd casts the
    # resulting gradient back to a lower-precision leaf only at that boundary.
    student_fp32 = student_logits.float()
    gradient_fp32 = dimo_gradient.float()
    target = (student_fp32 - gradient_fp32).detach()
    token_loss = 0.5 * (student_fp32 - target).square().sum(dim=-1)
    per_sample = (token_loss.flatten(1) * flat_mask).sum(1) / counts
    return per_sample.mean()
