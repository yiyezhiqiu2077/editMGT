"""Small, side-effect-free Region-DiMO diagnostics."""

from __future__ import annotations

import torch


def tensor_grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum())
    return total**0.5


def token_diagnostics(
    student_logits: torch.Tensor,
    student_tokens: torch.Tensor,
    source_tokens: torch.Tensor,
    edit_region_mask: torch.Tensor,
) -> dict[str, float | int]:
    region = edit_region_mask.bool()
    outside = ~region
    prob = torch.softmax(student_logits.detach().float(), dim=-1)
    entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum(-1)
    inside_count = max(int(region.sum()), 1)
    inside_tokens = student_tokens[region]
    return {
        "student_entropy_inside_roi": float(entropy[region].mean()),
        "student_source_copy_ratio_inside_roi": float((inside_tokens == source_tokens[region]).sum()) / inside_count,
        "student_unique_token_fraction_inside_roi": float(inside_tokens.unique().numel()) / inside_count,
        "outside_mismatch_count": int((student_tokens[outside] != source_tokens[outside]).sum()),
        "max_abs_student_logit": float(student_logits.detach().abs().max()),
        "nan_inf_count": int((~torch.isfinite(student_logits.detach())).sum()),
    }
