"""Loss reductions used by all corruption modes."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def per_sample_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 4 or labels.ndim != 3:
        raise ValueError("expected logits [B,V,H,W] and labels [B,H,W]")
    if logits.shape[0] != labels.shape[0] or logits.shape[2:] != labels.shape[1:]:
        raise ValueError("logit/label shapes do not align")
    per_token = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction="none")
    valid = labels.ne(ignore_index)
    counts = valid.flatten(1).sum(1)
    if torch.any(counts == 0):
        raise ValueError("every sample must supervise at least one token")
    per_sample = (per_token * valid).flatten(1).sum(1) / counts
    token_mean = (per_token * valid).sum() / counts.sum()
    return per_sample.mean(), token_mean
