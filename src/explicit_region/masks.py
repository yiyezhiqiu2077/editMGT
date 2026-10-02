"""Pixel-region to discrete-token-region conversion."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_bchw(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    elif mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("pixel mask must have shape [H,W], [B,H,W], or [B,1,H,W]")
    return mask.float()


def pixel_mask_to_token_mask(
    pixel_mask: torch.Tensor,
    token_size: tuple[int, int],
    *,
    mode: str = "any_overlap",
    coverage_threshold: float = 0.5,
    dilation_tokens: int = 0,
    minimum_edit_tokens: int = 1,
) -> torch.Tensor:
    mask = _as_bchw(pixel_mask)
    if mode == "any_overlap":
        pooled = F.adaptive_max_pool2d(mask, token_size)
        token_mask = pooled > 0
    elif mode == "coverage":
        if not 0 <= coverage_threshold <= 1:
            raise ValueError("coverage_threshold must be in [0,1]")
        pooled = F.adaptive_avg_pool2d(mask, token_size)
        token_mask = pooled >= coverage_threshold
    else:
        raise ValueError(f"unsupported token mask mode: {mode}")
    if dilation_tokens < 0:
        raise ValueError("dilation_tokens must be non-negative")
    if dilation_tokens:
        kernel = 2 * dilation_tokens + 1
        token_mask = F.max_pool2d(token_mask.float(), kernel, stride=1, padding=dilation_tokens).bool()
    token_mask = token_mask[:, 0]
    counts = token_mask.flatten(1).sum(1)
    if torch.any(counts < minimum_edit_tokens):
        raise ValueError(
            f"token mask has fewer than minimum_edit_tokens={minimum_edit_tokens}: {counts.tolist()}"
        )
    return token_mask
