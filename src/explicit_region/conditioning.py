"""Persistent region-conditioning operation shared by model and tests."""

from __future__ import annotations

import math
import torch


def duplicate_region_mask_for_cfg(
    edit_region_mask: torch.Tensor | None, *, guidance_enabled: bool
) -> torch.Tensor | None:
    """Mirror the unconditional/conditional batch ordering used by CFG."""
    if edit_region_mask is None or not guidance_enabled:
        return edit_region_mask
    return torch.cat((edit_region_mask, edit_region_mask), dim=0)


def initialize_hardlock_latents(
    clean_reference_tokens: torch.Tensor,
    edit_region_mask: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """Create target latents without ever aliasing/mutating the reference branch."""
    if clean_reference_tokens.shape != edit_region_mask.shape:
        raise ValueError("reference tokens and edit region must share shape")
    latents = clean_reference_tokens.clone()
    latents[edit_region_mask.bool()] = mask_token_id
    return latents


def region_scheduled_ratio(step_index: int, num_steps: int) -> float:
    """ROI masking ratio present at transformer input for a cosine scheduler."""
    if num_steps <= 0 or not 0 <= step_index < num_steps:
        raise ValueError("invalid inference step")
    return math.cos(step_index / num_steps * math.pi / 2)


def add_region_condition(
    hidden_states: torch.Tensor,
    edit_region_embedding: torch.Tensor,
    edit_region_mask: torch.Tensor | None,
    *,
    active: bool,
) -> torch.Tensor:
    if not active or edit_region_mask is None:
        return hidden_states
    expected = (hidden_states.shape[0], hidden_states.shape[2], hidden_states.shape[3])
    if tuple(edit_region_mask.shape) != expected:
        raise ValueError(f"edit_region_mask must have shape {expected}, got {tuple(edit_region_mask.shape)}")
    if edit_region_embedding.ndim != 1 or edit_region_embedding.shape[0] != hidden_states.shape[1]:
        raise ValueError("edit_region_embedding must match hidden-state channels")
    mask = edit_region_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
    return hidden_states + mask[:, None] * edit_region_embedding[None, :, None, None].to(hidden_states.dtype)
