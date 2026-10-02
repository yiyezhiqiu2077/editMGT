"""Exact full-target and ROI-hardlock corruption contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .deterministic import cpu_generator


@dataclass(frozen=True)
class CorruptionBatch:
    input_tokens: torch.Tensor
    labels: torch.Tensor
    selected_mask: torch.Tensor
    edit_region_mask: torch.Tensor | None
    conditioning_active: bool
    mode: str
    scheduled_roi_ratio: torch.Tensor
    actual_roi_mask_fraction: torch.Tensor
    actual_global_mask_fraction: torch.Tensor
    full_roi_branch: torch.Tensor
    resolved_modes: tuple[str, ...]
    condition_dropout_count: int = 0
    max_abs_logit: float = 0.0


def _choose_mode(
    requested: str,
    roi_probability: float,
    base_seed: int,
    global_sample_index: int,
    sample_key: str,
) -> str:
    if requested in {"full_target", "roi_hardlock"}:
        return requested
    if requested != "mixed" or not 0 <= roi_probability <= 1:
        raise ValueError("mode must be full_target, roi_hardlock, or mixed")
    generator = cpu_generator(base_seed, global_sample_index, sample_key, "corruption-mode")
    return "roi_hardlock" if torch.rand((), generator=generator).item() < roi_probability else "full_target"


def prepare_corruption(
    source_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    edit_region_mask: torch.Tensor,
    mask_token_id: int,
    *,
    mode: str,
    base_seed: int,
    global_sample_indices: list[int],
    sample_keys: list[str],
    roi_probability: float = 0.5,
    full_roi_mask_probability: float = 0.15,
) -> CorruptionBatch:
    if source_tokens.shape != target_tokens.shape or source_tokens.ndim != 3:
        raise ValueError("source and target tokens must share [B,H,W]")
    if edit_region_mask.shape != source_tokens.shape:
        raise ValueError("edit_region_mask must match token grids")
    batch = source_tokens.shape[0]
    if len(global_sample_indices) != batch or len(sample_keys) != batch:
        raise ValueError("sample identity length must match batch")
    if not 0 <= full_roi_mask_probability <= 1:
        raise ValueError("full_roi_mask_probability must be in [0,1]")

    outputs = []
    resolved_modes = []
    for b in range(batch):
        index, key = int(global_sample_indices[b]), str(sample_keys[b])
        resolved = _choose_mode(mode, roi_probability, base_seed, index, key)
        resolved_modes.append(resolved)
        region = edit_region_mask[b].bool() if resolved == "roi_hardlock" else torch.ones_like(edit_region_mask[b], dtype=torch.bool)
        candidates = region.flatten().nonzero(as_tuple=False).flatten().cpu()
        if candidates.numel() == 0:
            raise ValueError("corruption region is empty")
        draw_gen = cpu_generator(base_seed, index, key, "mask-ratio")
        full_gen = cpu_generator(base_seed, index, key, "full-roi")
        full_roi = resolved == "roi_hardlock" and torch.rand((), generator=full_gen).item() < full_roi_mask_probability
        if full_roi:
            rho = 1.0
            count = candidates.numel()
        else:
            u = torch.rand((), generator=draw_gen).item()
            rho = math.cos(u * math.pi / 2)
            count = min(candidates.numel(), max(1, round(rho * candidates.numel())))
        order_gen = cpu_generator(base_seed, index, key, "mask-permutation")
        order = torch.randperm(candidates.numel(), generator=order_gen)
        selected = torch.zeros(region.numel(), dtype=torch.bool)
        selected[candidates[order[:count]]] = True
        selected = selected.reshape_as(region).to(source_tokens.device)
        region = region.to(source_tokens.device)
        if resolved == "roi_hardlock":
            canvas = torch.where(region, target_tokens[b], source_tokens[b]).clone()
            condition = region
        else:
            canvas = target_tokens[b].clone()
            condition = None
        canvas[selected] = mask_token_id
        labels = torch.full_like(target_tokens[b], -100)
        labels[selected] = target_tokens[b][selected]
        outputs.append(
            (
                canvas,
                labels,
                selected,
                condition,
                rho,
                count / candidates.numel(),
                count / region.numel(),
                full_roi,
            )
        )
    if len(set(resolved_modes)) != 1:
        # A batch can mix modes; condition uses an all-false row for full-target.
        resolved_name = "mixed"
    else:
        resolved_name = resolved_modes[0]
    conditions = torch.stack(
        [torch.zeros_like(edit_region_mask[0], dtype=torch.bool) if row[3] is None else row[3] for row in outputs]
    )
    return CorruptionBatch(
        input_tokens=torch.stack([row[0] for row in outputs]),
        labels=torch.stack([row[1] for row in outputs]),
        selected_mask=torch.stack([row[2] for row in outputs]),
        edit_region_mask=conditions,
        conditioning_active=any(row[3] is not None for row in outputs),
        mode=resolved_name,
        scheduled_roi_ratio=torch.tensor([row[4] for row in outputs], device=source_tokens.device),
        actual_roi_mask_fraction=torch.tensor([row[5] for row in outputs], device=source_tokens.device),
        actual_global_mask_fraction=torch.tensor([row[6] for row in outputs], device=source_tokens.device),
        full_roi_branch=torch.tensor([row[7] for row in outputs], device=source_tokens.device),
        resolved_modes=tuple(resolved_modes),
    )
