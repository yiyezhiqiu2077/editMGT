"""Centralized role prediction and text-only classifier-free guidance."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def _logits(output, target_shape: tuple[int, int, int]) -> torch.Tensor:
    if hasattr(output, "sample"):
        output = output.sample
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor) or output.ndim != 4:
        raise TypeError("role forward must return a 4D logits tensor")
    batch, height, width = target_shape
    if output.shape[:3] == target_shape:
        return output
    if output.shape[0] == batch and output.shape[2:] == (height, width):
        return output.permute(0, 2, 3, 1)
    raise ValueError("cannot align role logits to target token grid")


def predict_role_logits(
    roles,
    role: str,
    target_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    prompt_condition: Mapping[str, Mapping[str, torch.Tensor]],
    region_mask: torch.Tensor,
    timestep: torch.Tensor,
    cfg_scale: float,
    *,
    model_kwargs: Mapping[str, object] | None = None,
) -> torch.Tensor:
    """Predict with CFG while keeping source, target, and region identical."""
    if cfg_scale < 1:
        raise ValueError("cfg_scale must be >= 1")
    if target_tokens.shape != reference_tokens.shape or region_mask.shape != target_tokens.shape:
        raise ValueError("target, reference, and region token grids must match")
    conditional = dict(prompt_condition["conditional"])
    common = dict(model_kwargs or {})
    common.update(
        hidden_states=target_tokens,
        reference_image_hidden_states=reference_tokens,
        edit_region_mask=region_mask,
        edit_region_conditioning_active=True,
        timestep=timestep,
    )
    if cfg_scale == 1:
        return _logits(
            roles.forward_role(role, **common, **conditional), tuple(target_tokens.shape)
        )
    unconditional = dict(prompt_condition["unconditional"])
    if conditional.keys() != unconditional.keys():
        raise ValueError("conditional and unconditional prompt payloads must have identical keys")
    batch_size = target_tokens.shape[0]
    combined_prompt = {}
    for key in conditional:
        if not isinstance(conditional[key], torch.Tensor) or not isinstance(unconditional[key], torch.Tensor):
            raise TypeError("CFG prompt payload values must be tensors")
        combined_prompt[key] = torch.cat((unconditional[key], conditional[key]), dim=0)
    combined_common = {}
    for key, value in common.items():
        if key in {"hidden_states", "reference_image_hidden_states", "edit_region_mask", "timestep"}:
            combined_common[key] = torch.cat((value, value), dim=0)
        elif isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch_size:
            combined_common[key] = torch.cat((value, value), dim=0)
        else:
            combined_common[key] = value
    combined_shape = (2 * batch_size, target_tokens.shape[1], target_tokens.shape[2])
    combined_logits = _logits(
        roles.forward_role(role, **combined_common, **combined_prompt), combined_shape
    )
    unconditional_logits, conditional_logits = combined_logits.chunk(2, dim=0)
    return unconditional_logits + cfg_scale * (conditional_logits - unconditional_logits)
