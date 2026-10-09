"""ROI-only one-step initialization and categorical student sampling."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch

from .rng import cpu_generator


@dataclass(frozen=True)
class EditingInitialState:
    initial_tokens: torch.Tensor
    initial_mask: torch.Tensor
    random_token_mask: torch.Tensor
    initial_roi_mask_ratio: torch.Tensor


def _validate_token_grids(source_tokens: torch.Tensor, edit_region_mask: torch.Tensor) -> torch.Tensor:
    if source_tokens.ndim != 3:
        raise ValueError("source_tokens must have shape [B,H,W]")
    if edit_region_mask.shape != source_tokens.shape:
        raise ValueError("edit_region_mask must match source token grids")
    if source_tokens.dtype == torch.bool or source_tokens.is_floating_point():
        raise TypeError("source_tokens must contain integer token IDs")
    region = edit_region_mask.bool()
    if (region.flatten(1).sum(1) == 0).any():
        raise ValueError("every sample must contain at least one editable ROI token")
    return region


def _seeds(value: int | Sequence[int], batch_size: int, name: str) -> list[int]:
    if isinstance(value, int):
        return [value] * batch_size
    result = [int(item) for item in value]
    if len(result) != batch_size:
        raise ValueError(f"{name} must provide one seed per sample")
    return result


def _mask_count(ratio: float, count: int) -> int:
    if not 0 <= float(ratio) <= 1:
        raise ValueError("mask ratio must be in [0,1]")
    return max(1, min(count, int(round(float(ratio) * count))))


def build_editing_initial_state(
    source_tokens: torch.Tensor,
    edit_region_mask: torch.Tensor,
    mask_token_id: int,
    codebook_size: int,
    r_init: float,
    *,
    mask_seeds: int | Sequence[int],
    token_seeds: int | Sequence[int],
) -> EditingInitialState:
    """Build exact-count MASK/random initialization inside ROI only."""
    if codebook_size <= 0 or not 0 <= int(mask_token_id):
        raise ValueError("invalid codebook or MASK token ID")
    region = _validate_token_grids(source_tokens, edit_region_mask)
    batch_size = source_tokens.shape[0]
    mask_seeds = _seeds(mask_seeds, batch_size, "mask_seeds")
    token_seeds = _seeds(token_seeds, batch_size, "token_seeds")
    initial = source_tokens.clone()
    selected = torch.zeros_like(region)

    for batch in range(batch_size):
        roi_flat = region[batch].flatten().nonzero(as_tuple=False).flatten().cpu()
        count = roi_flat.numel()
        k = _mask_count(r_init, count)
        order = torch.randperm(count, generator=cpu_generator(mask_seeds[batch]))
        selected_flat = roi_flat[order[:k]].to(source_tokens.device)
        selected[batch].view(-1)[selected_flat] = True

        random_flat = roi_flat[order[k:]]
        if random_flat.numel():
            values = torch.randint(
                0, codebook_size, (random_flat.numel(),),
                generator=cpu_generator(token_seeds[batch]), dtype=torch.long,
            ).to(device=source_tokens.device, dtype=source_tokens.dtype)
            initial[batch].view(-1)[random_flat.to(source_tokens.device)] = values

    random_mask = region & ~selected
    initial[selected] = int(mask_token_id)
    ratios = selected.flatten(1).sum(1).to(torch.float32) / region.flatten(1).sum(1)

    if (selected & ~region).any() or (random_mask & ~region).any():
        raise AssertionError("DiMO initialization escaped editable ROI")
    if (selected & random_mask).any() or not torch.equal(selected | random_mask, region):
        raise AssertionError("ROI initialization partition is invalid")
    if not torch.equal(initial[~region], source_tokens[~region]):
        raise RuntimeError("OUTSIDE_REGION_LOCK_VIOLATION")
    return EditingInitialState(initial, selected, random_mask, ratios)


def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k < 0 or top_k > logits.shape[-1]:
        raise ValueError("top_k must be between 0 and vocabulary size")
    if not 0 <= top_p <= 1:
        raise ValueError("top_p must be in [0,1]")
    filtered = logits.clone()
    if top_k:
        threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered.masked_fill_(filtered < threshold, -torch.inf)
    if top_p:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove = remove.scatter(-1, sorted_indices, remove)
        filtered.masked_fill_(remove, -torch.inf)
    return filtered


@torch.no_grad()
def sample_student_tokens(
    student_logits: torch.Tensor,
    source_tokens: torch.Tensor,
    edit_region_mask: torch.Tensor,
    *,
    seeds: int | Sequence[int],
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> torch.Tensor:
    """Sample detached student logits only in ROI and hard-lock the exterior."""
    region = _validate_token_grids(source_tokens, edit_region_mask)
    if student_logits.shape[:-1] != source_tokens.shape:
        raise ValueError("student_logits must have shape [B,H,W,V]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    seeds = _seeds(seeds, source_tokens.shape[0], "seeds")
    filtered = _filter_logits(student_logits.detach().float() / temperature, top_k, top_p)
    output = source_tokens.clone()
    for batch, seed in enumerate(seeds):
        flat_region = region[batch].flatten()
        probs = torch.softmax(filtered[batch].reshape(-1, filtered.shape[-1])[flat_region], -1).cpu()
        sampled = torch.multinomial(probs, 1, generator=cpu_generator(seed)).squeeze(-1)
        output[batch].view(-1)[flat_region] = sampled.to(output.device, output.dtype)
    if not torch.equal(output[~region], source_tokens[~region]):
        raise RuntimeError("OUTSIDE_REGION_LOCK_VIOLATION")
    return output.detach()


def build_role_lora_configs(lora: dict, model_roles: dict | None = None):
    """Build role configs without mutating the selected teacher policy."""
    from peft import LoraConfig

    policy = dict(model_roles or {})
    if policy.get("lora_dropout_policy", "zero_for_dimo") != "zero_for_dimo":
        raise ValueError("unsupported DiMO LoRA dropout policy")
    teacher_value = policy.get("teacher_lora_dropout", "inherit")
    if teacher_value != "inherit":
        raise ValueError("teacher_lora_dropout must be 'inherit'")
    student_dropout = float(policy.get("student_lora_dropout", 0.0))
    auxiliary_dropout = float(policy.get("auxiliary_lora_dropout", 0.0))
    if student_dropout != 0.0 or auxiliary_dropout != 0.0:
        raise ValueError("student and auxiliary LoRA dropout must be zero")

    def make(dropout: float):
        return LoraConfig(
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=dropout,
            target_modules=list(lora["target_modules"]),
            init_lora_weights=True,
        )

    return {
        "teacher": make(float(lora["dropout"])),
        "student": make(student_dropout),
        "auxiliary": make(auxiliary_dropout),
    }


def initialize_shared_model_roles(
    base_model,
    teacher_checkpoint: str,
    *,
    model_roles: dict | None = None,
):
    """Load a compatible explicit-region SFT checkpoint into three adapters.

    This is a storage adapter only. Formal selection is enforced separately by
    :func:`src.dimo.contracts.enforce_run_guard`.
    """
    import json
    from pathlib import Path

    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    from .roles import DiMOModelRoles

    root = Path(teacher_checkpoint)
    trainable = json.loads((root / "trainable_config.json").read_text(encoding="utf-8"))
    lora = trainable["lora"]
    adapter_configs = build_role_lora_configs(lora, model_roles)
    for role in ("teacher", "student", "auxiliary"):
        base_model.add_adapter(adapter_configs[role], adapter_name=role)
    teacher_state = load_file(root / "adapter_model.safetensors")
    result = set_peft_model_state_dict(base_model, teacher_state, adapter_name="teacher")
    if getattr(result, "unexpected_keys", None):
        raise RuntimeError(f"unexpected teacher LoRA keys: {result.unexpected_keys}")
    loaded = get_peft_model_state_dict(base_model, adapter_name='teacher')
    if (set(loaded) != set(teacher_state) or any(
            not torch.equal(value.detach().cpu(), teacher_state[name].to(dtype=value.dtype))
            for name, value in loaded.items())):
        raise RuntimeError('teacher LoRA was not loaded exactly into the shared backbone')
    region_state = load_file(root / "mask_conditioning.safetensors")["edit_region_embedding"]
    if region_state.shape != base_model.edit_region_embedding.shape:
        raise RuntimeError("teacher region embedding shape is incompatible")
    base_model.edit_region_embedding.data.copy_(
        region_state.to(base_model.edit_region_embedding.device, base_model.edit_region_embedding.dtype)
    )
    return DiMOModelRoles(base_model, initialize_from_teacher=True)
