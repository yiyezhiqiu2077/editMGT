"""LoRA, mask-conditioning, fingerprint, resume and warm-start state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from .deterministic import canonical_json


SCHEMA_VERSION = 1


def recipe_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def save_trainable_state(model, output_dir: str | Path, config: dict) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lora_state = {name: value.detach().cpu().contiguous() for name, value in get_peft_model_state_dict(model).items()}
    save_file(lora_state, output / "adapter_model.safetensors")
    save_file(
        {"edit_region_embedding": model.edit_region_embedding.detach().cpu().contiguous()},
        output / "mask_conditioning.safetensors",
    )
    mask_config = {
        "schema_version": SCHEMA_VERSION,
        "shape": list(model.edit_region_embedding.shape),
        "dtype": str(model.edit_region_embedding.dtype),
        "conditioning": "binary_mask_times_single_vector",
    }
    (output / "mask_conditioning_config.json").write_text(
        json.dumps(mask_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "trainable_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_trainable_state(model, input_dir: str | Path) -> None:
    root = Path(input_dir)
    lora_state = load_file(root / "adapter_model.safetensors")
    result = set_peft_model_state_dict(model, lora_state)
    if getattr(result, "unexpected_keys", None):
        raise RuntimeError(f"unexpected LoRA keys: {result.unexpected_keys}")
    mask_state = load_file(root / "mask_conditioning.safetensors")
    value = mask_state["edit_region_embedding"]
    if tuple(value.shape) != tuple(model.edit_region_embedding.shape):
        raise ValueError("mask-conditioning shape mismatch")
    model.edit_region_embedding.data.copy_(
        value.to(device=model.edit_region_embedding.device, dtype=model.edit_region_embedding.dtype)
    )


def save_resume_state(
    output_dir: str | Path,
    *,
    model,
    optimizer,
    scheduler,
    global_optimizer_step: int,
    committed_global_sample_count: int,
    sampler_schema: str,
    fingerprint_payload: dict,
    quality_state: dict | None = None,
    sampler_state: dict | None = None,
) -> None:
    output = Path(output_dir)
    save_trainable_state(model, output, fingerprint_payload)
    fingerprint = recipe_fingerprint(fingerprint_payload)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "global_optimizer_step": global_optimizer_step,
            "committed_global_sample_count": committed_global_sample_count,
            "sampler_schema": sampler_schema,
            "quality_state": quality_state or {},
            "sampler_state": sampler_state or {},
            "fingerprint": fingerprint,
        },
        output / "training_state.pt",
    )
    (output / "fingerprint.json").write_text(
        json.dumps({"sha256": fingerprint, "payload": fingerprint_payload}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_resume_state(input_dir, *, model, optimizer, scheduler, fingerprint_payload: dict) -> dict:
    root = Path(input_dir)
    state = torch.load(root / "training_state.pt", map_location="cpu")
    expected = recipe_fingerprint(fingerprint_payload)
    if state["fingerprint"] != expected:
        raise RuntimeError("checkpoint fingerprint mismatch; use warm_start for a new recipe")
    load_trainable_state(model, root)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state
