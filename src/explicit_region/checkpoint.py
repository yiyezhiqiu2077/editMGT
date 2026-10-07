"""LoRA, mask-conditioning, fingerprint, resume and warm-start state."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from .deterministic import canonical_json


SCHEMA_VERSION = 1
TRAINING_STATE_SCHEMA_VERSION = 2
RANK_RNG_SCHEMA_VERSION = 1


def distributed_rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("distributed checkpoint requires an initialized process group")
    return 0, 1


def capture_local_rng_state(rank: int) -> dict:
    return {
        "rank": rank,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        # One process owns one accelerator device.  Reading every visible CUDA
        # RNG here would make every rank initialize contexts on every GPU.
        "torch_cuda": torch.cuda.get_rng_state(torch.cuda.current_device())
        if torch.cuda.is_available() else None,
    }


def validate_rank_rng_state(bundle: dict, *, world_size: int) -> None:
    if not isinstance(bundle, dict) or bundle.get("schema_version") != RANK_RNG_SCHEMA_VERSION:
        raise RuntimeError("missing or unsupported per-rank RNG schema")
    if bundle.get("world_size") != world_size:
        raise RuntimeError("RNG world_size mismatch")
    states = bundle.get("states")
    if not isinstance(states, dict) or set(states) != {str(rank) for rank in range(world_size)}:
        raise RuntimeError("incomplete or unexpected RNG ranks")
    required = {"rank", "python", "numpy", "torch_cpu", "torch_cuda"}
    for rank, state in states.items():
        if not isinstance(state, dict) or set(state) != required or state["rank"] != int(rank):
            raise RuntimeError(f"invalid RNG state for rank {rank}")
        random.Random(0).setstate(state["python"])
        np.random.RandomState(0).set_state(state["numpy"])
        torch.Generator(device="cpu").set_state(state["torch_cpu"])
        cuda = state["torch_cuda"]
        if cuda is not None and not isinstance(cuda, torch.Tensor):
            raise RuntimeError(f"invalid CUDA RNG state for rank {rank}")


def capture_rank_rng_state() -> dict:
    """Collectively capture process RNG state at one optimizer boundary."""
    rank, world_size = distributed_rank_world()
    local = capture_local_rng_state(rank)
    states = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    bundle = {
        "schema_version": RANK_RNG_SCHEMA_VERSION,
        "world_size": world_size,
        "states": {str(state["rank"]): state for state in states},
    }
    validate_rank_rng_state(bundle, world_size=world_size)
    return bundle


def restore_rank_rng_state(bundle: dict) -> None:
    rank, world_size = distributed_rank_world()
    validate_rank_rng_state(bundle, world_size=world_size)
    local = bundle["states"][str(rank)]
    cuda_available = torch.cuda.is_available()
    if (local["torch_cuda"] is not None) != cuda_available:
        raise RuntimeError(f"CUDA RNG availability mismatch for rank {rank}")
    random.setstate(local["python"])
    np.random.set_state(local["numpy"])
    torch.set_rng_state(local["torch_cpu"])
    if cuda_available:
        torch.cuda.set_rng_state(local["torch_cuda"], device=torch.cuda.current_device())


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
    rank_rng = capture_rank_rng_state()
    rank, world_size = distributed_rank_world()
    if rank == 0:
        output = Path(output_dir)
        save_trainable_state(model, output, fingerprint_payload)
        fingerprint = recipe_fingerprint(fingerprint_payload)
        torch.save(
            {
                "schema_version": TRAINING_STATE_SCHEMA_VERSION,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rank_rng": rank_rng,
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
            json.dumps(
                {"sha256": fingerprint, "payload": fingerprint_payload},
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
    if world_size > 1:
        dist.barrier()


def read_resume_state(input_dir, *, fingerprint_payload: dict, world_size: int) -> dict:
    state = torch.load(
        Path(input_dir) / "training_state.pt", map_location="cpu", weights_only=False
    )
    if state.get("schema_version") != TRAINING_STATE_SCHEMA_VERSION:
        raise RuntimeError("checkpoint lacks complete per-rank RNG state; use warm_start")
    if state.get("fingerprint") != recipe_fingerprint(fingerprint_payload):
        raise RuntimeError("checkpoint fingerprint mismatch; use warm_start for a new recipe")
    validate_rank_rng_state(state.get("rank_rng"), world_size=world_size)
    return state


def load_resume_state(
    input_dir, *, model, optimizer, scheduler, fingerprint_payload: dict,
    state: dict | None = None,
) -> dict:
    _, world_size = distributed_rank_world()
    if state is None:
        state = read_resume_state(
            input_dir, fingerprint_payload=fingerprint_payload, world_size=world_size
        )
    load_trainable_state(model, input_dir)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rank_rng_state(state["rank_rng"])
    return state
