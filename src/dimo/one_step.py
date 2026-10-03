"""Exactly-one-student-forward token inference for Region-DiMO."""

from __future__ import annotations

from dataclasses import dataclass
import time
from collections.abc import Mapping, Sequence

import torch

from .conditioning import predict_role_logits
from .initialization import build_editing_initial_state, sample_student_tokens


@dataclass(frozen=True)
class OneStepTokenOutput:
    tokens: torch.Tensor
    logits: torch.Tensor
    initial_tokens: torch.Tensor
    metadata: dict[str, object]


@torch.inference_mode()
def one_step_edit_tokens(
    roles,
    *,
    source_tokens: torch.Tensor,
    edit_region_mask: torch.Tensor,
    prompt_condition: Mapping[str, Mapping[str, torch.Tensor]],
    timestep_model_kwargs: Mapping[str, object],
    mask_token_id: int,
    codebook_size: int,
    r_init: float,
    init_mask_seeds: int | Sequence[int],
    init_token_seeds: int | Sequence[int],
    sample_seeds: int | Sequence[int],
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    cfg_scale: float = 1.0,
    target_embedding_noise: torch.Tensor | None = None,
    embedding_noise_sigma: float = 0.0,
) -> OneStepTokenOutput:
    started = time.perf_counter()
    initial = build_editing_initial_state(
        source_tokens,
        edit_region_mask,
        mask_token_id,
        codebook_size,
        r_init,
        mask_seeds=init_mask_seeds,
        token_seeds=init_token_seeds,
    )
    model_kwargs = dict(timestep_model_kwargs)
    model_kwargs.update(
        target_embedding_noise=target_embedding_noise,
        target_embedding_noise_sigma=embedding_noise_sigma,
    )
    logits = predict_role_logits(
        roles,
        "student",
        initial.initial_tokens,
        source_tokens,
        prompt_condition,
        edit_region_mask,
        initial.initial_roi_mask_ratio.to(source_tokens.device),
        cfg_scale,
        model_kwargs=model_kwargs,
        training=False,
    )
    output = sample_student_tokens(
        logits,
        source_tokens,
        edit_region_mask,
        seeds=sample_seeds,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    if not torch.equal(output[~edit_region_mask.bool()], source_tokens[~edit_region_mask.bool()]):
        raise RuntimeError("OUTSIDE_REGION_LOCK_VIOLATION")
    return OneStepTokenOutput(
        output,
        logits,
        initial.initial_tokens,
        {
            "r_init": float(r_init),
            "embedding_noise_sigma": float(embedding_noise_sigma),
            "temperature": float(temperature),
            "latency": time.perf_counter() - started,
            "number_of_transformer_forwards": 1,
            "cfg_scale": float(cfg_scale),
            "effective_cfg_batch_multiplier": 1 if cfg_scale == 1 else 2,
            "outside_mismatch_count": 0,
        },
    )
