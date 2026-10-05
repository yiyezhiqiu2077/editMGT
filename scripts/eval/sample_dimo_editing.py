#!/usr/bin/env python3
"""Single-transformer-forward Region-DiMO image editing inference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from src.dataset_utils import encode_prompt, tokenize_prompt
from src.dimo.checkpoint import validate_inference_checkpoint
from src.dimo.contracts import (
    DIMO_MODEL_ROLES_V11, DIMO_UPSTREAM_COMMIT, build_inference_fingerprint,
    released_base_model_identity, teacher_bundle_fingerprint,
)
from src.dimo.ema import apply_ema_to_student_role
from src.dimo.initialization import initialize_shared_model_roles
from src.dimo.one_step import one_step_edit_tokens
from src.dimo.rng import normal_noise_per_sample, stable_seed
from src.explicit_region.geometry import image_to_tensor
from src.explicit_region.masks import pixel_mask_to_token_mask
from src.explicit_region.modeling import load_released_components
from src.transformer import get_text_encoder_length
from src.v2_utils import prepare_cond_token


def latent_ids(height, width, device, dtype):
    ids = torch.zeros(height // 2, width // 2, 3)
    ids[..., 1] += torch.arange(height // 2)[:, None]
    ids[..., 2] += torch.arange(width // 2)[None, :]
    return ids.reshape(-1, 3).to(device=device, dtype=dtype)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--edit-region-mask", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--r-init", type=float, default=0.5)
    parser.add_argument("--embedding-noise-sigma", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--weights", choices=("student", "ema"), default="student")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("one-step inference requires exactly one visible GPU")
    device = torch.device("cuda")
    components = load_released_components(
        args.model_root, torch_dtype=torch.bfloat16, vq_dtype=torch.float32
    )
    base_model_identity = released_base_model_identity(components.identity)
    teacher_bundle = teacher_bundle_fingerprint(
        args.teacher_checkpoint, base_model_identity=base_model_identity, formal=False
    )
    inference_identity = build_inference_fingerprint(
        teacher_bundle_sha256=teacher_bundle["bundle_sha256"],
        base_model_identity=base_model_identity,
        model_roles=DIMO_MODEL_ROLES_V11,
        upstream_commit=DIMO_UPSTREAM_COMMIT,
    )
    roles = initialize_shared_model_roles(
        components.transformer, args.teacher_checkpoint, model_roles=DIMO_MODEL_ROLES_V11
    )
    state = torch.load(Path(args.student_checkpoint) / "training_state.pt", map_location="cpu")
    validate_inference_checkpoint(
        state,
        expected_teacher_bundle_fingerprint=teacher_bundle,
        expected_inference_fingerprint=inference_identity,
        expected_upstream_commit=DIMO_UPSTREAM_COMMIT,
    )
    if args.weights == "student":
        roles.load_role_state_dict("student", state["student_state"])
    else:
        apply_ema_to_student_role(roles, state["student_ema"])
    roles.to(device)
    roles.eval()
    if roles.base_model.training:
        raise RuntimeError("student inference must run with base_model.training == False")
    components.text_encoder.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    components.llm_encoder.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    components.vqvae.to(device=device, dtype=torch.float32).eval().requires_grad_(False)

    source_pil = Image.open(args.source_image).convert("RGB").resize((1024, 1024), Image.Resampling.BICUBIC)
    mask_pil = Image.open(args.edit_region_mask).convert("L").resize((1024, 1024), Image.Resampling.NEAREST)
    source_image = image_to_tensor(source_pil).unsqueeze(0).to(device=device, dtype=torch.float32)
    pixel_mask = torch.from_numpy(np.asarray(mask_pil, dtype=np.uint8) > 0).unsqueeze(0)
    source_tokens = prepare_cond_token(None, source_image, components.vqvae)
    grid = int(math.sqrt(source_tokens.shape[1]))
    source_tokens = source_tokens.reshape(1, grid, grid)
    region = pixel_mask_to_token_mask(
        pixel_mask, (grid, grid), mode="any_overlap", coverage_threshold=0.5,
        dilation_tokens=0, minimum_edit_tokens=1,
    ).to(device)
    prompt_ids = tokenize_prompt(
        [components.tokenizer, components.llm_tokenizer], [args.instruction],
        "CLIP_Gemma2", device=device,
    )
    empty_ids = tokenize_prompt(
        [components.tokenizer, components.llm_tokenizer], [""], "CLIP_Gemma2", device=device
    )
    conditional_hidden, conditional_pooled = encode_prompt(
        [components.text_encoder, components.llm_encoder], prompt_ids, "CLIP_Gemma2"
    )
    unconditional_hidden, unconditional_pooled = encode_prompt(
        [components.text_encoder, components.llm_encoder], empty_ids, "CLIP_Gemma2"
    )
    ids = latent_ids(grid, grid, device, source_tokens.dtype)
    text_ids = torch.zeros(
        get_text_encoder_length("CLIP_Gemma2", return_main=True), 3,
        device=device, dtype=source_tokens.dtype,
    )
    micro = torch.tensor([1024, 1024, 0, 0, 6.0], device=device, dtype=conditional_hidden.dtype)[None]
    seed_args = (args.seed, 0, "inference-sample", 0)
    noise_seed = stable_seed(*seed_args, "dimo-embedding-noise")
    noise = normal_noise_per_sample(
        torch.empty(1, roles.base_model.inner_dim, grid, grid, device=device, dtype=torch.bfloat16),
        [noise_seed],
    )
    output = one_step_edit_tokens(
        roles,
        source_tokens=source_tokens,
        edit_region_mask=region,
        prompt_condition={
            "conditional": {"encoder_hidden_states": conditional_hidden, "pooled_projections": conditional_pooled},
            "unconditional": {"encoder_hidden_states": unconditional_hidden, "pooled_projections": unconditional_pooled},
        },
        timestep_model_kwargs={
            "img_ids": ids, "txt_ids": text_ids, "micro_conds": micro,
            "reference_image_ids": ids, "reference_image_timestep": 0, "lora_scope": "both",
        },
        mask_token_id=roles.base_model.config.vocab_size - 1,
        codebook_size=roles.base_model.config.codebook_size,
        r_init=args.r_init,
        init_mask_seeds=stable_seed(*seed_args, "dimo-init-mask"),
        init_token_seeds=stable_seed(*seed_args, "dimo-init-token"),
        sample_seeds=stable_seed(*seed_args, "dimo-student-sample"),
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        cfg_scale=args.cfg_scale, target_embedding_noise=noise,
        embedding_noise_sigma=args.embedding_noise_sigma,
    )
    decoded = components.vqvae.decode(
        output.tokens,
        force_not_quantize=True,
        shape=(1, grid, grid, components.vqvae.config.latent_channels),
    ).sample.clip(0, 1)
    array = (decoded[0].float().permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(destination)
    metadata = output.metadata | {
        "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
        "student_checkpoint": str(Path(args.student_checkpoint).resolve()),
        "seed": args.seed,
        "weights_source": args.weights,
        "nan_inf_count": int((~torch.isfinite(output.logits)).sum()) + int((~torch.isfinite(decoded)).sum()),
        "output": str(destination.resolve()),
        "DIMO_EDIT_FORMAL_READY": False,
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
