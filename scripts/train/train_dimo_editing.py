#!/usr/bin/env python3
"""Guarded one-GPU Region-DiMO preparation trainer.

The formal gate is intentionally impossible to pass in this milestone. The
only executable mode is ``--prep-smoke`` and it is limited to two complete
student+auxiliary+EMA steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader

from src.dataset_utils import encode_prompt, tokenize_prompt
from src.dimo.checkpoint import config_hash, load_dimo_checkpoint, save_dimo_checkpoint
from src.dimo.contracts import (
    DIMO_UPSTREAM_COMMIT, build_inference_fingerprint, enforce_run_guard,
    load_teacher_contract, resolve_teacher_checkpoint, teacher_bundle_fingerprint,
)
from src.dimo.ema import TrainableEMA
from src.dimo.diagnostics import run_nonzero_signal_diagnostic
from src.dimo.initialization import initialize_shared_model_roles
from src.dimo.roles import audit_role_optimizers
from src.dimo.step import complete_dimo_step
from src.explicit_region.config import load_config
from src.explicit_region.dataset import MagicBrushAlignedDataset
from src.explicit_region.epoch_sampler import DeterministicEpochSampler
from src.explicit_region.fixed_dataset import FixedCorpusDataset
from src.explicit_region.masks import pixel_mask_to_token_mask
from src.explicit_region.modeling import load_released_components
from src.transformer import get_text_encoder_length
from src.v2_utils import prepare_cond_token


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def latent_ids(height: int, width: int, device, dtype) -> torch.Tensor:
    ids = torch.zeros(height // 2, width // 2, 3)
    ids[..., 1] += torch.arange(height // 2)[:, None]
    ids[..., 2] += torch.arange(width // 2)[None, :]
    return ids.reshape(-1, 3).to(device=device, dtype=dtype)


def build_dataset(config: dict, *, cursor: int, epoch: int):
    data = config["data"]
    common = dict(
        resolution=int(config["resolution"]), base_seed=int(config["seed"]),
        max_resample_attempts=int(config["geometry"]["max_resample_attempts"]),
        minimum_mask_retention=float(config["geometry"]["minimum_mask_retention"]),
    )
    if data["name"] == "fixed_200k":
        dataset = FixedCorpusDataset(
            data["manifest"], data["roots"], verify_hashes=data.get("verify_hashes_at_read", True),
            **common,
        )
        manifest_hash = sha256_file(data["manifest"])
        sampler = DeterministicEpochSampler(
            len(dataset), base_seed=config["seed"], epoch=epoch,
            train_manifest_sha256=manifest_hash, samples_consumed_in_epoch=cursor,
        )
    elif data["name"] == "magicbrush":
        dataset = MagicBrushAlignedDataset(data["manifest"], **common)
        sampler = list(range(cursor, len(dataset)))
    else:
        raise ValueError("Region-DiMO v1 supports fixed_200k or prep-only magicbrush data")
    return dataset, sampler


@torch.no_grad()
def prepare_batch(batch: dict, components, config: dict, device):
    source_image = batch["source_image"].to(device=device, dtype=components.vqvae.dtype)
    source_tokens = prepare_cond_token(None, source_image, components.vqvae)
    batch_size = source_tokens.shape[0]
    grid = int(math.sqrt(source_tokens.shape[1]))
    if grid * grid != source_tokens.shape[1]:
        raise RuntimeError("VQ source tokens must form a square grid")
    source_tokens = source_tokens.reshape(batch_size, grid, grid)
    region = pixel_mask_to_token_mask(
        batch["edit_region_mask"], (grid, grid), **config["token_mask"]
    ).to(device)
    prompts = list(batch["instruction_en"])
    prompt_ids = tokenize_prompt(
        [components.tokenizer, components.llm_tokenizer], prompts, "CLIP_Gemma2", device=device
    )
    conditional_hidden, conditional_pooled = encode_prompt(
        [components.text_encoder, components.llm_encoder], prompt_ids, "CLIP_Gemma2"
    )
    empty_ids = tokenize_prompt(
        [components.tokenizer, components.llm_tokenizer], [""] * batch_size,
        "CLIP_Gemma2", device=device,
    )
    unconditional_hidden, unconditional_pooled = encode_prompt(
        [components.text_encoder, components.llm_encoder], empty_ids, "CLIP_Gemma2"
    )
    ids = latent_ids(grid, grid, device, source_tokens.dtype)
    text_ids = torch.zeros(
        get_text_encoder_length("CLIP_Gemma2", return_main=True), 3,
        device=device, dtype=source_tokens.dtype,
    )
    micro = torch.tensor(
        [config["resolution"], config["resolution"], 0, 0, 6.0],
        device=device, dtype=conditional_hidden.dtype,
    ).repeat(batch_size, 1)
    return {
        "source_tokens": source_tokens,
        "edit_region_mask": region,
        "sample_uids": list(batch.get("sample_uid", batch.get("sample_key"))),
        "prompt_condition": {
            "conditional": {
                "encoder_hidden_states": conditional_hidden,
                "pooled_projections": conditional_pooled,
            },
            "unconditional": {
                "encoder_hidden_states": unconditional_hidden,
                "pooled_projections": unconditional_pooled,
            },
        },
        "model_kwargs": {
            "img_ids": ids,
            "txt_ids": text_ids,
            "micro_conds": micro,
            "reference_image_ids": ids,
            "reference_image_timestep": 0,
            "lora_scope": config.get("lora_scope", "both"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prep-smoke", action="store_true")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument(
        "--stop-after-steps", type=int,
        help="prep diagnostic: stop this invocation after N additional complete steps",
    )
    args = parser.parse_args()
    if args.teacher_checkpoint:
        os.environ["DIMO_TEACHER_CHECKPOINT"] = args.teacher_checkpoint
    config = load_config(args.config)
    if args.max_optimizer_steps is not None:
        config["max_optimizer_steps"] = args.max_optimizer_steps
    teacher_path = resolve_teacher_checkpoint(config.get("teacher_checkpoint"))
    contract = None
    if teacher_path and (Path(teacher_path) / "dimo_teacher_manifest.json").is_file():
        contract = load_teacher_contract(teacher_path)
    gate = enforce_run_guard(
        contract, prep_smoke=args.prep_smoke,
        max_optimizer_steps=int(config["max_optimizer_steps"]),
        formal_ready=bool(config.get("formal_ready", False)),
    )
    if not teacher_path:
        raise RuntimeError("prep smoke requires a compatible temporary teacher checkpoint")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("prep smoke requires exactly one visible GPU")
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.use_deterministic_algorithms(True)

    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "prep_status.json").write_text(
        json.dumps(gate | {"DIMO_EDIT_FORMAL_READY": False}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    device = torch.device("cuda")
    dtype = torch.bfloat16 if config.get("precision") == "bf16" else torch.float32
    components = load_released_components(
        config["model"]["repo_or_root"], torch_dtype=dtype, vq_dtype=torch.float32,
        identity_output=output / "component_identity.json",
    )
    roles = initialize_shared_model_roles(
        components.transformer, teacher_path, model_roles=config["model_roles"]
    )
    roles.to(device)
    components.text_encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)
    components.llm_encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)
    components.vqvae.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    if config.get("gradient_checkpointing", True):
        roles.base_model.enable_gradient_checkpointing()

    student_parameters = [p for _, p in roles.role_named_parameters("student")]
    auxiliary_parameters = [p for _, p in roles.role_named_parameters("auxiliary")]
    optimizer_config = config["optimizer"]
    student_optimizer = torch.optim.AdamW(
        student_parameters, lr=float(optimizer_config["student_learning_rate"]),
        betas=tuple(optimizer_config["betas"]), weight_decay=float(optimizer_config["weight_decay"]),
        foreach=False,
    )
    auxiliary_optimizer = torch.optim.AdamW(
        auxiliary_parameters, lr=float(optimizer_config["auxiliary_learning_rate"]),
        betas=tuple(optimizer_config["betas"]), weight_decay=float(optimizer_config["weight_decay"]),
        foreach=False,
    )
    student_scheduler = get_scheduler(
        optimizer_config["scheduler"], optimizer=student_optimizer,
        num_warmup_steps=int(optimizer_config["warmup_steps"]),
        num_training_steps=int(optimizer_config["horizon_steps"]),
    )
    auxiliary_scheduler = get_scheduler(
        optimizer_config["scheduler"], optimizer=auxiliary_optimizer,
        num_warmup_steps=int(optimizer_config["warmup_steps"]),
        num_training_steps=int(optimizer_config["horizon_steps"]),
    )
    ema = TrainableEMA(roles.role_named_parameters("student"), decay=float(config["ema"]["decay"]))
    audit_role_optimizers(
        roles, student_optimizer, auxiliary_optimizer,
        output / "dimo_trainable_parameter_report.json",
    )
    teacher_bundle = teacher_bundle_fingerprint(
        teacher_path, base_model_identity=components.identity, formal=False
    )
    (output / "teacher_bundle_fingerprint.json").write_text(
        json.dumps(teacher_bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inference_identity = build_inference_fingerprint(
        teacher_bundle_sha256=teacher_bundle["bundle_sha256"],
        base_model_identity=components.identity,
        model_roles=config["model_roles"],
        upstream_commit=DIMO_UPSTREAM_COMMIT,
    )
    resolved_hash = config_hash(config)
    committed_step = cursor = epoch = 0
    if args.resume:
        state = load_dimo_checkpoint(
            args.resume, roles=roles, student_optimizer=student_optimizer,
            auxiliary_optimizer=auxiliary_optimizer, student_scheduler=student_scheduler,
            auxiliary_scheduler=auxiliary_scheduler, student_ema=ema,
            expected_config_hash=resolved_hash,
            expected_teacher_bundle_fingerprint=teacher_bundle,
            expected_inference_fingerprint=inference_identity,
            expected_upstream_commit=DIMO_UPSTREAM_COMMIT,
        )
        committed_step = int(state["global_dimo_step"])
        cursor = int(state["fixed_corpus_cursor"])
        epoch = int(state["epoch"])
    dataset, sampler = build_dataset(config, cursor=cursor, epoch=epoch)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(int(config["seed"]) + 10_000 + epoch)
    loader = DataLoader(
        dataset, batch_size=int(config.get("batch_per_gpu", 1)), sampler=sampler,
        num_workers=int(config.get("num_workers", 0)), drop_last=True,
        generator=loader_generator,
    )
    torch.cuda.reset_peak_memory_stats(device)
    log_path = output / "metrics.jsonl"
    with log_path.open("a", encoding="utf-8") as log:
        invocation_stop = (
            committed_step + int(args.stop_after_steps)
            if args.stop_after_steps is not None else int(config["max_optimizer_steps"])
        )
        invocation_stop = min(invocation_stop, int(config["max_optimizer_steps"]))
        for batch in loader:
            if committed_step >= invocation_stop:
                break
            prepared = prepare_batch(batch, components, config, device)
            diagnostic_path = output / "dimo_nonzero_signal_test.json"
            if committed_step == 0 and not args.resume and not diagnostic_path.exists():
                run_nonzero_signal_diagnostic(
                    roles,
                    target_tokens=prepared["source_tokens"],
                    reference_tokens=prepared["source_tokens"],
                    edit_region_mask=prepared["edit_region_mask"],
                    prompt_condition=prepared["prompt_condition"],
                    model_kwargs=prepared["model_kwargs"],
                    output_path=diagnostic_path,
                    epsilon=1e-3,
                )
            diagnostics, artifacts = complete_dimo_step(
                roles, **prepared, student_optimizer=student_optimizer,
                auxiliary_optimizer=auxiliary_optimizer, student_scheduler=student_scheduler,
                auxiliary_scheduler=auxiliary_scheduler, student_ema=ema,
                base_seed=int(config["seed"]), epoch=epoch,
                committed_dimo_step=committed_step,
                mask_token_id=roles.base_model.config.vocab_size - 1,
                codebook_size=roles.base_model.config.codebook_size,
                config=config,
            )
            committed_step += 1
            cursor += prepared["source_tokens"].shape[0]
            diagnostics["gpu_allocated_peak_bytes"] = torch.cuda.max_memory_allocated(device)
            diagnostics["gpu_reserved_peak_bytes"] = torch.cuda.max_memory_reserved(device)
            diagnostics["sample_uids"] = prepared["sample_uids"]
            diagnostics["deterministic_trace_sha256"] = {
                name: tensor_sha256(value) for name, value in artifacts.items()
            }
            diagnostics.update(gate)
            log.write(json.dumps(diagnostics, sort_keys=True) + "\n")
            log.flush()
            checkpoint = output / f"checkpoint-{committed_step}"
            save_dimo_checkpoint(
                checkpoint, roles=roles, student_optimizer=student_optimizer,
                auxiliary_optimizer=auxiliary_optimizer, student_scheduler=student_scheduler,
                auxiliary_scheduler=auxiliary_scheduler, student_ema=ema,
                committed_dimo_step=committed_step, fixed_corpus_cursor=cursor,
                epoch=epoch, samples_consumed=cursor, config_sha256=resolved_hash,
                teacher_bundle_fingerprint=teacher_bundle,
                inference_fingerprint=inference_identity,
                upstream_commit=DIMO_UPSTREAM_COMMIT,
            )
            print(json.dumps(diagnostics, sort_keys=True), flush=True)
    if committed_step == 0:
        raise RuntimeError("prep smoke dataset produced no complete batch")


if __name__ == "__main__":
    main()
