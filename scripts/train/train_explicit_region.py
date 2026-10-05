#!/usr/bin/env python3
"""Config-driven explicit-region EditMGT SFT entry point.

The local configs deliberately keep batch size and worker count small. Formal
launchers use the same semantics, but run only after cluster asset validation.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import subprocess
import hashlib
from collections import Counter
import sys

# Required by deterministic CUDA GEMM when torch deterministic algorithms are
# enabled. It must be set before the first CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object, set_seed
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from torch.utils.data import DataLoader, Subset

from src.dataset_utils import encode_prompt, tokenize_prompt
from src.explicit_region.checkpoint import (
    load_resume_state,
    load_trainable_state,
    save_resume_state,
    save_trainable_state,
)
from src.explicit_region.config import load_config
from src.explicit_region.contracts import audit_trainable_parameters, freeze_modules
from src.explicit_region.contracts import sha256_file
from src.explicit_region.corruption import prepare_corruption
from src.explicit_region.deterministic import stable_seed
from src.explicit_region.dataset import (
    DeterministicCoreDataset, InterEditArchiveDataset, MagicBrushAlignedDataset,
)
from src.explicit_region.epoch_sampler import DeterministicEpochSampler, SAMPLER_SCHEMA_VERSION
from src.explicit_region.fixed_corpus import verify_corpus_ready
from src.explicit_region.fixed_dataset import FixedCorpusDataset
from src.explicit_region.losses import per_sample_cross_entropy
from src.explicit_region.masks import pixel_mask_to_token_mask
from src.explicit_region.modeling import load_released_components
from src.explicit_region.quality import QualityGate
from src.explicit_region.validation import run_validation
from src.transformer import get_text_encoder_length
from src.v2_utils import prepare_cond_token


DEFAULT_LORA_TARGETS = [
    "to_q", "to_k", "to_v", "to_out.0", "ff.net.2", "proj_mlp", "proj_out"
]


def prepare_latent_image_ids(height, width, device, dtype, *, up_sample=False):
    if up_sample:
        height, width = 2 * height, 2 * width
    ids = torch.zeros(height // 2, width // 2, 3)
    ids[..., 1] += torch.arange(height // 2)[:, None]
    ids[..., 2] += torch.arange(width // 2)[None, :]
    return ids.reshape(-1, 3).to(device=device, dtype=dtype)


def assert_worktree() -> None:
    expected = Path(os.environ.get("EDITMGT_WORKTREE", ROOT)).resolve()
    if ROOT.resolve() != expected:
        raise SystemExit(f"WRONG_WORKTREE expected={expected} actual={ROOT.resolve()}")


def repository_identity() -> dict:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT,
        check=True, text=True, capture_output=True,
    ).stdout.splitlines()
    dirty_digest = hashlib.sha256(diff)
    for relative in sorted(untracked):
        if relative == "before_p0_fix.patch":
            continue
        path = ROOT / relative
        if path.is_file():
            dirty_digest.update(relative.encode())
            dirty_digest.update(path.read_bytes())
    return {
        "git_sha": sha, "git_dirty": bool(status),
        "git_dirty_diff_sha256": dirty_digest.hexdigest(),
        "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
    }


def write_provenance(output: Path, config_path: str, config: dict, warm_start=None) -> None:
    payload = repository_identity() | {
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256_file(config_path), "resolved_config": config,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_compile": False,
    }
    if warm_start:
        parent = Path(warm_start).resolve()
        parent_fingerprint = json.loads((parent / "fingerprint.json").read_text(encoding="utf-8"))
        payload["warm_start_parent"] = str(parent)
        payload["warm_start_parent_fingerprint"] = parent_fingerprint["sha256"]
    (output / "run_provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def matched_modules(model, targets: list[str]) -> list[str]:
    names = []
    for name, _ in model.named_modules():
        if any(name == target or name.endswith("." + target) for target in targets):
            names.append(name)
    if not names:
        raise RuntimeError("no LoRA target modules matched")
    return names


def fingerprint_payload(config: dict, identity: dict) -> dict:
    paths = {}
    def collect(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                collect(child, f"{prefix}.{key}" if prefix else key)
        elif isinstance(value, str) and any(
            prefix.lower().endswith(token) for token in (
                "manifest", "metadata", "cache", "translation_config", "ready", "report",
            )
        ):
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(f"fingerprint input missing: {prefix}={path}")
            paths[prefix] = {"resolved_path": str(path.resolve()), "sha256": sha256_file(path)}
    collect(config.get("data", {}), "data")
    collect(config.get("validation", {}), "validation")
    repo = repository_identity()
    model_identity = {
        "repo_id": identity.get("repo_id"),
        "resolved_revision": identity.get("resolved_revision"),
        "components": {
            name: {"config_sha256": value["config_sha256"]}
            for name, value in sorted(identity.get("components", {}).items())
        },
    }
    return {
        "schema": "explicit-region-sft-v1",
        "model_snapshot": identity["snapshot_identity"],
        "model_identity": model_identity,
        "data": config["data"],
        "data_content_hashes": paths,
        "validation": config.get("validation", {"enabled": False}),
        "resolution": config["resolution"],
        "seed": config["seed"],
        "corruption": config["corruption"],
        "token_mask": config["token_mask"],
        "lora": config["lora"],
        "optimizer": config["optimizer"],
        "scheduler_horizon_steps": config["scheduler_horizon_steps"],
        "warmup_steps": config["warmup_steps"],
        "precision": config["precision"],
        "component_dtypes": config["component_dtypes"],
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "per_device_batch": config["batch_per_gpu"],
        "gradient_accumulation": config["gradient_accumulation"],
        "sampler_schema": (
            SAMPLER_SCHEMA_VERSION if config["data"]["name"] == "fixed_200k"
            else "stateless-global-index-v1"
        ),
        "loss": "per_sample_normalized_ce",
        "torch_compile": False,
        "uv_lock_sha256": repo["uv_lock_sha256"],
        "git_sha": repo["git_sha"],
        "git_dirty_diff_sha256": repo["git_dirty_diff_sha256"],
    }


def grad_report(model) -> dict:
    buckets = {"q": 0.0, "k": 0.0, "v": 0.0, "out": 0.0, "ffn": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or "lora_" not in name:
            continue
        value = float(parameter.grad.float().norm().item())
        if "to_q" in name:
            buckets["q"] += value
        elif "to_k" in name:
            buckets["k"] += value
        elif "to_v" in name:
            buckets["v"] += value
        elif "to_out" in name or "proj_out" in name:
            buckets["out"] += value
        else:
            buckets["ffn"] += value
    buckets["total"] = sum(buckets.values())
    return buckets


def tensor_norm(parameters, *, gradients=False) -> float:
    total = torch.zeros((), device=next(iter(parameters)).device, dtype=torch.float64)
    for parameter in parameters:
        value = parameter.grad if gradients else parameter
        if value is not None:
            total += value.detach().double().square().sum()
    return float(total.sqrt())


def prepare_batch(batch, components, accelerator, config, *, scope: str):
    device = accelerator.device
    transformer = accelerator.unwrap_model(components.transformer)
    source = batch["source_image"].to(device=device, dtype=components.vqvae.dtype)
    target = batch["target_image"].to(device=device, dtype=components.vqvae.dtype)
    with torch.no_grad():
        source_tokens = prepare_cond_token(None, source, components.vqvae)
        target_tokens = prepare_cond_token(None, target, components.vqvae)
    batch_size = source.shape[0]
    grid = int(math.sqrt(source_tokens.shape[1]))
    if grid * grid != source_tokens.shape[1]:
        raise RuntimeError("VQ tokens are not a square grid")
    source_tokens = source_tokens.reshape(batch_size, grid, grid)
    target_tokens = target_tokens.reshape(batch_size, grid, grid)
    token_mask = pixel_mask_to_token_mask(
        batch["edit_region_mask"],
        (grid, grid),
        **config["token_mask"],
    ).to(device)
    sample_indices = [int(value) for value in batch["global_sample_index"]]
    sample_keys = list(batch["sample_key"])
    corruption = prepare_corruption(
        source_tokens,
        target_tokens,
        token_mask,
        transformer.config.vocab_size - 1,
        mode=config["corruption"]["mode"],
        roi_probability=config["corruption"].get("roi_probability", 0.5),
        full_roi_mask_probability=config["corruption"].get("full_roi_mask_probability", 0.15),
        base_seed=config["seed"],
        global_sample_indices=sample_indices,
        sample_keys=sample_keys,
    )
    prompts = list(batch["instruction_en"])
    dropout = config.get("condition_dropout", {"enabled": False})
    dropped = 0
    if dropout.get("enabled", False):
        probability = float(dropout["probability"])
        if not 0 <= probability <= 1:
            raise ValueError("condition dropout probability must be in [0,1]")
        for i, (index, key) in enumerate(zip(sample_indices, sample_keys)):
            draw = stable_seed(config["seed"], index, key, "text-condition-dropout") / (2**63 - 2)
            if draw < probability:
                prompts[i] = ""
                dropped += 1
    object.__setattr__(corruption, "condition_dropout_count", dropped)
    prompt_ids = tokenize_prompt(
        [components.tokenizer, components.llm_tokenizer],
        prompts,
        "CLIP_Gemma2",
        device=device,
    )
    with torch.no_grad():
        encoder_hidden_states, pooled = encode_prompt(
            [components.text_encoder, components.llm_encoder], prompt_ids, "CLIP_Gemma2"
        )
    img_ids = prepare_latent_image_ids(
        grid, grid, device, corruption.input_tokens.dtype, up_sample=config["resolution"] != 1024
    )
    txt_ids = torch.zeros(
        get_text_encoder_length("CLIP_Gemma2", return_main=True), 3,
        device=device, dtype=corruption.input_tokens.dtype,
    )
    micro = torch.tensor(
        [config["resolution"], config["resolution"], 0, 0, 6.0],
        device=device, dtype=encoder_hidden_states.dtype,
    ).repeat(batch_size, 1)
    reference_ids = prepare_latent_image_ids(
        grid, grid, device, source_tokens.dtype, up_sample=config["resolution"] != 1024
    )
    logits = components.transformer(
        hidden_states=corruption.input_tokens,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled,
        timestep=corruption.scheduled_roi_ratio,
        img_ids=img_ids,
        txt_ids=txt_ids,
        micro_conds=micro,
        reference_image_hidden_states=source_tokens,
        reference_image_ids=reference_ids,
        lora_scope=scope,
        edit_region_mask=corruption.edit_region_mask,
        edit_region_conditioning_active=(
            corruption.conditioning_active
            and config["corruption"].get("persistent_conditioning", True)
        ),
    )
    loss, token_mean = per_sample_cross_entropy(logits, corruption.labels)
    object.__setattr__(corruption, "max_abs_logit", float(logits.detach().abs().max()))
    return loss, token_mean, corruption, token_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--warm-start")
    parser.add_argument("--gradient-diagnostic", action="store_true")
    args = parser.parse_args()
    assert_worktree()
    if args.resume and args.warm_start:
        raise SystemExit("resume and warm-start are mutually exclusive")
    config = load_config(args.config)
    set_seed(int(config["seed"]), device_specific=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if config.get("torch_compile", False):
        raise SystemExit("torch_compile is forbidden by v1 correctness contract")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation"],
        mixed_precision=config["precision"],
    )
    if accelerator.is_main_process:
        write_provenance(output, args.config, config, args.warm_start)
    required_subfolders = {
        "transformer_subfolder": "editmgt", "text_encoder_subfolder": "text_encoder",
        "tokenizer_subfolder": "tokenizer", "llm_encoder_subfolder": "llm_encoder",
        "vqvae_subfolder": "vqvae", "scheduler_subfolder": "scheduler",
    }
    for key, expected in required_subfolders.items():
        if config["model"].get(key) != expected:
            raise RuntimeError(f"released pipeline identity requires model.{key}={expected}")
    load_dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float32
    components = load_released_components(
        config["model"]["repo_or_root"],
        identity_output=output / "component_identity.json" if accelerator.is_main_process else None,
        torch_dtype=load_dtype,
        vq_dtype=torch.float32,
    )
    freeze_modules(
        [components.transformer, components.text_encoder, components.llm_encoder, components.vqvae]
    )
    targets = config["lora"].get("target_modules", DEFAULT_LORA_TARGETS)
    matches = matched_modules(components.transformer, targets)
    if accelerator.is_main_process:
        (output / "matched_lora_modules.json").write_text(
            json.dumps(matches, indent=2) + "\n", encoding="utf-8"
        )
    components.transformer.add_adapter(
        LoraConfig(
            r=config["lora"]["rank"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=targets,
        )
    )
    components.transformer.edit_region_embedding.requires_grad_(True)
    if args.warm_start:
        load_trainable_state(components.transformer, args.warm_start)
    components.transformer.train()
    if config.get("gradient_checkpointing", True):
        components.transformer.enable_gradient_checkpointing()
    trainable = [p for p in components.transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["optimizer"]["learning_rate"]),
        betas=tuple(config["optimizer"]["betas"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
        foreach=False,
    )
    scheduler = get_scheduler(
        config["optimizer"].get("scheduler", "constant_with_warmup"),
        optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=config["scheduler_horizon_steps"],
    )
    audit_trainable_parameters(
        {
            "transformer": components.transformer,
            "text_encoder": components.text_encoder,
            "llm_encoder": components.llm_encoder,
            "vqvae": components.vqvae,
        },
        optimizer,
        output / "trainable_parameter_report.json" if accelerator.is_main_process else None,
    )
    payload = fingerprint_payload(config, components.identity)
    initial_cursor = 0
    initial_step = 0
    dataset_kwargs = dict(
        resolution=config["resolution"], base_seed=config["seed"],
        max_resample_attempts=config["geometry"]["max_resample_attempts"],
        minimum_mask_retention=config["geometry"]["minimum_mask_retention"],
    )
    fixed_sampler = None
    current_epoch = 0
    if config["data"]["name"] == "magicbrush":
        dataset = MagicBrushAlignedDataset(config["data"]["manifest"], **dataset_kwargs)
    elif config["data"]["name"] == "core":
        weights = config["data"]["weights"]
        if weights != {"interedit": 0.8, "magicbrush": 0.2}:
            raise RuntimeError("formal CORE weights must be exactly Inter-Edit 0.80 / MagicBrush 0.20")
        if not config["data"].get("interedit_only_better_data", False):
            raise RuntimeError("formal CORE requires Inter-Edit better_data only")
        interedit = InterEditArchiveDataset(
            config["data"]["interedit_manifest"], config["data"]["interedit_root"],
            **dataset_kwargs,
        )
        magicbrush = MagicBrushAlignedDataset(config["data"]["magicbrush_manifest"], **dataset_kwargs)
        required_samples = (
            config["max_optimizer_steps"] * config["batch_per_gpu"]
            * config["gradient_accumulation"] * accelerator.num_processes
        )
        dataset = DeterministicCoreDataset(
            interedit, magicbrush, interedit_weight=0.8,
            length=required_samples, base_seed=config["seed"],
        )
    elif config["data"]["name"] == "fixed_200k":
        ready = verify_corpus_ready(config["data"]["corpus_ready"])
        if int(ready.get("total_rows", -1)) != 200000:
            raise RuntimeError("CORPUS_NOT_READY: attestation total_rows must equal 200000")
        dataset = FixedCorpusDataset(
            config["data"]["manifest"], config["data"]["roots"],
            resolution=config["resolution"], base_seed=config["seed"],
            verify_hashes=config["data"].get("verify_hashes_at_read", True),
            max_random_attempts=config["geometry"]["max_resample_attempts"],
            minimum_mask_retention=config["geometry"]["minimum_mask_retention"],
        )
        if len(dataset) != 200000:
            raise RuntimeError(f"fixed corpus must have 200000 rows, got {len(dataset)}")
    else:
        raise ValueError(f"unsupported data recipe: {config['data']['name']}")
    if args.resume:
        metadata = torch.load(Path(args.resume) / "training_state.pt", map_location="cpu")
        initial_cursor = int(metadata["committed_global_sample_count"])
        initial_step = int(metadata["global_optimizer_step"])
    if config["data"]["name"] == "fixed_200k":
        if args.resume:
            sampler_state = metadata.get("sampler_state", {})
            expected = {
                "sampler_schema_version": SAMPLER_SCHEMA_VERSION,
                "world_size": accelerator.num_processes,
                "batch_per_gpu": config["batch_per_gpu"],
                "gradient_accumulation": config["gradient_accumulation"],
            }
            for key, value in expected.items():
                if sampler_state.get(key) != value:
                    raise RuntimeError(f"RESUME_FINGERPRINT_MISMATCH sampler {key}")
            current_epoch = int(sampler_state["epoch"])
            samples_in_epoch = int(sampler_state["samples_consumed_in_epoch"])
        else:
            samples_in_epoch = 0
        dataset.set_epoch(current_epoch)
        manifest_sha = sha256_file(config["data"]["manifest"])
        fixed_sampler = DeterministicEpochSampler(
            len(dataset), base_seed=config["seed"], epoch=current_epoch,
            train_manifest_sha256=manifest_sha, rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            samples_consumed_in_epoch=samples_in_epoch,
        )
        if args.resume and fixed_sampler.epoch_permutation_sha256 != sampler_state["epoch_permutation_sha256"]:
            raise RuntimeError("RESUME_FINGERPRINT_MISMATCH epoch permutation")
        loader = DataLoader(
            dataset, sampler=fixed_sampler, batch_size=config["batch_per_gpu"],
            num_workers=config.get("num_workers", 0), pin_memory=True,
        )
        components.transformer, optimizer, scheduler = accelerator.prepare(
            components.transformer, optimizer, scheduler
        )
    else:
        indices = list(range(initial_cursor, len(dataset)))
        if not indices:
            raise RuntimeError("dataset exhausted at committed cursor")
        loader = DataLoader(
            Subset(dataset, indices), batch_size=config["batch_per_gpu"], shuffle=False,
            num_workers=config.get("num_workers", 0), pin_memory=True,
        )
        components.transformer, optimizer, loader, scheduler = accelerator.prepare(
            components.transformer, optimizer, loader, scheduler
        )
    dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float32
    components.text_encoder.to(accelerator.device, dtype=dtype)
    components.llm_encoder.to(accelerator.device, dtype=dtype)
    components.vqvae.to(accelerator.device, dtype=torch.float32)
    if args.resume:
        resumed_state = load_resume_state(
            args.resume,
            model=accelerator.unwrap_model(components.transformer),
            optimizer=optimizer,
            scheduler=scheduler,
            fingerprint_payload=payload,
        )
    else:
        resumed_state = None
    if args.gradient_diagnostic:
        batch = next(iter(loader))
        report = {}
        for scope in ("target_only", "reference_only", "both"):
            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                loss, _, _, _ = prepare_batch(batch, components, accelerator, config, scope=scope)
            accelerator.backward(loss)
            model = accelerator.unwrap_model(components.transformer)
            report[scope] = grad_report(model) | {
                "loss": float(loss.detach()),
                "mask_embedding_grad_norm": float(model.edit_region_embedding.grad.float().norm())
                if model.edit_region_embedding.grad is not None else 0.0,
            }
        if accelerator.is_main_process:
            (output / "branch_gradient_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps(report, indent=2, sort_keys=True))
        return

    global_step = initial_step
    committed = initial_cursor
    metrics_path = output / "train_metrics.jsonl"
    quality_config = config.get("quality_gate", {"enabled": False})
    quality_gate = QualityGate.from_state(
        quality_config, resumed_state.get("quality_state") if resumed_state else None
    )
    dataset_counts, corruption_counts, edit_type_counts = Counter(), Counter(), Counter()
    pending_sample_uids, pending_dataset_names, pending_edit_types = [], [], []
    pending_geometry_seeds, pending_corruption_seeds, pending_modes = [], [], []
    pending_dropout = 0
    torch.cuda.reset_peak_memory_stats(accelerator.device)
    for batch in loader:
        with accelerator.accumulate(components.transformer):
            with accelerator.autocast():
                loss, token_mean, corruption, token_mask = prepare_batch(
                    batch, components, accelerator, config, scope=config["lora"]["scope"]
                )
            batch_keys = list(batch.get("sample_uid", batch["sample_key"]))
            batch_indices = [int(value) for value in batch["global_sample_index"]]
            pending_sample_uids.extend(str(value) for value in batch_keys)
            pending_dataset_names.extend(str(value) for value in batch["dataset_name"])
            pending_edit_types.extend(str(value) for value in batch["edit_type"])
            if "geometry_seed" in batch:
                pending_geometry_seeds.extend(int(value) for value in batch["geometry_seed"])
            else:
                pending_geometry_seeds.extend(None for _ in batch_keys)
            pending_corruption_seeds.extend(
                stable_seed(config["seed"], index, key, "corruption-mode")
                for index, key in zip(batch_indices, batch_keys)
            )
            pending_modes.extend(corruption.resolved_modes)
            pending_dropout += corruption.condition_dropout_count
            accelerator.backward(loss)
            before_update = [parameter.detach().clone() for parameter in trainable]
            grad_norm_pre = tensor_norm(trainable, gradients=True)
            clip_threshold = float(config["optimizer"]["gradient_clip"])
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    components.transformer.parameters(), clip_threshold
                )
            grad_norm_post = tensor_norm(trainable, gradients=True)
            diagnostic_interval = int(config.get("gradient_diagnostic_interval", 10))
            capture_diagnostic = accelerator.sync_gradients and (
                global_step + 1 == 1 or (global_step + 1) % diagnostic_interval == 0
            )
            current_grad_report = (
                grad_report(accelerator.unwrap_model(components.transformer))
                if capture_diagnostic else None
            )
            mask_grad = accelerator.unwrap_model(components.transformer).edit_region_embedding.grad
            current_mask_grad = (
                float(mask_grad.float().norm()) if capture_diagnostic and mask_grad is not None else None
            )
            applied_learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if not accelerator.sync_gradients:
            continue
        committed_sample_uids = list(gather_object(pending_sample_uids))
        committed_dataset_names = list(gather_object(pending_dataset_names))
        committed_edit_types = list(gather_object(pending_edit_types))
        committed_geometry_seeds = list(gather_object(pending_geometry_seeds))
        committed_corruption_seeds = list(gather_object(pending_corruption_seeds))
        committed_modes = list(gather_object(pending_modes))
        committed_dropout = sum(gather_object([pending_dropout]))
        pending_sample_uids.clear(); pending_dataset_names.clear(); pending_edit_types.clear()
        pending_geometry_seeds.clear(); pending_corruption_seeds.clear(); pending_modes.clear()
        pending_dropout = 0
        global_step += 1
        update_norm = math.sqrt(sum(
            float((parameter.detach().double() - before.double()).square().sum())
            for parameter, before in zip(trainable, before_update)
        ))
        parameter_norm = tensor_norm(trainable)
        update_ratio = update_norm / max(parameter_norm, 1e-12)
        finite_parameters = all(bool(torch.isfinite(parameter).all()) for parameter in trainable)
        dataset_counts.update(committed_dataset_names)
        edit_type_counts.update(committed_edit_types)
        corruption_counts.update(committed_modes)
        committed += (
            config["batch_per_gpu"] * accelerator.num_processes * config["gradient_accumulation"]
        )
        row = {
            "global_step": global_step,
            "committed_global_sample_count": committed,
            "sample_mean_ce": float(loss.detach()),
            "token_mean_ce": float(token_mean.detach()),
            "finite": bool(torch.isfinite(loss)),
            "learning_rate": applied_learning_rate,
            "next_learning_rate": scheduler.get_last_lr()[0],
            "scheduled_roi_ratio": corruption.scheduled_roi_ratio.detach().float().cpu().tolist(),
            "actual_roi_mask_fraction": corruption.actual_roi_mask_fraction.detach().float().cpu().tolist(),
            "actual_global_mask_fraction": corruption.actual_global_mask_fraction.detach().float().cpu().tolist(),
            "token_mask_count": token_mask.flatten(1).sum(1).cpu().tolist(),
            "valid_token_count": corruption.labels.ne(-100).flatten(1).sum(1).cpu().tolist(),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(accelerator.device),
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(accelerator.device),
            "grad_norm_pre_clip": grad_norm_pre,
            "grad_norm_post_clip": grad_norm_post,
            "clip_applied": grad_norm_pre > clip_threshold,
            "parameter_norm": parameter_norm,
            "optimizer_update_norm": update_norm,
            "update_ratio": update_ratio,
            "max_abs_logit": corruption.max_abs_logit,
            "finite_parameters": finite_parameters,
            "dataset_sample_counts": dict(dataset_counts),
            "dataset_sample_fraction": {
                key: value / sum(dataset_counts.values()) for key, value in dataset_counts.items()
            },
            "corruption_mode_counts": dict(corruption_counts),
            "edit_type_counts": dict(edit_type_counts),
            "mean_roi_fraction": float(token_mask.float().mean()),
            "mean_masked_roi_ratio": float(corruption.actual_roi_mask_fraction.float().mean()),
            "mean_global_mask_fraction": float(corruption.actual_global_mask_fraction.float().mean()),
            "full_roi_count": int(corruption.full_roi_branch.sum()),
            "condition_dropout_count": committed_dropout,
            "committed_sample_uids": committed_sample_uids,
            "committed_geometry_seeds": committed_geometry_seeds,
            "committed_corruption_seeds": committed_corruption_seeds,
        }
        if current_grad_report is not None:
            row["lora_grad_diagnostic"] = current_grad_report
            row["mask_embedding_grad_norm"] = current_mask_grad
        quality_ok = quality_gate.update(
            global_step, loss=row["sample_mean_ce"], max_abs_logit=row["max_abs_logit"],
            clip_applied=row["clip_applied"], update_ratio=update_ratio,
            finite_parameters=finite_parameters,
        ) if quality_config.get("enabled", False) else True
        row["quality_status"] = quality_gate.status if quality_config.get("enabled", False) else "DISABLED"
        if accelerator.is_main_process:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
            (output / "quality_status.json").write_text(
                json.dumps(quality_gate.state_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if not row["finite"] or not quality_ok:
            raise FloatingPointError(f"training quality failure: {quality_gate.reason or 'non-finite loss'}")
        validation_config = config.get("validation", {"enabled": False})
        validation_due = (
            validation_config.get("enabled", False)
            and global_step % int(validation_config["interval_steps"]) == 0
        )
        if global_step in set(config.get("checkpoint_steps", [])) or validation_due:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_resume_state(
                    output / f"checkpoint-{global_step}",
                    model=accelerator.unwrap_model(components.transformer),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_optimizer_step=global_step,
                    committed_global_sample_count=committed,
                    sampler_schema=(
                        SAMPLER_SCHEMA_VERSION if fixed_sampler is not None
                        else "stateless-global-index-v1"
                    ),
                    fingerprint_payload=payload,
                    quality_state=quality_gate.state_dict(),
                    sampler_state=(
                        {
                            **fixed_sampler.state_dict(),
                            "global_optimizer_step_in_epoch": global_step,
                            "global_microbatch_in_epoch": global_step * config["gradient_accumulation"],
                            "samples_consumed_in_epoch": committed,
                            "batch_per_gpu": config["batch_per_gpu"],
                            "gradient_accumulation": config["gradient_accumulation"],
                        }
                        if fixed_sampler is not None else None
                    ),
                )
        if validation_due:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                run_validation(
                    components=components,
                    transformer=accelerator.unwrap_model(components.transformer),
                    train_config=config, step=global_step, output_dir=output,
                    device=accelerator.device,
                )
            accelerator.wait_for_everyone()
        if global_step >= config["max_optimizer_steps"]:
            break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_trainable_state(
            accelerator.unwrap_model(components.transformer), output / "final", payload
        )


if __name__ == "__main__":
    main()
