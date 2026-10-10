#!/usr/bin/env python3
"""Independent Dense E3 orchestration; reuse the original batch/objective code."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NCCL_ALGO", "Ring")
os.environ.setdefault("NCCL_PROTO", "Simple")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, gather_object, set_seed
from diffusers.optimization import get_scheduler

from scripts.train.train_explicit_region import (prepare_batch, fingerprint_payload,
                                                repository_identity, assert_worktree)
from src.explicit_region.config import load_config
from src.explicit_region.contracts import freeze_modules, sha256_file
from src.explicit_region.dense_checkpoint import (save_dense_checkpoint, read_dense_resume,
                                                 restore_dense_resume, write_json)
from src.explicit_region.dense_runtime import (validate_dense_config, audit_released_fp32,
    check_direct_initialization, trainable_report, assert_rank_parameters, gradient_report,
    UpdateSample, DenseQualityGate, memory_record, collective_require, DenseEpochLoader)
from src.explicit_region.epoch_sampler import DeterministicEpochSampler, EpochConsumptionTracker
from src.explicit_region.fixed_corpus import verify_corpus_ready, verify_mini_corpus_ready
from src.explicit_region.fixed_dataset import FixedCorpusDataset
from src.explicit_region.modeling import load_released_components


def dense_fingerprint(config, identity, released_audit):
    # The reused fingerprint preserves the exact old data/RNG/math contract.
    compatible = dict(config, lora={"enabled": False})
    result = fingerprint_payload(compatible, identity)
    return result | {"schema": "dense-e3-fp32-recipe-v1", "training_mode": "full_transformer",
                     "released_weight_hashes": released_audit["files"],
                     "dense": config["dense"], "max_optimizer_steps": config["max_optimizer_steps"]}


def build_loader(config, accelerator, resume=None):
    data = config["data"]
    if data["name"] == "fixed_200k":
        verify_corpus_ready(data["corpus_ready"])
        expected = 200000
    elif data["name"] == "fixed_corpus_dev":
        expected = int(data["expected_rows"])
        verify_mini_corpus_ready(data["corpus_ready"], expected_rows=expected)
    else:
        raise RuntimeError("DENSE_REQUIRES_FROZEN_CORPUS")
    dataset = FixedCorpusDataset(data["manifest"], data["roots"], resolution=config["resolution"],
        base_seed=config["seed"], verify_hashes=True,
        max_random_attempts=config["geometry"]["max_resample_attempts"],
        minimum_mask_retention=config["geometry"]["minimum_mask_retention"])
    if len(dataset) != expected:
        raise RuntimeError("DENSE_FROZEN_ROW_COUNT_MISMATCH")
    world = accelerator.num_processes
    global_batch = world * config["batch_per_gpu"] * config["gradient_accumulation"]
    if len(dataset) % global_batch:
        raise RuntimeError("DENSE_SAMPLER_CANNOT_PAD_OR_DROP")
    state = resume.get("sampler_state", {}) if resume else {}
    epoch, cursor = int(state.get("epoch", 0)), int(state.get("samples_consumed_in_epoch", 0))
    sampler = DeterministicEpochSampler(len(dataset), base_seed=config["seed"], epoch=epoch,
        train_manifest_sha256=sha256_file(data["manifest"]), rank=accelerator.process_index,
        world_size=world, samples_consumed_in_epoch=cursor)
    if resume:
        for key, value in sampler.state_dict().items():
            if state.get(key) != value:
                raise RuntimeError(f"DENSE_RESUME_SAMPLER_MISMATCH: {key}")
        if resume["committed_global_sample_count"] != epoch * len(dataset) + cursor:
            raise RuntimeError("DENSE_RESUME_GLOBAL_CURSOR_MISMATCH")
        if resume["global_optimizer_step"] * global_batch != resume["committed_global_sample_count"]:
            raise RuntimeError("DENSE_RESUME_STEP_CURSOR_MISMATCH")
    generator = torch.Generator().manual_seed(config["seed"] + 10000)
    loader = DenseEpochLoader(dataset, sampler, epochs=config["epochs"],
        batch_size=config["batch_per_gpu"], num_workers=config.get("num_workers", 0),
        generator=generator, resume_epoch_start_rng=resume.get("loader_epoch_start_rng") if resume else None)
    tracker = EpochConsumptionTracker(dataset.rows, sampler, config["batch_per_gpu"],
                                       config["gradient_accumulation"], state)
    return dataset, sampler, loader, generator, tracker


def configure_determinism():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    parser.add_argument("--stop-at-global-step", type=int)
    args = parser.parse_args()
    assert_worktree()
    config = load_config(args.config)
    validate_dense_config(config, formal=config["data"]["name"] == "fixed_200k")
    configure_determinism()
    set_seed(config["seed"], device_specific=False)
    accelerator = Accelerator(mixed_precision="bf16",
        gradient_accumulation_steps=config["gradient_accumulation"], step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True,
                                                       gradient_as_bucket_view=True)])
    if accelerator.device.type != "cuda" or accelerator.num_processes != 8:
        raise RuntimeError("REAL_DENSE_E3_REQUIRES_EXCLUSIVE_8_GPU_ALLOCATION")
    output = Path(args.output_dir or config["output_dir"])
    if not args.resume and output.exists() and any(output.iterdir()):
        raise RuntimeError("DENSE_FRESH_OUTPUT_NOT_EMPTY")
    output.mkdir(parents=True, exist_ok=True)
    repo = repository_identity()
    if repo["git_dirty"]:
        raise RuntimeError("DENSE_TRAINING_REQUIRES_COMMITTED_CLEAN_CODE")
    if config["data"]["name"] == "fixed_200k":
        from scripts.train.verify_e3_dense_config import verify_training_admission
        verify_training_admission(config, args.resume, check_gpu=False)
    audit = audit_released_fp32(config["model"]["repo_or_root"])
    components = load_released_components(config["model"]["repo_or_root"], torch_dtype=torch.bfloat16,
                                          transformer_dtype=torch.float32, vq_dtype=torch.float32)
    direct = check_direct_initialization(components.transformer, config["model"]["repo_or_root"], audit)
    freeze_modules([components.text_encoder, components.llm_encoder, components.vqvae])
    components.transformer.requires_grad_(True).train()
    components.transformer.enable_gradient_checkpointing()
    opt = config["optimizer"]
    optimizer = torch.optim.AdamW(components.transformer.parameters(), lr=opt["learning_rate"],
        betas=tuple(opt["betas"]), weight_decay=opt["weight_decay"], foreach=False)
    report = trainable_report(components, optimizer)
    scheduler = get_scheduler(opt["scheduler"], optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"], num_training_steps=config["scheduler_horizon_steps"])
    payload = dense_fingerprint(config, components.identity, audit)
    resume = read_dense_resume(args.resume, fingerprint_payload=payload, world_size=8) if args.resume else None
    dataset, sampler, loader, generator, tracker = build_loader(config, accelerator, resume)
    components.transformer.to(accelerator.device)
    components.text_encoder.to(accelerator.device, dtype=torch.bfloat16)
    components.llm_encoder.to(accelerator.device, dtype=torch.bfloat16)
    components.vqvae.to(accelerator.device, dtype=torch.float32)
    rank_memory = output / f"memory-rank{accelerator.process_index}.jsonl"

    def record_memory(phase, step=0):
        with rank_memory.open("a") as handle:
            handle.write(json.dumps(memory_record(accelerator.device, phase, step)) + "\n")

    record_memory("fp32_loaded")
    components.transformer, optimizer, scheduler = accelerator.prepare(components.transformer, optimizer, scheduler)
    record_memory("ddp_wrapped")
    model = accelerator.unwrap_model(components.transformer)
    if resume:
        # Prime DDP bucket topology without committing samples; restore RNG and
        # every state afterwards, matching the existing E3 resume semantics.
        with accelerator.autocast():
            loss, _, _, _ = prepare_batch(next(iter(loader)), components, accelerator, config, scope="both")
        accelerator.backward(loss)
        optimizer.zero_grad(set_to_none=True)
        restore_dense_resume(model, optimizer, scheduler, resume, args.resume, loader_generator=generator)
    assert_rank_parameters(model)
    step = int(resume["global_optimizer_step"]) if resume else 0
    committed = int(resume["committed_global_sample_count"]) if resume else 0
    stop = args.stop_at_global_step or config["max_optimizer_steps"]
    if not step < stop <= config["max_optimizer_steps"]:
        raise RuntimeError("DENSE_INVALID_STOP_BOUNDARY")
    gate = DenseQualityGate(config["quality_gate"], resume["quality_state"] if resume else None)
    if accelerator.is_main_process:
        write_json(output / "trainable_parameter_report.json", report)
        write_json(output / "initialization_report.json", audit | {"direct_loading": direct})
        write_json(output / "run_provenance.json", repo | {"resolved_config": config, "fingerprint": payload})
    pending, maximum, started = [], 0.0, time.perf_counter()
    samples_per_step = config["gradient_accumulation"] * config["batch_per_gpu"]
    for batch in loader:
        diagnostic = step + 1 in config["dense"]["diagnostic_steps"] or (step + 1) % config["dense"]["diagnostic_interval"] == 0
        with accelerator.accumulate(components.transformer):
            with accelerator.autocast():
                loss, token_mean, corruption, _ = prepare_batch(batch, components, accelerator, config, scope="both")
            collective_require(bool(torch.isfinite(loss)), "DENSE_NONFINITE_LOSS", accelerator.device)
            if step == 0 and not pending:
                record_memory("first_forward")
            accelerator.backward(loss)
            if step == 0 and not pending:
                record_memory("first_backward")
            pending.extend([{ "uid": str(uid), "loss": float(loss.detach()),
                "geometry_seed": int(geometry_seed), "corruption_seed": int(seed)}
                for uid, geometry_seed, seed in zip(batch["sample_uid"], batch["geometry_seed"], corruption_seed_trace(batch, config))])
            maximum = max(maximum, corruption.max_abs_logit)
            if not accelerator.sync_gradients:
                continue
            gradient = gradient_report(model, allowed_unused=config["dense"]["allowed_unused_parameters"]) if diagnostic else None
            preclip = accelerator.clip_grad_norm_(components.transformer.parameters(), opt["gradient_clip"])
            collective_require(bool(torch.isfinite(preclip)), "DENSE_NONFINITE_GRADIENT_NORM", accelerator.device)
            sampled = UpdateSample(model) if diagnostic else None
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            collective_require(not accelerator.optimizer_step_was_skipped, "DENSE_OPTIMIZER_UPDATE_SKIPPED", accelerator.device)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        step += 1
        committed += samples_per_step * accelerator.num_processes
        gathered = gather_object(pending)
        epoch_row = tracker.commit([r["uid"] for r in gathered], [r["loss"] for r in gathered])
        pending = []
        updates = sampled.finish(require_update=lr > 0) if sampled else {"dense_update_norm": None, "dense_update_ratio": None,
                                                   "update_diagnostic_scope": "NOT_COLLECTED", "representative_module_updates": None}
        finite = bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()) if diagnostic else None
        if diagnostic:
            collective_require(finite, "DENSE_NONFINITE_PARAMETER", accelerator.device)
            trainable_report(components, optimizer)
            consistency = assert_rank_parameters(model)
            for state in optimizer.state.values():
                if any(v.dtype != torch.float32 for k, v in state.items() if k in ("exp_avg", "exp_avg_sq")):
                    raise RuntimeError("DENSE_ADAM_MOMENT_NOT_FP32")
        else:
            consistency = None
        if step in (1, 4, 8, 16):
            record_memory("optimizer_update_complete", step)
        if scheduler.state_dict()["last_epoch"] != step:
            raise RuntimeError("DENSE_SCHEDULER_SUCCESS_COUNT_MISMATCH")
        mean_ce = sum(r["loss"] for r in gathered) / len(gathered)
        quality_ok = gate.update(step, loss=mean_ce, max_abs_logit=maximum,
            clip_applied=bool(preclip > opt["gradient_clip"]), update_ratio=updates["dense_update_ratio"],
            finite_parameters=finite)
        row = epoch_row | updates | {"global_step": step, "committed_global_sample_count": committed,
            "optimizer_updates": step, "scheduler_updates": scheduler.state_dict()["last_epoch"],
            "sample_mean_ce": mean_ce, "token_mean_ce_last_microbatch": float(token_mean.detach()),
            "max_abs_logit": maximum, "learning_rate": lr, "grad_norm_pre_clip": float(preclip),
            "finite_parameters": finite, "gradient_diagnostic": gradient, "rank_consistency": consistency,
            "quality": gate.state_dict(), "seconds_per_optimizer_step": time.perf_counter() - started,
            "memory": memory_record(accelerator.device, "step", step), "sample_trace": gathered}
        if accelerator.is_main_process:
            with (output / "train_metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(json.dumps(row, allow_nan=False), flush=True)
        collective_require(quality_ok, "DENSE_QUALITY_GATE_FAILED", accelerator.device)
        validation_due = config["validation"].get("enabled") and step % config["validation"]["interval_steps"] == 0
        if step in config["checkpoint_steps"] or step == stop or validation_due:
            state = sampler.state_dict() | tracker.state_dict() | {"batch_per_gpu": config["batch_per_gpu"],
                "gradient_accumulation": config["gradient_accumulation"]}
            save_dense_checkpoint(output / f"checkpoint-{step}", model=model, optimizer=optimizer,
                scheduler=scheduler, fingerprint_payload=payload, global_optimizer_step=step,
                committed_global_sample_count=committed, sampler_state=state, quality_state=gate.state_dict(),
                loader_generator=generator, loader_epoch_start_rng=loader.epoch_start_rng,
                max_shard_bytes=config["dense"]["max_shard_bytes"])
        if validation_due:
            from src.explicit_region.dense_inference import periodic_dense_validation
            accelerator.wait_for_everyone()
            error = None
            if accelerator.is_main_process:
                try:
                    validation_result = periodic_dense_validation(components, model, config, output, step, accelerator.device)
                    from src.explicit_region.validation import record_periodic_diagnostic
                    record_periodic_diagnostic(output, row, validation_result)
                except Exception as exc:
                    error = str(exc)
            collective_require(error is None, f"DENSE_VALIDATION_FAILED: {error}", accelerator.device)
        if step >= stop:
            break
        maximum, started = 0.0, time.perf_counter()
    if step != stop:
        raise RuntimeError("DENSE_DATA_EXHAUSTED_BEFORE_STOP")
    accelerator.wait_for_everyone()


def corruption_seed_trace(batch, config):
    from src.explicit_region.deterministic import stable_seed
    return [stable_seed(config["seed"], int(index), key, "corruption-mode")
            for index, key in zip(batch["global_sample_index"], batch["sample_key"])]


if __name__ == "__main__":
    main()
