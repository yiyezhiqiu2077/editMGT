#!/usr/bin/env python3
"""Complete FP32 DenseDiMO with DDP, committed cursors and cluster preflight."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import math

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.pop("GH_TOKEN", None)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from src.explicit_region.config import load_config
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.contracts import sha256_file, freeze_modules
from src.explicit_region.dense_checkpoint import write_json
from src.explicit_region.epoch_sampler import DeterministicEpochSampler
from src.explicit_region.dense_runtime import DenseEpochLoader
from src.dimo.dense_roles import DenseDiMOModelRoles, initialize_dense_model_roles, initial_logits_parity
from src.dimo.dense_distributed_checkpoint import save_checkpoint, load_checkpoint
from src.dimo.ema import TrainableEMA
from src.dimo.roles import audit_role_optimizers
from src.dimo.runtime import validate_full_config, validate_output, gpu_admission, MemoryRecorder, disk_budget, append_committed_log, prepare_resume_output
from src.dimo.transaction import StepCommit, collective_status, sampled_rank_state, set_control_group
from src.dimo.step import complete_dimo_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dimo/full_dense_d200k.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--stop-at-global-step", type=int)
    parser.add_argument("--inject-failure", choices=("student_optimizer", "auxiliary_optimizer", "ema_during", "ema_complete", "before_commit"))
    parser.add_argument("--failure-rank", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    validate_full_config(config, synthetic=args.synthetic)
    if args.inject_failure and not args.synthetic:
        raise RuntimeError("FAULT_INJECTION_SYNTHETIC_ONLY")
    world, rank, local = (int(os.environ.get(name, default)) for name, default in
                          (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    if world != config["distributed"]["world_size"]:
        raise RuntimeError("FULL_DIMO_WORLD_SIZE_MISMATCH")
    if not args.synthetic and (not torch.cuda.is_available() or torch.cuda.device_count() < 8):
        raise RuntimeError("FULL_DIMO_FORMAL_REQUIRES_8_VISIBLE_GPUS")
    if args.synthetic:
        device = torch.device("cpu")
        torch.set_num_threads(1)
    else:
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("gloo" if args.synthetic else "nccl",
            timeout=timedelta(seconds=config["distributed"]["timeout_seconds"]))
        set_control_group(dist.new_group(backend="gloo",
            timeout=timedelta(seconds=config["distributed"]["control_timeout_seconds"])))
    torch.manual_seed(config["seed"])
    import random
    import numpy as np
    random.seed(config["seed"]); np.random.seed(config["seed"])
    from scripts.train.train_dense_region import configure_determinism
    configure_determinism()
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not args.synthetic and subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("FULL_DIMO_FORMAL_REQUIRES_FIXED_CLEAN_COMMIT")
    output = validate_output(config, synthetic=args.synthetic)
    output.mkdir(parents=True, exist_ok=True)
    memory = MemoryRecorder(device, output / f"memory_rank{rank}.jsonl", rank)
    commit = StepCommit()
    try:
        if not args.synthetic:
            admission = gpu_admission()
            write_json(output / f"gpu_admission_rank{rank}.json", admission)
        if args.synthetic:
            from src.dimo.synthetic import SyntheticDenseTransformer, SyntheticDataset, synthetic_batch
            roles = DenseDiMOModelRoles(SyntheticDenseTransformer()).to(device)
            dataset = SyntheticDataset(config["data"]["expected_rows"])
            bundle = {"identity": "CPU_SYNTHETIC", "formal_teacher": False}
            manifest_hash = recipe_fingerprint(dataset.rows)
            data_hash = manifest_hash
            prepare = lambda batch: synthetic_batch(batch, dataset)
        else:
            from src.explicit_region.modeling import load_released_components
            from src.explicit_region.fixed_dataset import FixedCorpusDataset
            from src.explicit_region.fixed_corpus import verify_corpus_ready
            from src.dimo.contracts import released_base_model_identity, teacher_bundle_fingerprint
            from scripts.train.train_dimo_editing import prepare_batch
            ready = verify_corpus_ready(config["data"]["corpus_ready"])
            manifest_hash = sha256_file(config["data"]["manifest"])
            if ready.get("train_manifest_sha256") != manifest_hash:
                raise RuntimeError("FULL_DIMO_CORPUS_READY_MANIFEST_MISMATCH")
            components = load_released_components(config["model"]["repo_or_root"],
                torch_dtype=torch.bfloat16, transformer_dtype=torch.float32, vq_dtype=torch.float32)
            base = released_base_model_identity(components.identity)
            bundle = teacher_bundle_fingerprint(config["teacher_checkpoint"], base_model_identity=base, formal=True)
            record = json.loads((Path(config["teacher_checkpoint"]) / "selection_record.json").read_text())
            prereg = record["selection_preregistration"]
            if prereg.get("protocol") != "full-dense-selection-v2" or prereg["train_manifest_sha256"] != manifest_hash:
                raise RuntimeError("FULL_DIMO_TEACHER_SOURCE_GROUP_SELECTION_NOT_VERIFIED")
            roles = initialize_dense_model_roles(components.transformer, config["teacher_checkpoint"], expected_base_identity=base).to(device)
            freeze_modules([components.text_encoder, components.llm_encoder, components.vqvae])
            for model in (components.text_encoder, components.llm_encoder, components.vqvae):
                model.to(device)
            dataset = FixedCorpusDataset(config["data"]["manifest"], config["data"]["roots"],
                resolution=config["resolution"], base_seed=config["seed"], verify_hashes=True,
                max_random_attempts=config["geometry"]["max_resample_attempts"],
                minimum_mask_retention=config["geometry"]["minimum_mask_retention"])
            data_hash = recipe_fingerprint({"manifest": manifest_hash, "ready": sha256_file(config["data"]["corpus_ready"])})
            prepare = lambda batch: prepare_batch(batch, components, config, device)
        if len(dataset) != config["data"]["expected_rows"]:
            raise RuntimeError("FULL_DIMO_FROZEN_ROW_COUNT_MISMATCH")
        if not args.synthetic:
            roles.enable_gradient_checkpointing()
        if world > 1:
            roles.wrap_ddp(device)
        memory.record("roles_loaded_and_ddp_wrapped", commit.step)
        opt = config["optimizer"]
        optimizers = [torch.optim.AdamW([p for _, p in roles.role_named_parameters(role)],
            lr=opt[f"{role}_learning_rate"], betas=tuple(opt["betas"]), weight_decay=opt["weight_decay"], foreach=False)
            for role in ("student", "auxiliary")]
        schedulers = [torch.optim.lr_scheduler.LambdaLR(o, lambda _: 1.0) for o in optimizers]
        ema = TrainableEMA(roles.role_named_parameters("student"), decay=.9995,
            device=device if config["ema"]["device"] == "cuda" else "cpu")
        parameter_report = audit_role_optimizers(roles, *optimizers)
        identity = {"schema": "full-dense-dimo-identity-v2", "teacher_bundle_fingerprint": bundle,
            "data_sha256": data_hash, "config_sha256": recipe_fingerprint(config), "git_sha": git_sha,
            "world_size": world, "backend": "ddp", "surrogate": config["surrogate"],
            "precision": config["component_dtypes"], "formal": not args.synthetic}
        if rank == 0:
            manifest_path = output / "experiment_manifest.json"
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
                raise RuntimeError("FULL_DIMO_OUTPUT_IDENTITY_MISMATCH")
            if (output / "metrics.jsonl").exists() and not args.resume:
                raise RuntimeError("FULL_DIMO_EXISTING_RUN_REQUIRES_RESUME")
            write_json(manifest_path, identity)
            write_json(output / "resolved_config.json", config)
            write_json(output / "parameter_report.json", parameter_report)
            write_json(output / "disk_budget.json", disk_budget(parameter_report["student_trainable"], config, output))
        collective_status()
        generator = torch.Generator().manual_seed(config["seed"] + 10000)
        resumed = None
        if args.resume:
            resumed = load_checkpoint(args.resume, roles=roles, optimizers=optimizers, schedulers=schedulers, ema=ema,
                expected_identity=identity, expected_config_hash=identity["config_sha256"])
            commit = StepCommit(resumed["step"], resumed["cursor"])
            generator.set_state(resumed["loader_rng"]["current"])
            collective_status()
            error = None
            if rank == 0:
                try:
                    prepare_resume_output(output, commit.step)
                except Exception as exc:
                    error = str(exc)
            collective_status(error)
        sampler = DeterministicEpochSampler(len(dataset), base_seed=config["seed"], epoch=0,
            train_manifest_sha256=manifest_hash, rank=rank, world_size=world, samples_consumed_in_epoch=commit.cursor)
        if commit.cursor != commit.step * world:
            raise RuntimeError("FULL_DIMO_RESUME_CURSOR_STEP_MISMATCH")
        if resumed and resumed["sampler_state"] != sampler.state_dict():
            raise RuntimeError("FULL_DIMO_RESUME_SAMPLER_IDENTITY_MISMATCH")
        loader = DenseEpochLoader(dataset, sampler, epochs=1, batch_size=1, generator=generator,
            num_workers=config["num_workers"], pin_memory=device.type == "cuda",
            resume_epoch_start_rng=resumed["loader_rng"]["epoch_start"] if resumed else None)
        stop = args.stop_at_global_step or config["max_optimizer_steps"]
        if not commit.step <= stop <= config["max_optimizer_steps"]:
            raise RuntimeError("FULL_DIMO_INVALID_STOP_STEP")
        def checkpoint(inference_only=False):
            folder = output / ("inference" if inference_only else "checkpoints") / f"checkpoint-{commit.step}"
            if folder.exists():
                from src.dimo.full_checkpoint import verify
                verify(folder, full=True, expected_identity=identity)
                marker = json.loads((folder / "COMPLETED.json").read_text())
                if marker["committed_step"] != commit.step:
                    raise RuntimeError("FULL_DIMO_EXISTING_CHECKPOINT_STEP_MISMATCH")
                return
            save_checkpoint(folder, roles=roles, optimizers=optimizers, schedulers=schedulers, ema=ema,
                step=commit.step, cursor=commit.cursor, identity=identity, config_hash=identity["config_sha256"],
                commit=commit, sampler_state=sampler.state_dict(),
                loader_rng={"current": generator.get_state(), "epoch_start": loader.epoch_start_rng}, inference_only=inference_only)
            memory.record("checkpoint_save", commit.step)
        if commit.step == 0:
            checkpoint(inference_only=True)
        def failure(phase):
            if phase == args.inject_failure and rank == args.failure_rank:
                raise RuntimeError(f"INJECTED_FAILURE_{phase}_RANK_{rank}")
        for batch in loader:
            if commit.step >= stop:
                break
            prepared = prepare(batch)
            if commit.step == 0:
                kwargs = dict(prepared["model_kwargs"], **prepared["prompt_condition"]["conditional"],
                    hidden_states=prepared["source_tokens"], reference_image_hidden_states=prepared["source_tokens"],
                    edit_region_mask=prepared["edit_region_mask"], timestep=torch.ones(1, device=device))
                parity = initial_logits_parity(roles, kwargs, **config["parity"])
                write_json(output / f"initial_parity_rank{rank}.json", parity)
            expected_index = sampler.permutation[commit.cursor + rank]
            if prepared["sample_uids"] != [dataset.rows[expected_index]["sample_uid"]]:
                raise RuntimeError("FULL_DIMO_SAMPLER_SEQUENCE_MISMATCH")
            started = time.perf_counter()
            def stage(phase):
                memory.last_phase = phase
                if commit.step == 0 or commit.step + 1 == 20:
                    memory.record(phase, commit.step + 1)
            def finalize(next_step, next_cursor):
                if ema.update_count != next_step:
                    raise RuntimeError("FULL_DIMO_EMA_COUNT_MISMATCH")
                # Scalars each step; fixed-coordinate role/optimizer/EMA checks
                # at preregistered diagnostics, avoiding full state copies.
                if next_step in (1, 20) or next_step % config["diagnostic_interval"] == 0:
                    sampled_rank_state(roles, optimizers, schedulers, ema, next_step, next_cursor)
                collective_status()
            diagnostics, _ = commit.execute(lambda: complete_dimo_step(roles, **prepared,
                student_optimizer=optimizers[0], auxiliary_optimizer=optimizers[1],
                student_scheduler=schedulers[0], auxiliary_scheduler=schedulers[1], student_ema=ema,
                base_seed=config["seed"], epoch=0, committed_dimo_step=commit.step,
                mask_token_id=5 if args.synthetic else roles.base_model.config.vocab_size - 1,
                codebook_size=5 if args.synthetic else roles.base_model.config.codebook_size,
                config=config, phase_callback=stage, failure_injector=failure if args.inject_failure else None),
                samples=world, finalize=finalize, failure_injector=failure if args.inject_failure else None)
            sampler.samples_consumed_in_epoch = commit.cursor
            # Logging is after complete-step commit; monitor never participates.
            reduced = torch.tensor([diagnostics[k] for k in ("loss_dimo", "loss_aux", "distribution_energy",
                "student_grad_norm", "aux_grad_norm", "dimo_gradient_norm")], device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(reduced); reduced /= world
            diagnostics.update(zip(("loss_dimo", "loss_aux", "distribution_energy", "student_grad_norm", "aux_grad_norm", "dimo_gradient_norm"), reduced.cpu().tolist()))
            if rank == 0:
                warnings = []
                if commit.step % config["diagnostic_interval"] == 0:
                    if diagnostics["student_grad_norm"] < 1e-12:
                        warnings.append("SMALL_STUDENT_GRADIENT_RUN_LOGIT_DIAGNOSTICS")
                    if abs(diagnostics["loss_aux"] - math.log(5 if args.synthetic else roles.base_model.config.codebook_size)) < .2:
                        warnings.append("AUX_CE_NEAR_UNIFORM_REVIEW_FIXED_BATCH_FITTING")
                    if diagnostics["distribution_energy"] < 1e-12:
                        warnings.append("SMALL_DISTRIBUTION_ENERGY_REVIEW_LOGIT_GRADIENT_AUDIT")
                append_committed_log(output / "metrics.jsonl", diagnostics | {"global_step": commit.step,
                    "committed": True, "sampler_cursor": commit.cursor, "ema_update_count": ema.update_count,
                    "warnings": warnings,
                    "seconds_per_step": time.perf_counter() - started})
                print(json.dumps({"step": commit.step, "loss_dimo": diagnostics["loss_dimo"], "loss_aux": diagnostics["loss_aux"]}), flush=True)
            if commit.step in config["resume_checkpoint_steps"] or commit.step == stop:
                checkpoint()
            if commit.step in config["evaluation_steps"]:
                checkpoint(inference_only=True)
            if not args.synthetic and commit.step == 500:
                # Only rank0 uses existing student/EMA exports in a separate
                # evaluation process. Preserve all training RNG on every rank.
                error = None
                if rank == 0:
                    try:
                        subprocess.run([sys.executable, str(ROOT / "scripts/eval/evaluate_full_dense_dimo.py"),
                            "--plan", config["evaluation"]["plan"], "--run-root", str(output), "--steps", "500",
                            "--diagnostic-count", "4"], check=True)
                    except Exception as exc:
                        error = str(exc)
                collective_status(error)
        if commit.step != stop:
            raise RuntimeError("FULL_DIMO_DATA_EXHAUSTED_BEFORE_HORIZON")
        write_json(output / f"runtime_acceptance_rank{rank}.json", {"status": "PASS", "step": commit.step,
            "world_size": world, "evidence": "CPU_SYNTHETIC" if args.synthetic else "REAL_GPU",
            "gpu_acceptance": "NOT_RUN_RESOURCE_UNAVAILABLE" if args.synthetic else "PASS_FOR_COMPLETED_STEPS"})
    except BaseException as exc:
        write_json(output / f"failure_rank{rank}.json", {"error": type(exc).__name__, "message": str(exc),
            "phase": memory.last_phase, "last_committed_step": commit.step, "last_committed_cursor": commit.cursor,
            "recovery": "RELOAD_LAST_COMPLETE_CHECKPOINT"})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
