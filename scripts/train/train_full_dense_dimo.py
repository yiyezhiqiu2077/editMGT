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
from src.dimo.recipe import is_v2, batches_per_step, recipe_identity
from src.dimo.fresh_batch import PairedEpochLoader, verify_batch_pair
from src.explicit_region.magicbrush_experiment import is_magicbrush, verify_magicbrush_ready
from src.dimo.mixed_experiment import is_mixed_dimo, verify_ready as verify_mixed_dimo_ready, verify_teacher_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dimo/full_dense_d200k.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--fork-dimo-checkpoint", help="explicit weights-only Soft-DiMO fork")
    parser.add_argument("--continue-soft-reward-checkpoint", help="authorized same-recipe full-state continuation")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--stop-at-global-step", type=int)
    parser.add_argument("--inject-failure", choices=("student_optimizer", "fresh_student_forward", "auxiliary_optimizer", "ema_during", "ema_complete", "before_commit"))
    parser.add_argument("--failure-rank", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    validate_full_config(config, synthetic=args.synthetic)
    reward_config = config.get("soft_reward")
    continuation = config.get('soft_reward_continuation')
    if reward_config:
        if args.synthetic:
            raise RuntimeError("SOFT_REWARD_USE_CPU_UNIT_TESTS_NOT_FORMAL_SYNTHETIC_FLAG")
        step_cap = continuation['target_step'] if continuation else reward_config['max_new_steps']
        if args.stop_at_global_step is None or not 0 < args.stop_at_global_step <= step_cap:
            raise RuntimeError("SOFT_REWARD_EXPLICIT_STOP_REQUIRED_MAX_500_NO_LONG_TRAINING")
        if sum(bool(x) for x in (args.resume, args.fork_dimo_checkpoint, args.continue_soft_reward_checkpoint)) != 1:
            raise RuntimeError("SOFT_REWARD_EXACTLY_ONE_EXPLICIT_FORK_OR_RESUME_REQUIRED")
        if continuation and args.fork_dimo_checkpoint:
            raise RuntimeError('SOFT_REWARD_LONG_REQUIRES_FULL_STATE_CONTINUATION')
        if args.continue_soft_reward_checkpoint and not continuation:
            raise RuntimeError('SOFT_REWARD_LONG_REQUIRES_EXPLICIT_AUTHORIZATION')
        if args.fork_dimo_checkpoint and Path(args.fork_dimo_checkpoint).resolve() != Path(reward_config["fork_checkpoint"]).resolve():
            raise RuntimeError("SOFT_REWARD_USER_SELECTED_FORK_MISMATCH")
        if args.stop_at_global_step > 20:
            from scripts.train.run_soft_dimo_reward import validate_short_gate
            validate_short_gate(Path(continuation['source']).parent.parent if continuation else Path(config["output_dir"]))
    elif args.fork_dimo_checkpoint or args.continue_soft_reward_checkpoint:
        raise RuntimeError("SOFT_REWARD_FORK_REQUIRES_REWARD_CONFIG")
    v2 = is_v2(config)
    runtime_stats = {"student_clip_count": 0}
    if args.inject_failure and not args.synthetic:
        raise RuntimeError("FAULT_INJECTION_SYNTHETIC_ONLY")
    world, rank, local = (int(os.environ.get(name, default)) for name, default in
                          (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    if world != config["distributed"]["world_size"]:
        raise RuntimeError("FULL_DIMO_WORLD_SIZE_MISMATCH")
    exploratory = is_magicbrush(config) or is_mixed_dimo(config)
    required_world = 4 if exploratory else 8
    if not args.synthetic and (not torch.cuda.is_available() or torch.cuda.device_count() < required_world):
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
            admission = gpu_admission(magicbrush=exploratory,
                allow_other_jobs=exploratory and not v2 and os.environ.get('EDITMGT_ALLOW_SHARED_A800') == '1')
            write_json(output / f"gpu_admission_rank{rank}.json", admission)
        if args.synthetic:
            from src.dimo.synthetic import (SyntheticDenseTransformer, SyntheticDenseTransformerV2,
                                            SyntheticDataset, synthetic_batch, synthetic_v2_batch)
            model = SyntheticDenseTransformerV2() if v2 else SyntheticDenseTransformer()
            roles = DenseDiMOModelRoles(model, freeze_embed=v2, freeze_mlm_head=v2).to(device)
            dataset = SyntheticDataset(config["data"]["expected_rows"])
            bundle = {"identity": "CPU_SYNTHETIC", "formal_teacher": False}
            manifest_hash = recipe_fingerprint(dataset.rows)
            data_hash = manifest_hash
            prepare = lambda batch: (synthetic_v2_batch if v2 else synthetic_batch)(batch, dataset)
        else:
            from src.explicit_region.modeling import load_released_components
            from src.explicit_region.fixed_dataset import FixedCorpusDataset
            from src.explicit_region.fixed_corpus import verify_corpus_ready
            from src.dimo.contracts import released_base_model_identity, teacher_bundle_fingerprint
            from scripts.train.train_dimo_editing import prepare_batch
            ready = (verify_mixed_dimo_ready(config) if is_mixed_dimo(config) else
                     verify_magicbrush_ready(config) if is_magicbrush(config) else verify_corpus_ready(config["data"]["corpus_ready"]))
            manifest_hash = sha256_file(config["data"]["manifest"])
            if ready.get("train_manifest_sha256") != manifest_hash:
                raise RuntimeError("FULL_DIMO_CORPUS_READY_MANIFEST_MISMATCH")
            components = load_released_components(config["model"]["repo_or_root"],
                torch_dtype=torch.bfloat16, transformer_dtype=torch.float32, vq_dtype=torch.float32)
            base = released_base_model_identity(components.identity)
            bundle = teacher_bundle_fingerprint(config["teacher_checkpoint"], base_model_identity=base, formal=not exploratory)
            record = json.loads((Path(config["teacher_checkpoint"]) / "selection_record.json").read_text())
            if is_mixed_dimo(config):
                from src.explicit_region.dense_checkpoint import verify_dense_checkpoint
                verify_teacher_record(record, verify_dense_checkpoint(config['teacher_checkpoint'])['checkpoint_sha256'])
                if (record['authorization_sha256'] != sha256_file(config['user_transition_record'])
                        or record['data_identity']['train_manifest_sha256'] != manifest_hash
                        or record['data_identity']['corpus_ready_sha256'] != sha256_file(config['data']['corpus_ready'])):
                    raise RuntimeError('MIXED_DIMO_TEACHER_TRAIN_IDENTITY_CHANGED')
            elif is_magicbrush(config):
                from src.dimo.magicbrush_selection import verify_selection_record
                from src.explicit_region.dense_checkpoint import verify_dense_checkpoint
                verify_selection_record(record, verify_dense_checkpoint(config['teacher_checkpoint'])['checkpoint_sha256'])
                source_identity = (record['data_identity'] if record['selection_status']=='USER_SELECTED'
                                   else record['selection_preregistration'])
                if source_identity['train_manifest_sha256'] != manifest_hash:
                    raise RuntimeError('MAGICBRUSH_TEACHER_TRAIN_IDENTITY_CHANGED')
            else:
                prereg = record["selection_preregistration"]
                if prereg.get("protocol") != "full-dense-selection-v2" or prereg["train_manifest_sha256"] != manifest_hash:
                    raise RuntimeError("FULL_DIMO_TEACHER_SOURCE_GROUP_SELECTION_NOT_VERIFIED")
            roles = initialize_dense_model_roles(components.transformer, config["teacher_checkpoint"], expected_base_identity=base, model_roles=config["model_roles"]).to(device)
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
        optimizers = [torch.optim.AdamW([p for _, p in roles.role_named_parameters(role) if p.requires_grad],
            lr=opt[f"{role}_learning_rate"], betas=tuple(opt["betas"]), weight_decay=opt["weight_decay"], foreach=False)
            for role in ("student", "auxiliary")]
        schedulers = [torch.optim.lr_scheduler.LambdaLR(o, lambda _: 1.0) for o in optimizers]
        ema = TrainableEMA(roles.role_named_parameters("student"), decay=.9995,
            device=device if config["ema"]["device"] == "cuda" else "cpu")
        parameter_report = audit_role_optimizers(roles, *optimizers)
        identity = {"schema": "full-dense-dimo-identity-v2", "teacher_bundle_fingerprint": bundle,
            "data_sha256": data_hash, "config_sha256": recipe_fingerprint(config), "git_sha": git_sha,
            "world_size": world, "backend": "ddp", "surrogate": config["surrogate"],
            "precision": config["component_dtypes"], "initialization": config["initialization"],
            "formal": not args.synthetic and not exploratory}
        frozen_hashes = roles.frozen_parameter_hashes() if v2 else None
        if v2:
            identity.update(schema="full-dense-dimo-identity-v3", recipe="dimo-v2",
                            recipe_contract=recipe_identity(config, roles), frozen_parameter_hashes=frozen_hashes)
        reward_objective = None
        if reward_config:
            from src.dimo.soft_reward import build_reward_objective
            identity["reward_extension"] = reward_config
            if continuation:
                identity['reward_continuation'] = continuation
            reward_objective = build_reward_objective(components.vqvae, reward_config, device)
            runtime_stats["reward_calibration"] = reward_objective.calibration.state
            memory.record("frozen_reward_models_loaded", 0)
        if exploratory:
            identity.update(experiment_mode=config['experiment_mode'],initialization=config['initialization'],
                            teacher_source=record['teacher_source'],selection_scope=record['selection_scope'])
            if is_mixed_dimo(config):
                identity['authorization_sha256'] = sha256_file(config['user_transition_record'])
                identity['selected_sft_step'] = record['selected_step']
            # Selection metadata above is part of the immutable inference identity.

        if continuation:
            from src.dimo.reward_continuation import validate_continuation
            validate_continuation(config, identity)
        if rank == 0:
            manifest_path = output / "experiment_manifest.json"
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
                raise RuntimeError("FULL_DIMO_OUTPUT_IDENTITY_MISMATCH")
            if (output / "metrics.jsonl").exists() and not args.resume:
                raise RuntimeError("FULL_DIMO_EXISTING_RUN_REQUIRES_RESUME")
            write_json(manifest_path, identity)
            write_json(output / "resolved_config.json", config)
            write_json(output / "parameter_report.json", parameter_report)
            write_json(output / "disk_budget.json", disk_budget(parameter_report.get("student_total", parameter_report["student_trainable"]), config, output,
                trainable_parameters=parameter_report["student_trainable"], stop_at=args.stop_at_global_step))
        collective_status()
        generator = torch.Generator().manual_seed(config["seed"] + 10000)
        resumed = None
        if args.fork_dimo_checkpoint:
            from src.dimo.reward_fork import fork_weights
            fork_weights(args.fork_dimo_checkpoint, roles, optimizers, ema,
                reward_config["fork_identity"], identity)
            roles.assert_frozen_unchanged(frozen_hashes)
            memory.record("weights_only_fork_fresh_optimizers", 0)
        if args.resume or args.continue_soft_reward_checkpoint:
            if exploratory:
                # Prime the real DDP backward/bucket topology without any
                # optimizer or EMA update. Loading below restores every RNG.
                from torch.utils.data import default_collate
                prepared_prime=prepare(default_collate([dataset[0]]))
                for role in ('student','auxiliary'):
                    kwargs=dict(prepared_prime['model_kwargs'],**prepared_prime['prompt_condition']['conditional'],
                        hidden_states=prepared_prime['source_tokens'],reference_image_hidden_states=prepared_prime['source_tokens'],
                        edit_region_mask=prepared_prime['edit_region_mask'],timestep=torch.ones(1,device=device))
                    logits=roles.forward_role(role,training=True,**kwargs)
                    collective_status(None if torch.isfinite(logits).all().item() else 'NONFINITE_RESUME_DDP_PRIMING_LOGITS')
                    (logits.float().sum()*0).backward()
                    roles.model_for(role).zero_grad(set_to_none=True)
                    del logits
                memory.record('resume_ddp_backward_priming_no_optimizer',0)
                del prepared_prime
            if args.continue_soft_reward_checkpoint:
                from src.dimo.reward_continuation import load_continuation
                resumed = load_continuation(args.continue_soft_reward_checkpoint, config=config, new_identity=identity,
                    roles=roles, optimizers=optimizers, schedulers=schedulers, ema=ema)
                memory.record('authorized_full_state_continuation_loaded', resumed['step'])
            else:
                resumed = load_checkpoint(args.resume, roles=roles, optimizers=optimizers, schedulers=schedulers, ema=ema,
                    expected_identity=identity, expected_config_hash=identity["config_sha256"])
            commit = StepCommit(resumed["step"], resumed["cursor"])
            generator.set_state(resumed["loader_rng"]["current"])
            if v2:
                runtime_stats = resumed["runtime_stats"]
            if reward_objective is not None:
                reward_objective.calibration.load_state_dict(runtime_stats["reward_calibration"])
            collective_status()
            error = None
            if rank == 0:
                try:
                    prepare_resume_output(output, commit.step)
                except Exception as exc:
                    error = str(exc)
            collective_status(error)
        resumed_epoch = resumed['sampler_state']['epoch'] if resumed else 0
        resumed_cursor = resumed['sampler_state']['samples_consumed_in_epoch'] if resumed else 0
        if commit.cursor != resumed_epoch*len(dataset)+resumed_cursor:
            raise RuntimeError('FULL_DIMO_RESUME_GLOBAL_EPOCH_CURSOR_MISMATCH')
        sampler = DeterministicEpochSampler(len(dataset), base_seed=config["seed"], epoch=resumed_epoch,
            train_manifest_sha256=manifest_hash, rank=rank, world_size=world,
            samples_consumed_in_epoch=resumed_cursor)
        if commit.cursor != commit.step * world * config["batch_per_gpu"] * batches_per_step(config):
            raise RuntimeError("FULL_DIMO_RESUME_CURSOR_STEP_MISMATCH")
        if resumed and resumed["sampler_state"] != sampler.state_dict():
            raise RuntimeError("FULL_DIMO_RESUME_SAMPLER_IDENTITY_MISMATCH")
        loader = DenseEpochLoader(dataset, sampler, epochs=config['epochs'], batch_size=1, generator=generator,
            num_workers=config["num_workers"], pin_memory=device.type == "cuda",
            resume_epoch_start_rng=resumed["loader_rng"]["epoch_start"] if resumed else None)
        stop = args.stop_at_global_step if args.stop_at_global_step is not None else config["max_optimizer_steps"]
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
                loader_rng={"current": generator.get_state(), "epoch_start": loader.epoch_start_rng},
                runtime_stats=runtime_stats if v2 else None, inference_only=inference_only)
            memory.record("checkpoint_save", commit.step)
        if commit.step == 0:
            checkpoint(inference_only=True)
        def failure(phase):
            if phase == args.inject_failure and rank == args.failure_rank:
                raise RuntimeError(f"INJECTED_FAILURE_{phase}_RANK_{rank}")
        from itertools import islice
        iteration = PairedEpochLoader(loader) if v2 else loader
        for batch in islice(iteration, stop - commit.step):
            if v2:
                error, prepared, prepared_b = None, None, None
                try:
                    prepared, prepared_b = prepare(batch[0]), prepare(batch[1])
                except Exception as exc:
                    error = f"DIMO_FRESH_BATCH_PREPARATION: {exc}"
                collective_status(error)
                consumed = verify_batch_pair(prepared, prepared_b, dataset=dataset, sampler=sampler, cursor=commit.cursor)
            else:
                prepared, prepared_b, consumed = prepare(batch), None, world
            if commit.step == 0 and not reward_config:
                kwargs = dict(prepared["model_kwargs"], **prepared["prompt_condition"]["conditional"],
                    hidden_states=prepared["source_tokens"], reference_image_hidden_states=prepared["source_tokens"],
                    edit_region_mask=prepared["edit_region_mask"], timestep=torch.ones(1, device=device))
                parity = initial_logits_parity(roles, kwargs, **config["parity"])
                write_json(output / f"initial_parity_rank{rank}.json", parity)
            epoch_cursor = commit.cursor - sampler.epoch * len(dataset)
            expected_index = sampler.permutation[epoch_cursor + rank]
            if not v2 and prepared["sample_uids"] != [dataset.rows[expected_index]["sample_uid"]]:
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
                if exploratory and next_step in (2,20):
                    from src.explicit_region.dense_runtime import assert_rank_parameters
                    for role in ('student','auxiliary'):
                        check=assert_rank_parameters(roles.model_for(role))
                        write_json(output/f'full_rank_{role}_step{next_step}_rank{rank}.json',check)
                if v2 and next_step in config.get("full_update_diagnostic_steps", []):
                    roles.assert_frozen_unchanged(frozen_hashes)
                collective_status()
            diagnostics, _ = commit.execute(lambda: complete_dimo_step(roles, **prepared,
                student_optimizer=optimizers[0], auxiliary_optimizer=optimizers[1],
                student_scheduler=schedulers[0], auxiliary_scheduler=schedulers[1], student_ema=ema,
                base_seed=config["seed"], epoch=sampler.epoch, committed_dimo_step=commit.step,
                mask_token_id=5 if args.synthetic else roles.base_model.config.vocab_size - 1,
                codebook_size=5 if args.synthetic else roles.base_model.config.codebook_size,
                config=config, fresh_batch=prepared_b, phase_callback=stage, failure_injector=failure if args.inject_failure else None,
                reward_objective=reward_objective,
                reward_batch=({"target": batch[0]["target_image"].to(device=device, dtype=torch.float32),
                               "pixel_mask": batch[0]["edit_region_mask"].to(device=device)} if reward_config else None)),
                samples=consumed, finalize=finalize, failure_injector=failure if args.inject_failure else None)
            sampler.samples_consumed_in_epoch = commit.cursor - sampler.epoch * len(dataset)
            if v2:
                if reward_objective is not None:
                    runtime_stats["reward_calibration"] = reward_objective.calibration.state
                runtime_stats["student_clip_count"] += int(diagnostics["student_clip_triggered"])
                diagnostics.update(student_clip_fraction=runtime_stats["student_clip_count"] / commit.step,
                    budget_basis="total_data_consumption", total_sample_consumptions=commit.cursor,
                    student_update_count=commit.step, auxiliary_update_count=commit.step,
                    student_update_sample_consumptions=commit.step * world * config["batch_per_gpu"],
                    auxiliary_update_sample_consumptions=commit.step * world * config["batch_per_gpu"])
                append_committed_log(output/f'trace_rank{rank}.jsonl', {'global_step': commit.step,
                    'sample_uids_a': prepared['sample_uids'], 'sample_uids_b': prepared_b['sample_uids'],
                    'epoch': sampler.epoch, 'global_cursor': commit.cursor})
            elif exploratory:
                append_committed_log(output/f'trace_rank{rank}.jsonl',{'global_step':commit.step,
                    'sample_uids':prepared['sample_uids'],'epoch':sampler.epoch,'global_cursor':commit.cursor})
            # Logging is after complete-step commit; monitor never participates.
            reduced = torch.tensor([diagnostics[k] for k in ("loss_dimo", "loss_aux", "distribution_energy",
                "student_grad_norm", "aux_grad_norm", "dimo_gradient_norm")], device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(reduced); reduced /= world
            diagnostics.update(zip(("loss_dimo", "loss_aux", "distribution_energy", "student_grad_norm", "aux_grad_norm", "dimo_gradient_norm"), reduced.cpu().tolist()))
            if rank == 0:
                warnings = []
                if v2 and commit.step >= 20 and diagnostics["student_clip_fraction"] >= .95:
                    warnings.append("STUDENT_CLIPPING_FRACTION_NEAR_100_PERCENT_THRESHOLD_UNCHANGED")
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
            if reward_config and commit.step == stop and commit.step not in config["evaluation_steps"]:
                checkpoint(inference_only=True)
            # DEV runs are launched separately. In particular, step500 saves
            # its complete checkpoint/inference exports without spawning an
            # evaluator alongside live Student/Auxiliary optimizers and EMA.
        if commit.step != stop:
            raise RuntimeError("FULL_DIMO_DATA_EXHAUSTED_BEFORE_HORIZON")
        if exploratory and stop == 0:
            checkpoint()
        if v2:
            if reward_objective is not None:
                reward_objective.assert_frozen()
                write_json(output / f"reward_acceptance_rank{rank}.json", {
                    "status": "PASS", "step": commit.step, "gpu_experiment": "REAL_GPU",
                    "calibration": reward_objective.calibration.state,
                    "reward_models_frozen": True, "vq_decoder_frozen": True,
                    "trainable_student_embedding_and_head": False,
                    "soft_hard_gap_interpretation": "training batch diagnostics; fixed DEV gate required before expansion"})
            roles.assert_frozen_unchanged(frozen_hashes)
            write_json(output / f"v2_acceptance_rank{rank}.json", {"status": "PASS", "frozen_weights": "bitwise_hash_equal",
                "recipe_contract": identity["recipe_contract"], "completed_updates_per_role": commit.step,
                "completed_sample_consumptions": commit.cursor, "student_clip_count": runtime_stats["student_clip_count"],
                "student_clip_fraction": runtime_stats["student_clip_count"] / max(commit.step, 1),
                "evidence": "CPU_SYNTHETIC" if args.synthetic else "REAL_GPU"})
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
