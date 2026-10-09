"""Eight-rank exploratory trainer over an existing frozen D200K corpus."""
from __future__ import annotations

from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader

from .checkpoint import config_hash, load_dimo_checkpoint, read_dimo_state, save_dimo_checkpoint
from .contracts import (DIMO_UPSTREAM_COMMIT, build_inference_fingerprint, enforce_run_guard,
                        released_base_model_identity, teacher_bundle_fingerprint)
from .distributed import ExplicitGradientSync, assert_rank_consistency, collective_require, tensor_digest
from .ema import TrainableEMA
from .initialization import initialize_shared_model_roles
from .pilot_contract import (frozen_data_identity, protect_output, require_smoke_certificate,
                             smoke_compatibility, training_identity, validate_pilot_config)
from .pilot_contract import validate_output_prefix
from .provisional_teacher import load_provisional_teacher
from .rng import sample_seeds
from .roles import audit_role_optimizers
from .step import complete_dimo_step
from src.explicit_region.epoch_sampler import DeterministicEpochSampler
from src.explicit_region.deterministic import stable_seed as loader_seed
from src.explicit_region.fixed_corpus import assert_frozen_corpus
from src.explicit_region.fixed_dataset import FixedCorpusDataset


def git_identity():
    root = Path(__file__).resolve().parents[2]
    return {'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip())}


class PilotEpochLoader:
    """Stateless per-epoch worker RNG; no duplicated/padded rank samples."""
    def __init__(self, dataset, sampler, config):
        self.dataset, self.sampler, self.config = dataset, sampler, config
        self.initial_epoch = sampler.epoch
        self.initial_cursor = sampler.samples_consumed_in_epoch

    def __iter__(self):
        for epoch in range(self.initial_epoch, int(self.config.get('epochs', 1))):
            cursor = self.initial_cursor if epoch == self.initial_epoch else 0
            self.sampler.reset_epoch(epoch, cursor)
            self.dataset.set_epoch(epoch)
            generator = torch.Generator().manual_seed(loader_seed(
                self.config['seed'], epoch, f'rank-{self.sampler.rank}', 'dimo-loader-workers'))
            yield from DataLoader(self.dataset, sampler=self.sampler, batch_size=1,
                                  num_workers=int(self.config.get('num_workers', 0)),
                                  pin_memory=True, drop_last=False, generator=generator)


def audit_sample_update(dataset, sampler, global_uids):
    cursor = sampler.samples_consumed_in_epoch
    expected = [dataset.rows[i]['sample_uid'] for i in sampler.permutation[cursor:cursor + sampler.world_size]]
    if global_uids != expected or len(set(global_uids)) != sampler.world_size:
        raise RuntimeError('DIMO_SAMPLER_DUPLICATION_OMISSION_OR_SEQUENCE_MISMATCH')
    sampler.samples_consumed_in_epoch += sampler.world_size
    return sampler.samples_consumed_in_epoch


def _validate_loaded_progress(state, sampler, target_steps):
    step = int(state['global_dimo_step'])
    cursor = int(state['fixed_corpus_cursor'])
    if (step < 0 or step >= target_steps or cursor != step * 8
            or state['samples_consumed'] != cursor
            or cursor != sampler.epoch * sampler.length + sampler.samples_consumed_in_epoch
            or state['sampler_state'].get('epoch_permutation_sha256') != sampler.epoch_permutation_sha256
            or state['epoch'] != sampler.epoch):
        raise RuntimeError('DIMO_RESUME_STEP_OR_SAMPLER_CURSOR_MISMATCH')
    for role in ('student', 'auxiliary'):
        if state[role + '_scheduler'].get('last_epoch') != step:
            raise RuntimeError('DIMO_RESUME_SCHEDULER_STEP_MISMATCH')
        opt = state[role + '_optimizer']['state']
        if step and (not opt or any(int(s.get('step', -1)) != step for s in opt.values())):
            raise RuntimeError('DIMO_RESUME_OPTIMIZER_STEP_MISMATCH')


def run_pilot(args, config, *, prepare_batch):
    validate_pilot_config(config)
    if args.prep_smoke:
        raise RuntimeError('INVALID_OR_AMBIGUOUS_DIMO_EXPERIMENT_MODE')
    if int(os.environ.get('WORLD_SIZE', '1')) != 8:
        raise RuntimeError('PILOT_REQUIRES_EIGHT_RANKS')
    local_rank = int(os.environ['LOCAL_RANK'])
    if torch.cuda.device_count() != 8 or not 0 <= local_rank < 8:
        raise RuntimeError('PILOT_REQUIRES_EIGHT_VISIBLE_CUDA_DEVICES')
    os.environ.setdefault('NCCL_ALGO', 'Ring')
    os.environ.setdefault('NCCL_PROTO', 'Simple')
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=900))
    try:
        _train(args, config, prepare_batch)
    finally:
        dist.destroy_process_group()


def _train(args, config, prepare_batch):
    rank = dist.get_rank()
    device = torch.device('cuda', torch.cuda.current_device())
    seed = int(config['seed']) + rank
    random.seed(seed); np.random.seed(seed); torch.random.default_generator.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    teacher = Path(args.teacher_checkpoint or config['teacher_checkpoint']).resolve()
    contract = load_provisional_teacher(teacher)
    gate = enforce_run_guard(contract, prep_smoke=False, max_optimizer_steps=config['max_optimizer_steps'],
                             formal_ready=config['formal_ready'], experiment_mode='pilot_8gpu', world_size=8)
    protect_output(config, contract)
    source = git_identity()
    collective_require(not source['git_dirty'], 'PILOT_REQUIRES_CLEAN_EXPERIMENT_WORKTREE', device)
    data_identity = frozen_data_identity(config)
    if contract.manifest['source_d200k_sha256'] != data_identity['train_manifest_sha256']:
        raise RuntimeError('DIMO_CORPUS_DIFFERS_FROM_TEACHER_E3_D200K')
    if config['max_optimizer_steps'] > 20:
        require_smoke_certificate(os.environ.get('DIMO_SMOKE_REPORT'), config, contract.checkpoint_hash,
                                  data_identity, source['git_sha'])
    if config['max_optimizer_steps'] > 500 and os.environ.get('CONFIRM_DIMO_LONG_RUN') != 'YES':
        raise RuntimeError('FULL_EPOCH_REQUIRES_SEPARATE_USER_CONFIRMATION')
    output = Path(config['output_dir']).resolve()
    fresh_safe = args.resume is not None or not output.exists() or not any(output.iterdir())
    collective_require(fresh_safe, 'DIMO_FRESH_OUTPUT_ALREADY_EXISTS', device)
    prefix_state = read_dimo_state(args.resume) if args.resume else None
    validate_output_prefix(output, resume_state=prefix_state,
                           config_hash=prefix_state['config_hash'] if prefix_state else None)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    from src.explicit_region.modeling import load_released_components
    components = load_released_components(config['model']['repo_or_root'], torch_dtype=torch.bfloat16,
        vq_dtype=torch.float32, identity_output=output / 'component_identity.json' if rank == 0 else None)
    base_identity = released_base_model_identity(components.identity)
    contract = load_provisional_teacher(teacher, expected_base_identity=base_identity)
    roles = initialize_shared_model_roles(components.transformer, str(teacher), model_roles=config['model_roles'])
    roles.to(device)
    if config.get('gradient_checkpointing', True):
        roles.base_model.enable_gradient_checkpointing()
    for module, dtype in ((components.text_encoder, torch.bfloat16), (components.llm_encoder, torch.bfloat16),
                          (components.vqvae, torch.float32)):
        module.to(device=device, dtype=dtype).eval().requires_grad_(False)
    optimizer_config = config['optimizer']
    optimizers = [torch.optim.AdamW([p for _, p in roles.role_named_parameters(role)],
        lr=float(optimizer_config[role + '_learning_rate']), betas=tuple(optimizer_config['betas']),
        weight_decay=float(optimizer_config['weight_decay']), foreach=False)
        for role in ('student', 'auxiliary')]
    schedulers = [get_scheduler(optimizer_config['scheduler'], optimizer=optimizer,
        num_warmup_steps=int(optimizer_config['warmup_steps']),
        num_training_steps=int(optimizer_config['horizon_steps'])) for optimizer in optimizers]
    ema = TrainableEMA(roles.role_named_parameters('student'), decay=float(config['ema']['decay']))
    audit_role_optimizers(roles, *optimizers, output / 'dimo_trainable_parameter_report.json' if rank == 0 else None)
    teacher_bundle = teacher_bundle_fingerprint(teacher, base_model_identity=base_identity)
    inference = build_inference_fingerprint(teacher_bundle_sha256=teacher_bundle['bundle_sha256'],
        base_model_identity=base_identity, model_roles=config['model_roles'], upstream_commit=DIMO_UPSTREAM_COMMIT)
    identity = training_identity(config, teacher_bundle, base_identity, data_identity)
    identity['experiment_git_sha'] = source['git_sha']
    resolved_hash = config_hash(identity)
    dataset = FixedCorpusDataset(config['data']['manifest'], config['data']['roots'], resolution=config['resolution'],
        base_seed=config['seed'], verify_hashes=True, max_random_attempts=config['geometry']['max_resample_attempts'],
        minimum_mask_retention=config['geometry']['minimum_mask_retention'])
    assert_frozen_corpus(dataset.rows)
    sampler_identity = {'sampler_schema_version': 'fixed200k-hash-sort-v1', 'world_size': 8,
                        'train_200k_sha256': data_identity['train_manifest_sha256'],
                        'length': 200000, 'batch_per_gpu': 1, 'gradient_accumulation': 1}
    step = cursor = epoch = within = 0
    resumed = None
    if args.resume:
        resumed = load_dimo_checkpoint(args.resume, roles=roles, student_optimizer=optimizers[0],
            auxiliary_optimizer=optimizers[1], student_scheduler=schedulers[0], auxiliary_scheduler=schedulers[1],
            student_ema=ema, expected_config_hash=resolved_hash, expected_teacher_bundle_fingerprint=teacher_bundle,
            expected_inference_fingerprint=inference, expected_upstream_commit=DIMO_UPSTREAM_COMMIT,
            expected_sampler_identity=sampler_identity)
        step, cursor, epoch = (int(resumed[k]) for k in ('global_dimo_step', 'fixed_corpus_cursor', 'epoch'))
        within = int(resumed['sampler_state']['samples_consumed_in_epoch'])
    sampler = DeterministicEpochSampler(200000, base_seed=config['seed'], epoch=epoch,
        train_manifest_sha256=data_identity['train_manifest_sha256'], rank=rank, world_size=8,
        samples_consumed_in_epoch=within)
    if resumed is not None:
        _validate_loaded_progress(resumed, sampler, config['max_optimizer_steps'])
    else:
        from .diagnostics import run_nonzero_signal_diagnostic
        # This diagnostic restores auxiliary weights and process RNG; it does
        # not perform an optimizer update or alter the training initialization.
        initial_batch = next(iter(PilotEpochLoader(dataset, sampler, config)))
        initial = prepare_batch(initial_batch, components, config, device)
        run_nonzero_signal_diagnostic(roles, target_tokens=initial['source_tokens'],
            reference_tokens=initial['source_tokens'], edit_region_mask=initial['edit_region_mask'],
            prompt_condition=initial['prompt_condition'], model_kwargs=initial['model_kwargs'],
            output_path=output / 'nonzero-signal' / f'rank-{rank}.json', epsilon=1e-3)
        # The fresh iterator begins at the same immutable cursor.
        sampler.reset_epoch(epoch, within)
        del initial, initial_batch
    sync = ExplicitGradientSync(roles, bucket_bytes=config['distributed']['bucket_bytes'])
    metadata = gate | source | validate_pilot_config(config) | {
        'experiment': config['experiment'], 'resolved_config': config, 'training_identity': identity,
        'config_hash': resolved_hash, 'teacher_hash': contract.checkpoint_hash,
        'teacher_bundle_fingerprint': teacher_bundle, 'inference_fingerprint': inference,
        'data_identity': data_identity, 'compatibility_sha256': smoke_compatibility(config, contract.checkpoint_hash, data_identity),
        'actual_execution': {'backend': dist.get_backend(), 'device_type': device.type,
                             'visible_gpu_count': torch.cuda.device_count(), 'torch_version': torch.__version__},
        'DIMO_EDIT_FORMAL_READY': False}
    if rank == 0:
        if (output / 'run_manifest.json').exists() and json.loads((output / 'run_manifest.json').read_text())['config_hash'] != resolved_hash:
            raise RuntimeError('DIMO_OUTPUT_CONFIG_IDENTITY_MISMATCH')
        (output / 'run_manifest.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    def checkpoint():
        save_dimo_checkpoint(output / f'checkpoint-{step}', roles=roles, student_optimizer=optimizers[0],
            auxiliary_optimizer=optimizers[1], student_scheduler=schedulers[0], auxiliary_scheduler=schedulers[1],
            student_ema=ema, committed_dimo_step=step, fixed_corpus_cursor=cursor, epoch=sampler.epoch,
            samples_consumed=cursor, config_sha256=resolved_hash, teacher_bundle_fingerprint=teacher_bundle,
            inference_fingerprint=inference, upstream_commit=DIMO_UPSTREAM_COMMIT,
            sampler_state=sampler.state_dict() | sampler_identity)
    consistency = assert_rank_consistency(roles, ema, *optimizers, *schedulers, step)
    if rank == 0:
        with (output / 'rank_consistency.jsonl').open('a') as log:
            log.write(json.dumps(consistency, sort_keys=True) + '\n')
    if not args.resume and 0 in config['checkpoint_steps']:
        checkpoint()
    stop = int(config['max_optimizer_steps'])
    if args.stop_after_steps is not None:
        if args.stop_after_steps <= 0:
            raise ValueError('--stop-after-steps must be positive')
        stop = min(stop, step + args.stop_after_steps)
    torch.cuda.reset_peak_memory_stats(device)
    for batch in PilotEpochLoader(dataset, sampler, config):
        if step >= stop:
            break
        started = time.perf_counter()
        prepared = prepare_batch(batch, components, config, device)
        prior_step = step
        diagnostics, artifacts = complete_dimo_step(roles, **prepared,
            student_optimizer=optimizers[0], auxiliary_optimizer=optimizers[1],
            student_scheduler=schedulers[0], auxiliary_scheduler=schedulers[1], student_ema=ema,
            base_seed=config['seed'], epoch=sampler.epoch, committed_dimo_step=step,
            mask_token_id=roles.base_model.config.vocab_size - 1,
            codebook_size=roles.base_model.config.codebook_size, config=config, gradient_sync=sync)
        finite = (diagnostics['nan_inf_count'] == 0 and diagnostics['outside_mismatch_count'] == 0
                  and all(math.isfinite(v) for v in diagnostics.values() if isinstance(v, (int, float)))
                  and all(bool(torch.isfinite(p).all()) for role in ('student', 'auxiliary')
                          for _, p in roles.role_named_parameters(role)))
        collective_require(finite, 'DIMO_NONFINITE_OR_OUTSIDE_HARDLOCK_FAILURE', device)
        trace = {'sample_uids': prepared['sample_uids'], 'geometry_seeds': [int(s) for s in batch['geometry_seed']],
                 'artifacts': {k: tensor_digest(v) for k, v in artifacts.items()},
                 'rng_seeds': {purpose: sample_seeds(base_seed=config['seed'], epoch=sampler.epoch,
                    sample_uids=prepared['sample_uids'], global_dimo_step=prior_step, namespace=purpose)
                    for purpose in ('dimo-init-mask', 'dimo-init-token', 'dimo-student-sample',
                                    'dimo-teacher-query-mask', 'dimo-aux-query-mask', 'dimo-embedding-noise')}}
        ranks = [None] * 8
        local = {'rank': rank, 'diagnostics': diagnostics, 'trace': trace,
                 'peak_vram_bytes': torch.cuda.max_memory_allocated(device)}
        dist.all_gather_object(ranks, local)
        uids = [uid for r in ranks for uid in r['trace']['sample_uids']]
        within = audit_sample_update(dataset, sampler, uids)
        step += 1; cursor += 8
        for scheduler in schedulers:
            collective_require(scheduler.last_epoch == step, 'DIMO_SCHEDULER_UPDATE_BOUNDARY_MISMATCH', device)
        if step in set(config.get('rank_consistency_steps', [])) or step in config['checkpoint_steps']:
            consistency = assert_rank_consistency(roles, ema, *optimizers, *schedulers, step)
            if rank == 0:
                with (output / 'rank_consistency.jsonl').open('a') as log:
                    log.write(json.dumps(consistency, sort_keys=True) + '\n')
        row = {'global_dimo_step': step, 'epoch': sampler.epoch, 'samples_consumed': cursor,
               'samples_consumed_in_epoch': within, 'committed_sample_uids': uids, 'ranks': ranks,
               'dev_only': True, 'formal_teacher': False, 'world_size': 8, 'gradient_sync': 'explicit_allreduce',
               'finite': True, 'nan_inf_count': sum(r['diagnostics']['nan_inf_count'] for r in ranks),
               'role_isolation': True,
               'outside_mismatch_count': sum(r['diagnostics']['outside_mismatch_count'] for r in ranks),
               'student_scheduler_step': schedulers[0].last_epoch, 'auxiliary_scheduler_step': schedulers[1].last_epoch}
        for key in ('loss_dimo', 'loss_aux', 'student_grad_norm', 'aux_grad_norm', 'dimo_gradient_norm',
                    'student_parameter_update_ratio', 'aux_parameter_update_ratio', 'initial_roi_mask_ratio'):
            row[key] = sum(r['diagnostics'][key] for r in ranks) / 8
        row['ema_distance'] = math.sqrt(sum(float((p.detach().cpu().float() - ema.shadow[n]).square().sum())
                                        for n, p in roles.role_named_parameters('student')))
        if step in config['checkpoint_steps'] or step == stop:
            checkpoint()
        dist.barrier()
        row['step_wall_time'] = time.perf_counter() - started
        if rank == 0:
            with (output / 'metrics.jsonl').open('a') as log:
                log.write(json.dumps(row, sort_keys=True) + '\n')
            print(json.dumps({k: v for k, v in row.items() if k not in ('ranks', 'committed_sample_uids')}, sort_keys=True), flush=True)
    collective_require(step == stop, 'DIMO_DATA_EXHAUSTED_BEFORE_BUDGET', device)
    load_provisional_teacher(teacher, expected_base_identity=base_identity)
    collective_require(teacher_bundle_fingerprint(teacher, base_model_identity=base_identity) == teacher_bundle,
                       'DIMO_TEACHER_CHANGED_DURING_TRAINING', device)
    if rank == 0:
        (output / 'run_status.json').write_text(json.dumps({'status': 'COMPLETE' if step == config['max_optimizer_steps'] else 'PAUSED',
            'steps': step, 'samples_consumed': cursor, 'world_size': 8, 'dev_only': True, 'formal_teacher': False}, sort_keys=True) + '\n')
    dist.barrier()
