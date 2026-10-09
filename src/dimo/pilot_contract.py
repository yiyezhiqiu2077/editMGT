"""Exploratory configuration, frozen-data and smoke-certificate contracts."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess

from .checkpoint import config_hash
from .contracts import DIMO_MODEL_ROLES_V11, _sha256
from .provisional_teacher import load_provisional_teacher
from src.explicit_region.fixed_corpus import verify_corpus_ready

SMOKE_PROOFS = {'run_identities', 'step_sequences', 'unique_samples', 'rank_shards_and_cursor',
    'sample_seed_and_artifact_replay', 'finite', 'outside_hardlock', 'gradient_sync_and_role_isolation',
    'scheduler_updates', 'gradient_diagnostics', 'nonzero_path_diagnostic', 'all_rank_state_consistency',
    'checkpoint_progress', 'resume_state_equivalence', 'exact_rank_rng_and_sampler',
    'optimizer_step_counts', 'ema_state_available', 'ema_update_response'}


def validate_pilot_config(config):
    d = config.get('distributed', {})
    if (config.get('experiment_mode') != 'pilot_8gpu' or config.get('dev_only') is not True
            or config.get('formal_ready') is not False
            or d.get('world_size') != 8 or d.get('backend') != 'nccl'
            or d.get('gradient_sync') != 'explicit_allreduce'
            or config.get('batch_per_gpu') != 1 or config.get('gradient_accumulation') != 1
            or config.get('resolution') != 1024 or config.get('precision') != 'bf16'
            or config.get('data', {}).get('name') != 'fixed_200k'
            or config.get('model_roles') != DIMO_MODEL_ROLES_V11
            or config.get('lora_scope') != 'both'):
        raise RuntimeError('INVALID_PILOT_8GPU_CONTRACT')
    steps, epochs = int(config['max_optimizer_steps']), int(config.get('epochs', 1))
    optimizer = config['optimizer']
    if (not 0 < steps <= 25000 * epochs or epochs < 1
            or int(optimizer['horizon_steps']) < steps
            or optimizer['scheduler'] != 'constant_with_warmup'
            or int(optimizer['warmup_steps']) != 0
            or any(not math.isfinite(float(optimizer[k])) or float(optimizer[k]) <= 0
                   for k in ('student_learning_rate', 'auxiliary_learning_rate'))):
        raise RuntimeError('INVALID_PILOT_OPTIMIZER_BUDGET')
    checkpoints = config['checkpoint_steps']
    if (checkpoints != sorted(set(checkpoints)) or any(not isinstance(s, int) or not 0 <= s <= steps for s in checkpoints)
            or steps not in checkpoints):
        raise RuntimeError('INVALID_PILOT_CHECKPOINT_CADENCE')
    if config['auxiliary']['updates_per_student'] != 1 or config['auxiliary']['batch_policy'] != 'same_batch':
        raise RuntimeError('PILOT_REQUIRES_ONE_SAME_BATCH_AUXILIARY_UPDATE')
    return {'dev_only': True, 'formal_teacher': False, 'world_size': 8, 'global_batch': 8,
            'dataset_rows': 200000, 'steps_per_epoch': 25000, 'target_steps': steps,
            'sample_exposures': steps * 8, 'checkpoint_steps': checkpoints,
            'reference_hyperparameters_not_selected_optima': True}


def frozen_data_identity(config):
    data = config['data']
    corpus = verify_corpus_ready(data['corpus_ready'])
    manifest = Path(data['manifest']).resolve()
    digest = _sha256(manifest)
    if (corpus.get('schema_version') != 'fixed200k-ready-v3'
            or corpus.get('total_rows') != 200000
            or corpus.get('train_manifest_sha256') != digest
            or Path(corpus['files']['train_200k']['path']).resolve() != manifest):
        raise RuntimeError('PILOT_REQUIRES_EXISTING_FROZEN_D200K')
    return {'train_manifest_sha256': digest, 'corpus_ready_sha256': _sha256(Path(data['corpus_ready']))}


def training_identity(config, teacher_bundle, base_model_identity, data_identity):
    resolved = {k: v for k, v in config.items() if k != 'output_dir'}
    return {'resolved_config': resolved, 'teacher_bundle': teacher_bundle,
            'base_model_identity': base_model_identity, 'data_identity': data_identity,
            'world_size': 8, 'loader_rng_schema': 'stateless-epoch-worker-seed-v1',
            'nccl_algo': os.environ.get('NCCL_ALGO', 'Ring'),
            'nccl_proto': os.environ.get('NCCL_PROTO', 'Simple'), 'attention_backend': 'math_sdp'}


def smoke_compatibility(config, teacher_hash, data_identity):
    # Constant scheduler with zero warmup: horizons/cadence do not alter updates.
    recipe = {k: v for k, v in config.items() if k not in {
        'output_dir', 'experiment', 'max_optimizer_steps', 'epochs',
        'checkpoint_steps', 'rank_consistency_steps'}}
    recipe['optimizer'] = {k: v for k, v in config['optimizer'].items() if k != 'horizon_steps'}
    return config_hash({'recipe': recipe, 'teacher_hash': teacher_hash, 'data_identity': data_identity})


def require_smoke_certificate(path, config, teacher_hash, data_identity, git_sha):
    if not path or not Path(path).is_file():
        raise RuntimeError('REAL_8GPU_SMOKE_REQUIRED_BEFORE_PILOT')
    report = json.loads(Path(path).read_text())
    if (report.get('schema') != 'dimo-real-8gpu-smoke-v1' or report.get('status') != 'PASS'
            or report.get('world_size') != 8 or report.get('data_name') != 'fixed_200k'
            or report.get('git_sha') != git_sha
            or report.get('compatibility_sha256') != smoke_compatibility(config, teacher_hash, data_identity)
            or report.get('teacher_hash') != teacher_hash or report.get('data_identity') != data_identity
            or report.get('resume_equivalence') not in ('bitwise_exact', 'numerical_tolerance')
            or not SMOKE_PROOFS <= report.get('checks', {}).keys()
            or not all(v is True for v in report['checks'].values())):
        raise RuntimeError('REAL_8GPU_SMOKE_CERTIFICATE_MISMATCH')
    return report


def protect_output(config, contract):
    output = Path(config['output_dir']).resolve()
    teacher = contract.root
    e3 = Path(contract.manifest['source_checkpoint']).resolve().parent
    for protected in (teacher, e3):
        if output == protected or protected in output.parents or output in protected.parents:
            raise RuntimeError('DIMO_OUTPUT_OVERLAPS_E3_OR_TEACHER')


def validate_output_prefix(output, *, resume_state=None, config_hash=None):
    """Never append an older checkpoint onto an already advanced run."""
    output = Path(output)
    if not output.exists() or not any(output.iterdir()):
        return
    if resume_state is None:
        raise RuntimeError('DIMO_FRESH_OUTPUT_ALREADY_EXISTS')
    manifest = output / 'run_manifest.json'
    metrics = output / 'metrics.jsonl'
    if not manifest.is_file() or not metrics.is_file():
        raise RuntimeError('DIMO_RESUME_OUTPUT_PREFIX_INCOMPLETE')
    if json.loads(manifest.read_text()).get('config_hash') != config_hash:
        raise RuntimeError('DIMO_OUTPUT_CONFIG_IDENTITY_MISMATCH')
    steps = [json.loads(line)['global_dimo_step'] for line in metrics.read_text().splitlines() if line.strip()]
    step = resume_state['global_dimo_step']
    if (not steps or steps[-1] != step or steps != list(range(steps[0], step + 1))
            or any(int(path.name.split('-')[-1]) > step for path in output.glob('checkpoint-*')
                   if path.is_dir() and path.name.split('-')[-1].isdigit())):
        raise RuntimeError('DIMO_RESUME_OUTPUT_PREFIX_STEP_MISMATCH')


def gpu_allocation_report(config, *, expected_count=8):
    """Read NVML via nvidia-smi before CUDA allocations; never stop processes."""
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    devices = {}
    for line in rows:
        index, uuid, free, util = [s.strip() for s in line.split(',')]
        devices[index] = {'index': index, 'uuid': uuid, 'free_mib': int(free), 'utilization': int(util)}
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    ids = visible.split(',') if visible is not None else list(devices)
    chosen = []
    for key in ids:
        key = key.strip()
        matches = [d for d in devices.values() if key in (d['index'], d['uuid'])]
        if len(matches) != 1:
            raise RuntimeError('INVALID_DIMO_CUDA_VISIBLE_DEVICES')
        chosen.append(matches[0])
    if len(chosen) != expected_count or len({r['uuid'] for r in chosen}) != expected_count:
        raise RuntimeError(f'DIMO_REQUIRES_{expected_count}_DISTINCT_VISIBLE_GPUS')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader,nounits'], text=True).splitlines()
    occupied = {line.split(',')[0].strip() for line in active if ',' in line}
    minimum = float(config['distributed'].get('minimum_free_memory_gib', 24)) * 1024
    if any(r['uuid'] in occupied or r['free_mib'] < minimum for r in chosen):
        raise RuntimeError(f'NO_INDEPENDENT_{expected_count}GPU_ALLOCATION: existing processes are protected')
    return chosen
