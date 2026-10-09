#!/usr/bin/env python3
"""Read-only launch checks; --contract-only does not access assets or GPUs."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.dimo.pilot import git_identity
from src.dimo.pilot_contract import (frozen_data_identity, gpu_allocation_report, protect_output,
                                     require_smoke_certificate, validate_pilot_config)
from src.dimo.provisional_teacher import load_provisional_teacher
from src.dimo.contracts import released_base_model_identity
from src.dimo.checkpoint import config_hash, read_dimo_state
from src.dimo.contracts import teacher_bundle_fingerprint
from src.dimo.pilot_contract import training_identity, validate_output_prefix
from src.explicit_region.config import _load_unexpanded, load_config
from src.explicit_region.contracts import audit_component_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--contract-only', action='store_true')
    parser.add_argument('--resume')
    args = parser.parse_args()
    config = _load_unexpanded(Path(args.config)) if args.contract_only else load_config(args.config)
    budget = validate_pilot_config(config)
    if args.contract_only:
        print(json.dumps(budget, indent=2, sort_keys=True))
        return
    git = git_identity()
    if git['git_dirty']:
        raise RuntimeError('USE_A_CLEAN_INDEPENDENT_EXPERIMENT_WORKTREE')
    identity = released_base_model_identity(audit_component_identity(config['model']['repo_or_root']))
    teacher = load_provisional_teacher(config['teacher_checkpoint'], expected_base_identity=identity)
    data = frozen_data_identity(config)
    if teacher.manifest['source_d200k_sha256'] != data['train_manifest_sha256']:
        raise RuntimeError('DIMO_CORPUS_DIFFERS_FROM_TEACHER_E3_D200K')
    protect_output(config, teacher)
    output = Path(config['output_dir']).resolve()
    if not args.resume and output.exists() and any(output.iterdir()):
        raise RuntimeError('DIMO_OUTPUT_ALREADY_EXISTS')
    if args.resume and not (Path(args.resume) / 'CHECKPOINT_READY.json').is_file():
        raise RuntimeError('DIMO_RESUME_CHECKPOINT_NOT_PUBLISHED')
    if args.resume:
        state = read_dimo_state(args.resume)
        teacher_bundle = teacher_bundle_fingerprint(teacher.root, base_model_identity=identity)
        expected = training_identity(config, teacher_bundle, identity, data) | {'experiment_git_sha': git['git_sha']}
        if (state.get('world_size') != 8 or state.get('config_hash') != config_hash(expected)
                or state.get('teacher_bundle_fingerprint') != teacher_bundle
                or state.get('global_dimo_step', -1) >= config['max_optimizer_steps']):
            raise RuntimeError('DIMO_RESUME_PREFLIGHT_IDENTITY_OR_BUDGET_MISMATCH')
        validate_output_prefix(output, resume_state=state, config_hash=state['config_hash'])
    if config['max_optimizer_steps'] > 20:
        require_smoke_certificate(os.environ.get('DIMO_SMOKE_REPORT'), config, teacher.checkpoint_hash, data, git['git_sha'])
    if config['max_optimizer_steps'] > 500 and os.environ.get('CONFIRM_DIMO_LONG_RUN') != 'YES':
        raise RuntimeError('FULL_EPOCH_REQUIRES_SEPARATE_USER_CONFIRMATION')
    allocation = gpu_allocation_report(config)
    print(json.dumps(budget | git | {'teacher_hash': teacher.checkpoint_hash, 'data_identity': data,
                     'output_dir': str(output), 'gpu_allocation': allocation}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
