"""Explicit last-epoch user transition for the independent total20K run."""
import json
import os
from pathlib import Path
import re
import tempfile

from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.config import load_config
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dense_checkpoint import verify_dense_checkpoint, write_json
from src.explicit_region.mixed_experiment import MODE as SFT_MODE, ROWS, verify_mixed_ready

MODE = 'mixed20k_last_sft_dimo'
FINAL_SFT_STEP = 6250


def is_mixed_dimo(config):
    return config.get('experiment_mode') == MODE


def validate_budget(config):
    if (not is_mixed_dimo(config) or config['formal_ready'] is not False
            or (config['distributed']['world_size'], config['epochs'], config['batch_per_gpu'],
                config['gradient_accumulation'], config['max_optimizer_steps'], config['optimizer']['horizon_steps'],
                config['data']['expected_rows'], config['data']['name']) !=
                (4, 5, 1, 1, 25000, 25000, ROWS, 'fixed_mixed20k_total')
            or config['resume_checkpoint_steps'] != [5000, 15000, 25000]
            or config['evaluation_steps'] != [0, 5000, 15000, 25000]):
        raise RuntimeError('MIXED_DIMO_REGISTERED_5_EPOCH_BUDGET_CHANGED')


def verify_request(config):
    request = json.loads(Path(config['user_transition_record']).read_text())
    if (request.get('schema') != 'mixed20k-auto-last-sft-dimo-v1'
            or request.get('requested_action') != 'automatic_dimo_after_five_sft_epochs_using_final_checkpoint'
            or not request.get('user_instruction', '').strip()
            or request.get('sft_final_step') != FINAL_SFT_STEP
            or request.get('sft_additional_epochs') != 5 or request.get('dimo_epochs') != 5
            or request.get('dimo_training_dataset') != 'same_mixed20k'
            or Path(request.get('sft_final_checkpoint', '')).resolve() != Path(config['sft_final_checkpoint']).resolve()
            or request.get('dev_quality_gate') is not False):
        raise RuntimeError('MIXED_DIMO_EXPLICIT_FINAL_WEIGHT_REQUEST_MISMATCH')
    return request


def verify_ready(config):
    validate_budget(config); verify_request(config)
    sft = load_config(config['sft_admission_config'])
    ready = verify_mixed_ready(sft)
    if config['data'] != {**sft['data'], 'verify_hashes_at_read': True}:
        # Resolved templates already contain this flag; equality includes roots.
        raise RuntimeError('MIXED_DIMO_MUST_REUSE_EXACT_FROZEN_SFT_DATA')
    return ready


def verify_teacher_record(record, checkpoint_hash):
    if (record.get('schema') != 'mixed20k-user-last-teacher-v1'
            or record.get('selection_scope') != MODE or record.get('formal_teacher') is not False
            or record.get('selection_status') != 'USER_LAST_EPOCH'
            or record.get('teacher_source') != 'dense_sft_last_epoch_user_requested'
            or record.get('selected_step') != FINAL_SFT_STEP or record.get('selected_epoch') != 5
            or record.get('cumulative_sft_epoch') != 12
            or record.get('selected_checkpoint_hash') != checkpoint_hash
            or record.get('dev_quality_gate') is not False
            or record.get('record_sha256') != recipe_fingerprint({k: v for k, v in record.items() if k != 'record_sha256'})
            or record.get('data_identity', {}).get('train_rows') != ROWS
            or any(not re.fullmatch('[0-9a-f]{64}', str(record.get('data_identity', {}).get(k, '')))
                   for k in ('train_manifest_sha256', 'corpus_ready_sha256'))
            or not re.fullmatch('[0-9a-f]{64}', str(record.get('authorization_sha256', '')))):
        raise RuntimeError('MIXED_DIMO_LAST_TEACHER_RECORD_INVALID')


def select_last_teacher(config):
    ready = verify_ready(config)
    checkpoint = Path(config['sft_final_checkpoint']).resolve()
    identity = verify_dense_checkpoint(checkpoint)
    metadata = json.loads((checkpoint / 'metadata.json').read_text())
    payload = json.loads((checkpoint / 'fingerprint.json').read_text())['payload']
    sft = load_config(config['sft_admission_config'])
    if (checkpoint != Path(sft['output_dir']).resolve() / f'checkpoint-{FINAL_SFT_STEP}'
            or metadata['global_optimizer_step'] != FINAL_SFT_STEP
            or metadata['committed_global_sample_count'] != 5 * ROWS
            or payload.get('experiment_mode') != SFT_MODE
            or payload.get('training_mode') != 'full_transformer'
            or payload.get('max_optimizer_steps') != FINAL_SFT_STEP
            or payload.get('continuation') != sft['continuation']
            or payload.get('data_content_hashes', {}).get('data.manifest', {}).get('sha256') != ready['train_manifest_sha256']
            or payload.get('warm_start_dense_identity') != ready['warm_start_checkpoint_sha256']):
        raise RuntimeError('MIXED_DIMO_REQUIRES_COMPLETE_FINAL_SFT_IDENTITY')
    record = {'schema': 'mixed20k-user-last-teacher-v1', 'selection_scope': MODE,
              'selection_status': 'USER_LAST_EPOCH', 'formal_teacher': False, 'dev_quality_gate': False,
              'teacher_source': 'dense_sft_last_epoch_user_requested', 'selected_checkpoint': str(checkpoint),
              'selected_checkpoint_hash': identity['checkpoint_sha256'], 'selected_step': FINAL_SFT_STEP,
              'selected_epoch': 5, 'cumulative_sft_epoch': 12,
              'authorization_sha256': sha256_file(config['user_transition_record']),
              'data_identity': {'train_rows': ROWS, 'train_manifest_sha256': ready['train_manifest_sha256'],
                                'corpus_ready_sha256': sha256_file(config['data']['corpus_ready'])}}
    record['record_sha256'] = recipe_fingerprint(record)
    verify_teacher_record(record, identity['checkpoint_sha256'])
    return record


def export_last_teacher(record, output):
    from .dense_teacher import export_dense_teacher, load_dense_teacher_contract
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    verify_teacher_record(record, record['selected_checkpoint_hash'])
    temporary = Path(tempfile.mkdtemp(prefix='.mixed20k-teacher-', dir=output.parent))
    target = temporary / 'teacher'
    manifest = export_dense_teacher(record['selected_checkpoint'], target, status='provisional')
    write_json(target / 'selection_record.json', record)
    manifest.update(status='mixed20k_user_last', selection_scope=MODE, formal_teacher=False,
                    teacher_source=record['teacher_source'], selected_checkpoint_identity=record['selected_checkpoint_hash'],
                    selection_record_sha256=sha256_file(target / 'selection_record.json'))
    write_json(target / 'dimo_teacher_manifest.json', manifest)
    identity = json.loads((target / 'identity.json').read_text())
    ledger = {p.name: sha256_file(p) for p in target.iterdir() if p.name != 'COMPLETED.json'}
    write_json(target / 'COMPLETED.json', {'schema': identity['schema'], 'files': ledger})
    load_dense_teacher_contract(target)
    os.rename(target, output); temporary.rmdir()
    return manifest
