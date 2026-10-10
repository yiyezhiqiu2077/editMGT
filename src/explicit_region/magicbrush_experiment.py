"""Independent MagicBrush experiment admission; never writes D200K READY."""
import json
from pathlib import Path
from .contracts import sha256_file

MODE = 'magicbrush_exploratory'


def is_magicbrush(config):
    return config.get('experiment_mode') == MODE


def verify_magicbrush_ready(config):
    data=config['data']; root=Path(data['corpus_ready'])
    record=json.loads(root.read_text())
    if (record.get('schema')!='magicbrush-training-ready-v1' or record.get('status')!='VERIFIED'
            or record.get('selection_scope')!=MODE or record.get('total_rows')!=data['expected_rows']
            or data['name']!='fixed_magicbrush_train' or record.get('session_consistent') is not True
            or record.get('train_dev_leakage')!='PASS' or record.get('loader_geometry')!='PASS'
            or record.get('padding') is not False):
        raise RuntimeError('MAGICBRUSH_REAL_DATA_NOT_VERIFIED')
    for r in record['files'].values():
        if sha256_file(r['path'])!=r['sha256']: raise RuntimeError('MAGICBRUSH_FROZEN_FILE_CHANGED')
    if record['train_manifest_sha256']!=sha256_file(data['manifest']):
        raise RuntimeError('MAGICBRUSH_TRAIN_MANIFEST_CHANGED')
    if data['expected_rows'] != 8800 and record.get('budget_recalculation',{}).get('reason') != 'whole_session_geometry_exclusions':
        raise RuntimeError('MAGICBRUSH_ALTERNATIVE_BUDGET_NOT_DOCUMENTED')
    return record


def validate_magicbrush_budget(config, stage):
    if not is_magicbrush(config): raise RuntimeError('MAGICBRUSH_EXPLICIT_MODE_REQUIRED')
    world=config.get('world_size') if stage=='sft' else config['distributed']['world_size']
    accumulation=4 if stage=='sft' else 1
    rows=config['data']['expected_rows']
    if not isinstance(rows,int) or not 0 < rows <= 8800 or rows%16:
        raise RuntimeError('MAGICBRUSH_NO_PADDING_ROW_COUNT_CHANGED')
    steps=10*rows//(4*accumulation)
    if (world,config['epochs'],config['max_optimizer_steps'],
            config['batch_per_gpu'],config['gradient_accumulation'],config['data']['name']) != (
            4,10,steps,1,accumulation,'fixed_magicbrush_train'):
        raise RuntimeError('MAGICBRUSH_4A800_REGISTERED_BUDGET_CHANGED')
    if stage=='dimo' and (config.get('formal_ready') is not False or config['initialization'] !=
                         {'mask_ratio':1.0,'random_tokens_inside_roi':False}):
        raise RuntimeError('MAGICBRUSH_DIMO_IDENTITY_OR_INITIALIZATION_CHANGED')
    if stage=='sft' and not config.get('initial_dense_checkpoint'):
        raise RuntimeError('MAGICBRUSH_DENSE_WARM_START_REQUIRED')
    if stage=='sft' and config['checkpoint_steps'] != [e*(rows//16) for e in (1,2,3,5,7,10)]:
        raise RuntimeError('MAGICBRUSH_SFT_CANDIDATE_STEPS_CHANGED')
    if stage=='dimo' and config['optimizer']['horizon_steps'] != steps:
        raise RuntimeError('MAGICBRUSH_DIMO_HORIZON_CHANGED')
