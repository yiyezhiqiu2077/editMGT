"""Budgets for new mixed DiMO outputs; historical weights are retained."""
from collections import defaultdict
import os
from pathlib import Path
import shutil


def existing_parent(path):
    path = Path(path).resolve()
    while not path.exists(): path = path.parent
    return path


def segment_budget(parameters, config, output, stop):
    """Reserve only future writes of this invocation, separately per filesystem."""
    if not 0 <= stop <= config['max_optimizer_steps']:
        raise RuntimeError('MIXED_DIMO_INVALID_STOP_STEP')
    output = Path(output)
    resume = {step for step in config['resume_checkpoint_steps'] if step <= stop} | {stop}
    inference = {0} | {step for step in config['evaluation_steps'] if step <= stop}
    volumes = defaultdict(lambda: {'retained_bytes': 0, 'atomic_bytes': 0, 'paths': []})
    for kind, steps, factor in [('checkpoints', resume, 28), ('inference', inference, 8)]:
        parent = existing_parent(output / kind); device = os.stat(parent).st_dev
        record = volumes[device]; record['path'] = str(parent)
        missing = [step for step in steps if not (output / kind / f'checkpoint-{step}/COMPLETED.json').is_file()]
        record['retained_bytes'] += len(missing) * factor * parameters
        record['atomic_bytes'] = max(record['atomic_bytes'], factor * parameters if missing else 0)
        record['paths'].append(str((output / kind).resolve()))
    for record in volumes.values():
        record['required_bytes'] = int((record['retained_bytes'] + record['atomic_bytes']) * 1.08) + 4 * 1024**3
        record['free_bytes'] = shutil.disk_usage(record['path']).free
        if record['free_bytes'] < record['required_bytes']:
            raise RuntimeError(f'MIXED_DIMO_SEGMENT_DISK_BUDGET_FAILED: {record}')
    return {'status': 'PASS', 'stop_at_step': stop, 'parameters_per_role': parameters,
            'volumes': dict(volumes), 'retention': 'retain all checkpoints; no automatic deletion'}


def whole_handoff_budget(root, secondary, parameters, *, evaluation_root=None):
    """Include unfinished SFT saves and all new DiMO acceptance/training states."""
    root, secondary = Path(root), Path(secondary)
    primary_parent = existing_parent(root); secondary_parent = existing_parent(secondary)
    if os.stat(primary_parent).st_dev == os.stat(secondary_parent).st_dev:
        raise RuntimeError('MIXED_DIMO_STORAGE_PLAN_REQUIRES_TWO_FILESYSTEMS')
    skip_record = root / 'USER_SKIP_SFT_GPU_ACCEPTANCE.json'
    if skip_record.exists():
        from scripts.train.run_mixed30k_continuation import verify_skip_sft_acceptance
        verify_skip_sft_acceptance(root)
    sft_outputs = ([('smoke2', 2), ('fresh20', 20), ('resume20', 10), ('resume20', 20)] if not skip_record.exists() else [])
    sft_outputs += [('train', step) for step in (1250, 2500, 3750, 5000, 6250)]
    remaining_sft = sum(not (root / 'sft' / name / f'checkpoint-{step}/COMPLETED.json').exists()
                        for name, step in sft_outputs)
    # Primary: SFT, Teacher, smoke2 + initial inference, three resumes, atomic save.
    primary = int((remaining_sft * 12 + 4 + 36 + 3 * 28 + 28) * parameters * 1.05) + (8 if evaluation_root else 32) * 1024**3
    # Secondary peak: fresh/resume acceptance retained plus four final raw/EMA exports.
    # Their atomic acceptance save occurs before the final inference exports.
    secondary_bytes = int(max(3 * 28 + 2 * 8 + 4 * 8, 4 * 28 + 2 * 8) * parameters * 1.05) + 8 * 1024**3
    records = [{'path': str(primary_parent), 'required_bytes': primary},
               {'path': str(secondary_parent), 'required_bytes': secondary_bytes}]
    if evaluation_root is not None:
        evaluation_parent = existing_parent(evaluation_root)
        if os.stat(evaluation_parent).st_dev in (os.stat(primary_parent).st_dev, os.stat(secondary_parent).st_dev):
            raise RuntimeError('MIXED_DIMO_EVALUATION_STORAGE_REQUIRES_THIRD_FILESYSTEM')
        records.append({'path': str(evaluation_parent), 'required_bytes': 32 * 1024**3,
                        'purpose': 'post-training DEV predictions and case images'})
    for row in records:
        row['free_bytes'] = shutil.disk_usage(row['path']).free
        row['status'] = 'PASS' if row['free_bytes'] >= row['required_bytes'] else 'FAIL'
    return {'status': 'PASS' if all(row['status'] == 'PASS' for row in records) else 'FAIL',
            'remaining_sft_checkpoints': remaining_sft, 'filesystems': records,
            'retention': 'all old and new checkpoints retained; no deletion'}
