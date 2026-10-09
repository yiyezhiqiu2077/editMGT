#!/usr/bin/env python3
"""Summarize a real completed 500-update run, not a quality acceptance claim."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.dimo.checkpoint import read_dimo_state
from src.dimo.smoke import read_rows
from src.explicit_region.formal_pipeline import write_json_atomic


def summarize(directory):
    root = Path(directory)
    manifest = json.loads((root / 'run_manifest.json').read_text())
    execution = manifest.get('actual_execution', {})
    if (manifest.get('world_size') != 8 or manifest.get('target_steps') != 500
            or execution.get('backend') != 'nccl' or execution.get('device_type') != 'cuda'
            or manifest.get('dev_only') is not True or manifest.get('formal_teacher') is not False):
        raise RuntimeError('REQUIRES_REAL_EIGHT_GPU_PILOT_NOT_CPU_FIXTURE')
    metrics = read_rows(root / 'metrics.jsonl')
    checkpoints = [read_dimo_state(root / f'checkpoint-{step}') for step in (0, 50, 100, 250, 500)]
    uids = [uid for row in metrics for uid in row['committed_sample_uids']]
    consistency = read_rows(root / 'rank_consistency.jsonl')
    checks = {
        '500_steps': [row['global_dimo_step'] for row in metrics] == list(range(1, 501)),
        '4000_unique_samples': len(uids) == len(set(uids)) == 4000,
        'sample_cursor': all(row['samples_consumed'] == row['global_dimo_step'] * 8 for row in metrics),
        'finite': all(row['finite'] is True and row['nan_inf_count'] == 0 for row in metrics),
        'outside_hardlock': all(row['outside_mismatch_count'] == 0 for row in metrics),
        'gradient_roles': all(row['role_isolation'] is True and row['gradient_sync'] == 'explicit_allreduce' for row in metrics),
        'checkpoint_cadence': all(state['global_dimo_step'] == step and state['samples_consumed'] == step * 8
            and state['world_size'] == 8 and state['config_hash'] == manifest['config_hash']
            and state['teacher_bundle_fingerprint'] == manifest['teacher_bundle_fingerprint']
            for state, step in zip(checkpoints, (0, 50, 100, 250, 500))),
        'rank_consistency': {0, 1, 20, 100, 500} <= {row['step'] for row in consistency}
            and all(row['status'] == 'PASS' and row['world_size'] == 8
                    and len(row['ranks']) == 8 and all(r == row['ranks'][0] for r in row['ranks']) for row in consistency),
    }
    elapsed = sum(row['step_wall_time'] for row in metrics)
    return {'schema': 'dimo-real-8gpu-pilot-v1', 'status': 'PASS' if all(checks.values()) else 'FAIL',
        'checks': checks, 'steps_completed': len(metrics), 'samples_consumed': len(uids),
        'dev_only': True, 'formal_teacher': False, 'world_size': 8,
        'git_sha': manifest['git_sha'], 'teacher_hash': manifest['teacher_hash'],
        'data_identity': manifest['data_identity'], 'compatibility_sha256': manifest['compatibility_sha256'],
        'student_update_observed': any(row['student_parameter_update_ratio'] > 0 for row in metrics),
        'auxiliary_update_observed': any(row['aux_parameter_update_ratio'] > 0 for row in metrics),
        'ema_distance_last': metrics[-1]['ema_distance'] if metrics else None,
        'peak_vram_per_rank_bytes': [max(row['ranks'][rank]['peak_vram_bytes'] for row in metrics) for rank in range(8)] if metrics else [],
        'steps_per_second': len(metrics) / elapsed if elapsed else None,
        'estimated_epoch_seconds': 25000 * elapsed / len(metrics) if metrics else None,
        'quality_evaluated': False, 'DIMO_LEARNING_SIGNAL': 'UNCLEAR',
        'note': 'Optimizer movement and surrogate loss are not editing-quality evidence; inspect the fixed DEV comparison.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = summarize(args.run_dir)
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
