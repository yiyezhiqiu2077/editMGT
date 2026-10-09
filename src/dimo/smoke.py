"""Evidence verifier for real CUDA/NCCL fresh20 versus 10+resume20 runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import read_dimo_state


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compare_states(left, right, *, rtol=1e-5, atol=1e-7):
    exact, tolerant, maximum = True, True, 0.0
    def visit(a, b):
        nonlocal exact, tolerant, maximum
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                exact = tolerant = False; return
            same = torch.equal(a, b)
            exact &= same
            if a.is_floating_point():
                if a.numel(): maximum = max(maximum, float((a.double() - b.double()).abs().max()))
                tolerant &= bool(torch.isfinite(a).all() and torch.isfinite(b).all()
                                 and torch.allclose(a, b, rtol=rtol, atol=atol))
            else:
                tolerant &= same
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            same = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
            exact &= same; tolerant &= same
        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys(): exact = tolerant = False; return
            for k in a: visit(a[k], b[k])
        elif isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            if type(a) is not type(b) or len(a) != len(b): exact = tolerant = False; return
            for x, y in zip(a, b): visit(x, y)
        else:
            same = type(a) is type(b) and a == b
            exact &= same; tolerant &= same
    visit(left, right)
    return {'bitwise_exact': bool(exact), 'within_tolerance': bool(tolerant),
            'maximum_abs_difference': maximum, 'rtol': rtol, 'atol': atol}


def verify_real_smoke(fresh20, fresh10, resume20):
    roots = [Path(p) for p in (fresh20, fresh10, resume20)]
    manifests = [json.loads((r / 'run_manifest.json').read_text()) for r in roots]
    for m in manifests:
        execution = m.get('actual_execution', {})
        if (m.get('world_size') != 8 or m.get('dev_only') is not True or m.get('formal_teacher') is not False
                or m.get('resolved_config', {}).get('data', {}).get('name') != 'fixed_200k'
                or execution.get('backend') != 'nccl' or execution.get('device_type') != 'cuda'
                or execution.get('visible_gpu_count') != 8 or m.get('git_dirty') is not False):
            raise RuntimeError('SMOKE_EVIDENCE_IS_NOT_REAL_EIGHT_GPU_D200K')
    identities = ('git_sha', 'config_hash', 'teacher_hash', 'teacher_bundle_fingerprint',
                  'inference_fingerprint', 'data_identity', 'compatibility_sha256')
    checks = {'run_identities': all(all(m.get(k) == manifests[0].get(k) for k in identities)
                                    for m in manifests[1:])}
    full, first, resumed = [read_rows(r / 'metrics.jsonl') for r in roots]
    split = first + resumed
    checks['step_sequences'] = ([r['global_dimo_step'] for r in full] == list(range(1, 21))
        and [r['global_dimo_step'] for r in first] == list(range(1, 11))
        and [r['global_dimo_step'] for r in resumed] == list(range(11, 21)))
    uids = [u for r in full for u in r['committed_sample_uids']]
    checks['unique_samples'] = len(uids) == len(set(uids)) == 160
    checks['rank_shards_and_cursor'] = all(
        r['samples_consumed'] == r['global_dimo_step'] * 8 and len(r['ranks']) == 8
        and [x['rank'] for x in r['ranks']] == list(range(8))
        and [u for x in r['ranks'] for u in x['trace']['sample_uids']] == r['committed_sample_uids']
        for r in full + split)
    trace = lambda rows: [[x['trace'] for x in r['ranks']] for r in rows]
    checks['sample_seed_and_artifact_replay'] = trace(full) == trace(split)
    checks['finite'] = all(r['finite'] is True and r['nan_inf_count'] == 0 for r in full + split)
    checks['outside_hardlock'] = all(r['outside_mismatch_count'] == 0 for r in full + split)
    checks['gradient_sync_and_role_isolation'] = all(r['gradient_sync'] == 'explicit_allreduce'
        and r['role_isolation'] is True for r in full + split)
    checks['scheduler_updates'] = all(r['student_scheduler_step'] == r['auxiliary_scheduler_step']
                                    == r['global_dimo_step'] for r in full + split)
    checks['gradient_diagnostics'] = all(
        d['teacher_grad_norm'] == 0 and d['aux_grad_norm'] > 0
        and (d['dimo_gradient_norm'] <= 1e-12 or d['student_grad_norm'] > 0)
        for r in full + split for x in r['ranks'] for d in [x['diagnostics']])
    signal = [json.loads((roots[0] / 'nonzero-signal' / f'rank-{i}.json').read_text()) for i in range(8)]
    checks['nonzero_path_diagnostic'] = all(s.get('passed') is True for s in signal)
    consistency = read_rows(roots[0] / 'rank_consistency.jsonl')
    checks['all_rank_state_consistency'] = ({0, 1, 20} <= {r['step'] for r in consistency}
        and all(r['status'] == 'PASS' and r['world_size'] == 8 and r['equivalence'] == 'bitwise_exact'
                and len(r['ranks']) == 8 and all(x == r['ranks'][0] for x in r['ranks']) for r in consistency))
    states = [read_dimo_state(r / 'checkpoint-20') for r in (roots[0], roots[2])]
    checks['checkpoint_progress'] = all(s['global_dimo_step'] == 20 and s['samples_consumed'] == 160
                                       and s['world_size'] == 8 for s in states)
    fields = ('student_state', 'auxiliary_state', 'student_ema', 'student_optimizer', 'auxiliary_optimizer',
              'student_scheduler', 'auxiliary_scheduler', 'sampler_state', 'rank_rng', 'config_hash',
              'teacher_bundle_fingerprint', 'inference_fingerprint')
    comparisons = {k: compare_states(states[0][k], states[1][k]) for k in fields}
    checks['resume_state_equivalence'] = all(c['within_tolerance'] for c in comparisons.values())
    checks['exact_rank_rng_and_sampler'] = all(comparisons[k]['bitwise_exact'] for k in ('rank_rng', 'sampler_state'))
    checks['optimizer_step_counts'] = all(
        bool(s[role + '_optimizer']['state']) and
        all(float(v['step']) == 20 for v in s[role + '_optimizer']['state'].values())
        for s in states for role in ('student', 'auxiliary'))
    initial = read_dimo_state(roots[0] / 'checkpoint-0')
    checks['ema_state_available'] = (set(initial['student_ema']['shadow']) == set(states[0]['student_ema']['shadow'])
        and states[0]['student_ema']['decay'] == .9999)
    exact = all(c['bitwise_exact'] for c in comparisons.values())
    student_signal = any(r['student_parameter_update_ratio'] > 0 for r in full)
    ema_changed = not compare_states(initial['student_ema'], states[0]['student_ema'])['bitwise_exact']
    checks['ema_update_response'] = not student_signal or ema_changed
    elapsed = sum(r['step_wall_time'] for r in full)
    report = {'schema': 'dimo-real-8gpu-smoke-v1', 'status': 'PASS' if all(checks.values()) else 'FAIL',
        'world_size': 8, 'data_name': 'fixed_200k', 'git_sha': manifests[0]['git_sha'],
        'teacher_hash': manifests[0]['teacher_hash'], 'data_identity': manifests[0]['data_identity'],
        'compatibility_sha256': manifests[0]['compatibility_sha256'], 'checks': checks,
        'resume_equivalence': ('bitwise_exact' if exact else 'numerical_tolerance')
                              if checks['resume_state_equivalence'] else 'FAIL',
        'comparisons': comparisons, 'dev_only': True, 'formal_teacher': False,
        'student_learning_signal_observed': student_signal,
        'ema_changed': ema_changed,
        'peak_vram_per_rank_bytes': [max(r['ranks'][i]['peak_vram_bytes'] for r in full) for i in range(8)],
        'steps_per_second': len(full) / elapsed, 'estimated_epoch_seconds': 25000 * elapsed / len(full),
        'note': 'Runtime estimate includes measured logging/checkpoint cadence, not a formal quality claim.'}
    return report
