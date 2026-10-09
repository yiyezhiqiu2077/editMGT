"""Eight actual CPU/Gloo processes; never an 8-GPU acceptance certificate."""
from datetime import timedelta
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.dimo.checkpoint import load_dimo_checkpoint, save_dimo_checkpoint
from src.dimo.distributed import ExplicitGradientSync, assert_rank_consistency, state_digest
from src.dimo.smoke import compare_states
from src.dimo.step import complete_dimo_step
from src.explicit_region.checkpoint import capture_local_rng_state
from src.explicit_region.epoch_sampler import DeterministicEpochSampler
from tests.dimo.toy import batch, make_bundle, step_config


def _snapshot(bundle):
    roles, student, auxiliary, student_sched, auxiliary_sched, ema = bundle
    return dict(student=roles.role_state_dict('student'), auxiliary=roles.role_state_dict('auxiliary'),
                student_optimizer=student.state_dict(), auxiliary_optimizer=auxiliary.state_dict(),
                student_scheduler=student_sched.state_dict(), auxiliary_scheduler=auxiliary_sched.state_dict(),
                ema=ema.state_dict())


def _seed(rank):
    random.seed(100 + rank)
    np.random.seed(100 + rank)
    torch.manual_seed(100 + rank)


def _worker(rank, rendezvous, root):
    torch.set_num_threads(1)
    assert not torch.cuda.is_available(), 'This infrastructure test must be CPU-only'
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=8,
                            timeout=timedelta(seconds=120))
    try:
        bundle = make_bundle()
        sync = ExplicitGradientSync(bundle[0], bucket_bytes=32)
        named = sync.named['student']
        for _, parameter in named:
            parameter.grad = torch.full_like(parameter, rank + 1)
        sync('student')
        assert all(torch.equal(p.grad, torch.full_like(p, 4.5)) for _, p in named)
        # Rank-local unused parameters contribute zero; globally unused stay None.
        for _, parameter in named:
            parameter.grad = None
        named[0][1].grad = None if rank == 0 else torch.ones_like(named[0][1])
        sync('student')
        assert torch.equal(named[0][1].grad, torch.full_like(named[0][1], 7 / 8))
        assert all(p.grad is None for _, p in named[1:])
        for violation in ('nonfinite', 'teacher_grad'):
            for _, parameter in named:
                parameter.grad = torch.ones_like(parameter)
            teacher = bundle[0].role_named_parameters('teacher')[0][1]
            if rank == 7:
                if violation == 'nonfinite':
                    named[0][1].grad.fill_(float('nan'))
                else:
                    teacher.grad = torch.ones_like(teacher)
            try:
                sync('student')
                raise AssertionError('rank-local violation was not collectively rejected')
            except RuntimeError as exc:
                assert 'ISOLATION_OR_GRADIENT' in str(exc)
            teacher.grad = None
        # Parameter divergence must be detected, not averaged away after updates.
        bundle = make_bundle()
        with torch.no_grad():
            if rank == 7:
                bundle[0].role_named_parameters('student')[0][1].add_(.1)
        try:
            assert_rank_consistency(bundle[0], bundle[-1], *bundle[1:5], step=0)
            raise AssertionError('desynchronization was not detected')
        except RuntimeError as exc:
            assert 'DESYNCHRONIZATION' in str(exc)

        def run(bundle, start, stop, cursor):
            sync = ExplicitGradientSync(bundle[0], bucket_bytes=32)
            sampler = DeterministicEpochSampler(256, base_seed=42, epoch=0,
                train_manifest_sha256='cpu-fixture-only', rank=rank, world_size=8,
                samples_consumed_in_epoch=cursor)
            local_order = list(sampler)
            trace = []
            for step, index in zip(range(start, stop), local_order):
                uid = f'cpu-fixture-{index}'
                rng = [random.random(), float(np.random.rand()), float(torch.rand(()))]
                diagnostic, artifacts = complete_dimo_step(bundle[0], **batch(uid),
                    student_optimizer=bundle[1], auxiliary_optimizer=bundle[2],
                    student_scheduler=bundle[3], auxiliary_scheduler=bundle[4], student_ema=bundle[5],
                    base_seed=77, epoch=0, committed_dimo_step=step, mask_token_id=5,
                    codebook_size=5, config=step_config(), gradient_sync=sync)
                assert diagnostic['outside_mismatch_count'] == diagnostic['nan_inf_count'] == 0
                assert diagnostic['student_grad_norm'] > 0 and diagnostic['aux_grad_norm'] > 0
                gathered = [None] * 8
                dist.all_gather_object(gathered, uid)
                expected = [f'cpu-fixture-{i}' for i in sampler.permutation[step * 8:(step + 1) * 8]]
                assert gathered == expected and len(set(gathered)) == 8
                sampler.samples_consumed_in_epoch += 8
                trace.append(dict(uid=uid, rng=rng, artifacts=state_digest(artifacts)))
                if step + 1 in (1, 20):
                    assert_rank_consistency(bundle[0], bundle[-1], *bundle[1:5], step=step + 1)
            return trace, sampler

        _seed(rank)
        fresh = make_bundle()
        full_trace, _ = run(fresh, 0, 20, 0)
        full_state, full_rng = _snapshot(fresh), capture_local_rng_state(rank)
        _seed(rank)
        first = make_bundle()
        first_trace, sampler = run(first, 0, 10, 0)
        teacher = {'bundle_sha256': 'cpu-fixture-teacher'}
        save_dimo_checkpoint(Path(root) / 'checkpoint-10', roles=first[0],
            student_optimizer=first[1], auxiliary_optimizer=first[2],
            student_scheduler=first[3], auxiliary_scheduler=first[4], student_ema=first[5],
            committed_dimo_step=10, fixed_corpus_cursor=80, epoch=0, samples_consumed=80,
            config_sha256='cpu-fixture-config', teacher_bundle_fingerprint=teacher,
            inference_fingerprint={'fixture': True}, upstream_commit='cpu-fixture',
            sampler_state=sampler.state_dict() | {'length': 256})
        # Simulate unrelated process initialization between save and restore.
        random.random(); np.random.rand(); torch.rand(17)
        resumed = make_bundle()
        state = load_dimo_checkpoint(Path(root) / 'checkpoint-10', roles=resumed[0],
            student_optimizer=resumed[1], auxiliary_optimizer=resumed[2],
            student_scheduler=resumed[3], auxiliary_scheduler=resumed[4], student_ema=resumed[5],
            expected_config_hash='cpu-fixture-config', expected_teacher_bundle_fingerprint=teacher,
            expected_inference_fingerprint={'fixture': True}, expected_upstream_commit='cpu-fixture',
            expected_sampler_identity={'world_size': 8, 'length': 256})
        assert state['global_dimo_step'] == 10 and state['fixed_corpus_cursor'] == 80
        resumed_trace, _ = run(resumed, 10, 20, state['sampler_state']['samples_consumed_in_epoch'])
        assert full_trace == first_trace + resumed_trace
        comparison = compare_states(full_state, _snapshot(resumed))
        assert comparison['bitwise_exact'] and comparison['within_tolerance']
        assert compare_states(full_rng, capture_local_rng_state(rank))['bitwise_exact']
        assert state_digest(resumed[-1].state_dict()) != state_digest(make_bundle()[-1].state_dict())
        (Path(root) / f'rank-{rank}.json').write_text(json.dumps(dict(
            status='PASS', backend='gloo', device='cpu', fixture_only=True,
            world_size=8, fresh_steps=20, resumed_steps=20, equivalence='bitwise_exact')))
    finally:
        dist.destroy_process_group()


def test_eight_cpu_ranks_gradient_roles_rng_and_twenty_step_resume(tmp_path):
    mp.spawn(_worker, args=(f'file://{tmp_path}/rendezvous', str(tmp_path)), nprocs=8, join=True)
    rows = [json.loads((tmp_path / f'rank-{rank}.json').read_text()) for rank in range(8)]
    assert all(row['equivalence'] == 'bitwise_exact' and row['fixture_only'] for row in rows)


def test_full_200k_eight_rank_sampler_no_repeat_and_resume():
    shards = [list(DeterministicEpochSampler(200000, base_seed=42, epoch=0,
        train_manifest_sha256='identity-only-no-data-loaded', rank=rank, world_size=8)) for rank in range(8)]
    assert all(len(shard) == 25000 for shard in shards)
    assert len({index for shard in shards for index in shard}) == 200000
    for rank in range(8):
        resumed = list(DeterministicEpochSampler(200000, base_seed=42, epoch=0,
            train_manifest_sha256='identity-only-no-data-loaded', rank=rank, world_size=8,
            samples_consumed_in_epoch=4000))
        assert resumed == shards[rank][500:]
    next_epoch = list(DeterministicEpochSampler(200000, base_seed=42, epoch=1,
        train_manifest_sha256='identity-only-no-data-loaded', rank=0, world_size=8))
    assert next_epoch != shards[0]
