import ast
import copy
import io
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.train.verify_e3_five_epoch_config import verify
from src.explicit_region.config import _load_unexpanded
from src.explicit_region.epoch_sampler import (
    DeterministicEpochSampler, EpochConsumptionTracker, MultiEpochFixedLoader,
    epoch_permutation,
)
from src.explicit_region.reattestation import alpha_probe, compare_alpha, frozen_identity, publish_ready
from tests.fixed200k_helpers import fixture_record

ROOT = Path(__file__).resolve().parents[1]
SHA = 'a' * 64


def test_smoke_must_bind_current_git_corpus_and_exact_resume(tmp_path):
    from src.explicit_region.formal_pipeline import require_bound_smoke, sha256_file
    corpus = tmp_path / 'CORPUS_READY.json'
    corpus.write_text(json.dumps({'train_manifest_sha256': SHA}))
    smoke = {'status': 'PASS', 'git_sha': 'b' * 40, 'world_size': 8, 'data_name': 'fixed_200k',
             'train_manifest_sha256': SHA, 'corpus_ready_sha256': sha256_file(corpus)}
    proofs = ('lora_grad_finite_nonzero', 'region_grad_finite_nonzero',
              'frozen_base_gradients_zero', 'optimizer_scheduler_updates_match',
              'checkpoint_exact_replay', 'optimizer_exact_replay', 'scheduler_exact_replay',
              'sampler_state_exact_replay', 'rank_rng_exact_replay', 'committed_cursor_match',
              'fingerprint_match', 'dataset_identity_bound')
    smoke.update(dict.fromkeys(proofs, True))
    require_bound_smoke(smoke, corpus, 'b' * 40)
    for field in ('git_sha', 'corpus_ready_sha256', 'train_manifest_sha256', *proofs):
        invalid = dict(smoke); invalid[field] = None
        with pytest.raises(RuntimeError, match='exact smoke evidence required'):
            require_bound_smoke(invalid, corpus, 'b' * 40)


def test_selection_requires_preregistered_hash_bound_candidate(tmp_path, monkeypatch):
    import src.explicit_region.formal_pipeline as pipeline
    monkeypatch.setattr(pipeline, 'verify_formal_ready', lambda *args: {'world_size': 8})
    assets = tmp_path / 'assets.json'; assets.write_text('{}')
    selection = tmp_path / 'selection.yaml'
    selection.write_text('selection: {status: READY, threshold_preserve: 0.1, tau_edit: 0.2, tau_noop: 0.3}\n')
    checkpoint = tmp_path / 'checkpoint'; checkpoint.mkdir()
    names = ['adapter_model.safetensors', 'mask_conditioning.safetensors']
    for name in names:
        (checkpoint / name).write_bytes(name.encode())
    files = {name: pipeline.sha256_file(checkpoint / name) for name in names}
    candidate = tmp_path / 'candidate.json'
    candidate.write_text(json.dumps({'status': 'READY', 'git_sha': 'b' * 40,
        'formal_assets_sha256': pipeline.sha256_file(assets), 'files': files,
        'checkpoint_identity_sha256': pipeline.stable_json_hash(files), 'checkpoint_path': str(checkpoint)}))
    kwargs = dict(mode='select', formal_assets=assets, corpus_ready='unused', current_git='b' * 40,
                  formal_ready='unused', selection_config=selection, selected_checkpoint=candidate)
    assert pipeline.verify_pipeline_mode(**kwargs)['world_size'] == 8
    (checkpoint / names[0]).write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='CANDIDATE_HASH_MISMATCH'):
        pipeline.verify_pipeline_mode(**kwargs)
    # Training still ignores selection, including a missing candidate.
    kwargs.update(mode='train', selected_checkpoint=None, selection_config=None)
    assert pipeline.verify_pipeline_mode(**kwargs)['world_size'] == 8


def test_e3_gate_refuses_corpus_without_current_code_reattestation(tmp_path, monkeypatch):
    import src.explicit_region.formal_pipeline as pipeline
    monkeypatch.setattr(pipeline, 'verify_formal_asset_corpus', lambda *args: ({}, {}))
    corpus = tmp_path / 'CORPUS_READY.json'; corpus.write_text('{}')
    with pytest.raises(RuntimeError, match='re-attestation required'):
        pipeline.verify_pipeline_mode(mode='train', pre_smoke=True, formal_assets='unused',
            corpus_ready=corpus, current_git='b' * 40, require_reattestation=True)


def test_selection_cannot_use_pre_smoke_shortcut():
    from src.explicit_region.formal_pipeline import verify_pipeline_mode
    with pytest.raises(RuntimeError, match='selection cannot bypass the smoke gate'):
        verify_pipeline_mode(mode='select', pre_smoke=True, formal_assets='unused',
                             corpus_ready='unused', current_git='b' * 40)


def test_five_epoch_budget_scheduler_and_exact_ten_checkpoints():
    config = _load_unexpanded(ROOT / 'configs/train/cluster_8g_e3.yaml')
    resolved = verify(config)
    assert resolved['steps_per_epoch'] == 200000 // 32 == 6250
    assert config['epochs'] == 5
    assert config['max_optimizer_steps'] == config['scheduler_horizon_steps'] == 31250
    assert resolved['sample_exposures'] == 1000000
    assert config['checkpoint_steps'] == list(range(3125, 31251, 3125))
    assert len(config['checkpoint_steps']) == 10
    assert [s for s in config['checkpoint_steps'] if s % 6250 == 0] == [6250, 12500, 18750, 25000, 31250]
    assert config['warmup_steps'] == 200


def test_e3_recipe_is_inherited_unchanged():
    config = _load_unexpanded(ROOT / 'configs/train/cluster_8g_e3.yaml')
    config['corruption']['full_roi_mask_probability'] = .2
    with pytest.raises(RuntimeError, match='E3_CORE_RECIPE_CHANGED'):
        verify(config)


def test_e3_only_launcher_print_command_and_invalid_experiment():
    result = subprocess.run(['bash', 'scripts/train/run_cluster_e3.sh', '--print-command'],
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'configs/train/cluster_8g_e3.yaml' in result.stdout
    assert all(f'cluster_8g_{name}.yaml' not in result.stdout for name in ['e1', 'e2', 'e4'])
    result = subprocess.run(['bash', 'scripts/train/run_cluster_e3.sh', 'configs/train/cluster_8g_e1.yaml'],
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0


def test_periodic_probe_is_not_selection_and_does_not_change_recipe():
    config = _load_unexpanded(ROOT / 'configs/eval/e3_periodic_probe.yaml')
    assert config['guidance_scale'] == 5
    assert config['selection']['status'] == 'DIAGNOSTIC_ONLY'
    assert all(config['selection'][key] is None for key in ['threshold_preserve', 'tau_edit', 'tau_noop'])
    train = _load_unexpanded(ROOT / 'configs/train/cluster_8g_e3.yaml')
    assert train['validation']['interval_steps'] == 3125
    assert train['validation']['manifest'].endswith('magicbrush_probe128.jsonl')


def test_all_five_200k_epochs_unique_and_disjoint_ranks():
    previous = None
    for epoch in range(5):
        permutation = epoch_permutation(200000, 42, epoch, SHA)
        assert len(permutation) == len(set(permutation)) == 200000
        assert permutation != previous
        ranks = [list(DeterministicEpochSampler(200000, base_seed=42, epoch=epoch,
                    train_manifest_sha256=SHA, rank=rank, world_size=8)) for rank in range(8)]
        assert sum(map(len, ranks)) == 200000
        assert len(set().union(*map(set, ranks))) == 200000
        assert all(len(set(r)) == 25000 for r in ranks)
        previous = permutation


class ToyDataset:
    def __init__(self, length):
        self.rows = [{'sample_uid': str(i)} for i in range(length)]
        self.epoch = 0
    def __len__(self):
        return len(self.rows)
    def set_epoch(self, epoch):
        self.epoch = epoch
    def __getitem__(self, index):
        return {'sample_uid': str(index), 'index': index, 'epoch': self.epoch}


def sampler(length, epoch=0, cursor=0):
    return DeterministicEpochSampler(length, base_seed=42, epoch=epoch,
                                     train_manifest_sha256=SHA, samples_consumed_in_epoch=cursor)


@pytest.mark.parametrize('epoch,cursor', [(0, 0), (0, 32), (1, 8), (3, 32)])
def test_cross_epoch_loader_exact_suffix(epoch, cursor):
    def order(epoch, cursor):
        ds = ToyDataset(32)
        return [(int(e), int(i)) for batch in MultiEpochFixedLoader(ds, sampler(32, epoch, cursor),
                  epochs=5, batch_size=8, pin_memory=False)
                for e, i in zip(batch['epoch'], batch['index'])]
    all_rows = order(0, 0)
    assert len(all_rows) == 160
    assert order(epoch, cursor) == all_rows[epoch * 32 + cursor:]


def test_tracker_resume_and_epoch_boundary_audit():
    ds = ToyDataset(32); s = sampler(32)
    tracker = EpochConsumptionTracker(ds.rows, s, 8, 1)
    for batch in MultiEpochFixedLoader(ds, s, epochs=2, batch_size=8, pin_memory=False):
        row = tracker.commit(batch['sample_uid'], [1.] * 8)
        if row['epoch_boundary']:
            assert row['epoch_rows_consumed'] == row['epoch_unique_rows'] == 32
            assert row['epoch_train_ce_mean'] == 1
    assert row['epoch'] == 1 and row['epoch_boundary']
    with pytest.raises(RuntimeError, match='sampler duplication'):
        EpochConsumptionTracker(ds.rows, sampler(32), 8, 1).commit(['0'] * 8)


@pytest.mark.parametrize('stop', [4, 5])
def test_cross_epoch_resume_toy_optimizer_scheduler_rng_exact(stop):
    torch.manual_seed(42)
    ds = ToyDataset(32); s = sampler(32)
    model = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., step / 2))
    checkpoint = None
    def update(batch):
        x = (batch['index'].float() + batch['epoch'].float())[:, None] / 32
        optimizer.zero_grad(); loss = (model(x) - x).square().mean()
        loss.backward(); optimizer.step(); scheduler.step()
    for step, batch in enumerate(MultiEpochFixedLoader(ds, s, epochs=5, batch_size=8, pin_memory=False,
                                generator=torch.Generator().manual_seed(1)), 1):
        update(batch)
        if step == stop:
            checkpoint = copy.deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
                                        torch.get_rng_state(), int(batch['epoch'][0]), (step % 4 or 4) * 8))
    expected = copy.deepcopy(model.state_dict())
    weights, opt, sched, rng, epoch, cursor = checkpoint
    model.load_state_dict(weights); optimizer.load_state_dict(opt); scheduler.load_state_dict(sched)
    torch.set_rng_state(rng)
    # Dedicated loader generator must not advance dropout's global CPU RNG.
    generator = torch.Generator().manual_seed(1)
    for batch in MultiEpochFixedLoader(ds, sampler(32, epoch, cursor), epochs=5,
                                     batch_size=8, pin_memory=False, generator=generator):
        update(batch)
    assert all(torch.equal(expected[k], model.state_dict()[k]) for k in expected)
    assert scheduler.last_epoch == 20


def test_reattest_snapshot_never_changes_frozen_manifest(tmp_path):
    train = tmp_path / 'train.jsonl'; val = tmp_path / 'validation.jsonl'
    records = [dict(fixture_record(tmp_path / 'images', 'magicbrush', i), manifest_index=i) for i in range(4)]
    train.write_text(''.join(json.dumps(r) + '\n' for r in records)); val.write_text('')
    ready = {'files': {'train_200k': {'path': str(train)}, 'validation_dev': {'path': str(val)}}}
    before_bytes = train.read_bytes(), val.read_bytes()
    before, _, _ = frozen_identity(ready, expected_rows=4)
    after, _, _ = frozen_identity(ready, expected_rows=4)
    assert before == after and before_bytes == (train.read_bytes(), val.read_bytes())
    with pytest.raises(ValueError, match='exactly 200000'):
        frozen_identity(ready)


def test_alpha_semantics_never_use_rgb_brightness():
    raw = Image.new('RGBA', (2, 1)); raw.putdata([(0, 0, 0, 255), (255, 255, 255, 0)])
    actual = Image.new('L', (2, 1)); actual.putdata([0, 255])
    assert compare_alpha(raw, actual)['binary_disagreement'] == 0
    wrong = Image.new('L', (2, 1)); wrong.putdata([255, 0])
    with pytest.raises(RuntimeError, match='MASK_IDENTITY_UNVERIFIED'):
        compare_alpha(raw, wrong)
    with pytest.raises(RuntimeError, match='must be RGBA'):
        compare_alpha(raw.convert('RGB'), actual)


def test_alpha_real_parquet_32_records(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = tmp_path / 'canonical'; snapshot = tmp_path / 'snapshot/data'; snapshot.mkdir(parents=True)
    rows, exports, raw_rows = [], [], []
    for i in range(32):
        row = fixture_record(root, 'magicbrush', i)
        rows.append(row)
        exports.append({'source': row['source_locator']['relative_path'], 'img_id': str(i),
                        'turn_index': 0, 'instruction': row['instruction_original']})
        with Image.open(root / row['region_locator']['relative_path']) as mask:
            raw = Image.new('RGBA', mask.size, (30, 70, 200, 255))
            raw.putalpha(Image.fromarray(255 - np.asarray(mask)))
        buffer = io.BytesIO(); raw.save(buffer, format='PNG')
        raw_rows.append({'img_id': str(i), 'turn_index': 0, 'mask_img': {'bytes': buffer.getvalue(), 'path': None}})
    (root / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in exports))
    pq.write_table(pa.Table.from_pylist(raw_rows), snapshot / 'train-00000.parquet')
    result = alpha_probe(rows, root, {'local_path': str(snapshot.parent), 'resolved_revision': 'fixture-rev1'}, {})
    assert result['mode'] == 'RAW_COMPARE' and len(result['records']) == 32
    assert all(r['status'] == 'PASS' and r['exact_disagreement'] == 0 for r in result['records'])


def test_missing_raw_and_provenance_fail_closed(tmp_path):
    rows = [fixture_record(tmp_path, 'magicbrush', i) for i in range(32)]
    with pytest.raises(RuntimeError, match='MASK_IDENTITY_UNVERIFIED'):
        alpha_probe(rows, tmp_path, {'local_path': str(tmp_path)}, {'files': {}})


def test_training_math_ast_matches_personal_main():
    trainer = 'scripts/train/train_explicit_region.py'
    baseline = subprocess.check_output(['git', 'show', 'personal/main:' + trainer], cwd=ROOT, text=True)
    current = (ROOT / trainer).read_text()
    def function(source):
        return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_batch')
    assert ast.dump(function(baseline)) == ast.dump(function(current))
    corruption = 'src/explicit_region/corruption.py'
    assert (ROOT / corruption).read_bytes() == subprocess.check_output(['git', 'show', 'personal/main:' + corruption], cwd=ROOT)


@pytest.mark.parametrize('failure', [False, True])
def test_reattest_publication_preserves_data_and_rolls_back_bytes(tmp_path, monkeypatch, failure):
    import shutil
    from src.explicit_region.formal_pipeline import sha256_file, stable_json_hash
    train = tmp_path / 'train.jsonl'; train.write_text('immutable rows\n')
    marker = tmp_path / 'CORPUS_READY.json'; ledger = tmp_path / 'formal_assets.json'
    original_corpus = {'status': 'READY', 'files': {'train_200k': {'path': str(train), 'sha256': sha256_file(train)}}}
    marker.write_text(json.dumps(original_corpus, separators=(',', ':')))
    ledger.write_text('{"old":true}')
    old = marker.read_bytes(), ledger.read_bytes(), train.read_bytes()
    shutil.copyfile(marker, tmp_path / 'CORPUS_READY.previous.json')
    shutil.copyfile(ledger, tmp_path / 'formal_assets.previous.json')
    git = 'a' * 40; identity = stable_json_hash({'asset': 'pinned'})
    new_ready = dict(original_corpus, git_sha=git, formal_assets_sha256=identity)
    # Compute the candidate marker hash with exactly the production serializer.
    candidate = tmp_path / 'candidate.json'
    candidate.write_text(json.dumps(new_ready, indent=2, sort_keys=True) + '\n')
    new_assets = {'git_sha': git, 'formal_identity_sha256': identity,
                  'corpus_ready': {'sha256': sha256_file(candidate)}}
    if failure:
        def reject(*args): raise RuntimeError('simulated publication failure')
        monkeypatch.setattr('src.explicit_region.reattestation.verify_formal_asset_corpus', reject)
        with pytest.raises(RuntimeError, match='simulated'):
            publish_ready(marker, ledger, new_ready, new_assets, git, tmp_path)
        assert old == (marker.read_bytes(), ledger.read_bytes(), train.read_bytes())
    else:
        publish_ready(marker, ledger, new_ready, new_assets, git, tmp_path)
        assert json.loads(marker.read_text())['git_sha'] == git
        assert train.read_bytes() == old[2]


def test_periodic_quality_monitor_observes_but_never_selects(tmp_path):
    from src.explicit_region.validation import record_periodic_diagnostic
    for step, ce, lpips in [(3125, 8., .5), (6250, 7., .6), (9375, 6., .7)]:
        record_periodic_diagnostic(tmp_path, {'global_step': step, 'sample_mean_ce': ce},
                                   {'per_sample_after_seed_mean': [{'metrics': {'inside_masked_lpips': lpips}}]})
    report = json.loads((tmp_path / 'periodic_quality_trend.json').read_text())
    assert report['POSSIBLE_OVERFIT'] == 'YES'
    assert report['automatic_selection'] is False
    assert not (tmp_path / 'SELECTED_CHECKPOINT.json').exists()
