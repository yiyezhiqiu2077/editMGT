import json

from PIL import Image
import pytest

from scripts.eval.compare_dimo_dev128 import LABELS, METRICS, compare
from src.dimo.contracts import _sha256
from src.dimo.dev import verified_dev128
from tests.fixed200k_helpers import fixture_record


def test_dev128_requires_frozen_validation_not_test_and_no_train_leakage(tmp_path, monkeypatch):
    rows = [fixture_record(tmp_path / 'unit-fixture', 'magicbrush', index) for index in range(128)]
    manifest = tmp_path / 'magicbrush_probe128.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    train = tmp_path / 'train-fixture.jsonl'
    train.write_text(json.dumps({'source_sha256': 'nonoverlapping', 'group_id': 'training-only',
                                 'dataset_name': 'magicbrush'}) + '\n')
    marker = tmp_path / 'CORPUS_READY.json'; marker.write_text('{}')
    ready = {'schema_version': 'fixed200k-ready-v3', 'files': {
        'validation_magicbrush_probe128': {'path': str(manifest), 'sha256': _sha256(manifest)},
        'train_200k': {'path': str(train)}}}
    monkeypatch.setattr('src.dimo.dev.verify_corpus_ready', lambda _: ready)
    result, identity = verified_dev128(manifest, marker)
    assert len(result) == identity['dev_rows'] == 128
    test = tmp_path / 'test.jsonl'; test.write_bytes(manifest.read_bytes())
    with pytest.raises(RuntimeError, match='NOT_TEST'):
        verified_dev128(test, marker)
    train.write_text(json.dumps(rows[0]) + '\n')
    with pytest.raises(RuntimeError, match='LEAKAGE'):
        verified_dev128(manifest, marker)


def test_comparison_checks_input_seed_hashes_and_fixed_grids(tmp_path):
    # Synthetic CPU-only manifests/metrics test, not a quality result.
    picture = tmp_path / 'unit-fixture.png'
    Image.new('RGB', (16, 16), 'gray').save(picture)
    digest = _sha256(picture)
    aggregate = {name: {'mean': .1} for name in METRICS}
    directories = []
    for label in LABELS:
        root = tmp_path / label; root.mkdir(); directories.append(root)
        manifest = root / 'predictions.jsonl'
        records = [dict(source=str(picture), target=str(picture), mask=str(picture), output=str(picture),
            output_sha256=digest, sample_key=f'unit-fixture-{index}', seed=0,
            aligned_input_sha256={key: digest for key in ('source', 'target', 'mask')}) for index in range(128)]
        manifest.write_text(''.join(json.dumps(record) + '\n' for record in records))
        metrics = root / 'metrics.json'
        metrics.write_text(json.dumps({'aggregate': {'dataset:magicbrush': aggregate}}))
        metadata = dict(label=label, dev_only=True, formal_teacher=False, dev_rows=128,
            dev_manifest_sha256='unit-fixture', corpus_ready_sha256='unit-fixture',
            teacher_hash='unit-fixture', dataset_name='magicbrush', config={'synthetic_test': True},
            latency_scope='unit-fixture', prediction_manifest_sha256=_sha256(manifest),
            metrics_sha256=_sha256(metrics), aggregate=aggregate)
        (root / 'diagnostic.json').write_text(json.dumps(metadata))
    result = compare(directories, tmp_path / 'comparison')
    assert result['quality_selection_performed'] is False
    assert len(list((tmp_path / 'comparison').glob('fixed-dev-*.jpg'))) == 6
    corrupted = json.loads((directories[-1] / 'diagnostic.json').read_text())
    corrupted['teacher_hash'] = 'different'
    (directories[-1] / 'diagnostic.json').write_text(json.dumps(corrupted))
    with pytest.raises(RuntimeError, match='IDENTITY_MISMATCH'):
        compare(directories, tmp_path / 'wrong-comparison')
