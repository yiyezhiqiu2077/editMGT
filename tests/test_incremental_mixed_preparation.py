import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import pytest
import yaml

from scripts.data.build_schema_mapped_canonical import iter_rows, main as build_full
from scripts.data.incremental_mixed_preparation import (
    build_one, cache_spec, cached_report, early_translation_requests,
    planned_parquets, verify_payload,
)
from src.explicit_region.config import _load_unexpanded


def entry(path, root):
    return {'rfilename': path.relative_to(root).as_posix(), 'size': path.stat().st_size,
            'lfs': {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}


def image(color):
    buffer = io.BytesIO(); Image.new('RGB', (16, 16), color).save(buffer, format='PNG')
    return buffer.getvalue()


def parquet(path, color):
    mapping = yaml.safe_load(Path('configs/data/schema/crispedit.yaml').read_text())
    row = {key: '' for key in mapping['required_columns']}
    row.update(mapping['eligibility'])
    row.update(source_shard='s', row_idx=0, type='color', instruction='Make it blue',
               input_img={'bytes': image(color), 'path': None},
               output_img={'bytes': image('blue'), 'path': None}, mask__mask_png=image('white'))
    path.parent.mkdir(parents=True, exist_ok=True); pq.write_table(pa.Table.from_pylist([row]), path)
    return mapping


def test_mirror_final_total_and_bounded_translation_budget():
    c = _load_unexpanded(Path('configs/data/mixed20k_total.yaml'))
    assert c['train_total'] == 20000
    assert c['dataset_policy']['magicbrush']['fixed_count'] == 8784
    assert c['mirror']['interedit_asset_shards'] == 1
    assert c['mirror']['initial_candidate_rows'] == {'crispedit': 6000, 'scaleedit': 9000}
    assert c['translation']['early_translation_scope'] == 'bounded_provisional_interedit_candidates_and_reserves'


def test_only_planned_complete_files_and_safe_schema_paths(tmp_path):
    path = tmp_path / 'shards/a.parquet'; mapping = parquet(path, 'red')
    other = tmp_path / 'shards/b.parquet'; parquet(other, 'green')
    (tmp_path / 'shards/partial.parquet.partial').write_bytes(b'partial')
    plan = {'datasets': {'crispedit': {'files': [entry(path, tmp_path)]}}}
    assert [e['rfilename'] for e in planned_parquets(plan, 'crispedit')] == ['shards/a.parquet']
    assert [row[0] for row in iter_rows(tmp_path, mapping, ['shards/a.parquet'])] == ['shards/a.parquet']
    for unsafe in ['../escape.parquet', 'shards/partial.parquet', str(path.resolve())]:
        with pytest.raises(RuntimeError, match='outside schema glob'):
            list(iter_rows(tmp_path, mapping, [unsafe]))


def test_incremental_matches_full_builder_and_detects_cache_tampering(tmp_path, monkeypatch):
    dataset = tmp_path / 'datasets/crispedit'
    files = [dataset / 'shards/a.parquet', dataset / 'shards/b.parquet']
    for path, color in zip(files, ['red', 'green']): parquet(path, color)
    plan = {'datasets': {'crispedit': {'root': str(dataset), 'revision': 'a' * 40,
                                     'files': [entry(path, dataset) for path in files]}}}
    monkeypatch.setattr(sys, 'argv', ['builder'])
    outputs = [Path(build_one(str(tmp_path), plan, 'crispedit', [e]))
               for e in planned_parquets(plan, 'crispedit')]
    full = tmp_path / 'full.jsonl'
    monkeypatch.setattr(sys, 'argv', ['builder', '--root', str(dataset), '--mapping',
        'configs/data/schema/crispedit.yaml', '--edit-type-mapping', 'configs/data/edit_type_mapping.yaml',
        '--output', str(full)])
    build_full()
    assert ''.join(path.read_text() for path in outputs) == full.read_text()
    identity, output = cache_spec(tmp_path, plan, 'crispedit', [plan['datasets']['crispedit']['files'][0]])
    assert cached_report(identity, output)['accepted_rows'] == 1
    output.write_text('{}\n')
    with pytest.raises(RuntimeError, match='output changed'): cached_report(identity, output)
    mutated = json.loads(json.dumps(plan)); mutated['datasets']['crispedit']['files'][0]['lfs']['sha256'] = 'f' * 64
    assert cache_spec(tmp_path, mutated, 'crispedit', [mutated['datasets']['crispedit']['files'][0]])[1] != output


def test_payload_identity_detects_same_size_corruption(tmp_path):
    path = tmp_path / 'file'; path.write_bytes(b'valid')
    e = entry(path, tmp_path); verify_payload(path, e)
    path.write_bytes(b'wrong')
    with pytest.raises(RuntimeError, match='hash changed'): verify_payload(path, e)


def test_early_translation_is_bounded_and_does_not_claim_pixel_admission(tmp_path):
    metadata = tmp_path / 'metadata/first.jsonl.gz'; metadata.parent.mkdir()
    common = {'sample_id': 1, 'source_id': 2, 'edit_type': 'Add', 'better_data': True,
              'source_archive': 's.tar', 'source_file': 's.png', 'asset_archive': 'a.tar',
              'target_file': 't.png', 'mask_file': 'm.png'}
    with gzip.open(metadata, 'wt') as handle:
        for row in [dict(common, instruction='添加猫'), dict(common, instruction='坏例子', better_data=False),
                    dict(common, instruction='Add a dog')]: handle.write(json.dumps(row) + '\n')
    unplanned = tmp_path / 'metadata/other.jsonl.gz'; unplanned.write_bytes(b'ignored')
    plan = {'datasets': {'interedit': {'root': str(tmp_path), 'files': [entry(metadata, tmp_path)]}}}
    rows = early_translation_requests(plan)
    assert len(rows) == 1 and rows[0]['instruction_original'] == '添加猫'
    assert rows[0]['scope'] == 'provisional_candidate_and_reserve'
