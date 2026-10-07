"""Read-only frozen-corpus checks, plus transactional current-code attestation."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shlex

import numpy as np
from PIL import Image

from .canonical import image_from_locator, expected_locator_hash
from .fixed_corpus import assert_frozen_corpus, verify_corpus_ready
from .fixed_dataset import CanonicalAlignedDataset, FixedCorpusDataset
from .formal_pipeline import sha256_file
from .formal_pipeline import verify_formal_asset_corpus, write_json_atomic
from .language import contains_han, translation_qa_flags

DATASETS = ('magicbrush', 'crispedit', 'scaleedit', 'interedit')


def publish_ready(marker, ledger, new_ready, new_assets, current, backup_directory):
    """Atomic file publication with byte-exact rollback, never touches data."""
    marker, ledger, backups = map(Path, (marker, ledger, backup_directory))
    try:
        write_json_atomic(marker, new_ready)
        write_json_atomic(ledger, new_assets)
        verify_formal_asset_corpus(ledger, marker, current)
    except Exception:
        import shutil
        for target, backup in [(marker, backups / 'CORPUS_READY.previous.json'),
                               (ledger, backups / 'formal_assets.previous.json')]:
            temporary = target.with_suffix(target.suffix + '.rollback')
            shutil.copyfile(backup, temporary)
            temporary.replace(target)
        raise


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def discover_unique(root, name):
    candidates = sorted(Path(root).rglob(name))
    if len(candidates) != 1:
        raise RuntimeError(f'REATTEST_INPUT_AMBIGUOUS_OR_MISSING: {name}: {candidates}')
    return candidates[0].resolve()


def frozen_identity(ready, *, expected_rows=200000):
    train = Path(ready['files']['train_200k']['path'])
    rows = read_rows(train)
    assert_frozen_corpus(rows, expected_rows)
    validation = {name: Path(record['path']) for name, record in ready['files'].items()
                  if name.startswith('validation_') and Path(record['path']).suffix == '.jsonl'}
    if not validation:
        raise RuntimeError('FROZEN_VALIDATION_MISSING')
    digest = hashlib.sha256()
    for row in rows:
        digest.update((row['sample_uid'] + '\n').encode())
    return {'train_sha256': sha256_file(train), 'rows': len(rows),
            'dataset_counts': dict(Counter(r['dataset_name'] for r in rows)),
            'ordered_sample_uid_sha256': digest.hexdigest(),
            'validation_sha256': {name: sha256_file(path) for name, path in validation.items()}}, rows, validation


def recorded_environment(path):
    """Read generated export records without executing arbitrary shell code."""
    values = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith('export '):
            key, value = line[7:].split('=', 1)
            words = shlex.split(value)
            if len(words) != 1 or '$' in words[0] or '`' in words[0]:
                raise RuntimeError('REATTTEST_ENV_RECORD_NOT_LITERAL')
            values[key] = words[0]
    required = [name.upper() + '_ROOT' for name in DATASETS] + ['MAGICBRUSH_DEV_ROOT']
    if any(key not in values or not Path(values[key]).is_dir() for key in required):
        raise RuntimeError('CANONICAL_ROOTS_NOT_RECORDED_OR_MISSING')
    return values


def compare_alpha(raw, actual):
    if raw.mode != 'RGBA':
        raise RuntimeError('MAGICBRUSH_MASK_IDENTITY_UNVERIFIED: raw mask must be RGBA')
    expected = 255 - np.asarray(raw.getchannel('A'), dtype=np.uint8)
    actual = np.asarray(actual.convert('L'), dtype=np.uint8)
    if expected.shape != actual.shape:
        raise RuntimeError('MAGICBRUSH_MASK_IDENTITY_UNVERIFIED: dimensions')
    binary = int(np.count_nonzero((expected > 0) != (actual > 0)))
    exact = int(np.count_nonzero(expected != actual))
    result = {'mask_mode': raw.mode, 'expected_nonzero_count': int((expected > 0).sum()),
              'actual_nonzero_count': int((actual > 0).sum()),
              'binary_disagreement': binary, 'exact_disagreement': exact,
              'status': 'PASS' if binary == exact == 0 else 'FAIL'}
    if result['status'] != 'PASS':
        raise RuntimeError(f'MAGICBRUSH_MASK_IDENTITY_UNVERIFIED: {result}')
    return result


def alpha_probe(rows, root, raw_asset, ready, count=32):
    chosen = sorted((r for r in rows if r['dataset_name'] == 'magicbrush'),
                    key=lambda r: hashlib.sha256(('alpha-probe\0' + r['sample_uid']).encode()).digest())[:count]
    if len(chosen) != count:
        raise RuntimeError('MagicBrush alpha probe requires 32 frozen samples')
    raw_files = sorted(Path(raw_asset['local_path']).glob('data/train-*.parquet'))
    if not raw_files:
        evidence_record = ready['files'].get('magicbrush_alpha_export')
        if not evidence_record:
            raise RuntimeError('MAGICBRUSH_MASK_IDENTITY_UNVERIFIED: no pinned raw or hash-bound export evidence')
        evidence = json.loads(Path(evidence_record['path']).read_text())
        if (evidence.get('status') != 'PASS' or evidence.get('export_algorithm') != '255-alpha'
                or evidence.get('raw_revision') != raw_asset.get('resolved_revision')):
            raise RuntimeError('MAGICBRUSH_MASK_IDENTITY_UNVERIFIED: export provenance')
        records = {r['sample_uid']: r for r in evidence['records']}
        results = []
        for row in chosen:
            record = records.get(row['sample_uid'], {})
            if record.get('region_sha256') != row['region_sha256'] or expected_locator_hash(row['region_locator'], root) != row['region_sha256']:
                raise RuntimeError('MAGICBRUSH_MASK_IDENTITY_UNVERIFIED: export mask identity')
            if record.get('status') != 'PASS' or record.get('binary_disagreement') != 0 or record.get('exact_disagreement') != 0:
                raise RuntimeError('MAGICBRUSH_MASK_IDENTITY_UNVERIFIED: export alpha result')
            results.append(record)
        return {'status': 'PASS', 'mode': 'PROVENANCE_REUSE', 'records': results,
                'evidence_sha256': evidence_record['sha256']}
    exported = {r['source']: r for r in read_rows(Path(root) / 'manifest.jsonl')}
    requested = {}
    for row in chosen:
        original = exported.get(row['source_locator'].get('relative_path'))
        if original is None or original['instruction'].strip() != row['instruction_original'].strip():
            raise RuntimeError('MAGICBRUSH_MASK_IDENTITY_UNVERIFIED: raw/export mapping')
        requested[(str(original['img_id']), int(original['turn_index']))] = row
    import pyarrow.parquet as pq
    results = []
    found = set()
    for path in raw_files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=['img_id', 'turn_index', 'mask_img']):
            for item in batch.to_pylist():
                key = (str(item['img_id']), int(item['turn_index']))
                if key not in requested:
                    continue
                if key in found:
                    raise RuntimeError('MAGICBRUSH_MASK_IDENTITY_UNVERIFIED: ambiguous raw identity')
                found.add(key); row = requested[key]
                value = item['mask_img']
                if not isinstance(value, dict) or not value.get('bytes'):
                    raise RuntimeError('MAGICBRUSH_MASK_IDENTITY_UNVERIFIED: raw parquet image bytes missing')
                if expected_locator_hash(row['region_locator'], root) != row['region_sha256']:
                    raise RuntimeError('MAGICBRUSH_MASK_IDENTITY_UNVERIFIED: canonical mask SHA')
                with Image.open(io.BytesIO(value['bytes'])) as raw:
                    result = compare_alpha(raw, image_from_locator(row['region_locator'], root))
                results.append({'sample_uid': row['sample_uid'], 'region_sha256': row['region_sha256'],
                                'raw_identity': {'file': str(path), 'img_id': key[0], 'turn_index': key[1],
                                                 'revision': raw_asset['resolved_revision']}, **result})
    if len(results) != count:
        raise RuntimeError('MAGICBRUSH_MASK_IDENTITY_UNVERIFIED: incomplete raw coverage')
    return {'status': 'PASS', 'mode': 'RAW_COMPARE', 'records': results,
            'raw_file_hashes': {str(path): sha256_file(path) for path in raw_files}}


def verify_current_loader(train, rows, validation, roots, dev_root):
    """Use the actual production loaders/geometry, never a parallel checker."""
    dataset = FixedCorpusDataset(train, roots, resolution=1024, base_seed=42, verify_hashes=True)
    source_ids, group_ids = set(), set()
    for path in validation.values():
        for row in read_rows(path):
            source_ids.add(row['source_sha256'])
            group_ids.add((row['dataset_name'], row['group_id']))
    for index, row in enumerate(rows):
        if row['source_sha256'] in source_ids or (row['dataset_name'], row['group_id']) in group_ids:
            raise RuntimeError('FROZEN_TRAIN_DEV_LEAKAGE')
        if contains_han(row['instruction_en']) or translation_qa_flags(row['instruction_original'], row['instruction_en']):
            raise RuntimeError('FROZEN_TRANSLATION_CONTRACT_FAILED')
        dataset[index]
        if (index + 1) % 1000 == 0:
            print(f'CURRENT_CODE_LOADER_VERIFIED={index + 1}', flush=True)
    for path in validation.values():
        grouped = {}
        for row in read_rows(path):
            grouped.setdefault(row['dataset_name'], []).append(row)
        for name, records in grouped.items():
            data = CanonicalAlignedDataset(records, dev_root if name == 'magicbrush' else roots[name],
                                           resolution=1024, base_seed=42, verify_hashes=True)
            for index in range(len(data)):
                data[index]
    return {'status': 'PASS', 'rows': len(rows), 'geometry': 'PASS', 'translation': 'PASS',
            'train_dev_leakage': 'PASS', 'implementation': 'FixedCorpusDataset/CanonicalAlignedDataset'}
