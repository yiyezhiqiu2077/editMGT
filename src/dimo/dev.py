"""Frozen DEV128 identity, never TEST or a newly sampled validation set."""
import json
from pathlib import Path

from .contracts import _sha256
from src.explicit_region.canonical import validate_record
from src.explicit_region.fixed_corpus import verify_corpus_ready


def verified_dev128(manifest, corpus_ready, *, dataset='magicbrush'):
    ready = verify_corpus_ready(corpus_ready)
    path = Path(manifest).resolve()
    key = ('validation_magicbrush_probe128' if dataset == 'magicbrush'
           else f'validation_{dataset}_aux128')
    record = ready.get('files', {}).get(key, {})
    if (ready.get('schema_version') != 'fixed200k-ready-v3'
            or Path(record.get('path', '/missing')).resolve() != path
            or record.get('sha256') != _sha256(path)):
        raise RuntimeError('REQUIRES_EXISTING_FROZEN_DEV128_NOT_TEST')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if (len(rows) != 128 or len({r['sample_uid'] for r in rows}) != 128
            or any(r['dataset_name'] != dataset for r in rows)):
        raise RuntimeError('DEV128_COUNT_OR_DATASET_MISMATCH')
    for row in rows:
        validate_record(row)
    train = Path(ready['files']['train_200k']['path'])
    train_rows = [json.loads(line) for line in train.read_text().splitlines() if line.strip()]
    if ({r['source_sha256'] for r in rows} & {r['source_sha256'] for r in train_rows}
            or {r['group_id'] for r in rows} & {r['group_id'] for r in train_rows if r['dataset_name'] == dataset}):
        raise RuntimeError('DIMO_TRAIN_DEV_LEAKAGE')
    return rows, {'dev_manifest_sha256': _sha256(path), 'corpus_ready_sha256': _sha256(Path(corpus_ready)),
                  'train_manifest_sha256': _sha256(train),
                  'dataset_name': dataset, 'dev_rows': 128}
