#!/usr/bin/env python3
"""Read-only upstream integrity, native alignment and train/DEV lineage audit."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dataset import _align_to_mask_coordinates
from src.explicit_region.dense_checkpoint import write_json


def session_subset(records):
    groups = defaultdict(list)
    for row in records:
        if row['split']=='train': groups[row['group_id']].append(row)
    excluded = {group for group, rows in groups.items() if any(not r['valid'] for r in rows)}
    removed = sum(len(groups[g]) for g in excluded)
    retained = sum(map(len, groups.values())) - removed
    target = min(8800, retained // 16 * 16)
    singles = sorted((g for g, rows in groups.items() if len(rows)==1 and g not in excluded),
                     key=lambda g: hashlib.sha256(('magicbrush-4a800-exclude-v1:'+groups[g][0]['sample_key']).encode()).hexdigest())
    extra = retained-target
    if extra > len(singles) or target <= 0: raise RuntimeError('NO_SESSION_CONSISTENT_SUBSET')
    excluded.update(singles[:extra])
    kept = [r for r in records if r['split']=='train' and r['group_id'] not in excluded]
    dropped = [r for r in records if r['split']=='train' and r['group_id'] in excluded]
    return {'kept_rows': len(kept), 'excluded_rows': len(dropped),
            'excluded_groups': sorted(excluded), 'kept_sample_keys': [r['sample_key'] for r in kept],
            'excluded_sample_keys': [r['sample_key'] for r in dropped],
            'steps_per_sft_epoch': len(kept)//16, 'steps_per_dimo_epoch': len(kept)//4,
            'session_consistent': True, 'padding': False, 'replacement': False,
            'contains_invalid_rows': any(not r['valid'] for r in kept)}


def audit(root, upstream, output):
    if output.exists(): raise FileExistsError(output)
    upstream = json.loads(upstream.read_text())
    if upstream['sha'] != '1d8d4629150d18ca50afab66391866f2085be989':
        raise RuntimeError('MAGICBRUSH_REVISION_MISMATCH')
    records, failures, files = [], [], []
    for split, expected in (('train', 8807), ('dev', 528)):
        count = 0
        for path in sorted((root/'data').glob(f'{split}-*.parquet')):
            meta = upstream['files']['data/'+path.name]
            digest = sha256_file(path)
            if meta['size'] != path.stat().st_size or digest != meta['lfs']['sha256']:
                raise RuntimeError(f'UPSTREAM_PARQUET_HASH_MISMATCH: {path.name}')
            files.append({'name': path.name, 'sha256': digest, 'bytes': path.stat().st_size})
            parquet = pq.ParquetFile(path)
            offset = 0
            for batch in parquet.iter_batches(batch_size=8):
                for row in batch.to_pylist():
                    count += 1
                    key = f"{row['img_id']}_turn{row['turn_index']}"
                    identity = {'split': split, 'sample_key': key, 'group_id': str(row['img_id']),
                                'turn_index': int(row['turn_index']), 'file': path.name, 'row': offset,
                                'instruction': row['instruction']}
                    offset += 1
                    identity.update({k+'_sha256': hashlib.sha256(row[col]['bytes']).hexdigest()
                                     for k, col in (('source','source_img'), ('target','target_img'), ('native_mask','mask_img'))})
                    try:
                        with Image.open(io.BytesIO(row['source_img']['bytes'])) as source, \
                             Image.open(io.BytesIO(row['target_img']['bytes'])) as target, \
                             Image.open(io.BytesIO(row['mask_img']['bytes'])) as native:
                            if native.mode != 'RGBA': raise ValueError('OFFICIAL_MASK_NOT_RGBA')
                            mask = Image.fromarray(np.where(np.array(native.getchannel('A')) < 255, 255, 0).astype('uint8'))
                            if not np.any(np.asarray(mask)): raise ValueError('EMPTY_OFFICIAL_ROI')
                            aligned = _align_to_mask_coordinates(source, target, mask, key)
                            identity['alignment'] = aligned[2]
                            identity['image_sizes'] = [list(im.size) for im in (source,target,native)]
                    except (ValueError, OSError) as exc:
                        failures.append(dict(identity, reason=str(exc)))
                        identity['valid'] = False
                    else: identity['valid'] = True
                    records.append(identity)
            print(json.dumps({'split': split, 'count': count, 'file': path.name, 'invalid': len(failures)}), flush=True)
        if count != expected: raise RuntimeError(f'MAGICBRUSH_COUNT_MISMATCH: {split} {count}')
    groups = defaultdict(set)
    contents = defaultdict(set)
    for row in records:
        groups[row['group_id']].add(row['split'])
        for key in ('source_sha256', 'target_sha256'):
            contents[row[key]].add(row['split'])
    leaking_groups = [g for g,splits in groups.items() if len(splits)>1]
    leaking_content = [h for h,splits in contents.items() if len(splits)>1]
    by_group = defaultdict(list)
    for r in records:
        if r['split']=='train': by_group[r['group_id']].append(r)
    single = sorted((rs[0] for rs in by_group.values() if len(rs)==1 and rs[0]['valid']),
                    key=lambda r: hashlib.sha256(('magicbrush-4a800-exclude-v1:'+r['sample_key']).encode()).hexdigest())
    invalid_train = [r for r in failures if r['split']=='train']
    report = {'schema': 'magicbrush-real-input-audit-v1', 'dataset_revision': upstream['sha'],
              'status': 'STOP_DATA_GATE' if leaking_groups or leaking_content else (
                  'PASS_WITH_SESSION_GEOMETRY_EXCLUSIONS' if invalid_train else 'PASS'),
              'counts': {'train': 8807, 'dev': 528}, 'parquets': files,
              'invalid_train': invalid_train, 'invalid_dev': [r for r in failures if r['split']=='dev'],
              'train_dev_group_leakage': leaking_groups, 'train_dev_exact_content_leakage': leaking_content,
              'single_turn_sessions_available': len(single), 'proposed_seven_single_turn_exclusions': single[:7],
              'records': records, 'test_read': False, 'proposed_session_consistent_subset': session_subset(records)}
    write_json(output, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','parquets','invalid_train')}), flush=True)
    return report


def main():
    p=argparse.ArgumentParser(); p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--upstream-identity',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); report=audit(a.snapshot,a.upstream_identity,a.output)
    if report['status']=='STOP_DATA_GATE': raise RuntimeError('MAGICBRUSH_DATA_GATE_FAILED')


if __name__=='__main__': main()
