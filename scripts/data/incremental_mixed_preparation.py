#!/usr/bin/env python3
"""Screen verified shards and pretranslate a bounded cohort during mirror downloads.

These are candidate caches, never training admission. Final leakage checks,
selection, translation QA, loader/VQ checks and CUDA acceptance remain mandatory.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.explicit_region.contracts import sha256_file
from src.explicit_region.formal_pipeline import write_json_atomic
from src.explicit_region.interedit import iter_metadata
from src.explicit_region.language import JsonlTranslationCache, contains_han, translation_cache_key


def planned_parquets(plan, name):
    return sorted((entry for entry in plan['datasets'][name]['files']
                   if entry['rfilename'].endswith('.parquet')), key=lambda e: e['rfilename'])


def verify_payload(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        raise RuntimeError(f'incomplete planned payload: {path}')
    lfs = entry.get('lfs', {}).get('sha256')
    digest = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        digest.update(f'blob {entry["size"]}\0'.encode())
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != (lfs or entry['blobId']):
        raise RuntimeError(f'planned payload hash changed: {path}')


def contract(name):
    paths = ['src/explicit_region/canonical.py', 'src/explicit_region/dataset.py',
             'src/explicit_region/geometry.py', 'src/explicit_region/formal_pipeline.py',
             'configs/data/edit_type_mapping.yaml', 'uv.lock']
    paths += (['scripts/data/build_interedit_canonical_pool.py', 'src/explicit_region/interedit.py']
              if name == 'interedit' else ['scripts/data/build_schema_mapped_canonical.py',
                                          f'configs/data/schema/{name}.yaml'])
    return {path: sha256_file(ROOT / path) for path in paths}


def cache_spec(root, plan, name, entries):
    identity = {'schema': 'incremental-canonical-cache-v1', 'dataset': name,
                'revision': plan['datasets'][name]['revision'], 'files': entries,
                'contract': contract(name)}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = root / 'data/incremental' / name / f'{key}.jsonl'
    return identity, output


def cached_report(identity, output):
    marker = output.with_suffix('.complete.json')
    if not marker.exists():
        return None
    record = json.loads(marker.read_text())
    if record['identity'] != identity:
        raise RuntimeError('incremental cache identity changed')
    for entry in record['outputs']:
        if sha256_file(entry['path']) != entry['sha256']:
            raise RuntimeError('incremental cache output changed')
    return json.loads(output.with_suffix('.report.json').read_text())


def build_one(root, plan, name, entries):
    """Process workers reuse the unchanged canonical builder imports."""
    os.environ['OMP_NUM_THREADS'] = '2'
    import torch
    torch.set_num_threads(2)
    root = Path(root); dataset = Path(plan['datasets'][name]['root'])
    identity, output = cache_spec(root, plan, name, entries)
    for entry in entries:
        verify_payload(dataset / entry['rfilename'], entry)
    if cached_report(identity, output) is not None:
        return str(output)
    if name == 'interedit':
        from scripts.data.build_interedit_canonical_pool import main
        metadata = [dataset / e['rfilename'] for e in entries if e['rfilename'].endswith('.jsonl.gz')]
        argv = ['builder', *map(str, metadata), '--interedit-root', str(dataset),
                '--revision', plan['datasets'][name]['revision'], '--output', str(output)]
    else:
        from scripts.data.build_schema_mapped_canonical import main
        argv = ['builder', '--root', str(dataset), '--mapping', str(ROOT / f'configs/data/schema/{name}.yaml'),
                '--edit-type-mapping', str(ROOT / 'configs/data/edit_type_mapping.yaml'), '--output', str(output)]
        for entry in entries:
            argv += ['--file', entry['rfilename']]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
        sys.argv = argv
        main()
    outputs = [output, output.with_suffix('.report.json'), output.with_suffix('.rejections.jsonl')]
    write_json_atomic(output.with_suffix('.complete.json'), {'identity': identity,
        'outputs': [{'path': str(path), 'sha256': sha256_file(path)} for path in outputs]})
    return str(output)


def merge_cached_pools(root, plan):
    """Only preregistered files; stable file/row order equals the original builder."""
    pools = root / 'data/pools'; pools.mkdir(parents=True, exist_ok=True)
    totals = {}
    for name in ('crispedit', 'scaleedit', 'interedit'):
        batches = ([plan['datasets'][name]['files']] if name == 'interedit'
                   else [[entry] for entry in planned_parquets(plan, name)])
        rows, rejections = [], []
        for entries in batches:
            identity, output = cache_spec(root, plan, name, entries)
            if cached_report(identity, output) is None:
                raise RuntimeError(f'planned candidate cache missing: {name}')
            for entry in entries:
                verify_payload(Path(plan['datasets'][name]['root']) / entry['rfilename'], entry)
            rows.extend(json.loads(line) for line in output.open() if line.strip())
            rejections.extend(json.loads(line) for line in output.with_suffix('.rejections.jsonl').open() if line.strip())
        destination = pools / f'{name}.raw.jsonl'
        destination.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows))
        rejection_path = destination.with_suffix('.rejections.jsonl')
        rejection_path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rejections))
        totals[name] = {'accepted_rows': len(rows), 'rejected_rows': len(rejections),
                        'sha256': sha256_file(destination)}
    write_json_atomic(pools / 'INCREMENTAL_POOL_EVIDENCE.json', totals)
    return totals


def early_translation_requests(plan):
    """Bounded better_data metadata only; still provisional before pixel screening."""
    dataset = plan['datasets']['interedit']; rows = []
    for entry in sorted(dataset['files'], key=lambda e: e['rfilename']):
        if not entry['rfilename'].endswith('.jsonl.gz'):
            continue
        path = Path(dataset['root']) / entry['rfilename']; verify_payload(path, entry)
        for item in iter_metadata(path, only_better_data=True):
            if contains_han(item.instruction_original):
                rows.append({'instruction_original': item.instruction_original,
                             'sample_id': item.sample_id, 'scope': 'provisional_candidate_and_reserve'})
    return rows


def downloader_alive(root):
    pid = int((root / 'DOWNLOAD_PID').read_text()); proc = Path(f'/proc/{pid}')
    return (proc.exists() and not proc.joinpath('stat').read_text().split(') ')[1].startswith('Z')
            and b'download_mixed30k_mirror.py' in proc.joinpath('cmdline').read_bytes())


def reuse_complete_preparation(root, plan, plan_hash):
    """Verify completed candidate work once, without rescheduling hundreds of jobs."""
    marker = root / 'data/INCREMENTAL_COMPLETE.json'
    if not marker.exists():
        return False
    record = json.loads(marker.read_text())
    if record.get('status') != 'COMPLETE' or record.get('download_plan_sha256') != plan_hash:
        raise RuntimeError('completed preparation cohort changed')
    model = plan['datasets']['nllb']
    for entry in model['files']: verify_payload(Path(model['root']) / entry['rfilename'], entry)
    cache = JsonlTranslationCache(root / 'data/translation_cache.jsonl')
    decoding = {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 128}
    for row in early_translation_requests(plan):
        key = translation_cache_key(row['instruction_original'], 'nllb', model['revision'],
                                    'zho_Hans', 'eng_Latn', decoding)
        if cache.get(key) is None:
            return False
    totals = merge_cached_pools(root, plan)
    if totals != record['pools']:
        raise RuntimeError('completed preparation pool content changed')
    write_json_atomic(root / 'PREPARATION_STATUS.json', {'status': 'CANDIDATE_PREPARATION_COMPLETE',
        'accepted_candidates': {name: row['accepted_rows'] for name, row in totals.items()},
        'download_plan_sha256': plan_hash, 'translation': 'COMPLETE',
        'verified_completed_cache_reused': True, 'final_training_admission': 'PENDING', 'test_read': False})
    return True


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(); root = args.experiment_root.resolve()
    plan_path = root / 'DOWNLOAD_PLAN.json'; plan_hash = sha256_file(plan_path)
    plan = json.loads(plan_path.read_text()); submitted = set(); done = {}; running = {}
    if reuse_complete_preparation(root, plan, plan_hash):
        return
    jobs = [(name, [entry]) for name in ('crispedit', 'scaleedit') for entry in planned_parquets(plan, name)]
    jobs.append(('interedit', plan['datasets']['interedit']['files']))
    translation = None; translation_log = None; translated = False
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2',
                                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for key in list(env):
        if key.lower().endswith('_proxy'): env.pop(key)
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('incremental preparation interrupted')))
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'))
    try:
        while len(done) < len(jobs) or not translated:
            if sha256_file(plan_path) != plan_hash:
                raise RuntimeError('download cohort changed during incremental screening')
            for name, entries in jobs:
                key = (name, entries[0]['rfilename'])
                if key in submitted or len(running) >= args.workers * 2:
                    continue
                dataset = Path(plan['datasets'][name]['root'])
                if all((dataset / entry['rfilename']).is_file() for entry in entries):
                    future = pool.submit(build_one, str(root), plan, name, entries)
                    submitted.add(key); running[future] = key
            for future in list(running):
                if future.done():
                    key = running.pop(future); output = Path(future.result())
                    done[key] = json.loads(output.with_suffix('.report.json').read_text())
            if translation is None:
                model = plan['datasets']['nllb']
                if all((Path(model['root']) / entry['rfilename']).is_file() for entry in model['files']):
                    for entry in model['files']: verify_payload(Path(model['root']) / entry['rfilename'], entry)
                    required = root / 'data/early_translation_required.jsonl'; required.parent.mkdir(parents=True, exist_ok=True)
                    rows = early_translation_requests(plan)
                    required.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
                    translation_log = (root / 'coordinator_logs/early_translation.log').open('a')
                    translation = subprocess.Popen([sys.executable, ROOT / 'scripts/data/translate_canonical_candidates.py',
                        '--required', required, '--cache', root / 'data/translation_cache.jsonl',
                        '--qa-output', root / 'data/early_translation_qa.jsonl', '--backend', 'nllb',
                        '--model-path', model['root'], '--revision', model['revision'], '--formal'],
                        env=env, cwd=ROOT, stdout=translation_log, stderr=subprocess.STDOUT)
            if translation is not None and translation.poll() is not None:
                if translation.returncode:
                    raise RuntimeError('early NLLB translation failed; inspect early_translation.log')
                translated = True
            totals = {name: sum(row['accepted_rows'] for key, row in done.items() if key[0] == name)
                      for name in ('crispedit', 'scaleedit', 'interedit')}
            state = {'status': 'PREPARING_CANDIDATES', 'download_plan_sha256': plan_hash,
                     'screened_jobs': len(done), 'planned_jobs': len(jobs), 'accepted_candidates': totals,
                     'translation': 'COMPLETE' if translated else 'RUNNING' if translation else 'WAITING_FOR_MODEL',
                     'final_training_admission': 'PENDING', 'test_read': False}
            write_json_atomic(root / 'PREPARATION_STATUS.json', state)
            download_complete = json.loads((root / 'DOWNLOAD_STATUS.json').read_text()).get('status') == 'COMPLETE'
            if not download_complete and not downloader_alive(root):
                raise RuntimeError('mirror downloader stopped before cohort completion')
            if len(done) < len(jobs) or not translated: time.sleep(10)
        totals = merge_cached_pools(root, plan)
        write_json_atomic(root / 'data/INCREMENTAL_COMPLETE.json', {'status': 'COMPLETE',
                          'download_plan_sha256': plan_hash, 'pools': totals,
                          'early_translation_scope': 'bounded_provisional_interedit_candidates_and_reserves'})
        write_json_atomic(root / 'PREPARATION_STATUS.json', dict(state, status='CANDIDATE_PREPARATION_COMPLETE'))
    finally:
        if translation is not None and translation.poll() is None:
            translation.terminate(); translation.wait(timeout=30)
        if translation_log is not None: translation_log.close()
        pool.shutdown(wait=True, cancel_futures=True)


if __name__ == '__main__':
    main()
