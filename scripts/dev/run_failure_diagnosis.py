"""Dev-only immutable-checkpoint CE/CFG/timestep diagnostics; never trains."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ('magicbrush', 'crispedit', 'scaleedit', 'interedit')
MANIFESTS = ('magicbrush_official_dev', 'crispedit_aux8', 'scaleedit_aux8', 'interedit_aux8')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def freeze(mini, out):
    folder = out / 'manifests'
    folder.mkdir(parents=True, exist_ok=True)
    annotations = {}
    fingerprints = {}
    for dataset, name in zip(DATASETS, MANIFESTS):
        source = mini / 'corpus/validation' / f'{name}.jsonl'
        rows = [json.loads(l) for l in source.read_text().splitlines() if l]
        indices = [3, 8, 4, 25, 0, 2] if dataset == 'magicbrush' else list(range(6))
        chosen = [rows[i] for i in indices]
        if dataset == 'magicbrush':
            # Sidecar-only instruction interpretation; immutable canonical labels
            # remain 'other'. Compound replacement edits are not counted as remove.
            tags = ['local_attribute', 'add', 'replace', 'add', 'remove', 'replace',
                    'add', 'replace', 'add', 'texture', 'other', 'add',
                    'local_attribute', 'replace', 'replace', 'local_attribute',
                    'add', 'replace', 'other', 'add', 'local_attribute', 'add',
                    'local_attribute', 'add', 'replace', 'remove', 'remove',
                    'add', 'replace', 'other', 'remove', 'replace']
            if len(rows) != len(tags):
                raise RuntimeError('Manual sidecar applies only to frozen DEV32 MagicBrush')
            for i, tag in enumerate(tags):
                r = rows[i]
                annotations[r['sample_uid']] = {
                    'diagnostic_edit_type': tag, 'instruction_en': r['instruction_en'],
                    'basis': 'manual instruction interpretation, not canonical ground truth'}
        path = folder / f'{dataset}_dev24.jsonl'
        content = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in chosen)
        if path.exists() and path.read_text() != content:
            raise RuntimeError(f'Frozen manifest changed: {path}')
        if not path.exists():
            path.write_text(content)
        fingerprints[dataset] = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                 'selected_indices': indices,
                                 'selected_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'uids': [r['sample_uid'] for r in chosen]}
    dump(folder / 'DEV24_READY.json', {'dev_only': True, 'total': 24, 'datasets': fingerprints})
    dump(folder / 'diagnostic_annotations.json', annotations)


def config(out, cfg):
    path = out / 'configs' / f'cfg-{cfg:g}.json'
    dump(path, {'resolution': 1024, 'provided_region_hard_lock': True,
                'steps': 12, 'guidance_scale': cfg, 'reference_strength': 1.0,
                'generation_seeds': [0], 'bootstrap': {'seed': 42, 'resamples': 1000},
                'selection': {'status': 'DIAGNOSTIC_ONLY', 'tau_edit': None, 'tau_noop': None}})
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mini-root', required=True)
    p.add_argument('--model-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--stage', choices=['freeze', 'cfg24', 'cfg56', 'timestep'], required=True)
    p.add_argument('--cfg', type=float, default=5)
    p.add_argument('--gpus', nargs='+', type=int, default=list(range(8)))
    a = p.parse_args()
    mini, out = Path(a.mini_root), Path(a.output_root)
    freeze(mini, out)
    if a.stage == 'freeze':
        print('DEV24_FROZEN=YES'); return
    cells = ([(300, 'roi_relative', cfg, 24) for cfg in [1., 3., 5., 7.5, 10.]]
             if a.stage == 'cfg24' else
             [(300, 'roi_relative', cfg, 56) for cfg in sorted({a.cfg, 10.})]
             if a.stage == 'cfg56' else
             [(step, mode, a.cfg, 24) for step in [0, 300]
              for mode in ['official_upstream_timestep', 'roi_relative']])
    jobs = []
    for step, mode, cfg, count in cells:
        name = f'dev{count}/step-{step}_{mode}_cfg-{cfg:g}'
        for dataset, manifest_name in zip(DATASETS, MANIFESTS):
            manifest = (out / 'manifests' / f'{dataset}_dev24.jsonl' if count == 24 else
                        mini / 'corpus/validation' / f'{manifest_name}.jsonl')
            jobs.append((step, mode, cfg, name, dataset, manifest))
        config(out, cfg)

    def worker(i):
        env = os.environ | {'CUDA_VISIBLE_DEVICES': str(a.gpus[i]),
                            'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
        for step, mode, cfg, name, dataset, manifest in jobs[i::len(a.gpus)]:
            dest = out / name / dataset
            dest.mkdir(parents=True, exist_ok=True)
            expected = [json.loads(l)['sample_uid'] for l in manifest.read_text().splitlines() if l]
            pred = dest / 'predictions.jsonl'
            if pred.exists():
                actual = [json.loads(l)['sample_key'] for l in pred.read_text().splitlines() if l]
                if actual != expected:
                    raise RuntimeError(f'Incomplete/mismatched existing predictions {pred}')
                continue
            asset = mini / 'assets' / dataset
            if dataset == 'magicbrush': asset /= 'dev'
            cmd = [sys.executable, str(ROOT / 'scripts/eval/generate_evaluation.py'),
                   '--dataset', dataset, '--manifest', str(manifest), '--canonical-root', str(asset),
                   '--model-root', a.model_root, '--checkpoint',
                   str(mini / 'runs/e3-tiny64-overfit-300' / f'checkpoint-{step}'),
                   '--config', str(out / 'configs' / f'cfg-{cfg:g}.json'), '--timestep-mode', mode,
                   '--output-dir', str(dest)]
            with (dest / 'generate.log').open('w') as log:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f'GENERATED {name} {dataset}', flush=True)

    with ThreadPoolExecutor(max_workers=len(a.gpus)) as executor:
        list(executor.map(worker, range(len(a.gpus))))
    for step, mode, cfg, count in cells:
        name = f'dev{count}/step-{step}_{mode}_cfg-{cfg:g}'
        dest = out / name
        if (dest / 'metrics.json').exists(): continue
        rows = []
        for dataset in DATASETS:
            rows.extend(json.loads(l) for l in (dest / dataset / 'predictions.jsonl').read_text().splitlines() if l)
        (dest / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        with (dest / 'evaluate.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT / 'scripts/eval/formal_eval.py'),
                            '--experiment', 'e3-failure-diagnosis', '--checkpoint', str(step),
                            '--config', str(config(out, cfg)), '--predictions-manifest',
                            str(dest / 'predictions.jsonl'), '--output', str(dest / 'metrics.json')],
                           cwd=ROOT, env=os.environ | {'CUDA_VISIBLE_DEVICES': str(a.gpus[0]),
                                                      'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'METRICS {name}', flush=True)


if __name__ == '__main__':
    main()
