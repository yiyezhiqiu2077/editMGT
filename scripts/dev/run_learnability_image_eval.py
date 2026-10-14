"""Run fixed four-dataset checkpoint generation and metric evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mini-root', required=True)
    p.add_argument('--run-root', required=True)
    p.add_argument('--model-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--steps', nargs='+', type=int, required=True)
    p.add_argument('--tiny', action='store_true')
    p.add_argument('--gpus', nargs='+', type=int, default=list(range(8)))
    a = p.parse_args()
    mini, run, output = map(Path, (a.mini_root, a.run_root, a.output_root))
    datasets = ('magicbrush', 'crispedit', 'scaleedit', 'interedit')
    manifest_names = ('magicbrush_official_dev', 'crispedit_aux8', 'scaleedit_aux8', 'interedit_aux8')
    roots = {
        name: mini / 'assets' / name for name in datasets
    }
    roots['magicbrush'] /= 'train' if a.tiny else 'dev'
    jobs = []
    for step in a.steps:
        for name, dev_manifest in zip(datasets, manifest_names):
            manifest = (mini / 'learnability/tiny64' / f'{name}_64.jsonl'
                        if a.tiny else mini / 'corpus/validation' / f'{dev_manifest}.jsonl')
            jobs.append((step, name, manifest))

    def worker(worker_index):
        gpu = a.gpus[worker_index]
        for step, name, manifest in jobs[worker_index::len(a.gpus)]:
            destination = output / f'step-{step}' / name
            destination.mkdir(parents=True, exist_ok=True)
            if (destination / 'predictions.jsonl').is_file():
                rows = [json.loads(line) for line in (destination / 'predictions.jsonl').read_text().splitlines() if line]
                expected = sum(bool(line.strip()) for line in manifest.read_text().splitlines())
                if len(rows) != expected:
                    raise RuntimeError(f'incomplete existing predictions: {destination}')
                continue
            command = [sys.executable, str(ROOT / 'scripts/eval/generate_evaluation.py'),
                '--dataset', name, '--manifest', str(manifest),
                '--canonical-root', str(roots[name]), '--model-root', a.model_root,
                '--checkpoint', str(run / f'checkpoint-{step}'),
                '--config', str(ROOT / 'configs/dev/mini_eval.yaml'),
                '--timestep-mode', 'roi_relative', '--output-dir', str(destination)]
            with (destination / 'generate.log').open('w') as log:
                subprocess.run(command, cwd=ROOT,
                    env=os.environ | {'CUDA_VISIBLE_DEVICES': str(gpu)},
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f'GENERATION_COMPLETE step={step} dataset={name}', flush=True)

    with ThreadPoolExecutor(max_workers=len(a.gpus)) as executor:
        list(executor.map(worker, range(len(a.gpus))))
    for step in a.steps:
        destination = output / f'step-{step}'
        inputs = sum((['--input', str(destination / name / 'predictions.jsonl')] for name in datasets), [])
        subprocess.run([sys.executable, str(ROOT / 'scripts/dev/merge_prediction_manifests.py'),
            *inputs, '--output', str(destination / 'predictions.jsonl')], cwd=ROOT, check=True)
        with (destination / 'evaluate.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT / 'scripts/eval/formal_eval.py'),
                '--experiment', 'e3-learnability', '--checkpoint', str(run / f'checkpoint-{step}'),
                '--config', str(ROOT / 'configs/dev/mini_eval.yaml'),
                '--predictions-manifest', str(destination / 'predictions.jsonl'),
                '--output', str(destination / 'metrics.json')], cwd=ROOT,
                env=os.environ | {'CUDA_VISIBLE_DEVICES': str(a.gpus[0])},
                stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'METRICS_COMPLETE step={step}', flush=True)


if __name__ == '__main__':
    main()
