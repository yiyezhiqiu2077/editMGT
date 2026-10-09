#!/usr/bin/env python3
"""Compare fixed DEV predictions and produce small, deterministic image grids."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw
from src.dimo.contracts import _sha256

LABELS = ['teacher-3125', 'student-0', 'student-50', 'student-100', 'student-250', 'student-500',
          'ema-50', 'ema-100', 'ema-250', 'ema-500']
METRICS = ['inside_masked_lpips', 'inside_psnr', 'inside_ssim', 'outside_masked_lpips',
           'full_lpips_to_target', 'runtime_seconds']


def compare(directories, output):
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError('COMPARISON_OUTPUT_ALREADY_EXISTS')
    rows, predictions = {}, {}
    for directory in directories:
        directory = Path(directory)
        row = json.loads((directory / 'diagnostic.json').read_text())
        label = row['label']
        if label in rows or label not in LABELS or row.get('dev_only') is not True or row.get('formal_teacher') is not False:
            raise RuntimeError('INVALID_OR_DUPLICATED_DEV_DIAGNOSTIC')
        manifest = directory / 'predictions.jsonl'
        if row['prediction_manifest_sha256'] != _sha256(manifest):
            raise RuntimeError('DEV_PREDICTION_MANIFEST_CHANGED')
        metrics_path = directory / 'metrics.json'
        if (row['metrics_sha256'] != _sha256(metrics_path)
                or row['aggregate'] != json.loads(metrics_path.read_text())['aggregate'][f"dataset:{row['dataset_name']}"]):
            raise RuntimeError('DEV_METRICS_CHANGED')
        rows[label] = row
        predictions[label] = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if set(rows) != set(LABELS):
        raise RuntimeError('COMPLETE_TEACHER_RAW_EMA_DEV_COMPARISON_REQUIRED')
    reference = rows[LABELS[0]]
    identity = ['dev_manifest_sha256', 'corpus_ready_sha256', 'teacher_hash', 'config', 'dataset_name',
                'dev_rows', 'latency_scope']
    if any(any(row[k] != reference[k] for k in identity) for row in rows.values()):
        raise RuntimeError('DEV_DIAGNOSTIC_IDENTITY_MISMATCH')
    signature = lambda items: [(r['sample_key'], r['seed'], r['aligned_input_sha256']) for r in items]
    if any(len(p) != 128 or signature(p) != signature(predictions[LABELS[0]]) for p in predictions.values()):
        raise RuntimeError('DEV_INPUT_MASK_SEED_SEQUENCE_MISMATCH')
    for records in predictions.values():
        for record in records:
            if any(_sha256(Path(record[name])) != digest for name, digest in record['aligned_input_sha256'].items()):
                raise RuntimeError('ALIGNED_DEV_INPUT_CHANGED')
            if _sha256(Path(record['output'])) != record['output_sha256']:
                raise RuntimeError('GENERATED_DEV_IMAGE_CHANGED')
    root.mkdir(parents=True, exist_ok=True)
    table = ['| model | inside LPIPS | inside PSNR | inside SSIM | outside LPIPS-to-source | full LPIPS | seconds/image |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for label in LABELS:
        values = [rows[label]['aggregate'][metric]['mean'] for metric in METRICS]
        table.append('| ' + label + ' | ' + ' | '.join('N/A' if v is None else f'{v:.6f}' for v in values) + ' |')
    (root / 'comparison.md').write_text('\n'.join(table) + '\n')
    # Fixed first six rows of the frozen manifest, never quality-selected cases.
    visual_labels = ['source', 'mask', 'target'] + LABELS
    for index in range(6):
        grid = Image.new('RGB', (256 * len(visual_labels), 290), 'white')
        draw = ImageDraw.Draw(grid)
        for column, label in enumerate(visual_labels):
            record = predictions[LABELS[0]][index] if label in ('source', 'mask', 'target') else predictions[label][index]
            path = record[label] if label in ('source', 'mask', 'target') else record['output']
            picture = Image.open(path).convert('RGB').resize((256, 256))
            grid.paste(picture, (column * 256, 28))
            draw.text((column * 256 + 4, 6), label, fill='black')
        grid.save(root / f'fixed-dev-{index:03d}.jpg', quality=90)
    result = {'schema': 'dimo-dev-comparison-v1', 'dev_only': True, 'formal_teacher': False,
              'teacher_hash': reference['teacher_hash'], 'dev_manifest_sha256': reference['dev_manifest_sha256'],
              'diagnostics': rows, 'quality_selection_performed': False,
              'teacher_limit_note': 'Provisional teacher has only 0.5 E3 epoch; quality ceiling is not attributable to DiMO alone.'}
    (root / 'comparison.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    compare(args.runs, args.output_dir)


if __name__ == '__main__':
    main()
