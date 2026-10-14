"""Render all fixed DEV24 samples at CFG10: base versus Tiny-64 step300."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
DATASETS = ['magicbrush', 'crispedit', 'scaleedit', 'interedit']


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mini-root', required=True)
    parser.add_argument('--diagnosis-root', required=True)
    parser.add_argument('--model-root', required=True)
    args = parser.parse_args()
    mini, root = Path(args.mini_root), Path(args.diagnosis_root)
    base = root / 'dev24/step-0_roi_relative_cfg-10'
    trained = root / 'dev24/step-300_roi_relative_cfg-10'
    out = root / 'visuals_cfg10'
    out.mkdir(parents=True, exist_ok=True)

    def generate(index):
        dataset = DATASETS[index]
        dest = base / dataset
        dest.mkdir(parents=True, exist_ok=True)
        manifest = root / 'manifests' / f'{dataset}_dev24.jsonl'
        expected = [r['sample_uid'] for r in read_rows(manifest)]
        predictions = dest / 'predictions.jsonl'
        if predictions.exists():
            assert [r['sample_key'] for r in read_rows(predictions)] == expected
            return
        asset = mini / 'assets' / dataset
        if dataset == 'magicbrush':
            asset /= 'dev'
        command = [sys.executable, str(REPO / 'scripts/eval/generate_evaluation.py'),
                   '--dataset', dataset, '--manifest', str(manifest),
                   '--canonical-root', str(asset), '--model-root', args.model_root,
                   '--checkpoint', str(mini / 'runs/e3-tiny64-overfit-300/checkpoint-0'),
                   '--config', str(root / 'configs/cfg-10.json'),
                   '--timestep-mode', 'roi_relative', '--output-dir', str(dest)]
        with (dest / 'generate.log').open('w') as log:
            subprocess.run(command, cwd=REPO,
                           env=os.environ | {'CUDA_VISIBLE_DEVICES': str(index + 4),
                                              'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        assert [r['sample_key'] for r in read_rows(predictions)] == expected
        print(f'BASE_CFG10_COMPLETE {dataset}', flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(generate, range(4)))
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    manifest = []
    for dataset in DATASETS:
        original = read_rows(base / dataset / 'predictions.jsonl')
        tuned = read_rows(trained / dataset / 'predictions.jsonl')
        assert [r['sample_key'] for r in original] == [r['sample_key'] for r in tuned]
        frozen = read_rows(root / 'manifests' / f'{dataset}_dev24.jsonl')
        cell, header, rowheight = 256, 55, 310
        canvas = Image.new('RGB', (5 * cell, header + len(original) * rowheight), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 3), f'{dataset} | CFG10 | seed0 | 12 steps | ROI-relative', font=font, fill='black')
        for col, label in enumerate(['SOURCE', 'TARGET', 'MASK', 'BASE (step0)', 'E3 (step300)']):
            draw.text((col * cell + 5, 30), label, font=font, fill='black')
        for index, (a, b, raw) in enumerate(zip(original, tuned, frozen)):
            assert a['seed'] == b['seed'] == 0
            assert a['timestep_mode'] == b['timestep_mode'] == 'roi_relative'
            for role in ['source', 'target', 'mask']:
                assert hashlib.sha256(Path(a[role]).read_bytes()).digest() == hashlib.sha256(Path(b[role]).read_bytes()).digest()
            top = header + index * rowheight
            instruction = raw['instruction_en']
            draw.text((5, top), f"{index+1}. {instruction[:140]}", font=font, fill='black')
            images = [a['source'], a['target'], a['mask'], a['output'], b['output']]
            for col, path in enumerate(images):
                with Image.open(path) as image:
                    canvas.paste(image.convert('RGB').resize((cell, cell), Image.Resampling.LANCZOS),
                                 (col * cell, top + 24))
            manifest.append({'dataset': dataset, 'sample_uid': a['sample_key'],
                             'instruction': instruction, 'base_output': a['output'],
                             'step300_output': b['output']})
        path = out / f'{dataset}_cfg10_base_vs_step300.png'
        canvas.save(path)
        print(f'SHEET {path}', flush=True)
    (out / 'samples.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    (out / 'README.md').write_text(
        '# CFG10：原模型与 E3 300-step 对比\n\n'
        '固定 DEV24 全部样本，每数据集 6 条，不按生成质量筛选。'
        'CFG=10、seed=0、12 inference steps、1024、reference strength=1、roi_relative。'
        '从左到右：source、target、mask、原模型 step0、E3 step300。'
        '原模型 step0 的 LoRA B 和 region embedding 为零。'
        'E3 step300 来自 Tiny-64 训练，而这些 DEV24 是开发验证样本。'
        '图中源图/目标/掩码已逐文件校验一致。原始 1024 输出路径见 samples.json。\n')


if __name__ == '__main__':
    main()
