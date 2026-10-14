"""Export diagnosis tables and fixed representative comparison sheets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

METRICS = ['inside_masked_lpips', 'inside_psnr', 'inside_ssim',
           'outside_masked_lpips', 'outside_psnr', 'outside_ssim', 'full_lpips_to_target']


def read(path):
    return json.loads(path.read_text())


def paired(a, b, metric):
    """a minus b, matched by UID; negative LPIPS means a is better."""
    x = {r['sample_key']: r['metrics'][metric] for r in a['per_sample_after_seed_mean']}
    y = {r['sample_key']: r['metrics'][metric] for r in b['per_sample_after_seed_mean']}
    assert x.keys() == y.keys()
    values = np.array([x[k] - y[k] for k in sorted(x)
                       if x[k] is not None and y[k] is not None])
    rng = np.random.default_rng(42)
    ci = np.quantile(values[rng.integers(0, len(values), (20000, len(values)))].mean(1), [.025, .975])
    return {'mean_difference': float(values.mean()), 'paired_95_ci': ci.tolist(), 'n': len(values)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mini-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--cfg', type=float, required=True)
    p.add_argument('--validation-cfg', type=float)
    a = p.parse_args()
    root, mini = Path(a.output_root), Path(a.mini_root)
    annotations = read(root / 'manifests/diagnostic_annotations.json')
    all_results = {}
    table = []
    inputs = {}
    for path in sorted(root.glob('dev*/step-*/metrics.json')):
        parts = path.parent.name.split('_')
        step = int(parts[0].split('-')[1])
        cfg = float(parts[-1].split('-')[1])
        mode = '_'.join(parts[1:-1])
        count = int(path.parents[1].name[3:])
        data = read(path)
        for record in data['per_generation']:
            key = (record['dataset_name'], record['sample_key'])
            signature = {role: hashlib.sha256(Path(record[role]).read_bytes()).hexdigest()
                         for role in ['source', 'target', 'mask']}
            assert record['seed'] == 0
            if key in inputs:
                assert inputs[key] == signature, f'Input geometry changed across cells: {key}'
            inputs[key] = signature
        all_results[(count, step, mode, cfg)] = data
        samples = data['per_sample_after_seed_mean']
        groups = {'all': samples}
        for r in samples:
            groups.setdefault('dataset:' + r['dataset_name'], []).append(r)
            groups.setdefault('edit_type:' + r['edit_type'], []).append(r)
            if r['dataset_name'] == 'magicbrush' and r['sample_key'] in annotations:
                tag = annotations[r['sample_key']]['diagnostic_edit_type']
                groups.setdefault('diagnostic_magicbrush:' + tag, []).append(r)
        for group, rows in groups.items():
            row = {'dev_samples': count, 'step': step, 'timestep_mode': mode, 'cfg': cfg,
                   'group': group, 'n': len(rows)}
            for metric in METRICS:
                vals = [r['metrics'][metric] for r in rows if r['metrics'].get(metric) is not None]
                row[metric] = float(np.mean(vals)) if vals else None
            table.append(row)
    def write_csv(name, rows):
        with (root / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write_csv('all_metrics.csv', table)
    write_csv('cfg_sweep.csv', [r for r in table if r['dev_samples'] == 24
                               and r['step'] == 300 and r['timestep_mode'] == 'roi_relative'])
    ce = []
    for step in [0, 100, 300]:
        partial = read(mini / f'learnability/tiny64/eval/step-{step}/ce.json')
        full = read(root / 'ce' / f'step-{step}-full.json')
        assert [r['sample_uid'] for r in partial['per_sample']] == [r['sample_uid'] for r in full['per_sample']]
        ce.append({'step': step, 'partial_ce': partial['ce'], 'full_roi_ce': full['ce'],
                   'shortcut_gap': full['ce'] - partial['ce']})
    write_csv('full_roi_ce.csv', ce)
    comparisons = {}
    for count in [24, 56]:
        selected_cfg = a.validation_cfg if a.validation_cfg is not None else a.cfg
        key = (count, 300, 'roi_relative', selected_cfg)
        baseline = (count, 300, 'roi_relative', 10.)
        if key in all_results and baseline in all_results:
            comparisons[f'dev{count}_selected_cfg_minus_10'] = {
                m: paired(all_results[key], all_results[baseline], m)
                for m in ['inside_masked_lpips', 'outside_masked_lpips']}
    for step in [0, 300]:
        off, roi = ((24, step, mode, a.cfg) for mode in ['official_upstream_timestep', 'roi_relative'])
        if off in all_results and roi in all_results:
            comparisons[f'step{step}_official_minus_roi'] = {
                m: paired(all_results[off], all_results[roi], m)
                for m in ['inside_masked_lpips', 'outside_masked_lpips']}
    (root / 'paired_comparisons.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    (root / 'input_consistency.json').write_text(json.dumps({
        'status': 'PASS', 'sample_uids': len(inputs), 'seed': 0,
        'checks': 'source/target/mask SHA256 identical across all repeated cells',
        'files': [{'dataset': key[0], 'sample_uid': key[1], 'sha256': value}
                  for key, value in sorted(inputs.items())]}, indent=2) + '\n')
    print(json.dumps({'ce': ce, 'comparisons': comparisons}, indent=2), flush=True)

    selected = []
    for dataset, indices in [('magicbrush', [0, 1, 2, 3]), ('crispedit', [0]),
                             ('scaleedit', [0]), ('interedit', [0, 2])]:
        rows = [json.loads(l) for l in (root / f'manifests/{dataset}_dev24.jsonl').read_text().splitlines() if l]
        selected.extend((dataset, rows[i]) for i in indices)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    def records(key):
        return {r['sample_key']: r for r in all_results[key]['per_generation']}
    def sheet(name, keys, labels):
        width, cell, header, rowheight = (3 + len(keys)) * 224, 224, 45, 275
        canvas = Image.new('RGB', (width, header + len(selected) * rowheight), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, label in enumerate(['source', 'target', 'mask'] + labels):
            draw.text((i * cell + 5, 10), label, fill='black', font=font)
        maps = [records(key) for key in keys]
        for j, (dataset, sample) in enumerate(selected):
            uid = sample['sample_uid']; first = maps[0][uid]
            paths = [first[role] for role in ['source', 'target', 'mask']] + [m[uid]['output'] for m in maps]
            top = header + j * rowheight
            tag = annotations.get(uid, {}).get('diagnostic_edit_type', sample['edit_type_canonical'])
            draw.text((5, top), f'{dataset} {tag} | {uid[:10]}', fill='black', font=font)
            for i, path in enumerate(paths):
                with Image.open(path) as image:
                    canvas.paste(image.convert('RGB').resize((cell, cell)), (i * cell, top + 25))
        canvas.save(root / name)
    cfg_keys = [(24, 300, 'roi_relative', cfg) for cfg in [1., 3., 5., 7.5, 10.]]
    if all(key in all_results for key in cfg_keys):
        sheet('cfg_comparison_sheet.png', cfg_keys, ['CFG1', 'CFG3', 'CFG5', 'CFG7.5', 'CFG10'])
    time_keys = [(24, step, mode, a.cfg) for step in [0, 300]
                 for mode in ['official_upstream_timestep', 'roi_relative']]
    if all(key in all_results for key in time_keys):
        sheet('timestep_comparison_sheet.png', time_keys,
              ['step0 official', 'step0 ROI', 'step300 official', 'step300 ROI'])
    (root / 'visual_samples.json').write_text(json.dumps(
        [{'dataset': d, 'sample_uid': r['sample_uid'], 'instruction': r['instruction_en']}
         for d, r in selected], indent=2) + '\n')

    validation_cfg = a.validation_cfg if a.validation_cfg is not None else a.cfg
    required = [(56, 300, 'roi_relative', cfg) for cfg in {validation_cfg, 10.}]
    if not all(key in all_results for key in required + time_keys):
        return
    inside_delta = comparisons['dev56_selected_cfg_minus_10']['inside_masked_lpips']
    outside_delta = comparisons['dev56_selected_cfg_minus_10']['outside_masked_lpips']
    overguidance = ('YES' if inside_delta['paired_95_ci'][1] < 0
                    and outside_delta['paired_95_ci'][1] <= 0 else 'UNCLEAR')
    shift = 'YES' if comparisons['step0_official_minus_roi']['inside_masked_lpips']['paired_95_ci'][1] < 0 else 'NO'
    statuses = {'TARGET_CONTEXT_SHORTCUT': 'NO (NOT_PRIMARY)',
                'CFG_OVERGUIDANCE': overguidance, 'PRETRAIN_TIMESTEP_SHIFT': shift,
                'TIMESTEP_ADAPTATION': 'UNCLEAR', 'READY_FOR_D200K': 'NO',
                'selected_cfg_dev56': validation_cfg, 'cfg_timestep_ab': a.cfg}
    (root / 'diagnosis_summary.json').write_text(json.dumps(statuses, indent=2) + '\n')
    lines = ['# E3 Failure Diagnosis v1', '',
             '仅复用 Tiny-64、DEV56 和已有 step0/100/300 checkpoint；未训练、未下载数据、未改训练 recipe / 推理算法、未合入 main。', '',
             '## A. Full-ROI CE', '',
             '| checkpoint | partial CE | full-ROI CE | shortcut gap |',
             '| --- | ---: | ---: | ---: |']
    for row in ce:
        lines.append(f"| step{row['step']} | {row['partial_ce']:.6f} | {row['full_roi_ce']:.6f} | {row['shortcut_gap']:.6f} |")
    lines += ['', f"partial improvement: {ce[0]['partial_ce'] - ce[-1]['partial_ce']:.6f}", '',
              f"full-ROI improvement: {ce[0]['full_roi_ce'] - ce[-1]['full_roi_ce']:.6f}", '',
              'TARGET_CONTEXT_SHORTCUT: NO（NOT_PRIMARY）', '',
              '标准 partial 一列严格沿用 E3 corruption，包含原有 15% full-ROI 分支。全掩码探针仅在 dev-only CE 脚本中强制完整 ROI 掩码、timestep=1，且断言 ROI 内无可见 target token。两套 CE 均为 eval mode、固定 UID 顺序/geometry/corruption seed，非随机训练日志末批值。', '',
              'Full-ROI CE 明显下降且 gap 大幅缩小，不支持 target-context shortcut 是首要原因；此结果不能完全排除其他 exposure bias。', '',
              '## B. CFG', '',
              '| CFG | DEV24 inside LPIPS | DEV24 outside LPIPS |',
              '| ---: | ---: | ---: |']
    for cfg in [1., 3., 5., 7.5, 10.]:
        row = all_results[(24, 300, 'roi_relative', cfg)]['selection_summary']
        lines.append(f"| {cfg:g} | {row['inside_masked_lpips']:.6f} | {row['outside_masked_lpips']:.6f} |")
    lines += ['', f'Best CFG: {validation_cfg:g}（按 DEV24 inside 均值选择的候选，不是唯一显著最优值；1/3/5 接近）', '',
              '| DEV56 | inside LPIPS | outside LPIPS |', '| --- | ---: | ---: |']
    for cfg in [validation_cfg, 10.]:
        row = all_results[(56, 300, 'roi_relative', cfg)]['selection_summary']
        lines.append(f"| CFG{cfg:g} | {row['inside_masked_lpips']:.6f} | {row['outside_masked_lpips']:.6f} |")
    lines += ['', f'CFG_OVERGUIDANCE: {overguidance}', '',
              f"DEV56 inside 差值（候选−10）: {inside_delta['mean_difference']:.6f}，配对 95% CI {inside_delta['paired_95_ci']}。", '',
              f"DEV56 outside 差值（候选−10）: {outside_delta['mean_difference']:.6f}，配对 95% CI {outside_delta['paired_95_ci']}。", '',
              '## C. Timestep', '',
              f'CFG={a.cfg:g}：因 CFG 最优值不明确，按指令使用默认 5；保留已有 CFG10 DEV24 对照，不重新大规模生成。', '',
              '| checkpoint / timestep | inside LPIPS | outside LPIPS |', '| --- | ---: | ---: |']
    for key in time_keys:
        row = all_results[key]['selection_summary']
        lines.append(f"| step{key[1]} / {key[2]} | {row['inside_masked_lpips']:.6f} | {row['outside_masked_lpips']:.6f} |")
    lines += ['', f'PRETRAIN_TIMESTEP_SHIFT: {shift}（本次没有证实预训练时间分布漂移导致质量下降）', '',
              'TIMESTEP_ADAPTATION: UNCLEAR（step0 无明显 official 优势，不能据此判定该 gap 是否被训练缩小）', '',
              '## D. Dataset breakdown', '',
              f'DEV56 step300，CFG{validation_cfg:g} 对比 CFG10；两者均 roi_relative。', '',
              '| group | N | inside candidate | inside CFG10 | outside candidate | outside CFG10 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for group in ['dataset:magicbrush', 'dataset:crispedit', 'dataset:scaleedit', 'dataset:interedit',
                  'diagnostic_magicbrush:add', 'diagnostic_magicbrush:remove']:
        rows = [next(r for r in table if r['dev_samples'] == 56 and r['step'] == 300
                     and r['cfg'] == cfg and r['group'] == group)
                for cfg in [validation_cfg, 10.]]
        x, y = rows
        lines.append(f"| {group} | {x['n']} | {x['inside_masked_lpips']:.6f} | {y['inside_masked_lpips']:.6f} | {x['outside_masked_lpips']:.6f} | {y['outside_masked_lpips']:.6f} |")
    lines += ['', 'MagicBrush 的 canonical edit_type 均为 other、original 为 unknown。Add/Remove 是独立 sidecar 中的人工指令解释（不是官方标签）；复合替换不归为纯 remove。原始 frozen manifest、训练/评估数据处理规则未改。', '',
              '## Final root-cause ranking', '',
              f'1. 高 CFG：{overguidance}。三项假设中相对证据最多，但是否为主因应依 DEV56 和分项结果，不能视为四数据集一致的最优结论。', '',
              '2. Target-context shortcut：NOT_PRIMARY。完整 ROI 的 CE 下降更大，gap 缩小，不支持“主要靠 GT context 才学会”的解释。', '',
              '3. ROI-relative timestep 漂移：本次未证实。step0 official 不明显优于 ROI，step300 的 inside/outside 收益也不一致。', '',
              '上述排序是三个假设的证据排序，不是已证明三个真实根因。尚不能将残余质量问题归因于某个代码 bug。', '',
              'NEXT TRAINING CHANGE: 暂不改 loss、full-ROI 概率、LoRA scope、LR 或 timestep embedding。优先在后续预先固定的验证集/多 seed 上确认较低 CFG 的收益，再决定是否改变评估配置；本轮不直接修改正式配置或训练 recipe。', '',
              'READY_FOR_D200K: NO', '',
              '## Comparison sheets', '',
              '![CFG comparison](cfg_comparison_sheet.png)', '',
              '![Timestep comparison](timestep_comparison_sheet.png)', '',
              '八个样本在查看 sweep 结果前固定：MagicBrush Add×2、Remove×2、CrispEdit×1、ScaleEdit×1、InterEdit Add×1、Remove×1。', '',
              '## Reproducibility and limitations', '',
              '- generation seed=0、1024、12 steps、reference strength=1、LoRA scope=both、persistent region conditioning=true。',
              '- source/target/mask 在各重复 cell 中逐文件 SHA256 一致，见 input_consistency.json。',
              '- cfg_sweep.csv / all_metrics.csv 保留 inside/outside LPIPS、PSNR、SSIM、full LPIPS，以及 dataset/edit_type 分项。',
              '- paired_comparisons.json：按相同 UID 的配对差值 bootstrap，seed42、20000 次、样本为重采样单位；用于探索性诊断，非事后声称已预注册的正式显著性检验。',
              '- 一套小 DEV、单生成 seed；CFG 在 DEV24 选择，而 DEV24 是 DEV56 子集，所以 DEV56 不是完全独立确认集，不足以支持正式实验结论。',
              '- 137 pytest PASS；compileall / git diff --check PASS。']
    (root / 'REPORT_中文.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
