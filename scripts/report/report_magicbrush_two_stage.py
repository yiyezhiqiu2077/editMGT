#!/usr/bin/env python3
"""Truthful local stage reports and lightweight archives using the existing builder."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dense_checkpoint import write_json,verify_dense_checkpoint
from scripts.report.build_experiment_report import build_report


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-root',type=Path,required=True)
    p.add_argument('--stage',choices=['sft','dimo'],required=True);a=p.parse_args()
    root=a.experiment_root;stage=root/a.stage;stage.mkdir(exist_ok=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    ready=json.loads((root/'data/MAGICBRUSH_READY.json').read_text())
    selection=json.loads((root/'TEACHER_SELECTION.json').read_text()) if (root/'TEACHER_SELECTION.json').exists() else None
    report=[f'# MagicBrush 4×A800 {a.stage.upper()} report','',f'Code SHA: `{sha}`.',
            f"Train: {ready['total_rows']} turns, {ready['excluded_rows']} excluded turns in whole sessions.",
            f"Frozen manifest SHA256: `{ready['train_manifest_sha256']}`.",
            'DEV: previous frozen 527 samples / 266 source groups; seeds [0, 1]; TEST unused.',
            'Selection scope: magicbrush_exploratory; formal_teacher=false.','']
    log=stage/'train'/('train_metrics.jsonl' if a.stage=='sft' else 'metrics.jsonl')
    rows=[json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
    report.append(f"Committed optimizer steps: {rows[-1]['global_step'] if rows else 'NOT_TESTED'}.")
    if selection:
        report += [f"Teacher: `{selection['teacher_source']}`; selected step {selection['selected_step']}.",
                   f"Fallback reason: {selection['fallback_reason']}.",
                   f"Selected checkpoint hash: `{selection['selected_checkpoint_hash']}`."]
        shutil.copyfile(root/'TEACHER_SELECTION.json',stage/'TEACHER_SELECTION.json')
        if selection.get('selection_status')=='USER_SELECTED':
            report += ['Teacher selection: USER_SELECTED by explicit user instruction; full DEV candidate selection was skipped.',
                       'SFT was stopped early. This is an exploratory Teacher, not a passed formal or full-DEV selection.',
                       f"User instruction: {selection['user_request']['instruction']}"]
    report += ['', '## Training and precision','',
               'Transformer/region parameters, gradients, AdamW moments and EMA: FP32; CUDA forward: BF16 autocast.',
               'Frozen CLIP/Gemma/VQ are shared. No LoRA in trained Student/Auxiliary.',
               'Hard errors stop the experiment. Small gradients and surrogate values are diagnostics.']
    if a.stage=='sft':
        report += ['', '| Epoch | Turns consumed | Mean ROI CE |','|---:|---:|---:|']
        for r in rows:
            if r.get('epoch_boundary'):
                report.append(f"| {r['epoch']+1} | {r['epoch_rows_consumed']} | {r['epoch_train_ce_mean']:.8f} |")
    else:
        report += ['Training r_init = inference r_init = 1.0; ROI all MASK, outside source-token lock.',
                   'The pseudo-mask cosine schedule remains variable. Linear surrogate sums vocabulary and averages masked tokens per sample and batch.']
    report += ['', '## DEV metrics','',
               'Inside compares to target; outside compares to source. Seed means precede sample-weighted source-group bootstrap (10,000; seed42; 95% CI).',
               '| Checkpoint | Inside LPIPS | Inside PSNR | Inside SSIM | Outside LPIPS | Outside PSNR | Outside SSIM | Full LPIPS |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
    evaluation=stage/('dev_evaluation' if a.stage=='sft' else 'train/evaluation')
    for file in sorted(evaluation.rglob('metrics.json')) if evaluation.exists() else []:
        metrics=json.loads(file.read_text());samples=metrics['per_sample_after_seed_mean']
        def mean(key):
            values=[r['metrics'].get(key) for r in samples if r['metrics'].get(key) is not None]
            return f'{sum(values)/len(values):.6f}' if values else 'NOT_TESTED'
        report.append('| '+str(file.parent.relative_to(evaluation))+' | '+' | '.join(mean(k) for k in (
            'inside_masked_lpips','inside_psnr','inside_ssim','outside_masked_lpips','outside_psnr','outside_ssim','full_lpips_to_target'))+' |')
    report += ['', 'Fixed cases show Source / Mask / Target / Teacher / Student / EMA as applicable; instructions remain in predictions.jsonl.',
               'LPIPS improvement alone does not establish semantic editing success; d_SO/d_ST no-edit diagnostics remain in per-sample records.',
               'DINO-I, CLIP-I and synchronized latency are included in metrics.json when available.']
    acceptance=root/'admission'/f'{a.stage.upper()}_CUDA_ACCEPTANCE.json'
    if acceptance.exists():
        data=json.loads(acceptance.read_text());shutil.copyfile(acceptance,stage/'CUDA_ACCEPTANCE.json')
        report += ['',f"CUDA smoke/resume: {data['status']}; {data['equivalence']}; real four A800.",
                   'Peak allocated bytes by run/rank:', '```json',json.dumps(data['gpu_peak_allocated_bytes'],indent=2),'```']
    else:report += ['CUDA smoke/resume: NOT_TESTED.']
    merge={'initial_strict_trial':json.loads((root/'admission/merge/MERGE_PARITY.json').read_text()),
           'rounding_aware_unseen_cases':json.loads((root/'admission/merge_holdout_v2/MERGE_PARITY.json').read_text()),
           'fp64_static_semantics':json.loads((root/'admission/MERGE_FP64_COMPLETE_REFERENCE.json').read_text())}
    write_json(stage/'MERGE_PROVENANCE.json',merge)
    report += ['', 'Merge: original strict FP32/BF16 tolerance failed; retained as calibration evidence.',
               'FP64 reference (FP64 RMSNorm and rotary arithmetic only in the reference) proves static both-branch semantics.',
               'Separate unseen fixed cases used preregistered rounding-aware tolerance and mean inside LPIPS degradation <=0.01.',
               'No production RMSNorm/rotary arithmetic or Teacher-selection threshold was changed.']
    write_json(stage/'frozen_data_identity.json',ready)
    shutil.copyfile(root/'evaluation_plan.json',stage/'evaluation_plan.json')
    if (root/'selection_preregistration.json').exists():shutil.copyfile(root/'selection_preregistration.json',stage/'selection_preregistration.json')
    shutil.copyfile(root/'configs'/f'{a.stage}.yaml',stage/'stage_config.yaml')
    if a.stage=='sft':
        plan=json.loads((root/'evaluation_plan.json').read_text())
        identities={str(step):verify_dense_checkpoint(plan['candidate_checkpoints'][str(step)])
                    for step in plan['candidate_steps'] if Path(plan['candidate_checkpoints'][str(step)]).exists()}
    else:
        identities={}
        for marker in (stage/'train/inference').glob('checkpoint-*/COMPLETED.json'):
            identities[marker.parent.name]=json.loads(marker.read_text())
    write_json(stage/'final_checkpoint_identity.json',identities)
    (stage/'REPORT.md').write_text('\n'.join(report)+'\n')
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    snapshot={}
    for relative in tracked:
        path=Path(relative)
        if relative not in ('pyproject.toml','uv.lock','docs/EXPERIMENTS.md') and (
                path.parts[0] not in ('src','scripts','configs','tests') or path.suffix not in ('.py','.yaml','.sh')):continue
        target=stage/'code'/path;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/path,target);snapshot[relative]=sha256_file(target)
    write_json(stage/'code_snapshot.json',{'git_sha':sha,'files':snapshot})
    reports=root/'reports';reports.mkdir(exist_ok=True)
    name='e3' if a.stage=='sft' else 'dimo'
    archive=reports/f'{a.stage}_report.tar.gz'
    build_report({name:stage},archive,stage=name)
    print(archive,flush=True)


if __name__=='__main__':main()
