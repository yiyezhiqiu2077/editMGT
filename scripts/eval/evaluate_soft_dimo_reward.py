#!/usr/bin/env python3
"""Fixed DEV hard quality plus paired soft/hard gap; all figures are diagnostics.

Inference intentionally has no autograd. Training differentiability is tested
separately; PIL is only used to load fixed DEV inputs and save output artifacts.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.explicit_region.config import load_config
from src.explicit_region.contracts import sha256_file,freeze_modules
from src.explicit_region.dense_checkpoint import write_json
from src.explicit_region.group_statistics import cluster_bootstrap
from src.explicit_region.dense_evaluation import paired_comparison,comparison_visual
from src.dimo.dev_protocol import validate_plan,validate_predictions,STUDENT_PROTOCOL,rows_at
from src.dimo.native_evaluation import evaluate_roi_metrics
from src.dimo.soft_reward import SoftEmbeddingEditDecoder

KEYS=('inside_masked_lpips','outside_masked_lpips','roi_clip_to_target','roi_dino_to_target')


@torch.inference_mode()
def generate_pair(plan,rows,checkpoint,root,identity,weights,device):
    from src.explicit_region.modeling import load_released_components
    from src.explicit_region.dense_inference import pil
    from src.dimo.contracts import teacher_bundle_fingerprint,released_base_model_identity
    from src.dimo.inference import load_dense_inference_role
    from src.dimo.one_step import one_step_edit_tokens
    from src.dimo.rng import stable_seed,normal_noise_per_sample
    from scripts.train.train_dimo_editing import prepare_batch
    from scripts.eval.formal_eval import tensor
    root.mkdir(parents=True,exist_ok=False)
    components=load_released_components(plan['model_root'],torch_dtype=torch.bfloat16,
        transformer_dtype=torch.float32,vq_dtype=torch.float32)
    bundle=teacher_bundle_fingerprint(plan['teacher_checkpoint'],
        base_model_identity=released_base_model_identity(components.identity),formal=False)
    roles,loaded=load_dense_inference_role(components.transformer,checkpoint,weights=weights,teacher_bundle=bundle)
    if loaded!=identity:raise RuntimeError('SOFT_REWARD_DEV_CHECKPOINT_IDENTITY_MISMATCH')
    roles.to(device).eval()
    freeze_modules([components.text_encoder,components.llm_encoder,components.vqvae])
    for model in [components.text_encoder,components.llm_encoder,components.vqvae]:model.to(device)
    decoder=SoftEmbeddingEditDecoder(components.vqvae).eval()
    records={'hard':[],'soft':[]}
    for name in records:(root/name).mkdir()
    for index,row in enumerate(rows):
        batch={'source_image':tensor(row['source']),'edit_region_mask':tensor(row['mask'],mask=True)[:,0],
               'instruction_en':[row['instruction_en']],'sample_uid':[row['sample_uid']]}
        for seed in [0,1]:
            if device.type=='cuda':torch.cuda.synchronize(device)
            started=time.perf_counter()
            prepared=prepare_batch(batch,components,{'resolution':1024,'token_mask':STUDENT_PROTOCOL['token_mask']},device)
            source,region=prepared['source_tokens'],prepared['edit_region_mask'];args=(seed,0,row['sample_uid'],0)
            noise=normal_noise_per_sample(torch.empty(source.shape[0],roles.base_model.inner_dim,
                *source.shape[1:],device=device,dtype=torch.float32),[stable_seed(*args,'dimo-embedding-noise')])
            result=one_step_edit_tokens(roles,source_tokens=source,edit_region_mask=region,
                prompt_condition=prepared['prompt_condition'],timestep_model_kwargs=prepared['model_kwargs'],
                mask_token_id=roles.base_model.config.vocab_size-1,codebook_size=8192,r_init=1.,
                init_mask_seeds=stable_seed(*args,'dimo-init-mask'),init_token_seeds=stable_seed(*args,'dimo-init-token'),
                sample_seeds=stable_seed(*args,'dimo-student-sample'),cfg_scale=1.,
                target_embedding_noise=noise,embedding_noise_sigma=.3)
            images={'hard':decoder.hard(result.tokens),'soft':decoder(result.logits,source,region)}
            if device.type=='cuda':torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-started
            assert result.metadata['number_of_transformer_forwards']==1
            for name,image in images.items():
                if not torch.isfinite(image).all():raise RuntimeError('SOFT_REWARD_DEV_NONFINITE_DECODE')
                path=root/name/f'{index:04d}_seed{seed}.png';pil(image[0].float().cpu()).save(path)
                records[name].append(row|{'output':str(path.resolve()),'seed':seed,
                    'manifest_sha256':plan['manifest_sha256'],'runtime_seconds':elapsed,
                    'runtime_scope':'paired hard+soft probe, not standalone inference speed',
                    'transformer_forward_count':1,'token_outside_lock':'PASS',
                    'pixel_outside_lock':'NOT_GUARANTEED',
                    'inference_protocol':STUDENT_PROTOCOL|{'role':weights,'decode_mode':name}})
        if (index+1)%16==0:print(f'{weights}: {index+1}/128 hard+soft',flush=True)
    for name,rows_out in records.items():
        path=root/name/'predictions.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows_out))
        validate_predictions(path,rows,plan['manifest_sha256'],STUDENT_PROTOCOL)
    del decoder,roles,components;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()


def compare_metrics(candidate,baseline,options):
    result=paired_comparison(candidate,baseline,strict_groups=True,**options)['magicbrush_dev']
    a={r['sample_uid']:r for r in candidate['per_sample_after_seed_mean']}
    b={r['sample_uid']:r for r in baseline['per_sample_after_seed_mean']}
    for key in KEYS[2:]:
        result[key]=cluster_bootstrap([a[k]['metrics'][key]-b[k]['metrics'][key] for k in sorted(a)],
            [a[k]['cluster_id'] for k in sorted(a)],**options)
    return {k:result[k] for k in KEYS}


def decode_gap_delta(new_hard,new_soft,old_hard,old_soft,options):
    indexed=[{r['sample_uid']:r for r in v['per_sample_after_seed_mean']} for v in [new_hard,new_soft,old_hard,old_soft]]
    result={}
    for key in ('inside_masked_lpips','roi_clip_to_target','roi_dino_to_target'):
        sign=1 if key=='inside_masked_lpips' else -1
        result[key]=cluster_bootstrap([sign*(indexed[0][u]['metrics'][key]-indexed[1][u]['metrics'][key]
            -indexed[2][u]['metrics'][key]+indexed[3][u]['metrics'][key]) for u in sorted(indexed[0])],
            [indexed[0][u]['cluster_id'] for u in sorted(indexed[0])],**options)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--step',type=int,choices=[20,500],required=True)
    p.add_argument('--plan',required=True);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    from scripts.train.train_dense_region import configure_determinism
    configure_determinism();torch.set_num_threads(2)
    run=Path(args.run).resolve();identity=json.loads((run/'experiment_manifest.json').read_text())
    if not identity.get('reward_extension'):raise RuntimeError('SOFT_REWARD_RUN_IDENTITY_REQUIRED')
    plan=load_config(args.plan);rows=validate_plan(plan);device=torch.device(args.device)
    output=run/'evaluation/soft_reward';output.mkdir(parents=True,exist_ok=True)
    binding={'run_manifest_sha256':sha256_file(run/'experiment_manifest.json'),'plan_sha256':sha256_file(args.plan),
             'evaluator_sha256':sha256_file(__file__)}
    ledger=output/'EVALUATION_IDENTITY.json'
    if ledger.exists() and json.loads(ledger.read_text())!=binding:raise RuntimeError('SOFT_REWARD_EVAL_IDENTITY_CHANGED')
    write_json(ledger,binding)
    baseline=Path(identity['reward_extension']['fork_checkpoint']).parents[1]/'evaluation/dev128/step-4000'
    old_summary=json.loads((baseline/'CURRENT_COMPARISON.json').read_text()) if (baseline/'CURRENT_COMPARISON.json').exists() else json.loads((baseline.parent/'step-8000/CURRENT_COMPARISON.json').read_text())
    # Bind historical metric files to the completed previous evaluation ledger.
    for role,label in [('student','Raw'),('ema','EMA')]:
        path=baseline/role/'metrics.json';expected=old_summary['metric_sha256'][f'v2 {label} step4000']
        if sha256_file(path)!=expected:raise RuntimeError('SOFT_REWARD_HISTORICAL_METRICS_CHANGED')
        validate_predictions(baseline/role/'predictions.jsonl',rows,plan['manifest_sha256'],STUDENT_PROTOCOL)
    for step in [0,args.step]:
        marker=json.loads((run/f'inference/checkpoint-{step}/COMPLETED.json').read_text())
        if marker['committed_step']!=step:raise RuntimeError('SOFT_REWARD_DEV_CHECKPOINT_STEP_MISMATCH')
        for role in ['student','ema']:
            folder=output/f'step-{step}'/role
            if not folder.exists():generate_pair(plan,rows,run/f'inference/checkpoint-{step}',folder,identity,role,device)
            for name in ['hard','soft']:
                preds=folder/name/'predictions.jsonl';hashes=validate_predictions(preds,rows,plan['manifest_sha256'],STUDENT_PROTOCOL)
                metrics=folder/name/'metrics.json'
                if not metrics.exists():
                    result=evaluate_roi_metrics(preds,plan,metrics,device)
                    result.update(prediction_output_sha256=hashes,predictions_manifest_sha256=sha256_file(preds),
                        checkpoint_completed_sha256=sha256_file(run/f'inference/checkpoint-{step}/COMPLETED.json'))
                    write_json(metrics,result)
                else:
                    result=json.loads(metrics.read_text())
                    if (result.get('prediction_output_sha256')!=hashes
                            or result.get('predictions_manifest_sha256')!=sha256_file(preds)
                            or result.get('checkpoint_completed_sha256')!=sha256_file(run/f'inference/checkpoint-{step}/COMPLETED.json')):
                        raise RuntimeError('SOFT_REWARD_DEV_CACHED_IMAGE_OR_CHECKPOINT_CHANGED')
    options={k:v for k,v in plan['bootstrap'].items() if k in ['seed','resamples','confidence_level']}
    reasons=[];comparisons={};gaps={};models={}
    for role,label in [('student','Raw'),('ema','EMA')]:
        def read(step,kind):return json.loads((output/f'step-{step}'/role/kind/'metrics.json').read_text())
        hard,soft,hard0,soft0=read(args.step,'hard'),read(args.step,'soft'),read(0,'hard'),read(0,'soft')
        previous=json.loads((baseline/role/'metrics.json').read_text())
        # Confirm fork baseline really produces the same hard images, not just
        # compatible shapes or similar scores. PNG content is deterministic.
        a={(r['sample_uid'],r['seed']):r for r in rows_at(output/'step-0'/role/'hard/predictions.jsonl')}
        b={(r['sample_uid'],r['seed']):r for r in rows_at(baseline/role/'predictions.jsonl')}
        if any(sha256_file(a[k]['output'])!=sha256_file(b[k]['output']) for k in a):
            reasons.append(f'{label}: fork step0 hard images differ from original4000; investigate baseline')
        comparison=compare_metrics(hard,previous,options);comparisons[label]=comparison
        for key,m in comparison.items():
            lo,hi=m['bootstrap_95_ci']
            if (lo>0 if 'lpips' in key else hi<0):reasons.append(f'{label}: significant hard regression in {key}')
        gap=decode_gap_delta(hard,soft,hard0,soft0,options);gaps[label]=gap
        for key,m in gap.items():
            threshold=.01 if 'lpips' in key else .02
            if m['bootstrap_95_ci'][0]>threshold:reasons.append(f'{label}: widening soft/hard gap in {key}')
        for model,result in [(f'Original {label} 4000',previous),(f'Soft-DiMO {label} {args.step} HARD',hard),(f'Soft-DiMO {label} {args.step} SOFT diagnostic',soft)]:
            models[model]=result['aggregate']['dataset:magicbrush_dev']
    gate=binding|{'status':'STOP_REVIEW' if reasons else 'PASS','checkpoint_step':args.step,'reasons':reasons,
        'checkpoint_completed_sha256':sha256_file(run/f'inference/checkpoint-{args.step}/COMPLETED.json'),
        'models':models,'paired_hard_to_original4000':comparisons,'paired_soft_hard_gap_increase':gaps,
        'gap_thresholds':{'lpips':.01,'clip':.02,'dino':.02},
        'criteria':'cluster-bootstrap 95% CI; statistical non-significance is not proof of equivalence'}
    write_json(output/f'QUALITY_GATE_step{args.step}.json',gate)
    paths={'Teacher 1 (CFG1)':baseline/'teacher-one-step/predictions.jsonl',
        'Raw4000':baseline/'student/predictions.jsonl',f'Soft Raw{args.step} HARD':output/f'step-{args.step}/student/hard/predictions.jsonl',
        'EMA4000':baseline/'ema/predictions.jsonl',f'Soft EMA{args.step} HARD':output/f'step-{args.step}/ema/hard/predictions.jsonl'}
    comparison_visual(paths,output/f'comparison_step{args.step}.jpg',count=8)
    comparison_visual(paths,output/f'comparison_step{args.step}_preview.jpg',count=3)
    report=[f'# Soft-DiMO Reward step {args.step} 固定DEV', '', f"Gate: {gate['status']}", '',
        '| Model | Inside LPIPS ↓ | Outside LPIPS ↓ | ROI CLIP ↑ | ROI DINO ↑ |','|---|---:|---:|---:|---:|']
    for name,m in models.items():report.append('| '+name+' | '+' | '.join(f"{m[k]['mean_delta']:.5f}" for k in KEYS)+' |')
    report+=['','Reasons: '+('; '.join(reasons) or 'No detected significant hard regression or widening gap.'),
        '', 'Soft rows are diagnostics only. All quality decisions use categorical hard tokens. Raw/EMA: CFG1, r_init1, sigma0.3; 128 cases, seeds0/1; 106 source clusters.',
        '',f'![固定前8例](comparison_step{args.step}.jpg)']
    (output/f'REPORT_step{args.step}.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({'status':gate['status'],'step':args.step,'reasons':reasons,'report':str(output/f'REPORT_step{args.step}.md')}))


if __name__=='__main__':main()
