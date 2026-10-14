"""Inference and image metrics only; all weights remain read-only."""
from pathlib import Path
import argparse,gc,json,time
import numpy as np
from PIL import Image
import torch
from dimo_step0_runtime import make_generator,configure_torch,state_digest,sha,LABEL


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--previous-eval',type=Path,required=True)
    p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--stage',choices=['pilot128','full528'],required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();configure_torch()
    if a.stage=='full528':
        deadline=time.time()+300
        while not (a.work_root/'PREFLIGHT.json').exists():
            if time.time()>deadline:raise RuntimeError('ORIGINAL_INFERENCE_PREFLIGHT_TIMEOUT')
            time.sleep(1)
        preflight=json.loads((a.work_root/'PREFLIGHT.json').read_text())
        if preflight['status']!='PASS' or not preflight['student_unchanged']:raise RuntimeError('ORIGINAL_INFERENCE_PREFLIGHT_REQUIRED')
    manifest=a.previous_eval/'data'/f'{a.stage}.jsonl'
    rows=[json.loads(l) for l in manifest.read_text().splitlines()][a.shard::a.shards]
    out=a.work_root/a.stage/f'shard{a.shard}';out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETED.json').exists():print('ALREADY_COMPLETE',flush=True);return
    g,legacy=make_generator(a.previous_eval)
    legacy.atomic_json(out/'identity.json',g.identity)
    g.generate(rows[0],0,.5)
    records={};old={}
    for row in json.loads((a.previous_eval/'results'/a.stage/'raw/metrics.json').read_text())['per_generation']:
        old[(row['sample_key'],row['seed'])]=row
    for variant,rinit in [('r05',.5),('r10',1.)]:
        variant_out=out/variant;variant_out.mkdir(exist_ok=True)
        completed=[]
        with (variant_out/'predictions.jsonl').open('w') as f:
            for i,row in enumerate(rows):
                for seed in [0,1]:
                    dest=variant_out/f"{row['sample_key']}_seed{seed}.png"
                    cached=a.work_root/'pilot128/shard0'/variant/dest.name
                    cached_manifest=cached.parent/'predictions.jsonl'
                    if a.stage=='full528' and cached.exists():
                        if not hasattr(main,'pilot_records'):
                            main.pilot_records={}
                        if variant not in main.pilot_records:
                            main.pilot_records[variant]={(r['sample_key'],r['seed']):r for r in map(json.loads,cached_manifest.read_text().splitlines())}
                        record=main.pilot_records[variant][(row['sample_key'],seed)].copy()
                        for field in ['source_sha256','target_sha256','region_sha256','sample_uid']:
                            if row[field]!=record[field]:raise RuntimeError('PILOT_REUSE_IDENTITY_MISMATCH')
                        record.update(manifest_sha256=sha(manifest),reused_exact_pilot_output=True)
                    else:
                        image,recon,meta=g.generate(row,seed,rinit)
                        image.save(dest)
                        old_row=old[(row['sample_key'],seed)]
                        if not np.array_equal(np.asarray(recon),np.asarray(Image.open(old_row['reconstruction']).convert('RGB'))):
                            raise RuntimeError('VQ_BASELINE_NOT_EXACTLY_PREVIOUS_RECONSTRUCTION')
                        record=row|meta|{'seed':seed,'output':str(dest),'reconstruction':old_row['reconstruction'],
                            'manifest_sha256':sha(manifest),'vq_baseline_pixel_identity_verified':True}
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();completed.append(record)
                print(f'GENERATED {a.stage} shard{a.shard} {variant} {i+1}/{len(rows)}',flush=True)
        records[variant]=completed
    final=state_digest(g.roles.role_state_dict('student'))
    if final!=g.identity['role_state_sha256']['student']:raise RuntimeError('STUDENT_CHANGED_DURING_INFERENCE')
    legacy.atomic_json(out/'weight_integrity.json',{'student_unchanged':True,'role_state_sha256':final})
    # Release inference modules before loading the independent metric models.
    legacy.one_step_edit_tokens=lambda *args,**kwargs:None
    g.vq.decode=g.original_decode;g.handle.remove();del g;gc.collect();torch.cuda.empty_cache()
    config={'generation_seeds':[0,1],'bootstrap':{'seed':42,'resamples':10000},'selection':{},
        'embedding_models':{'dino':str(a.previous_eval/'assets/dinov2-base'),'clip':str(a.previous_eval/'assets/clip-vit-large-patch14')}}
    for variant,completed in records.items():
        dest=out/variant/'metrics.json'
        result=legacy.evaluate(out/variant/'predictions.jsonl',config,dest,'cuda')
        for record,m in zip(completed,result['per_generation']):
            baseline=old[(record['sample_key'],record['seed'])]['metrics']
            for key in ('vq_noedit_inside_lpips_to_target','vq_noedit_outside_lpips_to_source'):
                m['metrics'][key]=baseline[key]
            m['metrics']['editing_gain_lpips']=baseline['vq_noedit_inside_lpips_to_target']-m['metrics']['inside_masked_lpips']
            for key in ['transformer_forwards','outside_token_mismatch_count','initial_outside_token_mismatch_count',
                        'initial_roi_token_count','initial_mask_count','initial_random_count','initial_actual_mask_ratio']:
                m['metrics'][key]=record[key]
        # Independent-row aggregate in legacy evaluator is not the final CI.
        result['aggregate_note']='Use final source-group paired bootstrap reports; this legacy aggregate is not used for conclusions.'
        legacy.atomic_json(dest,result)
        print(f'METRICS_COMPLETE {a.stage} shard{a.shard} {variant}',flush=True)
    from src.explicit_region.metrics import MaskedLPIPS,masked_l1,masked_ssim
    from scripts.eval.formal_eval import tensor
    lpips=MaskedLPIPS(device='cuda');changes=[]
    for ra,rb in zip(records['r05'],records['r10']):
        for field in ['sample_key','seed','source_sha256','target_sha256','region_sha256',
                      'embedding_noise_sha256','init_mask_seed','init_token_seed','categorical_sample_seed']:
            if ra[field]!=rb[field]:raise RuntimeError('RINIT_ABLATION_CHANGED_EXTRA_FACTOR: '+field)
        x=tensor(ra['output']).to('cuda');y=tensor(rb['output']).to('cuda');m=tensor(ra['mask'],mask=True).to('cuda')
        values={'sample_key':ra['sample_key'],'seed':ra['seed'],'cluster_id':ra['cluster_id']}
        for label,region in [('inside',m),('outside',~m)]:
            for name,fn in [('l1',masked_l1),('ssim',masked_ssim),('lpips',lpips)]:
                v,valid=fn(x,y,region);values[f'output_change_{label}_{name}']=float(v[0]) if valid[0] else None
        values['output_change_full_l1']=float((x-y).abs().mean())
        changes.append(values)
    legacy.atomic_json(out/'output_changes.json',{'per_generation':changes,'only_rinit_changed_verified':True})
    legacy.atomic_json(out/'COMPLETED.json',{'checkpoint_status':LABEL,'samples':len(rows),'variants':['r05','r10'],
        'student_unchanged':True,'stage':a.stage,'shard':a.shard,'shards':a.shards})


if __name__=='__main__':main()
