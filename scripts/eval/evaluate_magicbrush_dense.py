#!/usr/bin/env python3
"""Dense multi-step or single-role one-step inference on the frozen previous DEV geometry."""
import argparse
import gc
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.contracts import sha256_file,freeze_modules
from src.explicit_region.dense_checkpoint import verify_dense_checkpoint,write_json
from src.explicit_region.dense_inference import load_evaluation_pipeline,inference_safety,pil
from src.explicit_region.latency import repeated_latency
from src.explicit_region.dense_evaluation import comparison_visual,assert_matching_predictions,paired_comparison
from src.explicit_region.modeling import load_released_components
from src.dimo.contracts import released_base_model_identity,teacher_bundle_fingerprint
from src.dimo.inference import load_dense_inference_role
from src.dimo.one_step import one_step_edit_tokens
from src.dimo.rng import stable_seed,normal_noise_per_sample
from scripts.train.train_dimo_editing import prepare_batch
from scripts.train.train_dense_region import configure_determinism
from scripts.eval.formal_eval import evaluate,tensor


@torch.inference_mode()
def generate(plan, manifest, checkpoint, output, *, weights=None):
    output=Path(output)
    if weights:
        components=load_released_components(plan['model_root'],torch_dtype=torch.bfloat16,
                                           transformer_dtype=torch.float32,vq_dtype=torch.float32)
        bundle=teacher_bundle_fingerprint(plan['teacher_checkpoint'],
            base_model_identity=released_base_model_identity(components.identity),formal=False)
        roles,identity=load_dense_inference_role(components.transformer,checkpoint,weights=weights,teacher_bundle=bundle)
        if identity.get('experiment_mode') not in ('magicbrush_exploratory', 'mixed20k_last_sft_dimo') or identity.get('initialization')!={'mask_ratio':1.,'random_tokens_inside_roi':False}:
            raise RuntimeError('MAGICBRUSH_ONE_STEP_TRAIN_INFERENCE_IDENTITY_MISMATCH')
        roles.to('cuda').eval()
        freeze_modules([components.text_encoder,components.llm_encoder,components.vqvae])
        for module in (components.text_encoder,components.llm_encoder,components.vqvae):module.to('cuda')
        modules={'transformer':(roles.base_model,'forward'),'vq_encode':(components.vqvae,'encode'),
                 'vq_decode':(components.vqvae,'decode'),'clip':(components.text_encoder,'forward'),'gemma':(components.llm_encoder,'forward')}
    else:
        pipe=load_evaluation_pipeline(plan['model_root'],backend='dense',checkpoint=checkpoint)
        identity=verify_dense_checkpoint(checkpoint)
        modules={'transformer':(pipe.transformer,'forward'),'vq_encode':(pipe.vqvae,'encode'),
                 'vq_decode':(pipe.vqvae,'decode'),'clip':(pipe.text_encoder,'forward'),'gemma':(pipe.text_encoder_t5,'forward')}
    evaluation_identity={'checkpoint_identity':identity,'plan_sha256':recipe_fingerprint(plan),
                         'manifest_sha256':sha256_file(manifest),'weights':weights,'inference':plan['inference']}
    if (output/'COMPLETED.json').exists():
        if json.loads((output/'evaluation_identity.json').read_text())!=evaluation_identity:
            raise RuntimeError('MAGICBRUSH_EVALUATION_IDENTITY_CHANGED')
        return output/'predictions.jsonl'
    if output.exists() and any(output.iterdir()):raise RuntimeError('MAGICBRUSH_EVALUATION_PARTIAL_OUTPUT_EXISTS')
    output.mkdir(parents=True)
    write_json(output/'evaluation_identity.json',evaluation_identity)
    rows=[json.loads(l) for l in Path(manifest).read_text().splitlines()]
    predictions=[]
    for i,row in enumerate(rows):
        for field,name in (('source','source_sha256'),('target','target_sha256'),('mask','region_sha256')):
            if sha256_file(row[field])!=row[name]:raise RuntimeError('MAGICBRUSH_FROZEN_DEV_PIXELS_CHANGED')
        for seed in plan['inference']['generation_seeds']:
            if weights:
                batch={'source_image':tensor(row['source']),'edit_region_mask':tensor(row['mask'],mask=True)[:,0],
                       'instruction_en':[row['instruction_en']],'sample_uid':[row['sample_uid']]}
                args=(seed,0,row['sample_uid'],0)
                def operation():
                    prepared=prepare_batch(batch,components,{'resolution':1024,'token_mask':plan['inference']['token_mask']},torch.device('cuda'))
                    source=prepared['source_tokens'];roi=prepared['edit_region_mask']
                    noise=normal_noise_per_sample(torch.empty(source.shape[0],roles.base_model.inner_dim,*source.shape[1:],
                        device='cuda',dtype=torch.float32),[stable_seed(*args,'dimo-embedding-noise')])
                    result=one_step_edit_tokens(roles,source_tokens=source,edit_region_mask=roi,
                        prompt_condition=prepared['prompt_condition'],timestep_model_kwargs=prepared['model_kwargs'],
                        mask_token_id=roles.base_model.config.vocab_size-1,codebook_size=roles.base_model.config.codebook_size,
                        r_init=1.,init_mask_seeds=stable_seed(*args,'dimo-init-mask'),init_token_seeds=stable_seed(*args,'dimo-init-token'),
                        sample_seeds=stable_seed(*args,'dimo-student-sample'),temperature=1.,top_k=0,top_p=0.,cfg_scale=1.,
                        target_embedding_noise=noise,embedding_noise_sigma=.3)
                    if (not torch.equal(result.tokens[~roi],source[~roi]) or
                        not bool((result.initial_tokens[roi]==roles.base_model.config.vocab_size-1).all())):
                        raise RuntimeError('MAGICBRUSH_ONE_STEP_MASK_OR_OUTSIDE_LOCK_FAILURE')
                    decoded=components.vqvae.decode(result.tokens,force_not_quantize=True,
                         shape=(*source.shape,components.vqvae.config.latent_channels)).sample.clip(0,1)
                    if not torch.isfinite(decoded).all():raise RuntimeError('MAGICBRUSH_NONFINITE_DECODE')
                    return pil(decoded[0].float().cpu())
            else:
                def operation():
                    with inference_safety(pipe):
                        return pipe(prompt=row['instruction_en'],reference_image=Image.open(row['source']).convert('RGB'),
                            mask_image=Image.open(row['mask']).convert('L'),height=1024,width=1024,num_inference_steps=12,
                            guidance_scale=5.,reference_strength=1.,generator=torch.Generator('cuda').manual_seed(seed),
                            lora_scope='both',inference_timestep_mode='roi_relative').images[0]
            image,latency=repeated_latency(operation,modules,torch.device('cuda'),warmup=1 if i==0 and seed==0 else 0,repeats=1)
            expected_calls=1 if weights else 12
            if any(p['transformer']['calls']!=expected_calls for p in latency['phases']):
                raise RuntimeError('MAGICBRUSH_TRANSFORMER_FORWARD_COUNT_FAILURE')
            path=output/f'{i:04d}_seed{seed}.png';image.save(path)
            predictions.append(row|{'dataset_name':'magicbrush_dev','output':str(path.resolve()),'seed':seed,
                'manifest_sha256':sha256_file(manifest),'runtime_seconds':latency['mean_seconds'],'latency_protocol':latency,
                'token_outside_lock':'PASS','transformer_forward_count':expected_calls,
                'inference_protocol':plan['inference']|{'role':weights or 'teacher'}})
        if (i+1)%32==0:print(json.dumps({'checkpoint':str(checkpoint),'weights':weights,'samples':i+1}),flush=True)
    path=output/'predictions.jsonl';path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in predictions))
    metric_config={'generation_seeds':plan['inference']['generation_seeds'],'bootstrap':plan['bootstrap'],
                   'selection':{},'embedding_models':plan['embedding_models']}
    evaluate(path,metric_config,output/'metrics.json','cuda')
    comparison_visual({'MODEL':path},output/'cases.jpg',count=8)
    write_json(output/'COMPLETED.json',{'samples':len(rows),'generations':len(predictions),'identity_sha256':recipe_fingerprint(evaluation_identity)})
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--dimo-root',type=Path);p.add_argument('--step',type=int);p.add_argument('--dev32',action='store_true')
    a=p.parse_args();configure_determinism();torch.set_num_threads(4)
    plan=json.loads((a.experiment_root/'evaluation_plan.json').read_text())
    manifest=plan['dev32_manifest'] if a.dev32 else plan['dev_manifest']
    if a.dimo_root:
        manifest=plan['dev32_manifest'] if a.dev32 else plan['dev_manifest']
        root=a.dimo_root/'evaluation'/('dev32' if a.dev32 else 'full_dev')
        teacher=generate(plan,manifest,plan['teacher_checkpoint'],root/'teacher')
        paths={'Teacher':teacher}
        for weights in ('student','ema'):
            path=generate(plan,manifest,a.dimo_root/'inference'/f'checkpoint-{a.step}',root/f'step-{a.step}'/weights,weights=weights)
            assert_matching_predictions([teacher,path],expected_seeds=plan['inference']['generation_seeds'])
            pairs=paired_comparison(json.loads((path.parent/'metrics.json').read_text()),
                                    json.loads((teacher.parent/'metrics.json').read_text()),strict_groups=True)
            write_json(path.parent/'paired_to_teacher.json',pairs)
            paths['Raw Student' if weights=='student' else 'EMA']=path
            gc.collect();torch.cuda.empty_cache()
        comparison_visual(paths,root/f'step-{a.step}'/'comparison.jpg',count=8)
    else:
        if a.checkpoint is None or a.output is None:p.error('--checkpoint and --output required for teacher evaluation')
        generate(plan,manifest,a.checkpoint,a.output)


if __name__=='__main__':main()
