"""Read-only LoRA step0 reconstruction using the pinned compatible runtime."""
from __future__ import annotations
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

COMPAT_REVISION='3132e24355f947aaeb604278514d9460806de0b6'
LABEL='RECONSTRUCTED_STEP0'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def state_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        v=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_legacy(previous):
    previous=Path(previous).resolve()
    code=previous/'code'
    if (code/'.git').exists():
        actual=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
        if actual!=COMPAT_REVISION or dirty:raise RuntimeError('COMPATIBLE_RUNTIME_IDENTITY_CHANGED')
    else:
        marker=code/'DIAGNOSIS_REVISION.json'
        package_info=previous/'PACKAGE_INFO.json'
        revision=json.loads(marker.read_text())['revision'] if marker.exists() else (
            json.loads(package_info.read_text()).get('inference_revision') if package_info.exists() else None)
        if revision!=COMPAT_REVISION:
            raise RuntimeError('Exported compatible runtime needs its pinned revision marker')
    sys.path.insert(0,str(code))
    spec=importlib.util.spec_from_file_location('step0_previous_worker',previous/'run_worker.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    return legacy


def verify_assets(previous,legacy):
    from src.explicit_region.contracts import audit_component_identity
    from src.dimo.contracts import released_base_model_identity
    previous=Path(previous)
    runtime_paths={name:str(Path(inspect.unwrap(getattr(legacy,name)).__code__.co_filename).resolve()) for name in
        ['initialize_shared_model_roles','one_step_edit_tokens','load_released_components']}
    if any(not path.startswith(str((previous/'code').resolve())+'/') for path in runtime_paths.values()):
        raise RuntimeError('COMPATIBLE_LORA_RUNTIME_IMPORT_CONTAMINATION')
    folder=previous/'assets/checkpoints'
    teacher=folder/'editmgt-e3-quality-epoch1-step6250'
    manifest=json.loads((folder/'editmgt-dimo-e3-step7500/run_manifest.json').read_text())
    expected=manifest['teacher_bundle_fingerprint']
    checks={name:sha(teacher/name) for name in expected['artifact_sha256']}
    if checks!=expected['artifact_sha256']:raise RuntimeError('TEACHER_ARTIFACT_MISMATCH')
    base=released_base_model_identity(audit_component_identity(previous/'assets/editmgt'))
    if base!=expected['base_model_identity']:raise RuntimeError('BASE_IDENTITY_MISMATCH')
    if manifest['inference_fingerprint']['base_model_identity']!=base:
        raise RuntimeError('INFERENCE_BASE_MISMATCH')
    if manifest['resolved_config']['model_roles']['backend']!='shared_base_adapters':
        raise RuntimeError('EXPECTED_LORA_SHARED_BASE_CONTRACT')
    protocol=json.loads((previous/'protocol.json').read_text())
    if protocol['implementation_revision']!=COMPAT_REVISION:raise RuntimeError('OLD_PROTOCOL_REVISION_MISMATCH')
    return teacher,manifest,{
        'checkpoint_status':LABEL,'real_step0_checkpoint_available':False,
        'global_dimo_step':0,'checkpoint_binary_created':False,
        'teacher_path':str(teacher),'teacher_artifact_sha256':checks,
        'teacher_bundle_fingerprint':expected,'base_model_identity':base,
        'lora_model_roles':manifest['resolved_config']['model_roles'],
        'legacy_inference_fingerprint':manifest['inference_fingerprint'],
        'compatible_inference_git_sha':COMPAT_REVISION,
        'runtime_function_paths':runtime_paths,
        'historical_training_git_sha':manifest['git_sha'],
        'historical_training_git_sha_available':False,
        'reconstruction_definition':'initialize_shared_model_roles copies verified Teacher adapters and region embedding exactly; no optimizer or update',
    }


def make_generator(previous,rinit=.5):
    legacy=load_legacy(previous)
    teacher,manifest,identity=verify_assets(previous,legacy)
    from src.dimo.ema import TrainableEMA
    original_one_step=legacy.one_step_edit_tokens

    class ReconstructedGenerator(legacy.Generator):
        def __init__(self):
            self.variant='raw'
            self.c=legacy.load_released_components(str(Path(previous)/'assets/editmgt'),torch_dtype=torch.bfloat16,vq_dtype=torch.float32)
            self.roles=legacy.initialize_shared_model_roles(self.c.transformer,teacher,model_roles=manifest['resolved_config']['model_roles'])
            states={role:self.roles.role_state_dict(role) for role in ['teacher','student','auxiliary']}
            for role in ['student','auxiliary']:
                if set(states[role])!=set(states['teacher']) or any(not torch.equal(v,states[role][k]) for k,v in states['teacher'].items()):
                    raise RuntimeError('RECONSTRUCTED_ROLE_NOT_EXACTLY_TEACHER')
            ema=TrainableEMA(self.roles.role_named_parameters('student'))
            if any(not torch.equal(ema.shadow[k],v.float()) for k,v in states['student'].items()):
                raise RuntimeError('RECONSTRUCTED_EMA_NOT_EXACTLY_STUDENT')
            identity.update(role_state_sha256={k:state_digest(v) for k,v in states.items()},
                role_tensor_counts={k:len(v) for k,v in states.items()},
                raw_student_exactly_teacher=True,auxiliary_exactly_teacher=True,
                ema_status='RECONSTRUCTED_INITIAL_EMA_NOT_A_SAVED_ARTIFACT',
                initial_ema_exactly_raw_student=True,ema_update_count=0,
                parameter_dtypes=sorted({str(v.dtype) for v in states['student'].values()}))
            del states,ema
            self.roles.to('cuda').eval()
            for m in [self.c.text_encoder,self.c.llm_encoder,self.c.vqvae]:m.to('cuda').eval().requires_grad_(False)
            self.vq=self.c.vqvae;self.model=self.roles.base_model
            self.calls=0;self.handle=self.model.register_forward_hook(self.count)
            self.last_tokens=None;self.last_result=None;self.last_inputs=None;self.initial_metadata=None
            self.original_decode=self.vq.decode
            def capture(tokens,*args,**kwargs):
                self.last_tokens=tokens.detach().clone()
                decoded=self.original_decode(tokens,*args,**kwargs)
                if not torch.isfinite(decoded.sample).all():raise RuntimeError('NONFINITE_DECODE')
                return decoded
            self.vq.decode=capture
            self.identity=identity
            def one_step(roles,**kwargs):
                result=original_one_step(roles,**kwargs)
                roi=kwargs['edit_region_mask'].bool();source=kwargs['source_tokens']
                mask_id=kwargs['mask_token_id'];count=int(roi.sum())
                masks=int((result.initial_tokens[roi]==mask_id).sum())
                expected=max(1,min(count,round(float(kwargs['r_init'])*count)))
                if masks!=expected:raise RuntimeError('INITIAL_MASK_COUNT_MISMATCH')
                if not torch.equal(result.initial_tokens[~roi],source[~roi]):raise RuntimeError('INITIAL_OUTSIDE_LOCK_FAILURE')
                self.initial_metadata={'initial_roi_token_count':count,'initial_mask_count':masks,
                    'initial_random_count':count-masks,'initial_actual_mask_ratio':masks/count,
                    'initial_outside_token_mismatch_count':0,'student_logits_dtype':str(result.logits.dtype),
                    'embedding_noise_sha256':state_digest({'noise':kwargs['target_embedding_noise']}),
                    'init_mask_seed':kwargs['init_mask_seeds'],'init_token_seed':kwargs['init_token_seeds'],
                    'categorical_sample_seed':kwargs['sample_seeds']}
                self.last_result=result;self.last_inputs=kwargs
                return result
            legacy.one_step_edit_tokens=one_step

        @torch.inference_mode()
        def generate(self,row,seed,rinit=rinit):
            from PIL import Image
            # Cache deterministic conditioning, not logits or stochastic outputs.
            # Geometry, text encoding, source tokens and the inference call are
            # copied exactly from the previous evaluator. Timing excludes cache.
            if getattr(self,'_cached_uid',None)!=row['sample_uid']:
                source_pil=Image.open(row['source']).convert('RGB')
                mask_pil=Image.open(row['mask']).convert('L')
                source=legacy.image_to_tensor(source_pil)[None].to('cuda',dtype=torch.float32)
                tokens=legacy.prepare_cond_token(None,source,self.vq)
                grid=int(math.sqrt(tokens.shape[1]));tokens=tokens.reshape(1,grid,grid)
                region=legacy.pixel_mask_to_token_mask(torch.from_numpy(np.asarray(mask_pil)>0)[None],(grid,grid),
                    mode='any_overlap',dilation_tokens=0,minimum_edit_tokens=1).to('cuda')
                recon=self.original_decode(tokens,force_not_quantize=True,
                    shape=(1,grid,grid,self.vq.config.latent_channels)).sample.clip(0,1)
                recon_pil=legacy.to_pil(recon)
                prompt_ids=legacy.tokenize_prompt([self.c.tokenizer,self.c.llm_tokenizer],
                    [row['instruction_en']],'CLIP_Gemma2',device='cuda')
                hidden,pooled=legacy.encode_prompt([self.c.text_encoder,self.c.llm_encoder],prompt_ids,'CLIP_Gemma2')
                ids=torch.zeros(grid//2,grid//2,3,device='cuda',dtype=tokens.dtype)
                ids[...,1]+=torch.arange(grid//2,device='cuda')[:,None]
                ids[...,2]+=torch.arange(grid//2,device='cuda')[None,:]
                ids=ids.reshape(-1,3)
                text_ids=torch.zeros(legacy.get_text_encoder_length('CLIP_Gemma2',return_main=True),3,device='cuda',dtype=tokens.dtype)
                micro=torch.tensor([1024,1024,0,0,6.0],device='cuda',dtype=hidden.dtype)[None]
                self._cached=(tokens,region,recon_pil,mask_pil,grid,hidden,pooled,ids,text_ids,micro)
                self._cached_uid=row['sample_uid']
            tokens,region,recon_pil,mask_pil,grid,hidden,pooled,ids,text_ids,micro=self._cached
            def rng(namespace):return legacy.stable_seed(seed,0,row['sample_uid'],0,namespace)
            noise=legacy.normal_noise_per_sample(torch.empty(1,self.model.inner_dim,grid,grid,device='cuda',dtype=torch.bfloat16),[rng('dimo-embedding-noise')])
            self.calls=0;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            result=legacy.one_step_edit_tokens(self.roles,source_tokens=tokens,edit_region_mask=region,
                prompt_condition={'conditional':{'encoder_hidden_states':hidden,'pooled_projections':pooled}},
                timestep_model_kwargs={'img_ids':ids,'txt_ids':text_ids,'micro_conds':micro,
                    'reference_image_ids':ids,'reference_image_timestep':0,'lora_scope':'both'},
                mask_token_id=self.model.config.vocab_size-1,codebook_size=self.model.config.codebook_size,
                r_init=float(rinit),init_mask_seeds=rng('dimo-init-mask'),init_token_seeds=rng('dimo-init-token'),
                sample_seeds=rng('dimo-student-sample'),temperature=1.,top_k=0,top_p=0,cfg_scale=1.,
                target_embedding_noise=noise,embedding_noise_sigma=.3)
            image=legacy.to_pil(self.vq.decode(result.tokens,force_not_quantize=True,
                shape=(1,grid,grid,self.vq.config.latent_channels)).sample)
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            mismatch=int((result.tokens[~region]!=tokens[~region]).sum())
            if mismatch or self.calls!=1:raise RuntimeError('ONE_STEP_OR_EXTERIOR_LOCK_FAILED')
            exterior=~(np.asarray(mask_pil)>0)
            a=np.asarray(image,dtype=np.float32)/255;b=np.asarray(recon_pil,dtype=np.float32)/255
            meta={'runtime_seconds':elapsed,'runtime_scope':'student_forward_and_decode; deterministic conditioning cached',
                'transformer_forwards':self.calls,'outside_token_mismatch_count':mismatch,
                'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30,
                'outside_l1_to_vq_reconstruction':float(np.abs(a-b)[exterior].mean()) if exterior.any() else None}
            return image,recon_pil,meta|self.initial_metadata|{'r_init':float(rinit),'checkpoint_status':LABEL}

    return ReconstructedGenerator(),legacy


def configure_torch():
    if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible GPU required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
