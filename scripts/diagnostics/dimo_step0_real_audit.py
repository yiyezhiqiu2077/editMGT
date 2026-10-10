"""Numerical audit on detached real logits, without training or weight saves."""
from contextlib import contextmanager
from pathlib import Path
import argparse,json,random,importlib.util
import numpy as np
import torch
from dimo_step0_runtime import make_generator,configure_torch,state_digest,sha,LABEL
from surrogate_precision import audit,compare


@contextmanager
def fp32_forward(roles):
    # Upcast only BF16 tensors. FP32 LoRA parameters retain all original bits.
    changed=[]
    for value in list(roles.parameters())+list(roles.buffers()):
        if value.dtype==torch.bfloat16:
            changed.append((value,value.data));value.data=value.data.float()
    try:yield
    finally:
        for value,original in changed:value.data=original


def floating_fp32(obj):
    if isinstance(obj,torch.Tensor):return obj.float() if obj.is_floating_point() else obj
    if isinstance(obj,dict):return {k:floating_fp32(v) for k,v in obj.items()}
    return obj


def fkl64(teacher,auxiliary,mask):
    return (auxiliary.double().softmax(-1)-teacher.double().softmax(-1)).masked_fill(~mask[...,None],0)


def summary(tensor,mask=None):
    t=tensor.detach()
    if mask is not None:t=t[mask]
    flat=t.flatten()
    quantile_sample=flat[::max(1,(flat.numel()+1048575)//1048576)].double().abs()
    return {'dtype':str(t.dtype),'shape':list(t.shape),'nonfinite_count':int((~torch.isfinite(t)).sum()),
        'norm':float(t.double().norm()),'max_absolute':float(t.abs().max()),
        'zero_fraction':float((t==0).double().mean()),
        'abs_quantiles':torch.quantile(quantile_sample,torch.tensor([0.,.25,.5,.75,.9,.99,1.],device=t.device,dtype=torch.float64)).tolist(),
        'quantile_sample_count':quantile_sample.numel(),'quantile_policy':'deterministic evenly strided sample <=1048576; norm/max/zero counts use all coordinates'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--previous-eval',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=int,default=4)
    a=p.parse_args();configure_torch();a.output.mkdir(parents=True,exist_ok=True)
    g,legacy=make_generator(a.previous_eval)
    from src.dimo.conditioning import predict_role_logits
    from src.dimo.forward_process import mask_student_prediction,schedule_mask_ratio
    from src.dimo.rng import stable_seed
    from src.dimo.divergence import dimo_divergence_gradient
    from src.dimo.surrogate import surrogate_logit_loss
    before={role:state_digest(g.roles.role_state_dict(role)) for role in ['teacher','student','auxiliary']}
    whole_model_before=state_digest(g.roles.state_dict())
    requires_grad_before={n:p.requires_grad for n,p in g.roles.named_parameters()}
    training_before={n:m.training for n,m in g.roles.named_modules()}
    adapter_before=list(g.model.active_adapters())
    if len(adapter_before)!=1:raise RuntimeError('Audit expects exactly one active adapter')
    raw_rows=[json.loads(l) for l in (a.previous_eval/'data/pilot128.jsonl').read_text().splitlines()][:a.cases]
    report={'checkpoint_status':LABEL,'identity':g.identity,'rows':[], 'cases':[],
        'model_parameter_backward':False,'optimizer_created':False,'optimizer_steps':0,
        'original_divergence_uses_nan_to_num':True,'controlled_g_is_not_training_g':True}
    python_rng=random.getstate();numpy_rng=np.random.get_state()
    cpu_rng=torch.random.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    with torch.random.fork_rng(devices=[0]):
        for index,row in enumerate(raw_rows):
            print(f'AUDIT_CASE {index+1}/{len(raw_rows)} {row["sample_key"]}',flush=True)
            image,_,_=g.generate(row,0,.5)
            if index==0:
                cached_logits=g.last_result.logits.detach().clone()
                legacy.PROTOCOL['dimo']['r_init']=.5
                original_image,_,_=legacy.Generator.generate(g,row,0)
                equal_images=np.array_equal(np.asarray(image),np.asarray(original_image))
                equal_logits=torch.equal(cached_logits,g.last_result.logits)
                report['cached_conditioning_matches_original_inference_exactly']={
                    'image_pixels_bitwise_equal':equal_images,'student_logits_bitwise_equal':equal_logits,
                    'sample_key':row['sample_key'],'seed':0,'r_init':.5}
                if not equal_images or not equal_logits:raise RuntimeError('CACHED_CONDITIONING_CHANGED_INFERENCE')
                del cached_logits
            inputs=g.last_inputs;student=g.last_result.logits.detach().clone()
            roi=inputs['edit_region_mask'];source=inputs['source_tokens']
            seed=stable_seed(0,0,row['sample_uid'],0,'dimo-teacher-query-mask')
            rho=schedule_mask_ratio(seed=seed,mode='cosine',r_min=.02,r_max=.98)
            query=mask_student_prediction(g.last_result.tokens,source,roi,rho,g.model.config.vocab_size-1,seeds=seed+1)
            mask=query.pseudo_mask
            common=dict(inputs['timestep_model_kwargs'])
            def predict(role,full_precision=False):
                condition=inputs['prompt_condition']
                kw=common
                if full_precision:condition=floating_fp32(condition);kw=floating_fp32(kw)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=not full_precision):
                    return predict_role_logits(g.roles,role,query.pseudo_tokens,source,condition,roi,
                        query.rho_actual.to(source.device),1.,model_kwargs=kw,training=False).detach()
            teacher=predict('teacher');auxiliary=predict('auxiliary')
            original=dimo_divergence_gradient(teacher,auxiliary,pseudo_mask=mask)
            case={'sample_key':row['sample_key'],'seed':0,'pseudo_rho':rho,
                'pseudo_selected_tokens':int(mask.sum()),'roi_tokens':int(roi.sum()),
                'student_logits':summary(student),'teacher_logits':summary(teacher),
                'teacher_aux_logits_bitwise_equal':torch.equal(teacher,auxiliary),
                'original_real_g':summary(original,mask),'distribution_tests':[], 'perturbation_tests':[]}
            for mode in ['FKL','RKL','Jeffreys']:
                measured=dimo_divergence_gradient(teacher,auxiliary,mode=mode,pseudo_mask=mask)
                lt=teacher.float().log_softmax(-1);la=auxiliary.float().log_softmax(-1)
                pt,pa=lt.exp(),la.exp();fkl=pa-pt
                rkl=pa*(la-lt-(pa*(la-lt)).sum(-1,keepdim=True))
                raw=fkl if mode=='FKL' else rkl if mode=='RKL' else .5*(fkl+rkl)
                case['distribution_tests'].append({'mode':mode,'raw_nonfinite_logits':int((~torch.isfinite(teacher)).sum()+(~torch.isfinite(auxiliary)).sum()),
                    'raw_distribution_gradient_nonfinite_count_before_nan_to_num':int((~torch.isfinite(raw)).sum()),
                    'original_g':summary(measured,mask)})
                del lt,la,pt,pa,fkl,rkl,raw,measured
            report['rows']+=audit(student,original,mask,{'kind':'real_original_g','sample_key':row['sample_key']})
            # Verify this is precisely the legacy production squared objective.
            probe=student.detach().float().requires_grad_(True)
            injected=torch.ones_like(probe)*1e-6
            old_grad=torch.autograd.grad(surrogate_logit_loss(probe,injected,mask),probe)[0]
            from surrogate_precision import squared
            probe2=probe.detach().clone().requires_grad_(True)
            local_grad=torch.autograd.grad(squared(probe2,injected,mask),probe2)[0]
            if not torch.equal(old_grad,local_grad):raise RuntimeError('DIAGNOSIS_SQUARED_DIFFERS_FROM_LEGACY')
            case['diagnosis_squared_matches_legacy_exactly']=True
            del probe,probe2,injected,old_grad,local_grad
            generator=torch.Generator(device=student.device).manual_seed(734+index)
            controlled=torch.randn(student.shape,device=student.device,dtype=torch.float32,generator=generator)
            # Distribution gradients sum to zero over vocabulary.
            controlled-=controlled.mean(-1,keepdim=True)
            controlled=controlled.masked_fill(~mask[...,None],0)
            for scale in [10.**(-i) for i in range(2,9)]:
                report['rows']+=audit(student,controlled*scale,mask,{'kind':'real_logits_controlled_nonzero_g',
                    'sample_key':row['sample_key'],'g_magnitude':scale})
            del controlled,auxiliary
            # Small reversible perturbation to one real Auxiliary output adapter.
            candidates=[(n,p) for n,p in g.roles.role_named_parameters('auxiliary') if 'lora_B' in n and 'proj_out' in n]
            if not candidates:candidates=[(n,p) for n,p in g.roles.role_named_parameters('auxiliary') if 'lora_B' in n]
            name,parameter=candidates[-1];saved=parameter.detach().clone()
            generator=torch.Generator(device=parameter.device).manual_seed(921+index)
            direction=torch.randn(parameter.shape,device=parameter.device,dtype=torch.float32,generator=generator)
            direction/=direction.square().mean().sqrt()
            with fp32_forward(g.roles):
                teacher32=predict('teacher',True)
                auxiliary_original32=predict('auxiliary',True)
            case['original_fp32_teacher_aux_logits_bitwise_equal']=torch.equal(teacher32,auxiliary_original32)
            case['original_fp32_distribution_g']=summary(fkl64(teacher32,auxiliary_original32,mask),mask)
            del auxiliary_original32
            try:
                for scale in [1e-5,1e-4,1e-3]:
                    with torch.no_grad():parameter.copy_((saved.float()+scale*direction).to(parameter.dtype))
                    auxiliary_bf=predict('auxiliary')
                    gb=dimo_divergence_gradient(teacher,auxiliary_bf,pseudo_mask=mask)
                    with fp32_forward(g.roles):auxiliary32=predict('auxiliary',True)
                    gf=dimo_divergence_gradient(teacher32,auxiliary32,pseudo_mask=mask)
                    reference=fkl64(teacher32,auxiliary32,mask)
                    bf_reference=fkl64(teacher,auxiliary_bf,mask)
                    perturb={'parameter':name,'perturbation_rms_requested':scale,
                        'actual_parameter_delta_rms':float((parameter.float()-saved.float()).square().mean().sqrt()),
                        'bf16_forward_g':summary(gb,mask),'fp32_forward_g':summary(gf,mask),
                        'fp32_distribution_algebra_vs_fp64_same_logits':compare(gf,reference),
                        'bf16_distribution_algebra_vs_fp64_same_logits':compare(gb,bf_reference),
                        'bf16_forward_vs_fp32_forward_distribution_gradient':compare(gb,reference),
                        'bf16_logits_equal_fraction_in_pseudo_roi':float((teacher[mask]==auxiliary_bf[mask]).double().mean())}
                    case['perturbation_tests'].append(perturb)
                    report['rows']+=audit(student,gb,mask,{'kind':'real_auxiliary_perturbation_bf16_g',
                        'sample_key':row['sample_key'],'perturbation_magnitude':scale})
                    report['rows']+=audit(student,gf,mask,{'kind':'real_auxiliary_perturbation_fp32_g',
                        'sample_key':row['sample_key'],'perturbation_magnitude':scale})
                    del auxiliary_bf,auxiliary32,gb,gf,reference,bf_reference
            finally:
                with torch.no_grad():parameter.copy_(saved)
            case['auxiliary_perturbation_restored_exactly']=torch.equal(parameter,saved)
            report['cases'].append(case)
            del teacher,teacher32,student,original,saved,direction
    random.setstate(python_rng);np.random.set_state(numpy_rng)
    g.model.set_adapter(adapter_before[0])
    for n,p in g.roles.named_parameters():p.requires_grad_(requires_grad_before[n])
    for n,m in g.roles.named_modules():m.training=training_before[n]
    after={role:state_digest(g.roles.role_state_dict(role)) for role in before}
    report['model_state_restored_exactly']=before==after
    report['whole_transformer_and_role_state_restored_exactly']=whole_model_before==state_digest(g.roles.state_dict())
    report['model_flags_restored_exactly']=all(p.requires_grad==requires_grad_before[n] for n,p in g.roles.named_parameters()) and all(m.training==training_before[n] for n,m in g.roles.named_modules())
    report['active_adapter_restored_exactly']=list(g.model.active_adapters())==adapter_before
    report['model_parameter_grads_all_none']=all(p.grad is None for p in g.roles.parameters())
    report['rng_restored_exactly']=bool(torch.equal(cpu_rng,torch.random.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state()))
    report['role_state_sha256_before']=before;report['role_state_sha256_after']=after
    if not all(report[k] for k in ['model_state_restored_exactly','whole_transformer_and_role_state_restored_exactly',
            'model_flags_restored_exactly','active_adapter_restored_exactly','model_parameter_grads_all_none','rng_restored_exactly']):raise RuntimeError('AUDIT_RESTORATION_FAILED')
    # Deliberately nonfinite inputs: test old sanitization on a tiny independent tensor.
    z=torch.zeros(1,2,4,device='cuda');other=z.clone();other[0,0,0]=float('nan')
    m=torch.tensor([[True,False]],device='cuda')
    sanitized=dimo_divergence_gradient(z,other,pseudo_mask=m)
    probability=other.float().log_softmax(-1).exp()
    report['nan_to_num_injection_test']={'input_nan_count':int(torch.isnan(other).sum()),
        'raw_probability_nonfinite_count':int((~torch.isfinite(probability)).sum()),
        'legacy_sanitized_output_nonfinite_count':int((~torch.isfinite(sanitized)).sum()),
        'legacy_sanitized_output_all_zero':bool((sanitized==0).all()),
        'real_finite_forward_observed_nonfinite':sum(c['teacher_logits']['nonfinite_count'] for c in report['cases'])}
    specification=importlib.util.spec_from_file_location('diagnosis_maintenance_divergence',
        Path(__file__).resolve().parents[2]/'src/dimo/divergence.py')
    maintenance=importlib.util.module_from_spec(specification);specification.loader.exec_module(maintenance)
    report['additional_nonfinite_injection_tests']=[]
    for label,value in [('NaN',float('nan')),('positive_Inf',float('inf')),('negative_Inf',-float('inf'))]:
        auxiliary=z.clone();auxiliary[0,0,0]=value
        sanitized=dimo_divergence_gradient(z,auxiliary,pseudo_mask=m)
        failed_fast=False;message=None
        try:maintenance.dimo_divergence_gradient(z,auxiliary,pseudo_mask=m)
        except FloatingPointError as exc:failed_fast=True;message=str(exc)
        report['additional_nonfinite_injection_tests'].append({'kind':label,
            'input_nonfinite_count':int((~torch.isfinite(auxiliary)).sum()),
            'legacy_output_nonfinite_count':int((~torch.isfinite(sanitized)).sum()),
            'legacy_output_all_zero':bool((sanitized==0).all()),
            'maintenance_failed_fast':failed_fast,'maintenance_error':message})
        if not failed_fast:raise RuntimeError('MAINTENANCE_NONFINITE_GUARD_FAILED')
    extreme=torch.tensor([[[-3e38,3e38]]],device='cuda')
    opposite=-extreme;extreme_mask=torch.ones(1,1,dtype=torch.bool,device='cuda')
    sanitized=dimo_divergence_gradient(extreme,opposite,mode='RKL',pseudo_mask=extreme_mask)
    failed_fast=False;message=None
    try:maintenance.dimo_divergence_gradient(extreme,opposite,mode='RKL',pseudo_mask=extreme_mask)
    except FloatingPointError as exc:failed_fast=True;message=str(exc)
    report['additional_nonfinite_injection_tests'].append({'kind':'finite_extreme_logits_RKL',
        'input_nonfinite_count':0,'legacy_output_all_zero':bool((sanitized==0).all()),
        'log_softmax_nonfinite_count':int((~torch.isfinite(extreme.log_softmax(-1))).sum()),
        'maintenance_failed_fast':failed_fast,'maintenance_error':message})
    if not failed_fast:raise RuntimeError('MAINTENANCE_NONFINITE_DIVERGENCE_GUARD_FAILED')
    report['source_file_sha256']={n:sha(a.previous_eval/'code/src/dimo'/n) for n in ['surrogate.py','divergence.py','step.py']}
    (a.output/'surrogate_real_logits.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'REAL_AUDIT_COMPLETE rows={len(report["rows"])} restored=True',flush=True)


if __name__=='__main__':main()
