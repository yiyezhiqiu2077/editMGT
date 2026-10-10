#!/usr/bin/env python3
"""Separate static merge semantics from FP32/BF16 evaluation roundoff."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from types import MethodType
from diffusers.models.normalization import RMSNorm
from peft.tuners.tuners_utils import BaseTunerLayer
from merge_e3_dense import inputs, Forward
from src.dimo.conditioning import predict_role_logits
from src.explicit_region.dense_inference import load_evaluation_pipeline
from src.explicit_region.dense_checkpoint import write_json
from scripts.train.train_dense_region import configure_determinism


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model-root',required=True);p.add_argument('--lora',required=True)
    p.add_argument('--dev-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    configure_determinism();torch.set_num_threads(4)
    # Independent precision audit, not a replacement for the failed CUDA admission.
    tolerance={'maximum_absolute_error': 1e-8, 'relative_rms_error': 1e-10}
    write_json(a.output.with_suffix('.preregistered.json'),{'precision':'fp64','tolerance':tolerance,
              'training_admission_override':False})
    pipe=load_evaluation_pipeline(a.model_root,backend='lora',checkpoint=a.lora)
    batch=inputs(pipe,json.loads(a.dev_manifest.read_text().splitlines()[0]))
    model=pipe.transformer.double()
    # Diffusers RMSNorm explicitly computes variance in FP32 even with FP64
    # weights. Use the same formula in FP64 ONLY in this independent reference.
    patched=[]
    def reference_rms(self, hidden):
        value=hidden*torch.rsqrt(hidden.square().mean(-1,keepdim=True)+self.eps)
        if self.weight is not None:
            value=value*self.weight
            if self.bias is not None: value=value+self.bias
        return value
    for module in model.modules():
        if isinstance(module,RMSNorm):
            patched.append((module,module.forward))
            module.forward=MethodType(reference_rms,module)
    import src.v2_utils as attention_runtime
    def reference_rotary(x, freqs, use_real=True, use_real_unbind_dim=-1):
        if not use_real or use_real_unbind_dim != -1: raise RuntimeError('UNTESTED_ROTARY_REFERENCE')
        real,imag=x.reshape(*x.shape[:-1],-1,2).unbind(-1)
        rotated=torch.stack((-imag,real),dim=-1).flatten(3)
        cos,sin=(v.to(device=x.device,dtype=x.dtype)[None,None] for v in freqs)
        return x*cos+rotated*sin
    attention_runtime.apply_rotary_emb=reference_rotary
    sampled={}
    for name,module in model.named_modules():
        if isinstance(module,BaseTunerLayer):
            key=module.active_adapters[0]
            if key not in module.lora_A: raise RuntimeError('UNSUPPORTED_MERGE_MODULE')
            base=module.base_layer.weight[:16,:16].clone()
            left=module.lora_B[key].weight[:16]
            right=module.lora_A[key].weight[:,:16]
            sampled[name]=(base+left@right*module.scaling[key]).cpu()
    z=batch['source_tokens'].clone();z[batch['edit_region_mask']]=8255
    conditions={k:{n:v.double() for n,v in sub.items()} for k,sub in batch['prompt_condition'].items()}
    kwargs=dict(target_tokens=z,reference_tokens=batch['source_tokens'],prompt_condition=conditions,
                region_mask=batch['edit_region_mask'],timestep=torch.ones(1,device='cuda',dtype=torch.float64),
                cfg_scale=5.,model_kwargs={k:v.double() if isinstance(v,torch.Tensor) and v.is_floating_point() else v
                                          for k,v in batch['model_kwargs'].items()},training=False)
    before=predict_role_logits(Forward(model,False),'teacher',**kwargs).cpu()
    model.fuse_lora(safe_fusing=True);model.unload_lora()
    named=dict(model.named_modules())
    weight_errors={k:float((named[k].weight[:16,:16].cpu()-v).abs().max()) for k,v in sampled.items()}
    after=predict_role_logits(Forward(model,False),'teacher',**kwargs).cpu()
    delta=before-after
    result={'precision':'FP64', 'tested_cfg':5.,'tested_roi_mask_ratio':1.,'dual_branches':True,
            'tested_modules':len(sampled),'sampled_weight_max_absolute_error':max(weight_errors.values()),
            'reference_only_fp64_rmsnorm_modules':len(patched),
            'reference_only_fp64_rotary':True,
            'production_rmsnorm_changed':False,
            'weight_errors':weight_errors,'logits_max_absolute_error':float(delta.abs().max()),
            'logits_relative_rms_error':float(delta.norm()/before.norm()),
            'tolerance':tolerance,'training_admission_override':False,
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    result['static_semantics_status']='PASS' if result['logits_max_absolute_error']<=1e-8 and result['logits_relative_rms_error']<=1e-10 else 'FAIL'
    write_json(a.output,result);print(json.dumps({k:v for k,v in result.items() if k!='weight_errors'}),flush=True)


if __name__=='__main__':main()
