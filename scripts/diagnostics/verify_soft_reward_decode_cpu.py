#!/usr/bin/env python3
"""Real local VQGAN/reward CPU autograd evidence; never GPU quality evidence."""
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.dimo.soft_reward import build_reward_objective
from diffusers import VQModel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(43)
    vq=VQModel.from_pretrained('/data01/zhangyuyang/editmgt_eval_20261010/assets/editmgt',
        subfolder='vqvae',local_files_only=True,torch_dtype=torch.float32)
    config={'models':{'dino':'/data01/zhangyuyang/editmgt_eval_20261010/assets/dinov2-base',
        'clip':'/data01/zhangyuyang/editmgt_eval_20261010/assets/clip-vit-large-patch14'},
        'lpips_resolution':512,'calibration':{'batches':4,'relative_scale':.1,'max_weight':100.},'gap_interval':20}
    engine=build_reward_objective(vq,config,'cpu')
    source=torch.arange(16).reshape(1,4,4).long()
    region=torch.zeros_like(source,dtype=torch.bool);region[:,1:3,1:3]=True
    logits=torch.randn(1,4,4,8193,requires_grad=True)
    image=engine.decoder(logits,source,region)
    mask=torch.nn.functional.interpolate(region[:,None].float(),size=image.shape[-2:],mode='nearest')
    with torch.no_grad():target=engine.decoder.hard((source+17)%8192)
    losses=engine.rewards(image,target,mask)
    gradients={k:torch.autograd.grad(v,logits,retain_graph=True)[0] for k,v in losses.items()}
    for k,g in gradients.items():
        assert torch.isfinite(g).all() and g.norm()>0,(k,g.norm())
        assert torch.count_nonzero(g[~region])==0
        assert torch.count_nonzero(g[...,8192])==0
    sum(losses.values()).backward();engine.assert_frozen()
    with torch.no_grad():
        peaked=torch.full_like(logits,-50.);peaked.scatter_(-1,source[...,None],50.)
        soft=engine.decoder(peaked,source,region)
        hard=engine.decoder.hard(source)
        native=vq.decode(source,force_not_quantize=True,shape=(1,4,4,64)).sample.clamp(0,1)
        torch.testing.assert_close(soft,hard,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(hard,native,atol=1e-6,rtol=1e-5)
    report={'status':'PASS','evidence':'REAL_MODELS_CPU_SMALL_GRID_NOT_GPU_SMOKE',
        'latent_grid':[4,4],'decoded_shape':list(image.shape),'valid_classes':8192,
        'losses':{k:float(v.detach()) for k,v in losses.items()},
        'logits_grad_norms':{k:float(v.norm()) for k,v in gradients.items()},
        'total_logits_grad_norm':float(logits.grad.norm()),'frozen_parameters_have_no_grad':True,
        'outside_roi_logits_grad_zero':True,'mask_class_grad_zero':True,
        'soft_one_hot_native_hard_max_abs_difference':float((soft-native).abs().max()),
        'gpu_smoke':'NOT_RUN','full_resolution_1024_backward':'NOT_RUN',
        'dev_quality':'NOT_RUN'}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
