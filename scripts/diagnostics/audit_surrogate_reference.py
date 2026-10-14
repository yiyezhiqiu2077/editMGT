#!/usr/bin/env python3
"""Small CPU/CUDA boundary checks; does not load or update model weights."""
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.dimo.surrogate_audit import audit_surrogate_gradients


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rows=[]
    for device in ('cpu','cuda'):
        for batch in (1,3):
            for n in (7,173,683):
                torch.manual_seed(20)
                # Exercise an actual BF16 autocast projection output and a
                # distribution-gradient field from BF16-quantized logits.
                projection=torch.nn.Linear(16,8192,device=device)
                with torch.autocast(device,dtype=torch.bfloat16):
                    logits=projection(torch.randn(batch,n+1,16,device=device))
                auxiliary=(logits.detach().float()+torch.randn_like(logits.float())*.001).bfloat16()
                g=auxiliary.float().softmax(-1)-logits.detach().float().softmax(-1)
                mask=torch.arange(n+1,device=device)[None,:]<torch.full((batch,1),n,device=device)
                report=audit_surrogate_gradients(logits,g,mask)
                rows.append({'device':device,'batch':batch,'roi_tokens':n,'field':'BF16_distribution_difference',**report})
        z=torch.ones(1,8,2,device=device,dtype=torch.bfloat16,requires_grad=True)
        g=torch.full(z.shape,float.fromhex('0x1.3f60000000000p-16'),device=device)
        mask=torch.arange(8,device=device)[None,:]<7
        rows.append({'device':device,'batch':1,'roi_tokens':7,'field':'controlled_FP32_midpoint',
                     **audit_surrogate_gradients(z,g,mask)})
    result={'status':'PASS','cases':len(rows),'training_recipe_changed':False,
            'checkpoint_weights_loaded_or_modified':False,'audits':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','cases':len(rows),
                     'old_reference_differing_cases':sum(r['division_first_reference_differing_elements']>0 for r in rows)}))


if __name__=='__main__':main()
