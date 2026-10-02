#!/usr/bin/env python3
"""Generate a frozen evaluation manifest for released or trainable-sidecar models."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from src.editmgt import init_edit_mgt
from src.explicit_region.config import load_config
from src.explicit_region.dataset import MagicBrushAlignedDataset,InterEditArchiveDataset
from src.explicit_region.fixed_dataset import CanonicalAlignedDataset

def pil(t):return Image.fromarray((t.numpy().transpose(1,2,0)*255).round().clip(0,255).astype(np.uint8))
def main():
 p=argparse.ArgumentParser();p.add_argument("--dataset",choices=("magicbrush","crispedit","scaleedit","interedit"),required=True);p.add_argument("--manifest",required=True);p.add_argument("--interedit-root");p.add_argument("--canonical-root");p.add_argument("--model-root",required=True);p.add_argument("--checkpoint");p.add_argument("--config",required=True);p.add_argument("--timestep-mode",choices=("roi_relative","official_upstream_timestep"),required=True);p.add_argument("--output-dir",required=True);p.add_argument("--count",type=int);a=p.parse_args()
 c=load_config(a.config);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 kw=dict(resolution=c["resolution"],base_seed=42,minimum_mask_retention=.75)
 if a.canonical_root:
  rows=[json.loads(line) for line in Path(a.manifest).open(encoding="utf-8") if line.strip()];ds=CanonicalAlignedDataset(rows,a.canonical_root,max_random_attempts=8,**kw)
 else:
  ds=MagicBrushAlignedDataset(a.manifest,max_resample_attempts=8,**kw) if a.dataset=="magicbrush" else InterEditArchiveDataset(a.manifest,a.interedit_root,max_resample_attempts=8,**kw)
 pipe=init_edit_mgt("cuda",enable_bf16=True,base_model_path=a.model_root,local_files_only=True,trainable_state_path=a.checkpoint);pipe.set_progress_bar_config(disable=True);rows=[]
 for i in range(min(len(ds),a.count or len(ds))):
  item=ds[i];source=pil(item["source_image"]);target=pil(item["target_image"]);mask=Image.fromarray(item["edit_region_mask"].numpy().astype(np.uint8)*255)
  for seed in c["generation_seeds"]:
   g=torch.Generator(device="cuda").manual_seed(seed);start=time.perf_counter();result=pipe(prompt=item["instruction_en"],reference_image=source,mask_image=mask,height=c["resolution"],width=c["resolution"],num_inference_steps=c["steps"],guidance_scale=c["guidance_scale"],reference_strength=c["reference_strength"],generator=g,lora_scope="both",inference_timestep_mode=a.timestep_mode).images[0];runtime=time.perf_counter()-start
   stem=f"{a.dataset}_{i:06d}_seed{seed}";paths={}
   for name,image in (("source",source),("target",target),("mask",mask),("output",result)):
    path=out/f"{stem}_{name}.png";image.save(path);paths[name]=str(path)
   rows.append(paths|{"sample_key":item["sample_key"],"dataset_name":a.dataset,"edit_type":item["edit_type"],"seed":seed,"runtime_seconds":runtime,"timestep_mode":a.timestep_mode,"timestep_diagnostics":pipe.last_inference_diagnostics})
 (out/"predictions.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows),encoding="utf-8");print(json.dumps({"generations":len(rows),"manifest":str(out/"predictions.jsonl")}))
if __name__=="__main__":main()
