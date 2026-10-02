#!/usr/bin/env python3
"""Audit whether source→target VQ changes lie inside the provided region."""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from diffusers import VQModel
from src.explicit_region.dataset import MagicBrushAlignedDataset, InterEditArchiveDataset
from src.explicit_region.masks import pixel_mask_to_token_mask
from src.v2_utils import prepare_cond_token


def percentile(values, q):
    return float(np.percentile(np.asarray(values), q))


def to_image(tensor):
    return Image.fromarray((tensor.numpy().transpose(1,2,0)*255).round().clip(0,255).astype(np.uint8))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("magicbrush","interedit"), required=True)
    p.add_argument("--manifest", required=True); p.add_argument("--interedit-root")
    p.add_argument("--model-root", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--count", type=int); p.add_argument("--seed", type=int, default=42)
    a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    kwargs=dict(resolution=1024,base_seed=a.seed,max_resample_attempts=8,minimum_mask_retention=.75)
    ds=(MagicBrushAlignedDataset(a.manifest,**kwargs) if a.dataset=="magicbrush" else
        InterEditArchiveDataset(a.manifest,a.interedit_root,**kwargs))
    vq=VQModel.from_pretrained(a.model_root,subfolder="vqvae",local_files_only=True).to("cuda").eval()
    rows=[]
    for i in range(min(len(ds),a.count or len(ds))):
        item=ds[i]; source=item["source_image"][None].to("cuda",dtype=vq.dtype); target=item["target_image"][None].to("cuda",dtype=vq.dtype)
        with torch.no_grad():
            st=prepare_cond_token(None,source,vq); tt=prepare_cond_token(None,target,vq)
        grid=int(math.sqrt(st.shape[1])); change=st.reshape(1,grid,grid).ne(tt.reshape(1,grid,grid))
        region=pixel_mask_to_token_mask(item["edit_region_mask"][None],(grid,grid)).to(change.device)
        total=max(int(change.sum()),1); inside=int((change&region).sum()); outside=int((change&~region).sum())
        rows.append({"index":i,"sample_key":item["sample_key"],"dataset_name":a.dataset,"edit_type":item["edit_type"],
                     "outside_change_ratio":outside/total,"inside_change_coverage":inside/total,
                     "source_image":item["source_image"],"target_image":item["target_image"],"mask":item["edit_region_mask"]})
    worst=sorted(rows,key=lambda x:x["outside_change_ratio"],reverse=True)[:100]
    tile=192; montage=Image.new("RGB",(3*tile,len(worst)*tile),"white")
    for r,row in enumerate(worst):
        images=[to_image(row["source_image"]),Image.fromarray(row["mask"].numpy().astype(np.uint8)*255).convert("RGB"),to_image(row["target_image"])]
        for c,image in enumerate(images): montage.paste(image.resize((tile,tile)),(c*tile,r*tile))
    montage.save(out/"worst_100.jpg",quality=90)
    clean=[{k:v for k,v in row.items() if not isinstance(v,torch.Tensor)} for row in rows]
    (out/"per_sample.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in clean),encoding="utf-8")
    report={"dataset":a.dataset,"samples":len(rows)}
    for metric in ("outside_change_ratio","inside_change_coverage"):
        values=[r[metric] for r in rows]; report[metric]={"mean":float(np.mean(values)),**{f"p{q}":percentile(values,q) for q in (50,75,90,95,99)}}
    (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
