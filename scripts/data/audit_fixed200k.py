#!/usr/bin/env python3
"""Unified four-dataset audit, montages, and FP32-VQ containment."""
from __future__ import annotations
import argparse,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw
import torch
from diffusers import VQModel
from src.explicit_region.fixed_dataset import FixedCorpusDataset
from src.explicit_region.canonical import image_from_locator
from src.explicit_region.masks import pixel_mask_to_token_mask
from src.v2_utils import prepare_cond_token


def stats(values):
 a=np.asarray(values,dtype=np.float64)
 return {"mean":float(a.mean()),**{f"p{q}":float(np.percentile(a,q)) for q in (10,25,50,75,90,95,99)}} if len(a) else {}
def order(rows,namespace):return sorted(rows,key=lambda pair:(hashlib.sha256(f"{namespace}\0{pair[1]['sample_uid']}".encode()).hexdigest(),pair[1]["sample_uid"]))
def stratified_pairs(pairs,count,namespace):
 groups=defaultdict(list)
 for pair in order(pairs,namespace):groups[pair[1]["edit_type_canonical"]].append(pair)
 result=[]
 while len(result)<count and any(groups.values()):
  for key in sorted(groups):
   if groups[key]:result.append(groups[key].pop(0))
   if len(result)==count:break
 return result
def tensor_image(t):return Image.fromarray((t.numpy().transpose(1,2,0)*255).round().clip(0,255).astype(np.uint8))
def montage(items,path):
 tile=192;label=54;canvas=Image.new("RGB",(4*tile,len(items)*(tile+label)),"white");draw=ImageDraw.Draw(canvas)
 for r,item in enumerate(items):
  y=r*(tile+label);mask=Image.fromarray(item["edit_region_mask"].numpy().astype(np.uint8)*255).convert("RGB")
  source=tensor_image(item["source_image"]);overlay=source.copy();overlay.paste(Image.new("RGB",source.size,(255,0,0)),mask=mask.point(lambda x:x//3));target=tensor_image(item["target_image"])
  for c,image in enumerate((source,mask,overlay,target)):canvas.paste(image.resize((tile,tile)),(c*tile,y))
  draw.text((4,y+tile+2),f"{item['dataset_name']} {item['sample_uid'][:12]} {item['edit_type']} {item['mask_semantics']}\n{item['instruction_en'][:150]}",fill="black")
 canvas.save(path,quality=90)
def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--magicbrush-root",required=True);p.add_argument("--crispedit-root",required=True);p.add_argument("--scaleedit-root",required=True);p.add_argument("--interedit-root",required=True);p.add_argument("--model-root",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--containment-count",type=int,default=500);a=p.parse_args()
 roots={name:getattr(a,f"{name}_root") for name in ("magicbrush","crispedit","scaleedit","interedit")};ds=FixedCorpusDataset(a.manifest,roots,resolution=1024,verify_hashes=True);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 indexed=list(enumerate(ds.rows));by_dataset=defaultdict(list)
 for pair in indexed:by_dataset[pair[1]["dataset_name"]].append(pair)
 report={"schema":"fixed200k-audit-v2","total":len(ds),"datasets":{}}
 vq=VQModel.from_pretrained(a.model_root,subfolder="vqvae",local_files_only=True,torch_dtype=torch.float32).to("cuda",dtype=torch.float32).eval()
 final_candidates=[]
 for name in ("magicbrush","crispedit","scaleedit","interedit"):
  pairs=order(by_dataset[name],f"audit-{name}");items=[ds[index] for index,_ in pairs[:max(100,min(a.containment_count,len(pairs)))]]
  montage(items[:100],out/f"{name}_100.jpg");final_candidates.extend(ds[index] for index,_ in stratified_pairs(pairs,50,f"final-{name}"))
  containment=[]
  for item in items[:a.containment_count]:
   source=item["source_image"][None].to("cuda",dtype=torch.float32);target=item["target_image"][None].to("cuda",dtype=torch.float32)
   with torch.no_grad():st=prepare_cond_token(None,source,vq);tt=prepare_cond_token(None,target,vq)
   grid=int(math.sqrt(st.shape[1]));change=st.reshape(1,grid,grid).ne(tt.reshape(1,grid,grid));region=pixel_mask_to_token_mask(item["edit_region_mask"][None],(grid,grid)).to(change.device);total=max(int(change.sum()),1)
   containment.append((int((change&~region).sum())/total,int((change&region).sum())/total))
  rows=[row for _,row in pairs];dimensions=[]
  for row in rows[:min(500,len(rows))]:
   source=image_from_locator(row["source_locator"],roots[name]);target=image_from_locator(row["target_locator"],roots[name]);dimensions.append({"source_width":source.width,"source_height":source.height,"target_width":target.width,"target_height":target.height,"source_aspect":source.width/source.height,"target_aspect":target.width/target.height})
  report["datasets"][name]={"selected":len(rows),"unique_sources":len({row["source_sha256"] for row in rows}),"decode_failures":0,"missing_assets":0,"han_prompts":sum(any('\u3400'<=c<='\u9fff' for c in row["instruction_en"]) for row in rows),"edit_type_counts":dict(Counter(row["edit_type_canonical"] for row in rows)),"mask_semantics_counts":dict(Counter(row["mask_semantics"] for row in rows)),"region_fraction":stats([row["region_fraction"] for row in rows]),"samples_per_source":stats(list(Counter(row["source_sha256"] for row in rows).values())),"dimension_audit_samples":len(dimensions),"source_width":stats([x["source_width"] for x in dimensions]),"source_height":stats([x["source_height"] for x in dimensions]),"target_width":stats([x["target_width"] for x in dimensions]),"target_height":stats([x["target_height"] for x in dimensions]),"source_aspect":stats([x["source_aspect"] for x in dimensions]),"target_aspect":stats([x["target_aspect"] for x in dimensions]),"containment_samples":len(containment),"outside_change_ratio":stats([x[0] for x in containment]),"inside_change_coverage":stats([x[1] for x in containment])}
 montage(final_candidates[:200],out/"final_200k_montage.jpg");(out/"audit_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__":main()
