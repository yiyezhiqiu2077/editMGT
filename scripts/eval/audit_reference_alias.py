#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from diffusers import VQModel
from src.explicit_region.conditioning import initialize_hardlock_latents
from src.explicit_region.dataset import MagicBrushAlignedDataset
from src.explicit_region.masks import pixel_mask_to_token_mask
from src.v2_utils import prepare_cond_token

def main():
 p=argparse.ArgumentParser();p.add_argument("--model-root",required=True);p.add_argument("--manifest",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 item=MagicBrushAlignedDataset(a.manifest,resolution=1024,base_seed=42)[0]
 vq=VQModel.from_pretrained(a.model_root,subfolder="vqvae",local_files_only=True,torch_dtype=torch.bfloat16).to("cuda").eval()
 with torch.no_grad():reference=prepare_cond_token(None,item["source_image"][None].to("cuda",dtype=vq.dtype),vq).reshape(1,64,64)
 before=reference.clone();mask=pixel_mask_to_token_mask(item["edit_region_mask"][None],(64,64)).to("cuda");target=initialize_hardlock_latents(reference,mask,8255)
 report={"reference_clean_unchanged":bool(torch.equal(before,reference)),"target_roi_is_mask":bool(torch.all(target[mask]==8255)),"target_outside_equals_source":bool(torch.equal(target[~mask],reference[~mask])),"reference_shape":list(reference.shape),"region_tokens":int(mask.sum())}
 path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
 if not all(report[k] for k in ("reference_clean_unchanged","target_roi_is_mask","target_outside_equals_source")):raise SystemExit(1)
if __name__=="__main__":main()
