#!/usr/bin/env python3
"""Compare FP32 versus BF16 VQ assignments and verify BF16 text finiteness."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from diffusers import VQModel
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTextModelWithProjection, CLIPTokenizer
from src.explicit_region.dataset import MagicBrushAlignedDataset
from src.dataset_utils import tokenize_prompt,encode_prompt
from src.v2_utils import prepare_cond_token

def main():
 p=argparse.ArgumentParser();p.add_argument("--model-root",required=True);p.add_argument("--manifest",required=True);p.add_argument("--output",required=True);p.add_argument("--count",type=int,default=32);a=p.parse_args()
 ds=MagicBrushAlignedDataset(a.manifest,resolution=1024,base_seed=42)
 fp=VQModel.from_pretrained(a.model_root,subfolder="vqvae",local_files_only=True,torch_dtype=torch.float32).to("cuda")
 bf=VQModel.from_pretrained(a.model_root,subfolder="vqvae",local_files_only=True,torch_dtype=torch.bfloat16).to("cuda");agree=total=0
 for i in range(min(a.count,len(ds))):
  x=ds[i]["source_image"][None].to("cuda")
  with torch.no_grad(): f=prepare_cond_token(None,x.float(),fp);b=prepare_cond_token(None,x.bfloat16(),bf)
  agree+=int(f.eq(b).sum());total+=f.numel()
 del fp,bf;torch.cuda.empty_cache()
 clip=CLIPTextModelWithProjection.from_pretrained(a.model_root,subfolder="text_encoder",local_files_only=True,torch_dtype=torch.bfloat16).to("cuda").eval()
 llm=AutoModelForCausalLM.from_pretrained(a.model_root,subfolder="llm_encoder",local_files_only=True,torch_dtype=torch.bfloat16,output_hidden_states=True).to("cuda").eval()
 clip_tok=CLIPTokenizer.from_pretrained(a.model_root,subfolder="tokenizer",local_files_only=True);llm_tok=AutoTokenizer.from_pretrained(a.model_root,subfolder="llm_encoder",local_files_only=True)
 prompts=[ds[i]["instruction_en"] for i in range(min(a.count,len(ds)))]
 ids=tokenize_prompt([clip_tok,llm_tok],prompts,"CLIP_Gemma2",device="cuda")
 with torch.no_grad(): hidden,pooled=encode_prompt([clip,llm],ids,"CLIP_Gemma2")
 report={"samples":min(a.count,len(ds)),"vq_token_agreement_rate":agree/total,"vq_tokens":total,"bf16_text_hidden_finite":bool(torch.isfinite(hidden).all()),"bf16_text_pooled_finite":bool(torch.isfinite(pooled).all())}
 path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
