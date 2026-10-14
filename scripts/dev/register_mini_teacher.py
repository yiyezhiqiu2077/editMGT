#!/usr/bin/env python3
"""Atomically register a dev-only Mini checkpoint as a DiMO prep teacher."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,tempfile
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.dimo.contracts import load_teacher_contract,released_base_model_identity
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.contracts import sha256_file

FILES=("adapter_model.safetensors","mask_conditioning.safetensors","trainable_config.json","fingerprint.json")

def stable_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--selection",required=True);p.add_argument("--output-dir",required=True);a=p.parse_args()
 checkpoint=Path(a.checkpoint).resolve();selection_path=Path(a.selection).resolve();output=Path(a.output_dir).resolve()
 selection=json.loads(selection_path.read_text())
 if selection.get("dev_only") is not True or selection.get("formal_eligible") is not False or selection.get("winner",{}).get("checkpoint_path")!=str(checkpoint):raise RuntimeError("MINI_TEACHER_SELECTION_MISMATCH")
 hashes={name:sha256_file(checkpoint/name) for name in FILES}
 fingerprint=json.loads((checkpoint/"fingerprint.json").read_text())
 if fingerprint.get("sha256")!=recipe_fingerprint(fingerprint.get("payload",{})):raise RuntimeError("MINI_TEACHER_FINGERPRINT_MISMATCH")
 base=released_base_model_identity(fingerprint["payload"]["model_identity"])
 manifest={"schema_version":"dimo-mini-teacher-v1","base_model_identity":base,
  "lora_state":{"file":"adapter_model.safetensors","sha256":hashes["adapter_model.safetensors"]},
  "edit_region_embedding_state":{"file":"mask_conditioning.safetensors","sha256":hashes["mask_conditioning.safetensors"]},
  "training_config_identity":{"file":"trainable_config.json","sha256":hashes["trainable_config.json"]},
  "fingerprint":{"file":"fingerprint.json","sha256":hashes["fingerprint.json"]},
  "source_git_sha":fingerprint["payload"]["git_sha"],"selected_checkpoint_identity":stable_hash(hashes),
  "formal_teacher":False,"dev_only":True,"formal_eligible":False,"mini_selection_sha256":sha256_file(selection_path)}
 output.parent.mkdir(parents=True,exist_ok=True);temporary=Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.",dir=output.parent));renamed=False
 try:
  for name in FILES:shutil.copy2(checkpoint/name,temporary/name)
  (temporary/"dimo_teacher_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
  contract=load_teacher_contract(temporary,expected_base_identity=base)
  if output.exists():raise RuntimeError("MINI_TEACHER_OUTPUT_CONFLICT")
  os.rename(temporary,output);renamed=True
  print(json.dumps({"status":"MINI_TEACHER_READY","output":str(output),"bundle_sha256":contract.checkpoint_hash},indent=2))
 finally:
  if not renamed and temporary.exists():shutil.rmtree(temporary)
if __name__=="__main__":main()
