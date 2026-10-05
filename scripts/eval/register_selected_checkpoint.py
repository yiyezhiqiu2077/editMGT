#!/usr/bin/env python3
"""Freeze the content identity of the preregistered formal TEST checkpoint."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.explicit_region.formal_pipeline import git_sha,require_selection_ready,sha256_file,stable_json_hash,write_json_atomic
p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--formal-assets",required=True);p.add_argument("--selection-config",default="configs/eval/formal.yaml");p.add_argument("--output",required=True);a=p.parse_args();require_selection_ready(a.selection_config);root=Path(a.checkpoint).resolve();names=("adapter_model.safetensors","mask_conditioning.safetensors","mask_conditioning_config.json","trainable_config.json","fingerprint.json")
files={name:sha256_file(root/name) for name in names if (root/name).is_file()}
if not {"adapter_model.safetensors","mask_conditioning.safetensors"}<=set(files):raise RuntimeError("selected checkpoint is missing trainable weights")
payload={"schema_version":"selected-checkpoint-v1","status":"READY","git_sha":git_sha(ROOT),"formal_assets_sha256":sha256_file(a.formal_assets),"checkpoint_path":str(root),"files":files};payload["checkpoint_identity_sha256"]=stable_json_hash(files);write_json_atomic(a.output,payload);print(json.dumps({"status":"READY","identity":payload["checkpoint_identity_sha256"]}))
