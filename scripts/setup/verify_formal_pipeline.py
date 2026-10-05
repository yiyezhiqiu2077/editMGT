#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.explicit_region.formal_pipeline import git_sha,require_selection_ready,verify_formal_asset_corpus,verify_formal_ready
p=argparse.ArgumentParser();p.add_argument("--formal-assets",required=True);p.add_argument("--corpus-ready",required=True);p.add_argument("--formal-ready");p.add_argument("--pre-smoke",action="store_true");p.add_argument("--selection-config");a=p.parse_args()
try:
 if a.pre_smoke:
  verify_formal_asset_corpus(a.formal_assets,a.corpus_ready,git_sha(ROOT));ready={"world_size":None}
 else:
  if not a.formal_ready:raise RuntimeError("--formal-ready is required")
  ready=verify_formal_ready(a.formal_assets,a.corpus_ready,a.formal_ready,git_sha(ROOT))
 if a.selection_config:require_selection_ready(a.selection_config)
except Exception as exc:
 message=str(exc)
 raise SystemExit(message if message.startswith(("SELECTION_RULE","FORMAL_PIPELINE_NOT_READY")) else f"FORMAL_PIPELINE_NOT_READY: {message}")
print(json.dumps({"status":"READY","world_size":ready["world_size"]}))
