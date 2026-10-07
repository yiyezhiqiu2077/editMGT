#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.explicit_region.formal_pipeline import git_sha,verify_pipeline_mode
p=argparse.ArgumentParser();p.add_argument("--formal-assets",required=True);p.add_argument("--corpus-ready",required=True);p.add_argument("--formal-ready");p.add_argument("--pre-smoke",action="store_true");p.add_argument("--selection-config");p.add_argument("--mode",choices=("train","select"));p.add_argument("--selected-checkpoint");p.add_argument("--require-reattestation",action="store_true");a=p.parse_args()
try:
 ready=verify_pipeline_mode(mode=a.mode or ("train" if a.pre_smoke else "select"),formal_assets=a.formal_assets,corpus_ready=a.corpus_ready,current_git=git_sha(ROOT),formal_ready=a.formal_ready,pre_smoke=a.pre_smoke,selection_config=a.selection_config,selected_checkpoint=a.selected_checkpoint,require_reattestation=a.require_reattestation)
except Exception as exc:
 message=str(exc)
 raise SystemExit(message if message.startswith(("SELECTION_RULE","FORMAL_PIPELINE_NOT_READY")) else f"FORMAL_PIPELINE_NOT_READY: {message}")
print(json.dumps({"status":"READY","world_size":ready["world_size"]}))
