#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.explicit_region.formal_pipeline import *

p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
check=sub.add_parser("check");check.add_argument("--stage-root",required=True);check.add_argument("--stage",required=True);check.add_argument("--git-sha",required=True);check.add_argument("--config",action="append",default=[]);check.add_argument("--input",action="append",default=[]);check.add_argument("--output",action="append",default=[])
write=sub.add_parser("write");write.add_argument("--stage-root",required=True);write.add_argument("--stage",required=True);write.add_argument("--git-sha",required=True);write.add_argument("--config",action="append",default=[]);write.add_argument("--input",action="append",default=[]);write.add_argument("--output",action="append",default=[]);write.add_argument("--asset-manifest",default="configs/formal_assets.yaml")
a=p.parse_args();expected=stage_fingerprint(stage=a.stage,git=a.git_sha,config_files=a.config,inputs=a.input);marker=Path(a.stage_root)/a.stage/"_SUCCESS.json"
if a.command=="check":
    valid=valid_stage_marker(marker,expected,a.output)
    if not valid: invalidate_from(a.stage_root,a.stage)
    print(json.dumps({"valid":valid,"marker":str(marker),"fingerprint":expected["fingerprint"]}));raise SystemExit(0 if valid else 10)
manifest=load_formal_assets(a.asset_manifest)
resolved={name:value["revision"] for name,value in manifest["assets"].items()}
write_stage_marker(marker,expected,resolved,a.output);print(json.dumps({"status":"SUCCESS","marker":str(marker)}))
