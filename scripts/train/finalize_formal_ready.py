#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.explicit_region.formal_pipeline import git_sha,sha256_file,write_json_atomic
from src.explicit_region.fixed_corpus import verify_corpus_ready
p=argparse.ArgumentParser();p.add_argument("--formal-assets",required=True);p.add_argument("--corpus-ready",required=True);p.add_argument("--smoke-verification",required=True);p.add_argument("--output",required=True);a=p.parse_args()
smoke=json.loads(Path(a.smoke_verification).read_text());assets=json.loads(Path(a.formal_assets).read_text());corpus=verify_corpus_ready(a.corpus_ready);current_git=git_sha(ROOT)
if smoke.get("status")!="PASS":raise RuntimeError("FORMAL_READY requires PASS smoke verification")
if assets.get("git_sha")!=current_git or corpus.get("git_sha")!=current_git or assets.get("formal_identity_sha256")!=corpus.get("formal_assets_sha256") or assets.get("corpus_ready",{}).get("sha256")!=sha256_file(a.corpus_ready):raise RuntimeError("FORMAL_READY provenance mismatch")
payload={"schema_version":"formal-ready-v1","status":"READY","git_sha":current_git,"world_size":8,"formal_assets_sha256":sha256_file(a.formal_assets),"corpus_ready_sha256":sha256_file(a.corpus_ready),"smoke_verification":{"path":str(Path(a.smoke_verification).resolve()),"sha256":sha256_file(a.smoke_verification)}}
write_json_atomic(a.output,payload);print(json.dumps({"status":"READY","output":a.output}))
