#!/usr/bin/env python3
"""Build the portable provenance ledger consumed by all formal launchers."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys,yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.explicit_region.formal_pipeline import asset_path,git_sha,identity_path,load_formal_assets,sha256_file,stable_json_hash,write_json_atomic
p=argparse.ArgumentParser();p.add_argument("--asset-root",required=True);p.add_argument("--manifest",default="configs/formal_assets.yaml");p.add_argument("--train-manifest");p.add_argument("--corpus-ready");p.add_argument("--output",required=True);a=p.parse_args();config=load_formal_assets(a.manifest);assets={}
for name,spec in config["assets"].items():
 target=asset_path(a.asset_root,spec);record={"provider":spec["provider"],"local_path":str(target),"requested_revision":spec["revision"]}
 marker=identity_path(target)
 if marker.is_file():record.update(json.loads(marker.read_text()))
 elif spec["provider"]=="python_package" and (target/"asset_identity.json").is_file():record.update(json.loads((target/"asset_identity.json").read_text()))
 else:raise RuntimeError(f"FORMAL_PIPELINE_NOT_READY: identity missing for {name}")
 assets[name]=record
contracts={path.stem:sha256_file(path) for path in sorted(Path("configs/data/schema").glob("*.yaml"))}
payload={"schema_version":"formal-assets-record-v3","git_sha":git_sha(ROOT),"asset_manifest_sha256":sha256_file(a.manifest),"assets":assets,"schema_contract_hashes":contracts}
if a.train_manifest:payload["d200k_manifest"]={"path":str(Path(a.train_manifest).resolve()),"sha256":sha256_file(a.train_manifest)}
payload["formal_identity_sha256"]=stable_json_hash(payload)
if a.corpus_ready:payload["corpus_ready"]={"path":str(Path(a.corpus_ready).resolve()),"sha256":sha256_file(a.corpus_ready)}
write_json_atomic(a.output,payload);print(json.dumps({"status":"PASS","output":a.output,"formal_identity_sha256":payload["formal_identity_sha256"]}))
