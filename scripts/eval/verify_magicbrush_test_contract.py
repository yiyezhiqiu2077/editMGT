#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.explicit_region.formal_pipeline import git_sha,sha256_file,stable_json_hash,verify_formal_ready,verify_magicbrush_test_counts
p=argparse.ArgumentParser();p.add_argument("--asset-root",required=True);p.add_argument("--checkpoint",required=True);a=p.parse_args();asset_root=Path(a.asset_root).resolve();formal=asset_root/"artifacts/formal_assets.json";corpus=asset_root/"derived/fixed200k/CORPUS_READY.json";ready=asset_root/"artifacts/FORMAL_READY.json";verify_formal_ready(formal,corpus,ready,git_sha(ROOT));ledger=json.loads(formal.read_text());test=ledger["assets"]["magicbrush_test"]
if test.get("requested_revision")!="8c8eb2cfeac96f635ed83512c60c3502b06e00c2" or test.get("resolved_revision")!=test.get("requested_revision"):raise RuntimeError("MAGICBRUSH_TEST_IDENTITY_MISMATCH")
meta=json.loads((asset_root/"datasets/magicbrush/test/canonical/dataset_meta.json").read_text());manifest=asset_root/"datasets/magicbrush/test/canonical/manifest.jsonl";rows=[json.loads(x) for x in manifest.read_text().splitlines() if x]
verify_magicbrush_test_counts(meta,rows)
selected=json.loads((asset_root/"artifacts/SELECTED_CHECKPOINT.json").read_text());checkpoint=Path(a.checkpoint).resolve();files={name:sha256_file(checkpoint/name) for name in selected["files"]}
if selected.get("status")!="READY" or selected.get("git_sha")!=git_sha(ROOT) or selected.get("formal_assets_sha256")!=sha256_file(formal) or selected.get("checkpoint_path")!=str(checkpoint) or selected.get("checkpoint_identity_sha256")!=stable_json_hash(files):raise RuntimeError("SELECTED_CHECKPOINT_IDENTITY_MISMATCH")
print(json.dumps({"status":"PASS","sessions":535,"turns":1053,"checkpoint_identity":selected["checkpoint_identity_sha256"]}))
