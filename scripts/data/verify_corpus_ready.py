#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.explicit_region.fixed_corpus import verify_corpus_ready

parser=argparse.ArgumentParser();parser.add_argument("marker")
args=parser.parse_args()
try:
    payload=verify_corpus_ready(args.marker)
except Exception as exc:
    raise SystemExit(f"CORPUS_NOT_READY: {exc}")
print(json.dumps({"status":"READY","total_rows":payload.get("total_rows"),"marker":str(Path(args.marker).resolve())}))
