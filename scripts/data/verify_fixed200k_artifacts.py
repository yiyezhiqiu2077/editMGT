#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.explicit_region.fixed_corpus import assert_frozen_corpus
from src.explicit_region.language import contains_han
p=argparse.ArgumentParser();p.add_argument("--fixed-root",required=True);p.add_argument("--phase",choices=("dedup","exact"),required=True);a=p.parse_args();root=Path(a.fixed_root);rows=[json.loads(x) for x in (root/"train_200k.jsonl").read_text().splitlines() if x]
if a.phase=="dedup":
 required=("candidate_selection.jsonl","reserve_order.jsonl","final_selection.jsonl","translation_rejections.jsonl","backfill_history.jsonl","selection_report.json","duplicate_report.json","translation_report.json")
 missing=[name for name in required if not (root/name).is_file()]
 if missing:raise RuntimeError(f"D200K_ARTIFACT_MISSING: {missing}")
 duplicate=json.loads((root/"duplicate_report.json").read_text())
 if not isinstance(duplicate.get("events"),list):raise RuntimeError("duplicate report is malformed")
else:
 assert_frozen_corpus(rows,200000)
 if any(contains_han(r["instruction_en"]) for r in rows):raise RuntimeError("Han remains in final corpus")
print(json.dumps({"status":"PASS","phase":a.phase,"rows":len(rows)}))
