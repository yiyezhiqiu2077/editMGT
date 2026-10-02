#!/usr/bin/env python3
"""Run final immutable-corpus invariants and write the only trusted READY marker."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.explicit_region.fixed_corpus import assert_frozen_corpus,corpus_counts,write_corpus_ready


def read(path):return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument("--fixed-root",required=True);p.add_argument("--selection-config",required=True);p.add_argument("--translation-cache",required=True);p.add_argument("--git-sha",required=True);a=p.parse_args();root=Path(a.fixed_root).resolve();train=root/"train_200k.jsonl";rows=read(train);assert_frozen_corpus(rows,200000)
 validation=sorted((root/"validation").glob("*.jsonl"));validation_rows=[row for path in validation for row in read(path)];train_sources={row["source_sha256"] for row in rows};val_sources={row["source_sha256"] for row in validation_rows}
 if train_sources&val_sources:raise RuntimeError("train/validation source SHA leakage")
 train_groups={(row["dataset_name"],row["group_id"]) for row in rows};val_groups={(row["dataset_name"],row["group_id"]) for row in validation_rows}
 if train_groups&val_groups:raise RuntimeError("train/validation group leakage")
 required={"train_200k":train,"train_meta":root/"train_200k.meta.json","candidate_selection":root/"candidate_selection.jsonl","reserve_order":root/"reserve_order.jsonl","final_selection":root/"final_selection.jsonl","translation_rejections":root/"translation_rejections.jsonl","backfill_history":root/"backfill_history.jsonl","selection_report":root/"selection_report.json","duplicate_report":root/"duplicate_report.json","translation_report":root/"translation_report.json","translation_cache":Path(a.translation_cache),"audit_report":root/"audit/audit_report.json","translation_manual_audit":root/"translation_manual_audit_500.jsonl","selection_config":Path(a.selection_config),"validation_meta":root/"validation/validation.meta.json"}
 for path in validation:required[f"validation_{path.stem}"]=path
 for path in required.values():
  if not path.is_file():raise FileNotFoundError(f"required readiness artifact missing: {path}")
 for name in ("magicbrush_100.jpg","crispedit_100.jpg","scaleedit_100.jpg","interedit_100.jpg","final_200k_montage.jpg"):
  path=root/"audit"/name
  if not path.is_file():raise FileNotFoundError(f"required montage missing: {path}")
  required[f"montage_{path.stem}"]=path
 metadata=json.loads((root/"train_200k.meta.json").read_text());payload=write_corpus_ready(root/"CORPUS_READY.json",files=required,metadata={"created_from_git_sha":a.git_sha,"total_rows":200000,"dataset_counts":corpus_counts(rows),"unique_source_counts":{name:len({row["source_sha256"] for row in rows if row["dataset_name"]==name}) for name in corpus_counts(rows)},"dataset_revisions":metadata["dataset_revisions"],"integrity_checks":{"exact_rows":"PASS","unique_sample_uid":"PASS","contiguous_manifest_index":"PASS","train_validation_source_overlap":"PASS","train_validation_group_overlap":"PASS","required_reports":"PASS","required_montages":"PASS"}});print(json.dumps({"status":payload["status"],"marker":str(root/"CORPUS_READY.json")},indent=2))
if __name__=="__main__":main()
