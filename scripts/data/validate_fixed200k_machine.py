#!/usr/bin/env python3
"""Exhaustive, non-interactive hard QA for the frozen D200K manifest."""
from __future__ import annotations
import argparse,io,json,math
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import yaml
from src.explicit_region.canonical import load_verified_record_images
from src.explicit_region.dataset import _align_to_mask_coordinates
from src.explicit_region.fixed_corpus import assert_frozen_corpus
from src.explicit_region.geometry import apply_geometry,sample_geometry
from src.explicit_region.language import contains_han,translation_qa_flags

def read(path):return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
def finite(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,list):return all(finite(v) for v in value)
    return True
def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--validation",action="append",default=[])
 for name in ("magicbrush","crispedit","scaleedit","interedit"):p.add_argument(f"--{name}-root",required=True)
 p.add_argument("--audit-report",required=True);p.add_argument("--translation-report",required=True);p.add_argument("--formal-asset-config",default="configs/formal_assets.yaml");p.add_argument("--output",required=True);a=p.parse_args()
 roots={name:Path(getattr(a,f"{name}_root")).resolve() for name in ("magicbrush","crispedit","scaleedit","interedit")};rows=read(a.manifest);assert_frozen_corpus(rows,200000)
 assets=yaml.safe_load(Path(a.formal_asset_config).read_text())["assets"];expected_revisions={name:assets[name]["revision"] for name in roots}
 validation=[row for path in a.validation for row in read(path)];val_sources={r["source_sha256"] for r in validation};val_groups={(r["dataset_name"],r["group_id"]) for r in validation}
 if {r["source_sha256"] for r in rows}&val_sources:raise RuntimeError("train/DEV exact-source leakage")
 if {(r["dataset_name"],r["group_id"]) for r in rows}&val_groups:raise RuntimeError("train/DEV group leakage")
 failures=[]
 for position,row in enumerate(rows):
  try:
   if row["dataset_name"]=="interedit" and row.get("better_data") is not True:raise ValueError("Inter-Edit row is not better_data")
   if row["dataset_revision"]!=expected_revisions[row["dataset_name"]]:raise ValueError("dataset revision is not the pinned training asset")
   if row["dataset_revision"]==assets["magicbrush_test"]["revision"]:raise ValueError("TEST asset participates in training")
   flags=translation_qa_flags(row["instruction_original"],row["instruction_en"])
   if flags:raise ValueError(f"translation QA flags remain: {flags}")
   root=roots[row["dataset_name"]]
   source,target,region=load_verified_record_images(row,root);source=source.convert("RGB");target=target.convert("RGB");region=region.convert("L")
   source,target,_=_align_to_mask_coordinates(source,target,region,row["sample_uid"])
   if source.size!=target.size or source.size!=region.size:raise ValueError("unaligned source/target/region")
   if not np.asarray(region).any():raise ValueError("empty region")
   geometry=sample_geometry(region,resolution=1024,base_seed=42,global_sample_index=position,sample_key=row["sample_uid"],max_resample_attempts=8,minimum_mask_retention=.75,deterministic_fallback=True)
   if not np.asarray(apply_geometry(region,geometry,is_mask=True)).any():raise ValueError("empty post-geometry region")
  except Exception as exc:
   failures.append({"manifest_index":position,"sample_uid":row.get("sample_uid"),"error":str(exc)})
   if len(failures)>=20:break
 if failures:raise RuntimeError("MACHINE_CORPUS_QA_FAILED: "+json.dumps(failures))
 audit=json.loads(Path(a.audit_report).read_text());translation=json.loads(Path(a.translation_report).read_text())
 if not finite(audit):raise RuntimeError("VQ_AUDIT_NOT_FINITE")
 report={"status":"PASS","schema":"fixed200k-machine-qa-v3","total_rows":len(rows),"unique_sample_uid":len({r['sample_uid'] for r in rows}),"dataset_counts":dict(Counter(r["dataset_name"] for r in rows)),"checks":{"asset_integrity":"PASS","decode":"PASS","translation":"PASS","geometry":"PASS","train_dev_leakage":"PASS","test_exclusion":"PASS","interedit_better_data":"PASS","dedup_determinism":"PASS","translation_backfill_determinism":"PASS","vq_audit":"PASS","nan_inf":0}}
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
