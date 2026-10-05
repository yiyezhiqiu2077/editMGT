#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,yaml
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.explicit_region.formal_pipeline import load_schema_contract,match_schema_contract,schema_columns,verify_contract_file
import gzip

p=argparse.ArgumentParser();p.add_argument("--magicbrush-snapshot",required=True);p.add_argument("--crispedit-root",required=True);p.add_argument("--scaleedit-root",required=True);p.add_argument("--interedit-root",required=True);p.add_argument("--output",required=True);a=p.parse_args()
reports=[verify_contract_file("configs/data/schema/crispedit.yaml",a.crispedit_root),verify_contract_file("configs/data/schema/scaleedit.yaml",a.scaleedit_root)]
mapping=yaml.safe_load(Path("configs/data/edit_type_mapping.yaml").read_text())["datasets"]
for name,root,count_key in (("crispedit",a.crispedit_root,"rows_by_source_type"),("scaleedit",a.scaleedit_root,"rows_by_final_task")):
 contract=load_schema_contract(f"configs/data/schema/{name}.yaml");manifest=json.loads((Path(root)/"dataset_manifest.json").read_text());actual=set(manifest[count_key]);expected=set(contract["known_edit_types"])
 if actual!=expected or actual!=set(mapping[name]):raise RuntimeError(f"UNMAPPED_EDIT_TYPE: {name} actual={sorted(actual)} contract={sorted(expected)} mapping={sorted(mapping[name])}")
magic=load_schema_contract("configs/data/schema/magicbrush.yaml");magic_files=sorted((Path(a.magicbrush_snapshot)/"data").glob("*.parquet"))
if not magic_files:raise RuntimeError("SCHEMA_CONTRACT_MISMATCH: MagicBrush parquet missing")
match_schema_contract(magic,schema_columns(magic_files[0]));reports.append({"dataset":"magicbrush","status":"PASS","files":len(magic_files)})
inter=load_schema_contract("configs/data/schema/interedit.yaml");metadata=sorted((Path(a.interedit_root)/"metadata").glob("*.jsonl.gz"))
if not metadata:raise RuntimeError("SCHEMA_CONTRACT_MISMATCH: Inter-Edit metadata missing")
with gzip.open(metadata[0],"rt",encoding="utf-8") as handle:row=json.loads(next(line for line in handle if line.strip()))
missing=sorted(set(inter["required_fields"])-set(row))
if missing:raise RuntimeError(f"SCHEMA_CONTRACT_MISMATCH: Inter-Edit missing {missing}")
if row.get("edit_type") not in inter["known_edit_types"]:raise RuntimeError("UNMAPPED_EDIT_TYPE: interedit")
reports.append({"dataset":"interedit","status":"PASS","files":len(metadata)})
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({"status":"PASS","contracts":reports},indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":"PASS","output":str(out)}))
