#!/usr/bin/env python3
"""Build canonical pools from an explicit mapping produced after real schema audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import yaml

from src.explicit_region.canonical import SCHEMA_VERSION,expected_locator_hash,image_from_locator,sample_uid_for
from src.explicit_region.language import contains_han


def locator(mapping,row,file_relative,row_index,role):
    spec=mapping["fields"][role];storage=spec["storage"]
    if storage=="embedded_parquet":
        return {"backend":"parquet","file":file_relative,"row_index":row_index,"column":spec["column"]}
    if storage=="file":
        return {"backend":"file","relative_path":str(row[spec["column"]])}
    if storage=="bbox":
        bbox=row[spec["column"]]
        return {"backend":"derived_bbox","bbox":list(bbox),"reference_width":int(row[spec["width_column"]]),"reference_height":int(row[spec["height_column"]])}
    raise ValueError(f"unsupported explicit storage mapping: {storage}")


def iter_rows(root,mapping):
    files=sorted(root.glob(mapping["files_glob"]));fmt=mapping["format"]
    if not files: raise RuntimeError("schema mapping glob matched no files")
    for file in files:
        relative=file.relative_to(root).as_posix()
        if fmt=="parquet":
            import pyarrow.parquet as pq
            table=pq.read_table(file)
            for index,row in enumerate(table.to_pylist()):yield relative,index,row
        elif fmt=="jsonl":
            for index,line in enumerate(file.open(encoding="utf-8")):
                if line.strip():yield relative,index,json.loads(line)
        else: raise ValueError("mapping format must be parquet or jsonl")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",required=True);parser.add_argument("--mapping",required=True);parser.add_argument("--edit-type-mapping",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();root=Path(args.root).resolve();mapping=yaml.safe_load(Path(args.mapping).read_text());types=yaml.safe_load(Path(args.edit_type_mapping).read_text())["datasets"]
    if mapping.get("schema_audit_status")!="REAL_SCHEMA_AUDITED":raise RuntimeError("mapping must bind a REAL_SCHEMA_AUDITED report")
    name=mapping["dataset_name"];revision=mapping["dataset_revision"]
    if name not in ("crispedit","scaleedit","interedit"):raise ValueError("schema-mapped builder is for audited non-MagicBrush pools")
    if not types.get(name):raise RuntimeError(f"formal edit-type mapping for {name} is empty")
    records=[]
    for file_relative,index,raw in iter_rows(root,mapping):
        better=mapping.get("better_data_column")
        if better and not bool(raw.get(better)):continue
        instruction=str(raw[mapping["fields"]["instruction"]]).strip()
        original_type=str(raw[mapping["fields"]["edit_type"]])
        if original_type not in types[name]:raise RuntimeError(f"unmapped {name} edit type: {original_type}")
        locators={role:locator(mapping,raw,file_relative,index,role) for role in ("source","target","region")}
        hashes={role:expected_locator_hash(value,root) for role,value in locators.items()}
        region=image_from_locator(locators["region"],root).convert("L")
        fraction=float((np.asarray(region)>0).mean())
        group_column=mapping["fields"].get("group_id")
        group=str(raw[group_column]) if group_column else hashes["source"]
        record={"schema_version":SCHEMA_VERSION,"sample_uid":"0"*64,"manifest_index":None,"dataset_name":name,"dataset_revision":revision,"group_id":group,
                **{f"{role}_locator":value for role,value in locators.items()},**{f"{role}_sha256":value for role,value in hashes.items()},
                "instruction_original":instruction,"instruction_en":"" if contains_han(instruction) else instruction,"language_original":"zho_Hans" if contains_han(instruction) else "en",
                "edit_type_original":original_type,"edit_type_canonical":types[name][original_type],"mask_semantics":mapping["mask_semantics"],"region_fraction":fraction,
                "translation_status":"pending" if contains_han(instruction) else "passthrough_en","translation_cache_key":None}
        record["sample_uid"]=sample_uid_for(record);records.append(record)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text("".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n" for row in records),encoding="utf-8")
    print(json.dumps({"dataset":name,"rows":len(records),"output":str(output)}))

if __name__=="__main__":main()
