#!/usr/bin/env python3
"""Build canonical pools from an explicit mapping produced after real schema audit."""

from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image, UnidentifiedImageError
import yaml

from src.explicit_region.canonical import (
    SCHEMA_VERSION, expected_locator_hash, image_from_locator, sample_uid_for,
    sha256_bytes, validate_record,
)
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dataset import _align_to_mask_coordinates
from src.explicit_region.geometry import SampleRejected, apply_geometry, sample_geometry
from src.explicit_region.language import contains_han
from src.explicit_region.formal_pipeline import load_schema_contract, match_schema_contract, schema_columns


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


def mapped_columns(mapping):
    fields=mapping["fields"]
    columns={fields["instruction"],fields["edit_type"],*mapping.get("eligibility",{}).keys()}
    for role in ("source","target","region"):
        spec=fields[role];columns.add(spec["column"])
        if spec["storage"]=="bbox":columns.update((spec["width_column"],spec["height_column"]))
    if fields.get("group_id"):columns.add(fields["group_id"])
    return sorted(columns)


def iter_rows(root,mapping):
    files=sorted(root.glob(mapping["files_glob"]));fmt=mapping["format"]
    if not files: raise RuntimeError("schema mapping glob matched no files")
    for file in files:
        relative=file.relative_to(root).as_posix()
        if fmt=="parquet":
            import pyarrow.parquet as pq
            match_schema_contract(mapping, schema_columns(file))
            parquet=pq.ParquetFile(file)
            offset=0
            for batch in parquet.iter_batches(batch_size=16,columns=mapped_columns(mapping),use_threads=False):
                for index,row in enumerate(batch.to_pylist(),offset):yield relative,index,row
                offset+=batch.num_rows
        elif fmt=="jsonl":
            for index,line in enumerate(file.open(encoding="utf-8")):
                if line.strip():yield relative,index,json.loads(line)
        else: raise ValueError("mapping format must be parquet or jsonl")


def embedded_bytes(raw, mapping, role):
    value=raw[mapping["fields"][role]["column"]]
    value=value.get("bytes") if isinstance(value,dict) else value
    if isinstance(value,memoryview):value=value.tobytes()
    if not isinstance(value,bytes):
        raise RuntimeError(f"SCHEMA_CONTRACT_MISMATCH: {role} cell is not bytes")
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",required=True);parser.add_argument("--mapping",required=True);parser.add_argument("--edit-type-mapping",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();root=Path(args.root).resolve();mapping=load_schema_contract(args.mapping);types=yaml.safe_load(Path(args.edit_type_mapping).read_text())["datasets"]
    name=mapping["dataset_name"];revision=mapping["dataset_revision"]
    if name not in ("crispedit","scaleedit","interedit"):raise ValueError("schema-mapped builder is for audited non-MagicBrush pools")
    if not types.get(name):raise RuntimeError(f"formal edit-type mapping for {name} is empty")
    records=[];rejections=[];rejection_counts=Counter()
    for file_relative,index,raw in iter_rows(root,mapping):
        if any(raw.get(column) != expected for column,expected in mapping.get("eligibility",{}).items()):
            continue
        instruction=str(raw[mapping["fields"]["instruction"]]).strip()
        original_type=str(raw[mapping["fields"]["edit_type"]])
        if original_type not in types[name]:raise RuntimeError(f"UNMAPPED_EDIT_TYPE: {name}/{original_type}")
        locators={role:locator(mapping,raw,file_relative,index,role) for role in ("source","target","region")}
        hashes={};embedded={};images={}
        for role,value in locators.items():
            if value["backend"]=="parquet":
                cell=embedded_bytes(raw,mapping,role)
                embedded[role]=cell;hashes[role]=sha256_bytes(cell)
            else:hashes[role]=expected_locator_hash(value,root)
        failed=None
        for role in ("source","target","region"):
            try:
                if role in embedded:
                    if not embedded[role]:raise UnidentifiedImageError("zero-byte image")
                    with Image.open(io.BytesIO(embedded[role])) as opened:
                        opened.load();images[role]=opened.copy()
                else:images[role]=image_from_locator(locators[role],root)
            except (UnidentifiedImageError,OSError,ValueError) as exc:
                failed={"reason":f"invalid_{role}_image","error":str(exc)};break
        if failed is None:
            region_array=np.asarray(images["region"].convert("L"))
            fraction=float((region_array>0).mean())
            if fraction==0:failed={"reason":"empty_region"}
        if failed is None:
            aligned_source=aligned_target=None
            try:
                aligned_source,aligned_target,_=_align_to_mask_coordinates(
                    images["source"],images["target"],images["region"],f"{file_relative}:{index}")
                geometry=sample_geometry(
                    images["region"],resolution=1024,base_seed=42,global_sample_index=index,
                    sample_key=f"{file_relative}:{index}",max_resample_attempts=8,
                    minimum_mask_retention=.75,
                )
                if not (np.asarray(apply_geometry(images["region"],geometry,is_mask=True))>0).any():
                    failed={"reason":"post_geometry_empty_region"}
            except SampleRejected as exc:
                failed={"reason":"post_geometry_empty_region","error":exc.reason}
            except ValueError as exc:
                if "unaligned aspect ratio" not in str(exc):raise
                failed={"reason":"unaligned_aspect_ratio","error":str(exc)}
            finally:
                if aligned_source is not None and aligned_source is not images.get("source"):
                    aligned_source.close()
                if aligned_target is not None and aligned_target is not images.get("target"):
                    aligned_target.close()
        for image in images.values():image.close()
        if failed is not None:
            rejection_counts[failed["reason"]]+=1
            rejections.append({"file":file_relative,"row_index":index,**failed})
            continue
        group_column=mapping["fields"].get("group_id")
        group=str(raw[group_column]) if group_column else hashes["source"]
        record={"schema_version":SCHEMA_VERSION,"sample_uid":"0"*64,"manifest_index":None,"dataset_name":name,"dataset_revision":revision,"group_id":group,
                **{f"{role}_locator":value for role,value in locators.items()},**{f"{role}_sha256":value for role,value in hashes.items()},
                "instruction_original":instruction,"instruction_en":"" if contains_han(instruction) else instruction,"language_original":"zho_Hans" if contains_han(instruction) else "en",
                "edit_type_original":original_type,"edit_type_canonical":types[name][original_type],"mask_semantics":mapping["mask_semantics"],"region_fraction":fraction,
                "translation_status":"pending" if contains_han(instruction) else "passthrough_en","translation_cache_key":None}
        record["sample_uid"]=sample_uid_for(record);validate_record(record);records.append(record)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text("".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n" for row in records),encoding="utf-8")
    rejection_path=output.with_suffix(".rejections.jsonl")
    rejection_path.write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in rejections),encoding="utf-8")
    report={"schema_version":"schema-mapped-canonical-v1","dataset":name,"accepted_rows":len(records),"rejected_rows":len(rejections),"rejection_counts":dict(sorted(rejection_counts.items())),"output":str(output.resolve()),"output_sha256":sha256_file(output),"rejections_sha256":sha256_file(rejection_path)}
    output.with_suffix(".report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,sort_keys=True))

if __name__=="__main__":main()
