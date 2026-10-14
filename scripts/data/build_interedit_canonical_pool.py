#!/usr/bin/env python3
"""Build the better_data Inter-Edit canonical eligible pool from audited metadata."""

from __future__ import annotations
import argparse,io,json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image, UnidentifiedImageError
from src.explicit_region.canonical import SCHEMA_VERSION,sample_uid_for,sha256_bytes
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dataset import _align_to_mask_coordinates
from src.explicit_region.geometry import SampleRejected, apply_geometry, sample_geometry
from src.explicit_region.interedit import TarMemberReader, iter_metadata
from src.explicit_region.language import contains_han

MAPPING={"Add":"add","Remove":"remove","Local":"local_attribute","Texture":"texture"}


def main():
 parser=argparse.ArgumentParser();parser.add_argument("metadata",nargs="+");parser.add_argument("--interedit-root",required=True);parser.add_argument("--revision",required=True);parser.add_argument("--output",required=True);args=parser.parse_args();root=Path(args.interedit_root).resolve();rows=[];rejections=[];rejection_counts=Counter();candidate_index=0;reader=TarMemberReader(root)
 for metadata in args.metadata:
  for item in iter_metadata(metadata,only_better_data=True):
   locators={"source":{"backend":"tar","archive":item.source_archive,"member":item.source_file},"target":{"backend":"tar","archive":item.asset_archive,"member":item.target_file},"region":{"backend":"tar","archive":item.asset_archive,"member":item.mask_file}}
   values={name:reader.read(locator["archive"],locator["member"]) for name,locator in locators.items()}
   images={};failed=None
   for role in ("source","target","region"):
    try:
     with Image.open(io.BytesIO(values[role])) as opened:
      opened.load();images[role]=opened.copy()
    except (UnidentifiedImageError,OSError,ValueError) as exc:
     failed={"reason":f"invalid_{role}_image","error":str(exc)};break
   if failed is None:
    mask=np.asarray(images["region"].convert("L"))>0
    if not mask.any():failed={"reason":"empty_region"}
   if failed is None:
    aligned_source=aligned_target=None
    try:
     aligned_source,aligned_target,_=_align_to_mask_coordinates(images["source"],images["target"],images["region"],str(item.sample_id))
     geometry=sample_geometry(images["region"],resolution=1024,base_seed=42,global_sample_index=candidate_index,sample_key=str(item.sample_id),max_resample_attempts=8,minimum_mask_retention=.75)
     if not (np.asarray(apply_geometry(images["region"],geometry,is_mask=True))>0).any():failed={"reason":"post_geometry_empty_region"}
    except SampleRejected as exc:
     failed={"reason":"post_geometry_empty_region","error":exc.reason}
    except ValueError as exc:
     if "unaligned aspect ratio" not in str(exc):raise
     failed={"reason":"unaligned_aspect_ratio","error":str(exc)}
    finally:
     if aligned_source is not None and aligned_source is not images.get("source"):aligned_source.close()
     if aligned_target is not None and aligned_target is not images.get("target"):aligned_target.close()
   for image in images.values():image.close()
   if failed is not None:
    rejection_counts[failed["reason"]]+=1;rejections.append({"sample_id":item.sample_id,**failed});candidate_index+=1;continue
   instruction=item.instruction_original.strip()
   if not instruction:
    rejection_counts["empty_instruction"]+=1;rejections.append({"sample_id":item.sample_id,"reason":"empty_instruction"});candidate_index+=1;continue
   row={"schema_version":SCHEMA_VERSION,"sample_uid":"0"*64,"manifest_index":None,"dataset_name":"interedit","dataset_revision":args.revision,"group_id":str(item.source_id),
        **{f"{name}_locator":locator for name,locator in locators.items()},**{f"{name}_sha256":sha256_bytes(value) for name,value in values.items()},
        "instruction_original":instruction,"instruction_en":"" if contains_han(instruction) else instruction,"language_original":"zho_Hans" if contains_han(instruction) else "en",
        "edit_type_original":item.edit_type,"edit_type_canonical":MAPPING[item.edit_type],"mask_semantics":"user_guidance_region","region_fraction":float(mask.mean()),
        "translation_status":"pending" if contains_han(instruction) else "passthrough_en","translation_cache_key":None,
        "better_data":True}
   row["sample_uid"]=sample_uid_for(row);rows.append(row);candidate_index+=1
 reader.close()
 output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text("".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n" for row in rows),encoding="utf-8")
 rejection_path=output.with_suffix(".rejections.jsonl");rejection_path.write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in rejections),encoding="utf-8")
 report={"schema_version":"interedit-canonical-pool-build-v1","accepted_rows":len(rows),"rejected_rows":len(rejections),"rejection_counts":dict(sorted(rejection_counts.items())),"output":str(output.resolve()),"output_sha256":sha256_file(output),"rejections_sha256":sha256_file(rejection_path)}
 output.with_suffix(".report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8");print(json.dumps(report,sort_keys=True))

if __name__=="__main__":main()
