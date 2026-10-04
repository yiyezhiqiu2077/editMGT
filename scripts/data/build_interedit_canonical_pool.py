#!/usr/bin/env python3
"""Build the better_data Inter-Edit canonical eligible pool from audited metadata."""

from __future__ import annotations
import argparse,io,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
from src.explicit_region.canonical import SCHEMA_VERSION,read_locator_bytes,sample_uid_for,sha256_bytes
from src.explicit_region.interedit import iter_metadata
from src.explicit_region.language import contains_han

MAPPING={"Add":"add","Remove":"remove","Local":"local_attribute","Texture":"texture"}


def main():
 parser=argparse.ArgumentParser();parser.add_argument("metadata",nargs="+");parser.add_argument("--interedit-root",required=True);parser.add_argument("--revision",required=True);parser.add_argument("--output",required=True);args=parser.parse_args();root=Path(args.interedit_root).resolve();rows=[]
 for metadata in args.metadata:
  for item in iter_metadata(metadata,only_better_data=True):
   locators={"source":{"backend":"tar","archive":item.source_archive,"member":item.source_file},"target":{"backend":"tar","archive":item.asset_archive,"member":item.target_file},"region":{"backend":"tar","archive":item.asset_archive,"member":item.mask_file}}
   values={name:read_locator_bytes(locator,root) for name,locator in locators.items()}
   mask=np.asarray(Image.open(io.BytesIO(values["region"])).convert("L"))>0
   if not mask.any():continue
   instruction=item.instruction_original.strip()
   if not instruction:continue
   row={"schema_version":SCHEMA_VERSION,"sample_uid":"0"*64,"manifest_index":None,"dataset_name":"interedit","dataset_revision":args.revision,"group_id":str(item.source_id),
        **{f"{name}_locator":locator for name,locator in locators.items()},**{f"{name}_sha256":sha256_bytes(value) for name,value in values.items()},
        "instruction_original":instruction,"instruction_en":"" if contains_han(instruction) else instruction,"language_original":"zho_Hans" if contains_han(instruction) else "en",
        "edit_type_original":item.edit_type,"edit_type_canonical":MAPPING[item.edit_type],"mask_semantics":"user_guidance_region","region_fraction":float(mask.mean()),
        "translation_status":"pending" if contains_han(instruction) else "passthrough_en","translation_cache_key":None,
        "better_data":True}
   row["sample_uid"]=sample_uid_for(row);rows.append(row)
 output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text("".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n" for row in rows),encoding="utf-8");print(json.dumps({"rows":len(rows),"output":str(output)}))

if __name__=="__main__":main()
