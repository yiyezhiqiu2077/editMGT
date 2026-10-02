#!/usr/bin/env python3
"""Freeze a deterministic dataset-stratified manual QA sample of translations."""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.explicit_region.language import contains_han,translation_qa_flags


def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--output-jsonl",required=True);p.add_argument("--output-csv",required=True);p.add_argument("--count",type=int,default=500);a=p.parse_args()
 groups=defaultdict(list)
 for line in Path(a.manifest).open(encoding="utf-8"):
  if line.strip():
   row=json.loads(line)
   if contains_han(row["instruction_original"]):groups[row["dataset_name"]].append(row)
 for name in groups:groups[name].sort(key=lambda row:(hashlib.sha256(f"translation-manual-qa\0seed=42\0{row['sample_uid']}".encode()).hexdigest(),row["sample_uid"]))
 selected=[]
 while len(selected)<a.count and any(groups.values()):
  progressed=False
  for name in sorted(groups):
   if groups[name]:selected.append(groups[name].pop(0));progressed=True
   if len(selected)==a.count:break
  if not progressed:break
 clean=[{"dataset":row["dataset_name"],"sample_uid":row["sample_uid"],"edit_type":row["edit_type_canonical"],"instruction_original":row["instruction_original"],"instruction_en":row["instruction_en"],"qa_flags":translation_qa_flags(row["instruction_original"],row["instruction_en"])} for row in selected]
 output=Path(a.output_jsonl);output.parent.mkdir(parents=True,exist_ok=True);output.write_text("".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n" for row in clean),encoding="utf-8")
 with Path(a.output_csv).open("w",encoding="utf-8",newline="") as handle:
  writer=csv.DictWriter(handle,fieldnames=("dataset","sample_uid","edit_type","instruction_original","instruction_en","qa_flags"));writer.writeheader()
  for row in clean:writer.writerow(row|{"qa_flags":"|".join(row["qa_flags"])})
 print(json.dumps({"rows":len(clean),"jsonl":a.output_jsonl,"csv":a.output_csv}))
if __name__=="__main__":main()
