#!/usr/bin/env python3
"""Verify 8-GPU fixed-corpus uninterrupted/resume smoke semantics."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from safetensors.torch import load_file


def rows(root):return [json.loads(line) for line in (Path(root)/"train_metrics.jsonl").open() if line.strip()]
def flatten(values,key):return [item for row in values for item in row[key]]
def state_diff(left,right,name):
 a=load_file(str(Path(left)/"checkpoint-20"/name));b=load_file(str(Path(right)/"checkpoint-20"/name));return max(float((a[key]-b[key]).abs().max()) for key in a)
def main():
 p=argparse.ArgumentParser();p.add_argument("--fresh20",required=True);p.add_argument("--fresh10",required=True);p.add_argument("--resume20",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 full=rows(a.fresh20);split=rows(a.fresh10)+rows(a.resume20)
 uids=flatten(full,"committed_sample_uids");split_uids=flatten(split,"committed_sample_uids")
 report={"fresh_steps":[row["global_step"] for row in full],"split_steps":[row["global_step"] for row in split],"samples":len(uids),"unique_samples":len(set(uids)),
         "sample_sequence_match":uids==split_uids,
         "geometry_seed_sequence_match":flatten(full,"committed_geometry_seeds")==flatten(split,"committed_geometry_seeds"),
         "corruption_seed_sequence_match":flatten(full,"committed_corruption_seeds")==flatten(split,"committed_corruption_seeds"),
         "learning_rate_sequence_match":[r["learning_rate"] for r in full]==[r["learning_rate"] for r in split],
         "adapter_max_abs_diff":state_diff(a.fresh20,a.resume20,"adapter_model.safetensors"),
         "mask_max_abs_diff":state_diff(a.fresh20,a.resume20,"mask_conditioning.safetensors")}
 report["status"]="PASS" if report["fresh_steps"]==list(range(1,21)) and report["split_steps"]==list(range(1,21)) and report["samples"]==report["unique_samples"]==640 and all(report[key] for key in ("sample_sequence_match","geometry_seed_sequence_match","corruption_seed_sequence_match","learning_rate_sequence_match")) else "FAIL"
 output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2,sort_keys=True))
 if report["status"]!="PASS":raise SystemExit(2)
if __name__=="__main__":main()
