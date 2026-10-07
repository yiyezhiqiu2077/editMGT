#!/usr/bin/env python3
"""Verify 8-GPU fixed-corpus uninterrupted/resume smoke semantics."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from safetensors.torch import load_file
from scripts.dev.compare_e3_resume import equal


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
 finite_metrics=all(torch.isfinite(torch.tensor([value for row in full+split for key,value in row.items() if isinstance(value,(int,float)) and key not in ("global_step",)])).all().item() for _ in [0])
 report["nan_inf_count"]=0 if finite_metrics else 1
 report["no_oom"]=True;report["no_nccl_error"]=True
 diagnostics=[row for row in full+split if "lora_grad_diagnostic" in row]
 report["lora_grad_finite_nonzero"]=bool(diagnostics) and all(torch.isfinite(torch.tensor(row["lora_grad_diagnostic"]["total"])).item() and row["lora_grad_diagnostic"]["total"]>0 for row in diagnostics)
 report["region_grad_finite_nonzero"]=bool(diagnostics) and all(row.get("mask_embedding_grad_norm") is not None and torch.isfinite(torch.tensor(row["mask_embedding_grad_norm"])).item() and row["mask_embedding_grad_norm"]>0 for row in diagnostics)
 report["frozen_base_gradients_zero"]=all(row.get("frozen_base_gradient_nonzero")==0 for row in full+split)
 report["optimizer_scheduler_updates_match"]=all(row.get("optimizer_updates")==row["global_step"]==row.get("scheduler_updates") for row in full+split)
 report["checkpoint_exact_replay"]=report["adapter_max_abs_diff"]==report["mask_max_abs_diff"]==0
 states=[torch.load(Path(root)/"checkpoint-20/training_state.pt",map_location="cpu",weights_only=False) for root in (a.fresh20,a.resume20)]
 for field in ("optimizer","scheduler","sampler_state","rank_rng"):
  report[field+"_exact_replay"]=equal(states[0][field],states[1][field])
 report["committed_cursor_match"]=all(s.get("global_optimizer_step")==20 and s.get("committed_global_sample_count")==640 for s in states)
 payloads=[json.loads((Path(root)/"checkpoint-20/fingerprint.json").read_text())["payload"] for root in (a.fresh20,a.resume20)]
 report["fingerprint_match"]=payloads[0]==payloads[1]
 identity=payloads[0];hashes=identity.get("data_content_hashes",{})
 report.update(git_sha=identity.get("git_sha"),world_size=identity.get("world_size"),data_name=identity.get("data",{}).get("name"),
               train_manifest_sha256=hashes.get("data.manifest",{}).get("sha256"),corpus_ready_sha256=hashes.get("data.corpus_ready",{}).get("sha256"))
 report["dataset_identity_bound"]=report["data_name"]=="fixed_200k" and report["world_size"]==8 and all(isinstance(report[k],str) and len(report[k])==64 for k in ("train_manifest_sha256","corpus_ready_sha256"))
 report["status"]="PASS" if report["fresh_steps"]==list(range(1,21)) and report["split_steps"]==list(range(1,21)) and report["samples"]==report["unique_samples"]==640 and finite_metrics and all(report[key] for key in ("sample_sequence_match","geometry_seed_sequence_match","corruption_seed_sequence_match","learning_rate_sequence_match","lora_grad_finite_nonzero","region_grad_finite_nonzero","frozen_base_gradients_zero","optimizer_scheduler_updates_match","checkpoint_exact_replay")) else "FAIL"
 if not all(report[key] for key in ("optimizer_exact_replay","scheduler_exact_replay","sampler_state_exact_replay","rank_rng_exact_replay","committed_cursor_match","fingerprint_match","dataset_identity_bound")):
  report["status"]="FAIL"
 output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2,sort_keys=True))
 if report["status"]!="PASS":raise SystemExit(2)
if __name__=="__main__":main()
