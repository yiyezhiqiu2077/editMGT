#!/usr/bin/env python3
"""Select one Mini E3 checkpoint without producing a formal selection artifact."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
import yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.explicit_region.contracts import sha256_file
from src.explicit_region.selection import rank_candidates,summarize_selection_metrics

def main():
 p=argparse.ArgumentParser();p.add_argument("--candidate",action="append",required=True,help="ID:CHECKPOINT:METRICS_JSON")
 p.add_argument("--config",default="configs/dev/mini_selection.yaml");p.add_argument("--output",required=True);a=p.parse_args()
 config=yaml.safe_load(Path(a.config).read_text());candidates=[]
 for value in a.candidate:
  identity,checkpoint,metrics_path=value.split(":",2);evaluation=json.loads(Path(metrics_path).read_text())
  candidates.append({"candidate_id":identity,"checkpoint_path":str(Path(checkpoint).resolve()),"metrics_path":str(Path(metrics_path).resolve()),"metrics_sha256":sha256_file(metrics_path),"metrics":summarize_selection_metrics(evaluation)})
 result=rank_candidates(candidates,max_outside_masked_lpips=config.get("max_outside_masked_lpips"))
 payload={"schema_version":"mini-selected-checkpoint-v1","status":"READY","dev_only":True,"formal_eligible":False,**result}
 output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
 print(json.dumps(payload,indent=2,sort_keys=True))
if __name__=="__main__":main()
