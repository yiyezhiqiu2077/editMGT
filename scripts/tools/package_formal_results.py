#!/usr/bin/env python3
"""Create a bounded provenance/metrics archive without data or model payloads."""
from __future__ import annotations
import argparse,json,tarfile,tempfile
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.explicit_region.formal_pipeline import git_sha,sha256_file

ALLOWED={".json",".jsonl",".yaml",".yml",".csv",".tsv",".txt",".log",".md"}
BLOCKED=("checkpoint","weights","hf_cache","token_cache","generated","images","predictions")
def files(root):
 root=Path(root).resolve()
 for path in sorted(root.rglob("*")):
  relative=path.relative_to(root)
  if path.is_file() and path.suffix.lower() in ALLOWED and not any(part.lower() in BLOCKED for part in relative.parts):yield path,relative
def main():
 p=argparse.ArgumentParser();p.add_argument("--experiment-root",required=True);p.add_argument("--eval-root",required=True);p.add_argument("--formal-assets",required=True);p.add_argument("--output",required=True);p.add_argument("--max-mib",type=int,default=1024);a=p.parse_args();sources=[]
 for label,root in (("experiments",a.experiment_root),("evaluation",a.eval_root)):
  sources.extend((path,Path(label)/relative) for path,relative in files(root))
 sources.extend((path,Path("configs")/path.relative_to(ROOT/"configs")) for path in (ROOT/"configs").rglob("*.yaml"))
 formal=Path(a.formal_assets).resolve();sources.append((formal,Path("provenance/formal_assets.json")));ledger=json.loads(formal.read_text())
 for name in ("FORMAL_READY.json","SELECTED_CHECKPOINT.json"):
  path=formal.parent/name
  if path.is_file():sources.append((path,Path("provenance")/name))
 corpus_record=ledger.get("corpus_ready",{});corpus_path=Path(corpus_record.get("path",""))
 if corpus_path.is_file():
  sources.append((corpus_path,Path("provenance/CORPUS_READY.json")));corpus=json.loads(corpus_path.read_text())
  include_names={"train_meta","selection_report","duplicate_report","translation_report","audit_report","machine_qa","validation_meta"}
  for name,record in corpus.get("files",{}).items():
   path=Path(record.get("path",""))
   if name in include_names and path.is_file():sources.append((path,Path("audit")/path.name))
 total=sum(path.stat().st_size for path,_ in sources);limit=a.max_mib*1024*1024
 if total>limit:raise RuntimeError(f"PACKAGE_SIZE_GUARD: {total} bytes exceeds {limit}")
 manifest={"schema":"formal-results-package-v1","git_sha":git_sha(ROOT),"formal_assets_sha256":sha256_file(formal),"files":[{"path":str(arc),"sha256":sha256_file(path),"bytes":path.stat().st_size} for path,arc in sources],"excluded":"raw data, weights, checkpoints, caches, token payloads, generated images"}
 output=Path(a.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile("w",suffix=".json",delete=False) as handle:json.dump(manifest,handle,indent=2,sort_keys=True);handle.write("\n");manifest_path=Path(handle.name)
 try:
  with tarfile.open(output,"w:gz") as archive:
   archive.add(manifest_path,arcname="provenance/package_manifest.json")
   for path,arc in sources:archive.add(path,arcname=str(arc),recursive=False)
 finally:manifest_path.unlink(missing_ok=True)
 if output.stat().st_size>limit:output.unlink();raise RuntimeError("PACKAGE_SIZE_GUARD: compressed archive exceeds limit")
 print(json.dumps({"status":"PASS","output":str(output),"files":len(sources)+1,"bytes":output.stat().st_size},indent=2))
if __name__=="__main__":main()
