#!/usr/bin/env python3
"""Materialize evaluator weights during prepare so formal eval is offline."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import torch
import torchvision
import lpips
from importlib.metadata import version
p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();root=Path(a.root).resolve();root.mkdir(parents=True,exist_ok=True);os.environ["TORCH_HOME"]=str(root)
if version("lpips")!="0.1.4":raise RuntimeError("LPIPS_PACKAGE_IDENTITY_MISMATCH")
model=lpips.LPIPS(net="alex");model.eval()
files=sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.name not in {"asset_identity.json","_SUCCESS"})
if not files: raise RuntimeError("LPIPS_WEIGHTS_NOT_MATERIALIZED")
(root/"asset_identity.json").write_text(json.dumps({"package":"lpips","version":"0.1.4","backbone":"alex","torchvision":torchvision.__version__,"files":files},indent=2,sort_keys=True)+"\n")
(root/"_SUCCESS").write_text("lpips==0.1.4 alex\n");print(json.dumps({"status":"PASS","root":str(root),"files":len(files)}))
