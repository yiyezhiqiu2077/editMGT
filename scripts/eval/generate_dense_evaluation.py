#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.explicit_region.config import load_config
from src.explicit_region.dense_inference import load_evaluation_pipeline, generate_fixed_evaluation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("released", "lora", "dense"), required=True)
    for name in ("model-root", "dataset", "manifest", "canonical-root", "config", "output-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--group-identity-contract")
    p.add_argument("--count", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--timestep-mode", choices=("roi_relative", "official_upstream_timestep"), required=True)
    a = p.parse_args()
    pipe = load_evaluation_pipeline(a.model_root, backend=a.backend, checkpoint=a.checkpoint, device=a.device)
    print(generate_fixed_evaluation(pipe, manifest=a.manifest, canonical_root=a.canonical_root,
        dataset_name=a.dataset, config=load_config(a.config), timestep_mode=a.timestep_mode,
        output_dir=a.output_dir, device=a.device, count=a.count,
        group_contract=json.loads(Path(a.group_identity_contract).read_text()) if a.group_identity_contract else None))


if __name__ == "__main__":
    main()
