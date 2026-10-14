#!/usr/bin/env python3
"""Strict predeclared bitwise verification; no post-hoc tolerance relaxation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from safetensors.torch import load_file
from src.explicit_region.dense_checkpoint import verify_dense_checkpoint, write_json


def exact(a, b, path="state"):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or not torch.equal(a, b):
            raise RuntimeError(f"DENSE_RESUME_NOT_BITWISE_EXACT: {path}")
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            raise RuntimeError(f"DENSE_RESUME_NOT_BITWISE_EXACT: {path}")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise RuntimeError(f"DENSE_RESUME_KEYS_MISMATCH: {path}")
        for key in a:
            exact(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise RuntimeError(f"DENSE_RESUME_LENGTH_MISMATCH: {path}")
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, f"{path}.{i}")
    elif a != b:
        raise RuntimeError(f"DENSE_RESUME_VALUE_MISMATCH: {path}")


def compare(fresh, resumed, *, fresh_log=None, interrupted_logs=(), output=None):
    fresh, resumed = Path(fresh), Path(resumed)
    left = verify_dense_checkpoint(fresh, inference_only=False)
    right = verify_dense_checkpoint(resumed, inference_only=False)
    a = json.loads((fresh / "model.safetensors.index.json").read_text())["weight_map"]
    b = json.loads((resumed / "model.safetensors.index.json").read_text())["weight_map"]
    exact(a, b, "model_index")
    for filename in set(a.values()):
        exact(load_file(fresh / filename), load_file(resumed / filename), filename)
    # Resume files are trusted local products, not downloaded checkpoints.
    exact(torch.load(fresh / "training_state.pt", map_location="cpu", weights_only=False),
          torch.load(resumed / "training_state.pt", map_location="cpu", weights_only=False))
    if fresh_log:
        def traces(paths):
            return [(r["global_step"], r["sample_trace"]) for p in paths
                    for line in Path(p).read_text().splitlines() if line.strip() for r in [json.loads(line)]]
        exact(traces([fresh_log]), traces(interrupted_logs), "sample_geometry_corruption_sequence")
    result = {"status": "PASS", "scope": "REAL_DENSE" if fresh_log else "CHECKPOINT_ONLY",
              "equivalence": "bitwise_exact", "fresh": str(fresh), "resumed": str(resumed),
              "fresh_identity": left["checkpoint_sha256"], "resumed_identity": right["checkpoint_sha256"]}
    if output:
        write_json(output, result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fresh", required=True); p.add_argument("--resumed", required=True)
    p.add_argument("--fresh-log"); p.add_argument("--interrupted-logs", nargs="*", default=[])
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(compare(a.fresh, a.resumed, fresh_log=a.fresh_log,
                             interrupted_logs=a.interrupted_logs, output=a.output), indent=2))


if __name__ == "__main__":
    main()
