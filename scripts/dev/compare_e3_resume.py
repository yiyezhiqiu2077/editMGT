#!/usr/bin/env python3
"""Verify exact fresh versus split/resume equivalence at any frozen horizon."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


TRACE_FIELDS = (
    "global_step", "committed_sample_uids", "committed_geometry_seeds",
    "committed_corruption_seeds", "learning_rate", "next_learning_rate",
)


def metrics(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "train_metrics.jsonl").read_text().splitlines() if line]


def equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh8", required=True)
    parser.add_argument("--fresh4", required=True)
    parser.add_argument("--resumed8", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    fresh8, fresh4, resumed8 = map(Path, (args.fresh8, args.fresh4, args.resumed8))
    direct_trace = metrics(fresh8)
    resumed_trace = metrics(fresh4) + metrics(resumed8)
    checks = {
        field: [row[field] for row in direct_trace] == [row[field] for row in resumed_trace]
        for field in TRACE_FIELDS
    }
    for field in ("epoch", "within_epoch_cursor", "epoch_train_ce_mean", "scheduler_updates"):
        if field in direct_trace[0]:
            checks[field] = [row[field] for row in direct_trace] == [row[field] for row in resumed_trace]
    checks['optimizer_step_sequence'] = [r['global_step'] for r in direct_trace] == list(range(1, args.steps + 1))
    direct_checkpoint = fresh8 / f"checkpoint-{args.steps}"
    resumed_checkpoint = resumed8 / f"checkpoint-{args.steps}"
    direct_lora = load_file(direct_checkpoint / "adapter_model.safetensors")
    resumed_lora = load_file(resumed_checkpoint / "adapter_model.safetensors")
    direct_region = load_file(direct_checkpoint / "mask_conditioning.safetensors")
    resumed_region = load_file(resumed_checkpoint / "mask_conditioning.safetensors")
    checks["lora_state"] = equal(direct_lora, resumed_lora)
    checks["region_embedding_state"] = equal(direct_region, resumed_region)
    direct_state = torch.load(direct_checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    resumed_state = torch.load(resumed_checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    for field in ("optimizer", "scheduler", "sampler_state", "rank_rng"):
        checks[field] = equal(direct_state[field], resumed_state[field])
    payload = {
        "schema_version": "e3-resume-equivalence-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
