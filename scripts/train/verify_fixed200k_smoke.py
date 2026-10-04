#!/usr/bin/env python3
"""Fail closed unless the real 8-rank smoke has exact finite resume state."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.explicit_region.checkpoint import (
    TRAINING_STATE_SCHEMA_VERSION, recipe_fingerprint, validate_rank_rng_state,
)
from src.explicit_region.contracts import sha256_file
from src.explicit_region.epoch_sampler import SAMPLER_SCHEMA_VERSION
from src.explicit_region.gates import SMOKE_CONFIGS, build_gate_identity


class SmokeMismatch(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise SmokeMismatch(message)


def assert_finite(value, name="state"):
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), f"{name}: non-finite tensor")
    elif isinstance(value, np.ndarray):
        require(bool(np.isfinite(value).all()), f"{name}: non-finite array")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{name}[{index}]")
    elif isinstance(value, (float, np.floating)):
        require(math.isfinite(value), f"{name}: non-finite number")


def assert_exact(left, right, name="state"):
    """Strict keys, shape, dtype, finite values, and exact equality (no tolerances)."""
    assert_finite(left, name)
    assert_finite(right, name)
    if isinstance(left, torch.Tensor):
        require(isinstance(right, torch.Tensor), f"{name}: tensor type mismatch")
        require(left.shape == right.shape and left.dtype == right.dtype, f"{name}: tensor schema mismatch")
        require(torch.equal(left, right), f"{name}: tensor values differ")
    elif isinstance(left, np.ndarray):
        require(isinstance(right, np.ndarray), f"{name}: array type mismatch")
        require(left.shape == right.shape and left.dtype == right.dtype, f"{name}: array schema mismatch")
        require(np.array_equal(left, right), f"{name}: array values differ")
    elif isinstance(left, dict):
        require(isinstance(right, dict) and left.keys() == right.keys(), f"{name}: dictionary keys differ")
        for key in left:
            assert_exact(left[key], right[key], f"{name}.{key}")
    elif isinstance(left, (list, tuple)):
        require(type(left) is type(right) and len(left) == len(right), f"{name}: sequence schema mismatch")
        for index, (a, b) in enumerate(zip(left, right)):
            assert_exact(a, b, f"{name}[{index}]")
    else:
        require(type(left) is type(right) and left == right, f"{name}: values differ")


def rows(root):
    with (Path(root) / "train_metrics.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten(values, key):
    return [item for row in values for item in row[key]]


def verify_metric_rows(values, expected_steps):
    require([row["global_step"] for row in values] == list(expected_steps), "optimizer step sequence differs")
    required = {
        "global_step", "committed_global_sample_count", "committed_sample_uids",
        "committed_geometry_seeds", "committed_corruption_seeds", "learning_rate",
        "next_learning_rate", "finite", "finite_parameters", "sample_mean_ce",
        "token_mean_ce", "quality_status", "grad_norm_pre_clip", "grad_norm_post_clip",
        "parameter_norm", "optimizer_update_norm", "update_ratio", "max_abs_logit",
    }
    for row in values:
        step = row["global_step"]
        require(required <= row.keys(), f"step {step}: required metric missing")
        assert_finite(row, f"metrics.step{step}")
        require(row["finite"] is True and row["finite_parameters"] is True, f"step {step}: non-finite training")
        require(row["quality_status"] in {"QUALITY_OK", "INSUFFICIENT_BASELINE"}, f"step {step}: quality failed/disabled")
        require(row["committed_global_sample_count"] == step * 32, f"step {step}: cursor mismatch")
        for key in ("committed_sample_uids", "committed_geometry_seeds", "committed_corruption_seeds"):
            require(isinstance(row[key], list) and len(row[key]) == 32, f"step {step}: partial {key}")
        require(all(isinstance(uid, str) and uid for uid in row["committed_sample_uids"]), "missing sample UID")
        for key in ("committed_geometry_seeds", "committed_corruption_seeds"):
            require(all(type(seed) is int and seed >= 0 for seed in row[key]), f"missing/invalid {key}")


def verify_provenances(roots):
    provenances = [json.loads((root / "run_provenance.json").read_text(encoding="utf-8")) for root in roots]
    ready = provenances[0]["resolved_config"]["data"]["corpus_ready"]
    identity = build_gate_identity(ready, repo_root=ROOT)
    for root, relative, provenance in zip(roots, SMOKE_CONFIGS, provenances):
        assert_exact(provenance["gate_identity"], identity, f"{root.name}.gate_identity")
        for key in ("git_sha", "git_dirty_diff_sha256", "uv_lock_sha256", "corpus_ready_sha256"):
            require(provenance[key] == identity[key], f"{root.name}: stale {key}")
        require(provenance["world_size"] == 8, "smoke provenance must record eight ranks")
        require(provenance["deterministic_algorithms"] is True and provenance["cudnn_deterministic"] is True, "deterministic algorithms were not enabled")
        require(provenance["cudnn_benchmark"] is False and provenance["allow_tf32"] is False and provenance["torch_compile"] is False, "runtime correctness flags changed")
        expected = identity["smoke_configs"][relative]
        config = provenance["resolved_config"]
        digest = recipe_fingerprint(config)
        require(provenance["resolved_config_sha256"] == digest == expected["resolved_sha256"], f"{root.name}: wrong resolved smoke config")
        require(provenance["config_sha256"] == expected["sha256"], f"{root.name}: wrong raw smoke config")
        require(Path(provenance["config_path"]).resolve() == (ROOT / relative).resolve(), f"{root.name}: wrong smoke config path")
        require(Path(config["output_dir"]).resolve() == root.resolve(), f"{root.name}: output provenance mismatch")
        require(config["data"]["corpus_ready"] == ready, "different corpus READY paths")
        require(config["batch_per_gpu"] == 1 and config["gradient_accumulation"] == 4, "smoke requires global32")
        require(config["resolution"] == 1024 and config["precision"] == "bf16", "smoke precision/resolution changed")
        require(config["torch_compile"] is False and config["component_dtypes"]["vqvae"] == "fp32", "smoke correctness contract changed")
        require(config["warmup_steps"] == 200 and config["scheduler_horizon_steps"] == 6250, "smoke scheduler recipe changed")
        require(config["corruption"]["mode"] == "roi_hardlock" and config["corruption"]["persistent_conditioning"] is True and config["lora"]["scope"] == "both", "smoke must use E3")
    return identity, provenances


def read_checkpoint(root, step, provenance):
    directory = root / f"checkpoint-{step}"
    weights = {}
    for filename in ("adapter_model.safetensors", "mask_conditioning.safetensors"):
        weights[filename] = load_file(str(directory / filename))
        require(bool(weights[filename]), f"empty {filename}")
        assert_finite(weights[filename], filename)
    require(set(weights["mask_conditioning.safetensors"]) == {"edit_region_embedding"}, "invalid mask state keys")
    state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=False)
    required = {
        "schema_version", "optimizer", "scheduler", "rank_rng", "global_optimizer_step",
        "committed_global_sample_count", "sampler_schema", "quality_state", "sampler_state",
        "metrics_state", "fingerprint",
    }
    require(required <= state.keys(), "missing required training state")
    require(state["schema_version"] == TRAINING_STATE_SCHEMA_VERSION, "unsupported training state schema")
    require(state["global_optimizer_step"] == step and state["committed_global_sample_count"] == step * 32, "checkpoint step/cursor mismatch")
    assert_finite(state)
    validate_rank_rng_state(state["rank_rng"], world_size=8)
    require(all(rank["torch_cuda"] for rank in state["rank_rng"]["states"].values()), "8-GPU smoke requires CUDA RNG on every rank")
    sampler = state["sampler_state"]
    expected_sampler = {
        "sampler_schema_version": SAMPLER_SCHEMA_VERSION, "epoch": 0,
        "samples_consumed_in_epoch": step * 32, "world_size": 8,
        "batch_per_gpu": 1, "gradient_accumulation": 4,
        "global_optimizer_step_in_epoch": step, "global_microbatch_in_epoch": step * 4,
    }
    require(state["sampler_schema"] == SAMPLER_SCHEMA_VERSION, "sampler schema mismatch")
    for key, value in expected_sampler.items():
        require(sampler[key] == value, f"sampler {key} mismatch")
    require(isinstance(sampler["epoch_permutation_sha256"], str) and len(sampler["epoch_permutation_sha256"]) == 64, "missing epoch permutation")
    fingerprint = json.loads((directory / "fingerprint.json").read_text(encoding="utf-8"))
    payload = fingerprint["payload"]
    require(payload["schema"] == "explicit-region-sft-v2", "unsupported recipe fingerprint schema")
    require(recipe_fingerprint(payload) == fingerprint["sha256"] == state["fingerprint"], "fingerprint mismatch")
    assert_exact(json.loads((directory / "trainable_config.json").read_text(encoding="utf-8")), payload, "trainable config")
    config = provenance["resolved_config"]
    expected_recipe = {
        key: value for key, value in config.items()
        if key not in {"output_dir", "max_optimizer_steps", "checkpoint_steps", "run_type"}
    }
    assert_exact(payload["training_config"], expected_recipe, "checkpoint training recipe")
    for key in ("git_sha", "git_dirty_diff_sha256", "uv_lock_sha256"):
        require(payload[key] == provenance[key], f"checkpoint provenance mismatch: {key}")
    require(payload["world_size"] == 8, "checkpoint world size mismatch")
    require(payload["data_content_hashes"]["data.corpus_ready"]["sha256"] == provenance["corpus_ready_sha256"], "checkpoint READY mismatch")
    manifest_sha = payload["data_content_hashes"]["data.manifest"]["sha256"]
    require(sampler["train_200k_sha256"] == manifest_sha, "sampler manifest identity mismatch")
    metadata = json.loads((directory / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    for key in ("schema_version", "global_optimizer_step", "committed_global_sample_count", "fingerprint"):
        require(metadata[key] == state[key], f"checkpoint metadata mismatch: {key}")
    require(metadata["resolved_config_sha256"] == provenance["resolved_config_sha256"], "checkpoint resolved config mismatch")
    optimizer = state["optimizer"]
    require(bool(optimizer["state"]) and bool(optimizer["param_groups"]), "missing optimizer moments/groups")
    optimizer_ids = [parameter for group in optimizer["param_groups"] for parameter in group["params"]]
    require(len(optimizer_ids) == len(set(optimizer_ids)) and set(optimizer_ids) == set(optimizer["state"]), "missing/duplicate optimizer parameter state")
    for parameter in optimizer["state"].values():
        require({"step", "exp_avg", "exp_avg_sq"} <= parameter.keys(), "missing AdamW state")
        require(float(parameter["step"]) == step, "optimizer update count mismatch")
    require(state["scheduler"]["last_epoch"] == step, "scheduler advanced more/less than once per optimizer update")
    next_lr = float(config["optimizer"]["learning_rate"]) * min(1.0, step / 200)
    require(state["scheduler"]["_last_lr"] == [next_lr], "incorrect 200-update warmup")
    require(all(group["lr"] == next_lr for group in optimizer["param_groups"]), "optimizer/scheduler LR mismatch")
    quality = state["quality_state"]
    require({"losses", "clips", "bad_windows", "status", "reason"} <= quality.keys(), "quality state incomplete")
    require(quality["status"] in {"QUALITY_OK", "INSUFFICIENT_BASELINE"} and quality["reason"] is None, "checkpoint quality failure")
    require([entry[0] for entry in quality["losses"]] == list(range(1, step + 1)), "quality history missing steps")
    require(len(quality["clips"]) == step, "quality clip history missing steps")
    return weights, state, {filename: sha256_file(directory / filename) for filename in (
        "adapter_model.safetensors", "mask_conditioning.safetensors", "training_state.pt",
        "fingerprint.json", "checkpoint_metadata.json",
    )}


def verify_smoke(fresh20, fresh10, resume20):
    roots = [Path(value) for value in (fresh20, fresh10, resume20)]
    report = {"schema": "fixed200k-smoke-verification-v2", "status": "FAIL"}
    try:
        require(len({root.resolve() for root in roots}) == 3, "smoke runs must use separate directories")
        identity, provenances = verify_provenances(roots)
        full, first, resumed = [rows(root) for root in roots]
        verify_metric_rows(full, range(1, 21))
        verify_metric_rows(first, range(1, 11))
        verify_metric_rows(resumed, range(11, 21))
        split = first + resumed
        # Memory peaks are process-lifetime diagnostics, not scientific state.
        ignore = {"peak_vram_bytes", "peak_reserved_vram_bytes"}
        assert_exact([{k: v for k, v in row.items() if k not in ignore} for row in full],
                     [{k: v for k, v in row.items() if k not in ignore} for row in split], "training metrics")
        uids = flatten(full, "committed_sample_uids")
        require(len(uids) == len(set(uids)) == 640, "smoke must commit exactly 640 unique records")
        base_lr = float(provenances[0]["resolved_config"]["optimizer"]["learning_rate"])
        for row in full:
            require(row["learning_rate"] == base_lr * ((row["global_step"] - 1) / 200), "applied LR sequence has world-size warmup factor")
            require(row["next_learning_rate"] == base_lr * (row["global_step"] / 200), "next LR sequence has world-size warmup factor")
        evidence = {}
        for step, right_index in ((10, 1), (20, 2)):
            left_weights, left_state, left_hashes = read_checkpoint(roots[0], step, provenances[0])
            right_weights, right_state, right_hashes = read_checkpoint(roots[right_index], step, provenances[right_index])
            assert_exact(left_weights, right_weights, f"checkpoint-{step}.weights")
            assert_exact(left_state, right_state, f"checkpoint-{step}.training_state")
            evidence[str(step)] = {"uninterrupted": left_hashes, "split": right_hashes}
        # JSON sidecar and the serialized quality history must agree too.
        for index, root in enumerate(roots):
            step = 10 if index == 1 else 20
            quality = json.loads((root / "quality_status.json").read_text(encoding="utf-8"))
            state = torch.load(root / f"checkpoint-{step}" / "training_state.pt", map_location="cpu", weights_only=False)
            require(quality == json.loads(json.dumps(state["quality_state"])), "quality sidecar mismatch")
        assert_exact(build_gate_identity(provenances[0]["resolved_config"]["data"]["corpus_ready"], repo_root=ROOT),
                     identity, "identity changed during smoke verification")
        report.update(
            status="PASS", gate_identity=identity,
            fresh_steps=[row["global_step"] for row in full],
            split_steps=[row["global_step"] for row in split], samples=640, unique_samples=640,
            sample_sequence_match=True, geometry_seed_sequence_match=True,
            corruption_seed_sequence_match=True, learning_rate_sequence_match=True,
            finite_state=True, exact_weights_match=True, optimizer_match=True,
            scheduler_match=True, quality_state_match=True, sampler_cursor_match=True,
            rank_rng_match=True, exact_metrics_match=True,
            adapter_max_abs_diff=0.0, mask_max_abs_diff=0.0, checkpoint_evidence=evidence,
        )
    except Exception as exc:
        # Always replace any old PASS artifact with FAIL, including missing or
        # malformed files/keys and unsupported schemas.
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh20", required=True)
    parser.add_argument("--fresh10", required=True)
    parser.add_argument("--resume20", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = verify_smoke(args.fresh20, args.fresh10, args.resume20)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
