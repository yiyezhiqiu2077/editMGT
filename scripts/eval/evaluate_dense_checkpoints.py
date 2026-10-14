#!/usr/bin/env python3
"""Evaluate every candidate and both timestep baselines; selection defaults pending."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.explicit_region.config import load_config
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dense_checkpoint import verify_dense_checkpoint, write_json
from src.explicit_region.dense_evaluation import (METRICS, CANDIDATES, selection_registration,
    paired_comparison, assert_matching_predictions, comparison_visual, select_checkpoint)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", default="configs/eval/dense_dev_plan.yaml")
    p.add_argument("--preregister-selection", action="store_true")
    p.add_argument("--checkpoint-steps", type=int, nargs="+")
    p.add_argument("--count", type=int)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    plan = load_config(a.plan)
    out = Path(plan["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    registration_path = out / "selection_preregistration.json"
    if a.preregister_selection:
        if any(p.name != "selection_preregistration.json" for p in out.iterdir()):
            raise RuntimeError("SELECTION_PREREGISTRATION_MUST_PRECEDE_EVALUATION")
        print(json.dumps(selection_registration(plan, registration_path)))
        return
    registration = json.loads(registration_path.read_text()) if registration_path.exists() else None
    if registration is None and not (a.count or a.checkpoint_steps):
        raise RuntimeError("FORMAL_DEV_REQUIRES_PRIOR_IMMUTABLE_PREREGISTRATION")
    if registration is not None:
        expected = selection_registration(plan, registration_path)
        if registration != expected:
            raise RuntimeError("DENSE_SELECTION_PLAN_CHANGED")
    if a.count or a.checkpoint_steps:
        if registration is not None:
            raise RuntimeError("PARTIAL_DEV_DIAGNOSTIC_CANNOT_SELECT_FORMAL_TEACHER")
    steps = a.checkpoint_steps or plan["candidate_steps"]
    if not a.checkpoint_steps and steps != CANDIDATES:
        raise RuntimeError("DENSE_FORMAL_CANDIDATE_SET_MISMATCH")
    models = [{"name": "E0-official", "backend": "released", "timestep": "official_upstream_timestep"},
              {"name": "E0-region", "backend": "released", "timestep": "roi_relative"}]
    if plan.get("lora_checkpoint"):
        models.append({"name": "E3-LoRA", "backend": "lora", "checkpoint": plan["lora_checkpoint"], "timestep": "roi_relative"})
    for step in steps:
        root = Path(plan["checkpoint_root"]) / f"checkpoint-{step}"
        verify_dense_checkpoint(root)
        models.append({"name": f"Dense-{step}", "backend": "dense", "checkpoint": str(root), "step": step, "timestep": "roi_relative"})
    infer = load_config(plan["inference_config"])
    identity = {"plan_sha256": recipe_fingerprint(plan), "inference_sha256": recipe_fingerprint(infer),
                "selection_preregistration_sha256": registration["sha256"] if registration else None,
                "count": a.count, "steps": steps,
                "validation_sha256": {d["name"]: sha256_file(d["manifest"]) for d in plan["datasets"]}}
    from src.explicit_region.dense_runtime import audit_released_fp32
    identity["released_transformer_sha256"] = audit_released_fp32(plan["model_root"])["files"]
    identity["model_checkpoints"] = {}
    for model in models:
        if model["backend"] == "dense":
            identity["model_checkpoints"][model["name"]] = verify_dense_checkpoint(model["checkpoint"])["checkpoint_sha256"]
        elif model["backend"] == "lora":
            root = Path(model["checkpoint"])
            identity["model_checkpoints"][model["name"]] = {n: sha256_file(root / n) for n in
                ("adapter_model.safetensors", "mask_conditioning.safetensors", "fingerprint.json", "trainable_config.json")}
    ledger = out / "evaluation_identity.json"
    if ledger.exists() and json.loads(ledger.read_text()) != identity:
        raise RuntimeError("DENSE_EVALUATION_IDENTITY_CHANGED")
    write_json(ledger, identity)
    results, predictions = {}, {}
    for model in models:
        results[model["name"]], predictions[model["name"]] = {}, {}
        for dataset in plan["datasets"]:
            name = dataset["name"]
            folder = out / model["name"] / name
            manifest = folder / "predictions.jsonl"
            metrics = folder / "metrics.json"
            if not manifest.exists():
                command = [sys.executable, str(ROOT / "scripts/eval/generate_dense_evaluation.py"),
                    "--backend", model["backend"], "--model-root", plan["model_root"], "--dataset", name,
                    "--manifest", dataset["manifest"], "--canonical-root", dataset["root"],
                    "--config", plan["inference_config"], "--timestep-mode", model["timestep"],
                    "--output-dir", str(folder), "--device", a.device]
                if model.get("checkpoint"):
                    command += ["--checkpoint", model["checkpoint"]]
                if plan.get("group_identity_contract"):
                    command += ["--group-identity-contract", plan["group_identity_contract"]]
                if a.count:
                    command += ["--count", str(a.count)]
                subprocess.run(command, cwd=ROOT, check=True)
            if not metrics.exists():
                from scripts.eval.formal_eval import evaluate
                evaluate(manifest, infer, metrics, a.device)
            results[model["name"]][name] = json.loads(metrics.read_text())
            predictions[model["name"]][name] = manifest
    comparison = []
    for dataset in plan["datasets"]:
        name = dataset["name"]
        assert_matching_predictions([predictions[m["name"]][name] for m in models], expected_seeds=infer["generation_seeds"])
        comparison_visual({m["name"]: predictions[m["name"]][name] for m in models}, out / f"{name}_fixed_cases.jpg")
    for model in models:
        item = dict(model, metrics={}, **{"paired_to_E0-region": {}})
        for dataset in plan["datasets"]:
            name = dataset["name"]
            result = results[model["name"]][name]
            samples = result["per_sample_after_seed_mean"]
            from src.explicit_region.metrics import finite_mean
            item["metrics"][name] = {k: finite_mean(r["metrics"].get(k) for r in samples) for k in METRICS}
            item["paired_to_E0-region"].update(paired_comparison(result, results["E0-region"][name], **{
                "seed": plan["selection"].get("bootstrap_seed", infer["bootstrap"]["seed"]),
                "resamples": plan["selection"].get("bootstrap_resamples", infer["bootstrap"]["resamples"]),
                "confidence_level": plan["selection"].get("confidence_level", .95),
                "strict_groups": registration is not None and registration.get("protocol") == "full-dense-selection-v2"}))
        comparison.append(item)
    write_json(out / "checkpoint_comparison.json", comparison)
    with (out / "checkpoint_comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "dataset", "step", *METRICS])
        writer.writeheader()
        for item in comparison:
            for dataset, metrics in item["metrics"].items():
                writer.writerow({"model": item["name"], "dataset": dataset, "step": item.get("step"), **metrics})
    write_json(out / "quality_trend.json", [x for x in comparison if x["backend"] == "dense"])
    print(json.dumps(select_checkpoint(plan, registration, comparison, out / "SELECTED_DENSE_CHECKPOINT.json")))


if __name__ == "__main__":
    main()
