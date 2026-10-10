"""Fixed-identity comparison, paired bootstrap and preregistered selection."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .checkpoint import recipe_fingerprint
from .contracts import sha256_file
from .dense_checkpoint import verify_dense_checkpoint, write_json
from .group_statistics import audited_clusters, audit_split_isolation, cluster_bootstrap, read_rows

METRICS = ("inside_masked_lpips", "inside_psnr", "inside_ssim", "outside_masked_lpips",
           "outside_psnr", "outside_ssim", "full_lpips_to_target", "runtime_seconds")
CANDIDATES = list(range(3125, 31251, 3125))


def selection_registration(plan, output):
    rules = plan["selection"]
    required = ("maximum_outside_lpips_degradation", "minimum_inside_lpips_improvement")
    if rules.get("status") != "PREREGISTERED" or any(type(rules.get(k)) not in (int, float) or not math.isfinite(rules[k]) or rules[k] < 0 for k in required):
        raise RuntimeError("DENSE_SELECTION_PENDING: explicitly define thresholds before evaluation")
    for key, expected in {"primary": "magicbrush", "baseline": "E0-region",
                          "rule": "minimum_inside_lpips_subject_to_preservation",
                          "tie_break": "earlier_step"}.items():
        if rules.get(key, expected) != expected:
            raise RuntimeError(f"DENSE_SELECTION_RULE_MISMATCH: {key}")
    if plan.get("candidate_steps") != CANDIDATES:
        raise RuntimeError("DENSE_SELECTION_CANDIDATE_SET_MISMATCH")
    payload = {"schema": "dense-selection-prereg-v1", "plan": plan,
               "validation_sha256": {r["name"]: sha256_file(r["manifest"]) for r in plan["datasets"]}}
    if rules.get("bootstrap_unit") == "source_group_cluster":
        for key, expected in {"confidence_level": .95, "bootstrap_resamples": 10000,
                "require_inside_ci_improvement": True,
                "bootstrap_seed": 42, "estimand": "sample_weighted_mean_delta",
                "inside_ci_rule": "positive_improvement_lower_bound", "outside_rule": "mean_degradation",
                "on_no_eligible_checkpoint": "STOP_TEACHER_EXPORT"}.items():
            if rules.get(key) != expected:
                raise RuntimeError(f"DENSE_SELECTION_STATISTICAL_CONTRACT_MISMATCH: {key}")
        contract = json.loads(Path(plan["group_identity_contract"]).read_text())
        tests = plan["test_manifests"]
        tests = json.loads(tests) if isinstance(tests, str) else tests
        if not isinstance(tests, list) or not tests:
            raise RuntimeError("DENSE_SELECTION_TEST_ISOLATION_MANIFESTS_REQUIRED")
        payload["source_group_audit"] = audit_split_isolation(plan["train_manifest"],
            [r["manifest"] for r in plan["datasets"]], tests, contract)
        payload["protocol"] = "full-dense-selection-v2"
        payload["group_identity_contract_sha256"] = sha256_file(plan["group_identity_contract"])
        payload["selection_config_sha256"] = recipe_fingerprint(rules)
        payload["train_manifest_sha256"] = sha256_file(plan["train_manifest"])
        payload["test_manifest_sha256"] = {str(p): sha256_file(p) for p in tests}
        from .dense_runtime import audit_released_fp32
        payload["baseline_identity"] = audit_released_fp32(plan["model_root"])
        from .contracts import audit_component_identity
        payload["baseline_model_revision"] = audit_component_identity(plan["model_root"])
    if plan.get("inference_config"):
        from .config import load_config
        payload["inference_protocol_sha256"] = recipe_fingerprint(load_config(plan["inference_config"]))
    result = payload | {"sha256": recipe_fingerprint(payload)}
    if Path(output).exists():
        if json.loads(Path(output).read_text()) != result:
            raise RuntimeError("DENSE_SELECTION_PREREGISTRATION_IMMUTABLE")
    else:
        write_json(output, result)
    return result


def paired_comparison(candidate, baseline, *, seed=42, resamples=10000, strict_groups=False, confidence_level=.95):
    def keyed(result):
        rows = result["per_sample_after_seed_mean"]
        keys = [(r["dataset_name"], r.get("sample_uid", r["sample_key"])) for r in rows]
        if len(keys) != len(set(keys)):
            raise RuntimeError("DENSE_PAIRED_DUPLICATE_SAMPLE")
        return dict(zip(keys, rows))
    a, b = keyed(candidate), keyed(baseline)
    if a.keys() != b.keys():
        raise RuntimeError("DENSE_PAIRED_SAMPLE_IDENTITY_MISMATCH")
    output = {}
    for dataset in sorted({k[0] for k in a}):
        keys = sorted(k for k in a if k[0] == dataset)
        if strict_groups:
            for key in keys:
                for field in ("sample_uid", "group_id", "cluster_id", "generation_seeds", "source_sha256", "target_sha256", "region_sha256", "manifest_sha256"):
                    if a[key].get(field) is None or a[key][field] != b[key].get(field):
                        raise RuntimeError(f"DENSE_PAIRED_GROUP_OR_SEED_IDENTITY_MISMATCH: {field}")
        output[dataset] = {}
        for name in METRICS:
            valid_keys = [k for k in keys if a[k]["metrics"].get(name) is not None and b[k]["metrics"].get(name) is not None]
            if strict_groups and any((a[k]["metrics"].get(name) is None) != (b[k]["metrics"].get(name) is None) for k in keys):
                raise RuntimeError("DENSE_PAIRED_METRIC_VALIDITY_MISMATCH")
            values = np.asarray([a[k]["metrics"][name] - b[k]["metrics"][name] for k in valid_keys], dtype=float)
            if strict_groups and not np.isfinite(values).all():
                raise RuntimeError("DENSE_PAIRED_NONFINITE_METRIC")
            values = values[np.isfinite(values)]
            if not len(values):
                output[dataset][name] = {"n": 0, "mean_delta": None, "bootstrap_95_ci": None}
                continue
            if strict_groups:
                output[dataset][name] = cluster_bootstrap(values, [a[k]["cluster_id"] for k in valid_keys],
                    seed=seed, resamples=resamples, confidence_level=confidence_level)
                output[dataset][name]["excluded_empty_region_samples"] = len(keys) - len(valid_keys)
                continue
            rng = np.random.default_rng(seed)
            means = np.empty(resamples)
            for start in range(0, resamples, 1000):
                n = min(1000, resamples - start)
                means[start:start + n] = values[rng.integers(len(values), size=(n, len(values)))].mean(1)
            output[dataset][name] = {"n": len(values), "mean_delta": float(values.mean()),
                "bootstrap_95_ci": np.quantile(means, [.025, .975]).tolist()}
    return output


def assert_matching_predictions(paths, expected_seeds=None):
    canonical = None
    for path in paths:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        identities = [(r["sample_uid"], r["seed"]) for r in rows]
        if not rows or len(identities) != len(set(identities)):
            raise RuntimeError("DENSE_COMPARISON_DUPLICATE_OR_EMPTY_PREDICTIONS")
        if expected_seeds is not None:
            for uid in {r["sample_uid"] for r in rows}:
                if sorted(r["seed"] for r in rows if r["sample_uid"] == uid) != sorted(expected_seeds):
                    raise RuntimeError("DENSE_COMPARISON_INCOMPLETE_SEED_SET")
        keys = [(r["sample_uid"], r["seed"],
                 *(sha256_file(r[name]) for name in ("source", "target", "mask"))) for r in rows]
        if canonical is None:
            canonical = keys
        elif canonical != keys:
            raise RuntimeError("DENSE_COMPARISON_INPUTS_OR_SEEDS_MISMATCH")
    return True


def comparison_visual(paths, output, *, count=4):
    rows = {name: [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
            for name, path in paths.items()}
    first = next(iter(rows.values()))
    # Select first fixed identities, not best-looking results; first seed only.
    indices, seen = [], set()
    for i, row in enumerate(first):
        if row["sample_uid"] not in seen:
            seen.add(row["sample_uid"]); indices.append(i)
        if len(indices) == count:
            break
    labels = ["SOURCE", "MASK", "TARGET"] + list(rows)
    canvas = Image.new("RGB", (256 * len(labels), 280 * len(indices)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, i in enumerate(indices):
        images = [first[i][k] for k in ("source", "mask", "target")] + [r[i]["output"] for r in rows.values()]
        for j, (label, path) in enumerate(zip(labels, images)):
            draw.text((j * 256 + 4, row_index * 280 + 4), label, fill="black")
            with Image.open(path) as image:
                canvas.paste(image.convert("RGB").resize((256, 256)), (j * 256, row_index * 280 + 24))
    canvas.save(output, quality=95)


def select_checkpoint(plan, preregistration, comparison, output):
    if preregistration is None:
        result = {"status": "PENDING", "reason": "selection thresholds not preregistered"}
        write_json(output, result)
        return result
    rules = preregistration["plan"]["selection"]
    if preregistration.get("protocol") == "full-dense-selection-v2":
        actual = sorted(item.get("step") for item in comparison if item.get("backend") == "dense")
        if actual != sorted(plan["candidate_steps"]):
            raise RuntimeError("DENSE_FORMAL_SELECTION_INCOMPLETE_CANDIDATES")
    passed = []
    for item in comparison:
        if item.get("backend") != "dense":
            continue
        primary = item["paired_to_E0-region"]["magicbrush"]
        inside, outside = primary["inside_masked_lpips"], primary["outside_masked_lpips"]
        if preregistration.get("protocol") == "full-dense-selection-v2" and any(
                metric.get("bootstrap_unit") != "source_group_cluster" or metric.get("estimand") != "sample_weighted_mean_delta"
                for metric in (inside, outside)):
            raise RuntimeError("DENSE_FORMAL_SELECTION_UNAUDITED_STATISTICS")
        if inside["mean_delta"] is None or outside["mean_delta"] is None:
            continue
        ci_ok = (inside.get("bootstrap_95_ci") is not None and inside["bootstrap_95_ci"][1] < 0) if rules.get("require_inside_ci_improvement", True) else True
        if (-inside["mean_delta"] >= rules["minimum_inside_lpips_improvement"] and ci_ok
                and outside["mean_delta"] <= rules["maximum_outside_lpips_degradation"]):
            passed.append(item)
    if not passed:
        result = {"status": "PENDING", "reason": "NO_ELIGIBLE_CHECKPOINT", "action": "STOP_TEACHER_EXPORT"}
    else:
        best = min(passed, key=lambda x: (x["metrics"]["magicbrush"]["inside_masked_lpips"], x["step"]))
        identity = verify_dense_checkpoint(best["checkpoint"])
        payload = json.loads((Path(best["checkpoint"]) / "fingerprint.json").read_text())["payload"]
        result = {"schema": "selected-dense-checkpoint-v1", "status": "READY", "dev_only": True,
                  "checkpoint_path": str(Path(best["checkpoint"]).resolve()),
                  "checkpoint_sha256": identity["checkpoint_sha256"], "git_sha": payload["git_sha"],
                  "selection_preregistration": preregistration,
                  "comparison_sha256": recipe_fingerprint(comparison), "selected_step": best["step"]}
    write_json(output, result)
    return result
