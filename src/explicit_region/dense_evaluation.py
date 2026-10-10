"""Fixed-identity comparison, paired bootstrap and preregistered selection."""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .checkpoint import recipe_fingerprint
from .contracts import sha256_file
from .dense_checkpoint import verify_dense_checkpoint, write_json

METRICS = ("inside_masked_lpips", "inside_psnr", "inside_ssim", "outside_masked_lpips",
           "outside_psnr", "outside_ssim", "full_lpips_to_target", "runtime_seconds")
CANDIDATES = list(range(3125, 31251, 3125))


def selection_registration(plan, output):
    rules = plan["selection"]
    required = ("maximum_outside_lpips_degradation", "minimum_inside_lpips_improvement")
    if rules.get("status") != "PREREGISTERED" or any(not isinstance(rules.get(k), (int, float)) or rules[k] < 0 for k in required):
        raise RuntimeError("DENSE_SELECTION_PENDING: explicitly define thresholds before evaluation")
    if plan.get("candidate_steps") != CANDIDATES:
        raise RuntimeError("DENSE_SELECTION_CANDIDATE_SET_MISMATCH")
    payload = {"schema": "dense-selection-prereg-v1", "plan": plan,
               "validation_sha256": {r["name"]: sha256_file(r["manifest"]) for r in plan["datasets"]}}
    result = payload | {"sha256": recipe_fingerprint(payload)}
    if Path(output).exists():
        if json.loads(Path(output).read_text()) != result:
            raise RuntimeError("DENSE_SELECTION_PREREGISTRATION_IMMUTABLE")
    else:
        write_json(output, result)
    return result


def paired_comparison(candidate, baseline, *, seed=42, resamples=10000):
    def keyed(result):
        return {(r["dataset_name"], r["sample_key"]): r["metrics"]
                for r in result["per_sample_after_seed_mean"]}
    a, b = keyed(candidate), keyed(baseline)
    if a.keys() != b.keys():
        raise RuntimeError("DENSE_PAIRED_SAMPLE_IDENTITY_MISMATCH")
    output = {}
    for dataset in sorted({k[0] for k in a}):
        keys = sorted(k for k in a if k[0] == dataset)
        output[dataset] = {}
        for name in METRICS:
            values = np.asarray([a[k][name] - b[k][name] for k in keys
                                 if a[k].get(name) is not None and b[k].get(name) is not None], dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                output[dataset][name] = {"n": 0, "mean_delta": None, "bootstrap_95_ci": None}
                continue
            rng = np.random.default_rng(seed)
            means = np.empty(resamples)
            for start in range(0, resamples, 1000):
                n = min(1000, resamples - start)
                means[start:start + n] = values[rng.integers(len(values), size=(n, len(values)))].mean(1)
            output[dataset][name] = {"n": len(values), "mean_delta": float(values.mean()),
                "bootstrap_95_ci": np.quantile(means, [.025, .975]).tolist()}
    return output


def assert_matching_predictions(paths):
    canonical = None
    for path in paths:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
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
    labels = ["SOURCE", "TARGET", "MASK"] + list(rows)
    canvas = Image.new("RGB", (256 * len(labels), 280 * len(indices)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, i in enumerate(indices):
        images = [first[i][k] for k in ("source", "target", "mask")] + [r[i]["output"] for r in rows.values()]
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
    passed = []
    for item in comparison:
        if item.get("backend") != "dense":
            continue
        primary = item["paired_to_E0-region"]["magicbrush"]
        inside, outside = primary["inside_masked_lpips"], primary["outside_masked_lpips"]
        if inside["mean_delta"] is None or outside["mean_delta"] is None:
            continue
        bound = inside["bootstrap_95_ci"][1] if rules.get("require_inside_ci_improvement", True) else inside["mean_delta"]
        if bound < -rules["minimum_inside_lpips_improvement"] and outside["mean_delta"] <= rules["maximum_outside_lpips_degradation"]:
            passed.append(item)
    if not passed:
        result = {"status": "PENDING", "reason": "no checkpoint satisfies preregistered DEV rules"}
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
