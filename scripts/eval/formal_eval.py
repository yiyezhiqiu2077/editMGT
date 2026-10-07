#!/usr/bin/env python3
"""Executable formal evaluator over a frozen generated-image manifest."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from src.explicit_region.config import load_config
from src.explicit_region.metrics import MaskedLPIPS, masked_l1, masked_psnr, masked_ssim, finite_mean
from src.explicit_region.selection import summarize_selection_metrics


def tensor(path, *, mask=False):
    image = Image.open(path).convert("L" if mask else "RGB")
    array = np.asarray(image, dtype=np.float32) / 255
    if mask:
        return torch.from_numpy(array > 0)[None, None]
    return torch.from_numpy(array.transpose(2, 0, 1))[None]


def embedding_backend(config, device):
    paths = config.get("embedding_models", {})
    if not paths:
        return None
    from transformers import AutoImageProcessor, AutoModel, CLIPModel
    result = {}
    for name in ("dino", "clip"):
        root = paths.get(name)
        if root:
            processor = AutoImageProcessor.from_pretrained(root, local_files_only=True)
            cls = CLIPModel if name == "clip" else AutoModel
            result[name] = (processor, cls.from_pretrained(root, local_files_only=True).to(device).eval())
    return result


@torch.no_grad()
def embed(image_path, backend, name, device):
    processor, model = backend[name]
    inputs = processor(images=Image.open(image_path).convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if name == "clip":
        return model.get_image_features(**inputs)
    output = model(**inputs)
    return getattr(output, "pooler_output", output.last_hidden_state[:, 0])


def evaluate(manifest, config, output, device):
    lpips_metric = MaskedLPIPS(device=device)
    embeddings = embedding_backend(config, device)
    selection = config.get("selection", {})
    rows = []
    for raw in Path(manifest).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        source, target, generated = (tensor(record[key]).to(device) for key in ("source", "target", "output"))
        mask = tensor(record["mask"], mask=True).to(device)
        if source.shape != target.shape or source.shape != generated.shape or source.shape[-2:] != mask.shape[-2:]:
            raise ValueError(f"unaligned evaluator row: {record.get('sample_key')}")
        outside = ~mask
        metrics = {}
        for prefix, a, b, region in (
            ("inside", generated, target, mask), ("outside", generated, source, outside)
        ):
            for metric_name, function in (("l1", masked_l1), ("psnr", masked_psnr), ("ssim", masked_ssim)):
                value, valid = function(a, b, region)
                metrics[f"{prefix}_{metric_name}"] = float(value[0]) if valid[0] else None
            value, valid = lpips_metric(a, b, region)
            metrics[f"{prefix}_masked_lpips"] = float(value[0]) if valid[0] else None
        d_st, valid_st = lpips_metric(source, target, mask)
        d_so, valid_so = lpips_metric(source, generated, mask)
        d_ot, valid_ot = lpips_metric(generated, target, mask)
        metrics.update({
            "full_lpips_to_target": float(lpips_metric.full(generated, target)[0]),
            "d_ST": float(d_st[0]) if valid_st[0] else None,
            "d_SO": float(d_so[0]) if valid_so[0] else None,
            "d_OT": float(d_ot[0]) if valid_ot[0] else None,
        })
        metrics["progress"] = (
            metrics["d_SO"] / max(metrics["d_ST"], 1e-12)
            if metrics["d_SO"] is not None and metrics["d_ST"] is not None else None
        )
        tau_edit, tau_noop = selection.get("tau_edit"), selection.get("tau_noop")
        metrics["no_op"] = (
            metrics["d_ST"] >= tau_edit and metrics["progress"] <= tau_noop
            if tau_edit is not None and tau_noop is not None and metrics["progress"] is not None else None
        )
        if embeddings:
            from src.explicit_region.metrics import cosine_similarity
            for name in embeddings:
                eo, et, es = (embed(record[key], embeddings, name, device) for key in ("output", "target", "source"))
                label = "DINO-I" if name == "dino" else "CLIP-I"
                metrics[f"{label}-to-Target"] = float(cosine_similarity(eo, et)[0])
                metrics[f"{label}-to-Source"] = float(cosine_similarity(eo, es)[0])
        metrics["runtime_seconds"] = record.get("runtime_seconds")
        rows.append(record | {"metrics": metrics})
    # Generation seeds are repeated measures, not independent bootstrap units.
    sample_rows = {}
    for row in rows:
        key = (row.get("dataset_name", "unknown"), row.get("sample_key"))
        sample_rows.setdefault(key, []).append(row)
    samples = []
    for (dataset_name, sample_key), repeated in sample_rows.items():
        names = {name for row in repeated for name in row["metrics"]}
        averaged = {name: finite_mean(row["metrics"].get(name) for row in repeated) for name in names}
        if all(row["metrics"].get("no_op") is not None for row in repeated):
            averaged["no_op"] = sum(bool(row["metrics"]["no_op"]) for row in repeated) / len(repeated)
        samples.append({"dataset_name": dataset_name, "sample_key": sample_key,
                        "edit_type": repeated[0].get("edit_type") or "unknown", "metrics": averaged})
    grouped = defaultdict(list)
    for row in samples:
        grouped[f"dataset:{row['dataset_name']}"].append(row)
        grouped[f"edit_type:{row['edit_type']}"].append(row)
    aggregate = {}
    bootstrap = config.get("bootstrap", {"seed": 42, "resamples": 10000})
    for group, members in grouped.items():
        names = sorted({key for row in members for key in row["metrics"]})
        aggregate[group] = {}
        for name in names:
            values = np.asarray([row["metrics"].get(name) for row in members
                                 if row["metrics"].get(name) is not None and np.isfinite(row["metrics"].get(name))],dtype=np.float64)
            if not len(values):
                aggregate[group][name] = {"mean": None, "standard_error": None, "bootstrap_95_ci": None}
                continue
            rng=np.random.default_rng(int(bootstrap["seed"])); n=int(bootstrap["resamples"])
            boot=values[rng.integers(0,len(values),size=(n,len(values)))].mean(1)
            aggregate[group][name]={"mean":float(values.mean()),"standard_error":float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.0,
                                    "bootstrap_95_ci":[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))]}
        aggregate[group]["samples"] = len(members)
    result = {
        "schema": "formal-evaluator-v1", "manifest": str(Path(manifest).resolve()),
        "thresholds_preregistered": selection.get("tau_edit") is not None and selection.get("tau_noop") is not None,
        "per_generation": rows, "per_sample_after_seed_mean": samples, "aggregate": aggregate,
    }
    result["selection_summary"] = summarize_selection_metrics(result)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    result = evaluate(args.predictions_manifest, load_config(args.config), args.output, "cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"generations": len(result["per_generation"]), "samples": len(result["per_sample_after_seed_mean"]), "output": args.output, "wall_seconds": time.perf_counter()-started}))


if __name__ == "__main__":
    main()
