#!/usr/bin/env python3
"""Exhaustive frozen-record integrity; bounded-memory montages and sampled VQ."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image, ImageDraw
import torch

from src.explicit_region.canonical import image_from_locator
from src.explicit_region.contracts import sha256_file
from src.explicit_region.fixed_corpus import (
    DATASET_PRIORITY, INTEGRITY_CHECK_NAMES, VALIDATION_NAMES, validate_frozen_language,
)
from src.explicit_region.fixed_dataset import CanonicalAlignedDataset, FixedCorpusDataset


INTEGRITY_CHECKS = INTEGRITY_CHECK_NAMES


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return ({"mean": float(array.mean()),
             **{f"p{q}": float(np.percentile(array, q)) for q in (10, 25, 50, 75, 90, 95, 99)}}
            if len(array) else {})


def order(rows, namespace):
    return sorted(rows, key=lambda pair: (
        hashlib.sha256(f"{namespace}\0{pair[1]['sample_uid']}".encode()).hexdigest(),
        pair[1]["sample_uid"],
    ))


def stratified_pairs(pairs, count, namespace):
    groups = defaultdict(deque)
    for pair in order(pairs, namespace):
        groups[pair[1]["edit_type_canonical"]].append(pair)
    result = []
    while len(result) < count and any(groups.values()):
        for key in sorted(groups):
            if groups[key]:
                result.append(groups[key].popleft())
            if len(result) == count:
                break
    return result


def tensor_image(tensor):
    return Image.fromarray((tensor.numpy().transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8))


def montage(items, path, *, count):
    """Consume one full-resolution item at a time; retain only the small canvas."""
    tile, label, header = 192, 54, 20
    canvas = Image.new("RGB", (4 * tile, header + max(count, 1) * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(("Source", "Region (white = edit)", "Region overlay", "Target")):
        draw.text((column * tile + 4, 3), title, fill="black")
    actual = 0
    for index, item in enumerate(items):
        if index >= count:
            raise ValueError("montage count differs from iterator")
        y = header + index * (tile + label)
        # PIL paste alpha must stay single-channel L, not an RGB display mask.
        mask = Image.fromarray(item["edit_region_mask"].numpy().astype(np.uint8) * 255)
        source = tensor_image(item["source_image"])
        overlay = source.copy()
        overlay.paste(Image.new("RGB", source.size, (255, 0, 0)), mask=mask.point(lambda x: x // 3))
        target = tensor_image(item["target_image"])
        for column, image in enumerate((source, mask, overlay, target)):
            canvas.paste(image.resize((tile, tile)), (column * tile, y))
        draw.text((4, y + tile + 2),
                  f"{item['dataset_name']} {item['sample_uid'][:12]} {item['edit_type']} {item['mask_semantics']}\n"
                  f"{item['instruction_en'][:150]}", fill="black")
        actual += 1
    if actual != count:
        raise ValueError("montage count differs from iterator")
    canvas.save(path, quality=90)
    canvas.close()


def failure_kind(exc):
    text = str(exc).lower()
    if isinstance(exc, FileNotFoundError) or "no such file" in text or "member missing" in text:
        return "missing_assets"
    if "hash mismatch" in text:
        return "hash_failures"
    if "image" in text and any(word in text for word in ("identify", "truncated", "broken", "decode")):
        return "decode_failures"
    if "instruction_en" in text:
        return "english_failures"
    if "unaligned aspect ratio" in text or "geometry" in text:
        return "geometry_failures"
    if "mask" in text or "region" in text:
        return "mask_failures"
    return "runtime_read_failures"


def audit_dataset(dataset, roots, *, error_sink=None, max_examples=20):
    """Attempt every row, recording actual completed checks (never assumed zero).

    The strict runtime reader is the authority for hashes, decoding, alignment,
    geometry, and post-transform nonempty masks. On a failed read downstream
    checks are explicitly incomplete, rather than marked as successful.
    """
    rows = dataset.rows
    completed = Counter({key: 0 for key in INTEGRITY_CHECKS})
    failures = Counter()
    examples = []
    dimensions = defaultdict(list)
    dataset_completed = defaultdict(Counter)
    dataset_failures = defaultdict(Counter)
    for index, row in enumerate(rows):
        errors = []
        name = row["dataset_name"]
        try:
            validate_frozen_language(row)
            completed["english"] += 1; dataset_completed[name]["english"] += 1
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(("english_failures", str(exc)))
        try:
            item = dataset[index]
        except Exception as exc:
            kind = failure_kind(exc)
            if kind not in {item[0] for item in errors}:
                errors.append((kind, str(exc)))
        else:
            for check in ("assets", "decode", "hash", "geometry"):
                completed[check] += 1; dataset_completed[name][check] += 1
            sizes = item["geometry"]["prealignment"]["original_sizes"]
            for role in ("source", "target", "mask"):
                width, height = sizes[role]
                dimensions[f"{name}/{role}_width"].append(width)
                dimensions[f"{name}/{role}_height"].append(height)
                dimensions[f"{name}/{role}_aspect"].append(width / height)
            del item  # No list of 1024px tensors survives this iteration.
            try:
                with image_from_locator(row["region_locator"], roots[name]) as mask:
                    fraction = float((np.asarray(mask.convert("L")) > 0).mean())
                if fraction <= 0 or not math.isclose(fraction, float(row["region_fraction"]), abs_tol=1e-7):
                    raise ValueError("native region_fraction mismatch or empty region")
                semantics = row["mask_semantics"]
                if not isinstance(semantics, str) or not semantics.strip() or any(
                    token in semantics.upper() for token in ("REPLACE", "NOT_RUN", "UNKNOWN")
                ):
                    raise ValueError("unreviewed mask_semantics")
                completed["mask"] += 1; dataset_completed[name]["mask"] += 1
            except Exception as exc:
                errors.append(("mask_failures", str(exc)))
        for kind, text in errors:
            failures[kind] += 1; dataset_failures[name][kind] += 1
            detail = {"index": index, "sample_uid": row["sample_uid"], "kind": kind, "error": text}
            if len(examples) < max_examples:
                examples.append(detail)
            if error_sink is not None:
                error_sink.write(json.dumps(detail, sort_keys=True) + "\n")

    by_dataset = {}
    for name in sorted({row["dataset_name"] for row in rows}):
        members = [row for row in rows if row["dataset_name"] == name]
        by_dataset[name] = {
            "selected": len(members),
            "unique_sources": len({row["source_sha256"] for row in members}),
            "completed_checks": {key: dataset_completed[name][key] for key in INTEGRITY_CHECKS},
            "failures": dict(dataset_failures[name]),
            "edit_type_counts": dict(Counter(row["edit_type_canonical"] for row in members)),
            "mask_semantics_counts": dict(Counter(row["mask_semantics"] for row in members)),
            "region_fraction": stats([row["region_fraction"] for row in members]),
            "samples_per_source": stats(list(Counter(row["source_sha256"] for row in members).values())),
            **{key.split("/", 1)[1]: stats(values) for key, values in dimensions.items()
               if key.startswith(name + "/")},
        }
    return {
        "status": "PASS" if not failures and all(completed[k] == len(rows) for k in INTEGRITY_CHECKS) else "FAIL",
        "mode": "exhaustive", "total_rows": len(rows), "attempted_rows": len(rows),
        "completed_checks": dict(completed), "failure_counts": dict(failures),
        "failure_examples": examples, "datasets": by_dataset,
    }


def sampled_containment(dataset, pairs, vq, device):
    from src.explicit_region.masks import pixel_mask_to_token_mask
    from src.v2_utils import prepare_cond_token

    values = []
    for index, _ in pairs:
        item = dataset[index]
        with torch.no_grad():
            source = prepare_cond_token(None, item["source_image"][None].to(device, dtype=torch.float32), vq)
            target = prepare_cond_token(None, item["target_image"][None].to(device, dtype=torch.float32), vq)
        grid = math.isqrt(source.shape[1])
        change = source.reshape(1, grid, grid).ne(target.reshape(1, grid, grid))
        region = pixel_mask_to_token_mask(item["edit_region_mask"][None], (grid, grid)).to(change.device)
        total = max(int(change.sum()), 1)
        values.append((int((change & ~region).sum()) / total, int((change & region).sum()) / total))
    return {"status": "PASS", "mode": "sampled", "samples": len(values),
            "sample_uids": [row["sample_uid"] for _, row in pairs],
            "outside_change_ratio": stats([value[0] for value in values]),
            "inside_change_coverage": stats([value[1] for value in values])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    for name in DATASET_PRIORITY:
        parser.add_argument(f"--{name}-root", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--magicbrush-dev-root", required=True)
    parser.add_argument("--model-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--containment-count", type=int, default=500)
    parser.add_argument("--skip-vq", action="store_true", help="Integrity-only diagnostics; cannot finalize READY")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.skip_vq and (not args.model_root or args.containment_count <= 0):
        parser.error("sampled VQ requires --model-root and positive --containment-count")
    roots = {name: getattr(args, f"{name}_root") for name in DATASET_PRIORITY}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out.parent / "CORPUS_READY.json").unlink(missing_ok=True)
    report = {"schema": "fixed200k-audit-v2", "status": "RUNNING", "integrity": {
        "mode": "exhaustive", "resolution": 1024, "manifests": {}},
        "vq_containment": {"status": "NOT_RUN", "mode": "sampled"}}

    def save():
        (out / "audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    save()  # Invalidate any previous success even if dataset construction fails.
    train = FixedCorpusDataset(args.manifest, roots, resolution=1024, verify_hashes=True)
    manifests = [("train_200k", Path(args.manifest), roots)] + [
        (f"validation_{name}", Path(args.validation_dir) / f"{name}.jsonl",
         {**roots, "magicbrush": args.magicbrush_dev_root}) for name in VALIDATION_NAMES
    ]
    with (out / "integrity_failures.jsonl").open("w", encoding="utf-8") as errors:
        for name, path, manifest_roots in manifests:
            before = sha256_file(path)
            if name == "train_200k":
                dataset = train
            else:
                with path.open(encoding="utf-8") as handle:
                    rows = [json.loads(line) for line in handle if line.strip()]
                dataset_names = {row["dataset_name"] for row in rows}
                if len(dataset_names) != 1:
                    raise ValueError(f"validation must contain one dataset: {name}")
                dataset = CanonicalAlignedDataset(rows, manifest_roots[next(iter(dataset_names))],
                                                  resolution=1024, verify_hashes=True)
            result = audit_dataset(dataset, manifest_roots, error_sink=errors)
            if sha256_file(path) != before:
                raise RuntimeError(f"manifest changed during audit: {path}")
            report["integrity"]["manifests"][name] = {"sha256": before, **result}
            save()
            if name != "train_200k":
                del dataset
    results = report["integrity"]["manifests"]
    report["integrity"]["status"] = "PASS" if all(r["status"] == "PASS" for r in results.values()) else "FAIL"
    report["integrity"]["total_rows"] = sum(r["total_rows"] for r in results.values())
    report["total"] = len(train)
    report["datasets"] = results["train_200k"]["datasets"]
    if report["integrity"]["status"] != "PASS":
        report["status"] = "FAIL"; save()
        raise SystemExit("frozen integrity audit failed; see integrity_failures.jsonl")

    by_dataset = defaultdict(list)
    for pair in enumerate(train.rows):
        by_dataset[pair[1]["dataset_name"]].append(pair)
    final_pairs, ordered = [], {}
    for name in DATASET_PRIORITY:
        ordered[name] = order(by_dataset[name], f"audit-{name}")
        chosen = ordered[name][:100]
        montage((train[index] for index, _ in chosen), out / f"{name}_100.jpg", count=len(chosen))
        final_pairs.extend(stratified_pairs(by_dataset[name], 50, f"final-{name}"))
    montage((train[index] for index, _ in final_pairs), out / "final_200k_montage.jpg", count=len(final_pairs))
    report["montages"] = {path.name: sha256_file(path) for path in sorted(out.glob("*.jpg"))}
    if not args.skip_vq:
        from diffusers import VQModel
        vq = VQModel.from_pretrained(args.model_root, subfolder="vqvae", local_files_only=True,
                                    torch_dtype=torch.float32).to(args.device, dtype=torch.float32).eval()
        vq_results = {name: sampled_containment(train, ordered[name][:args.containment_count], vq, args.device)
                      for name in DATASET_PRIORITY}
        report["vq_containment"] = {"status": "PASS", "mode": "sampled", "dtype": "float32",
                                    "requested_per_dataset": args.containment_count, "datasets": vq_results}
    report["status"] = "PASS" if not args.skip_vq else "INTEGRITY_ONLY"
    save()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
