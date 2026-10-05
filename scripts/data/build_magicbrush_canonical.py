#!/usr/bin/env python3
"""Convert an existing MagicBrush manifest into portable canonical records."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image

from src.explicit_region.canonical import (
    SCHEMA_VERSION, expected_locator_hash, sample_uid_for,
)
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dataset import _align_to_mask_coordinates


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest, root = Path(args.manifest), Path(args.dataset_root).resolve()
    rows, rejections = [], []
    rejection_counts = Counter()
    for index, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines()):
        if not raw.strip():
            continue
        source = json.loads(raw)
        sample_key = source["sample_key"]
        locators = {
            "source": {"backend": "file", "relative_path": source["source"]},
            "target": {"backend": "file", "relative_path": source["target"]},
            "region": {"backend": "file", "relative_path": source["mask_edit"]},
        }
        with (
            Image.open(root / source["source"]) as source_image,
            Image.open(root / source["target"]) as target_image,
            Image.open(root / source["mask_edit"]) as mask_image,
        ):
            region_fraction = float((np.asarray(mask_image.convert("L")) > 0).mean())
            try:
                aligned_source, aligned_target, _ = _align_to_mask_coordinates(
                    source_image, target_image, mask_image, sample_key,
                )
                if aligned_source is not source_image:
                    aligned_source.close()
                if aligned_target is not target_image:
                    aligned_target.close()
            except ValueError as exc:
                if "unaligned aspect ratio" not in str(exc):
                    raise
                rejection_counts["unaligned_aspect_ratio"] += 1
                rejections.append({
                    "manifest_index": index,
                    "sample_key": sample_key,
                    "img_id": source.get("img_id"),
                    "reason": "unaligned_aspect_ratio",
                    "error": str(exc),
                })
                continue
        row = {
            "schema_version": SCHEMA_VERSION, "sample_uid": "0" * 64,
            "manifest_index": None, "dataset_name": "magicbrush",
            "dataset_revision": args.revision,
            "group_id": str(source.get("img_id", source["sample_key"].split("_")[0])),
            **{f"{name}_locator": locator for name, locator in locators.items()},
            **{f"{name}_sha256": expected_locator_hash(locator, root)
               for name, locator in locators.items()},
            "instruction_original": source["instruction"],
            "instruction_en": source["instruction"], "language_original": "en",
            "edit_type_original": "unknown", "edit_type_canonical": "other",
            "mask_semantics": "edit_region", "region_fraction": region_fraction,
            "translation_status": "passthrough_en", "translation_cache_key": None,
        }
        row["sample_uid"] = sample_uid_for(row)
        rows.append(row)
        if args.limit is not None and len(rows) >= args.limit:
            break
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, rows)
    rejections_path = output.with_suffix(".rejections.jsonl")
    _write_jsonl(rejections_path, rejections)
    report = {
        "schema_version": "magicbrush-canonical-pool-build-v1",
        "status": "PASS",
        "manifest": str(manifest.resolve()),
        "dataset_root": str(root),
        "dataset_revision": args.revision,
        "accepted_rows": len(rows),
        "rejected_rows": len(rejections),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "rejections": str(rejections_path),
        "rejections_sha256": sha256_file(rejections_path),
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": len(rows),
        "rejected_rows": len(rejections),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "output": str(output),
        "report": str(report_path),
    }))


if __name__ == "__main__":
    main()
