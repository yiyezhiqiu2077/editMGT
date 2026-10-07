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
from PIL import Image, UnidentifiedImageError

from src.explicit_region.canonical import (
    SCHEMA_VERSION, expected_locator_hash, sample_uid_for,
)
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dataset import _align_to_mask_coordinates
from src.explicit_region.geometry import SampleRejected, apply_geometry, sample_geometry


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def geometry_rejection(
    source_path: Path, target_path: Path, mask_path: Path, *, sample_key: str, index: int,
) -> tuple[float | None, dict | None]:
    try:
        with (
            Image.open(source_path) as source_raw,
            Image.open(target_path) as target_raw,
            Image.open(mask_path) as mask_raw,
        ):
            source, target, mask = source_raw.copy(), target_raw.copy(), mask_raw.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return None, {"reason": "decode_failure", "error": str(exc)}
    try:
        mask_array = np.asarray(mask.convert("L"))
        if not (mask_array > 0).any():
            return None, {"reason": "empty_region"}
        try:
            aligned_source, aligned_target, _ = _align_to_mask_coordinates(
                source, target, mask, sample_key
            )
        except ValueError as exc:
            if "unaligned aspect ratio" not in str(exc):
                raise
            return None, {"reason": "unaligned_aspect_ratio", "error": str(exc)}
        try:
            geometry = sample_geometry(
                mask, resolution=1024, base_seed=42, global_sample_index=index,
                sample_key=sample_key, max_resample_attempts=8,
                minimum_mask_retention=0.75,
            )
            transformed = apply_geometry(mask, geometry, is_mask=True)
            if not (np.asarray(transformed) > 0).any():
                return None, {"reason": "post_geometry_empty_region"}
        except SampleRejected as exc:
            return None, {"reason": "post_geometry_empty_region", "error": exc.reason}
        finally:
            if "aligned_source" in locals() and aligned_source is not source:
                aligned_source.close()
            if "aligned_target" in locals() and aligned_target is not target:
                aligned_target.close()
        return float((mask_array > 0).mean()), None
    finally:
        source.close(); target.close(); mask.close()


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
        locators = {
            "source": {"backend": "file", "relative_path": source["source"]},
            "target": {"backend": "file", "relative_path": source["target"]},
            "region": {"backend": "file", "relative_path": source["mask_edit"]},
        }
        region_fraction, rejection = geometry_rejection(
            root / source["source"], root / source["target"], root / source["mask_edit"],
            sample_key=source["sample_key"], index=index,
        )
        if rejection is not None:
            rejection_counts[rejection["reason"]] += 1
            rejections.append({
                "manifest_index": index,
                "sample_key": source["sample_key"],
                "img_id": source.get("img_id"),
                **rejection,
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
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    rejections_path = output.with_suffix(".rejections.jsonl")
    write_jsonl(rejections_path, rejections)
    report = {
        "schema_version": "magicbrush-canonical-pool-build-v1",
        "status": "PASS", "accepted_rows": len(rows), "rejected_rows": len(rejections),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "output": str(output.resolve()), "output_sha256": sha256_file(output),
        "rejections": str(rejections_path.resolve()),
        "rejections_sha256": sha256_file(rejections_path),
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
