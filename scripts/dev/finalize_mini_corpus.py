#!/usr/bin/env python3
"""Exhaustively verify Mini-512 and write a dev-only readiness marker."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.explicit_region.canonical import load_verified_record_images, validate_record
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dataset import _align_to_mask_coordinates
from src.explicit_region.geometry import apply_geometry, sample_geometry
from src.explicit_region.language import contains_han
from src.explicit_region.interedit import TarMemberReader


EXPECTED = {"magicbrush": 256, "crispedit": 64, "scaleedit": 64, "interedit": 128}


def read(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validation", action="append", default=[])
    parser.add_argument("--magicbrush-root", required=True)
    parser.add_argument("--crispedit-root", required=True)
    parser.add_argument("--scaleedit-root", required=True)
    parser.add_argument("--interedit-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve(); rows = read(manifest)
    roots = {name: Path(getattr(args, f"{name}_root")).resolve() for name in EXPECTED}
    if len(rows) != 512 or Counter(row["dataset_name"] for row in rows) != Counter(EXPECTED):
        raise RuntimeError("MINI_CORPUS_NOT_READY: exact dataset counts failed")
    if len({row["sample_uid"] for row in rows}) != 512:
        raise RuntimeError("MINI_CORPUS_NOT_READY: duplicate sample_uid")
    validation = [row for path in args.validation for row in read(path)]
    validation_sources = {row["source_sha256"] for row in validation}
    validation_groups = {(row["dataset_name"], row["group_id"]) for row in validation}
    if {row["source_sha256"] for row in rows} & validation_sources:
        raise RuntimeError("MINI_CORPUS_NOT_READY: source SHA validation leakage")
    if {(row["dataset_name"], row["group_id"]) for row in rows} & validation_groups:
        raise RuntimeError("MINI_CORPUS_NOT_READY: group validation leakage")
    tar_readers = {
        name: TarMemberReader(root) for name, root in roots.items()
        if any(
            row["dataset_name"] == name
            and any(row[f"{role}_locator"]["backend"] == "tar" for role in ("source", "target", "region"))
            for row in rows
        )
    }
    for index, row in enumerate(rows):
        validate_record(row, require_frozen_index=True)
        if row["manifest_index"] != index:
            raise RuntimeError("MINI_CORPUS_NOT_READY: non-contiguous manifest_index")
        if not row["instruction_en"] or contains_han(row["instruction_en"]):
            raise RuntimeError("MINI_CORPUS_NOT_READY: invalid instruction_en")
        if not math.isfinite(float(row["region_fraction"])):
            raise RuntimeError("MINI_CORPUS_NOT_READY: non-finite metadata")
        source, target, region = load_verified_record_images(
            row, roots[row["dataset_name"]], tar_reader=tar_readers.get(row["dataset_name"])
        )
        aligned_source = aligned_target = None
        try:
            aligned_source, aligned_target, _ = _align_to_mask_coordinates(
                source, target, region, row["sample_uid"]
            )
            geometry = sample_geometry(
                region, resolution=1024, base_seed=42, global_sample_index=index,
                sample_key=row["sample_uid"], max_resample_attempts=8,
                minimum_mask_retention=.75,
            )
            if not (np.asarray(apply_geometry(region, geometry, is_mask=True)) > 0).any():
                raise RuntimeError("MINI_CORPUS_NOT_READY: post-geometry empty mask")
        finally:
            if aligned_source is not None and aligned_source is not source: aligned_source.close()
            if aligned_target is not None and aligned_target is not target: aligned_target.close()
            source.close(); target.close(); region.close()
    for reader in tar_readers.values():
        reader.close()
    files = {"train_512": manifest}
    for index, path in enumerate(args.validation):
        files[f"validation_{index}"] = Path(path).resolve()
    payload = {
        "schema_version": "mini-corpus-ready-v1", "status": "READY", "dev_only": True,
        "formal_eligible": False, "total_rows": 512, "dataset_counts": EXPECTED,
        "files": {name: {"path": str(path), "sha256": sha256_file(path)}
                  for name, path in sorted(files.items())},
        "checks": {"asset_hashes": "PASS", "decode": "PASS", "geometry": "PASS",
                   "translation": "PASS", "duplicates": "PASS", "validation_leakage": "PASS"},
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
