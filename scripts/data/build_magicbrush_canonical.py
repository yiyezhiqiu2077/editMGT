#!/usr/bin/env python3
"""Convert an existing MagicBrush manifest into portable canonical records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image

from src.explicit_region.canonical import (
    SCHEMA_VERSION, expected_locator_hash, sample_uid_for,
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
    rows = []
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        source = json.loads(raw)
        locators = {
            "source": {"backend": "file", "relative_path": source["source"]},
            "target": {"backend": "file", "relative_path": source["target"]},
            "region": {"backend": "file", "relative_path": source["mask_edit"]},
        }
        with Image.open(root / source["mask_edit"]) as mask:
            region_fraction = float((np.asarray(mask.convert("L")) > 0).mean())
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
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()
