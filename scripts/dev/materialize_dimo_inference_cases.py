#!/usr/bin/env python3
"""Materialize one verified Mini validation case per dataset for DiMO inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.explicit_region.canonical import load_verified_record_images
from src.explicit_region.interedit import TarMemberReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    mini = Path(args.mini_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    specifications = {
        "magicbrush": ("magicbrush_probe8.jsonl", mini / "assets/magicbrush/dev"),
        "crispedit": ("crispedit_aux8.jsonl", mini / "assets/crispedit"),
        "scaleedit": ("scaleedit_aux8.jsonl", mini / "assets/scaleedit"),
        "interedit": ("interedit_aux8.jsonl", mini / "assets/interedit"),
    }
    cases = []
    for dataset, (manifest_name, dataset_root) in specifications.items():
        manifest = mini / "corpus/validation" / manifest_name
        record = json.loads(next(line for line in manifest.read_text().splitlines() if line))
        reader = TarMemberReader(dataset_root) if dataset == "interedit" else None
        source, _, mask = load_verified_record_images(record, dataset_root, tar_reader=reader)
        source_path = output / f"{dataset}_source.png"
        mask_path = output / f"{dataset}_mask.png"
        source.convert("RGB").save(source_path)
        mask.convert("L").save(mask_path)
        cases.append({
            "dataset_name": dataset,
            "sample_uid": record["sample_uid"],
            "instruction": record["instruction_en"],
            "source_image": str(source_path),
            "edit_region_mask": str(mask_path),
        })
    manifest_path = output / "cases.json"
    manifest_path.write_text(json.dumps(cases, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cases": len(cases), "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
