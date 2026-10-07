#!/usr/bin/env python3
"""Freeze the first 64 already-verified Mini rows for the learnability probe."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.explicit_region.contracts import sha256_file


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    mini = Path(args.mini_root).resolve()
    output = Path(args.output_dir).resolve()
    source = mini / "corpus/train_512.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line][:64]
    counts = Counter(row["dataset_name"] for row in rows)
    required = {"magicbrush", "crispedit", "scaleedit", "interedit"}
    if len(rows) != 64 or set(counts) != required:
        raise RuntimeError(f"Tiny-64 must contain all four datasets: {dict(counts)}")
    output.mkdir(parents=True, exist_ok=True)
    train = output / "train_64.jsonl"
    write_jsonl(train, rows)
    files = {"train_64": {"path": str(train), "sha256": sha256_file(train)}}
    for dataset in sorted(required):
        path = output / f"{dataset}_64.jsonl"
        write_jsonl(path, [row for row in rows if row["dataset_name"] == dataset])
        files[f"eval_{dataset}"] = {"path": str(path), "sha256": sha256_file(path)}
    ready = {
        "schema_version": "mini-corpus-ready-v1",
        "status": "READY",
        "dev_only": True,
        "formal_eligible": False,
        "total_rows": 64,
        "dataset_counts": dict(sorted(counts.items())),
        "source_manifest": {"path": str(source), "sha256": sha256_file(source)},
        "selection": "first_64_rows_no_resampling",
        "files": files,
    }
    marker = output / "TINY64_READY.json"
    marker.write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "READY", "counts": counts, "marker": str(marker)}, default=dict))


if __name__ == "__main__":
    main()
