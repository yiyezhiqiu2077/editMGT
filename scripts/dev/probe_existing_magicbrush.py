#!/usr/bin/env python3
"""Audit an existing materialized MagicBrush train/dev without changing it."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image


def probe_split(root: Path, split: str) -> dict:
    base = root / split
    rows = [json.loads(line) for line in (base / "manifest.jsonl").open(encoding="utf-8") if line.strip()]
    sizes = {"source": Counter(), "target": Counter(), "mask_edit": Counter()}
    modes = Counter()
    for row in rows[:32]:
        for role in sizes:
            with Image.open(base / row[role]) as image:
                sizes[role][str(image.size)] += 1
                if role == "mask_edit": modes[image.mode] += 1
    return {"rows": len(rows), "sample_keys": [row["sample_key"] for row in rows[:8]],
            "sampled_sizes": {name: dict(values) for name, values in sizes.items()},
            "sampled_mask_modes": dict(modes)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); root = Path(args.root).resolve()
    result = {"root": str(root), "format": "materialized_train_dev",
              "train": probe_split(root, "train"), "dev": probe_split(root, "dev")}
    if result["train"]["rows"] != 8807 or result["dev"]["rows"] != 528:
        raise RuntimeError("existing MagicBrush row counts do not match pinned release")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
