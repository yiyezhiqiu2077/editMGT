#!/usr/bin/env python3
"""Create deterministic group-disjoint train/validation manifests with hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-key", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be in (0,1)")
    rows = [json.loads(line) for line in Path(args.input).open(encoding="utf-8") if line.strip()]
    groups = {}
    for row in rows:
        groups.setdefault(str(row[args.group_key]), []).append(row)
    threshold = int(args.validation_fraction * (2**256 - 1))
    split_rows = {"train": [], "validation": []}
    for group, members in sorted(groups.items()):
        value = int(hashlib.sha256(f"{args.seed}:{group}".encode()).hexdigest(), 16)
        split_rows["validation" if value <= threshold else "train"].extend(members)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {"seed": args.seed, "group_key": args.group_key, "input": str(Path(args.input).resolve())}
    for split, members in split_rows.items():
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in members:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        report[split] = {
            "rows": len(members), "groups": len({str(r[args.group_key]) for r in members}),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "path": str(path.resolve()),
        }
    overlap = {str(r[args.group_key]) for r in split_rows["train"]} & {
        str(r[args.group_key]) for r in split_rows["validation"]
    }
    if overlap:
        raise AssertionError("group leakage between train and validation")
    (output / "split_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
