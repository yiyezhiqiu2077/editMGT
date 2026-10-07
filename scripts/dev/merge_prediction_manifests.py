#!/usr/bin/env python3
"""Merge disjoint evaluation JSONL manifests with duplicate-key checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    for source in map(Path, args.input):
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row.get("dataset_name")),
                str(row.get("sample_key")),
                int(row.get("seed", 0)),
            )
            if key in seen:
                raise RuntimeError(f"duplicate prediction key: {key}")
            seen.add(key)
            rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"inputs": len(args.input), "rows": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()
