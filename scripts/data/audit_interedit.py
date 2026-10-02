#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explicit_region.interedit import iter_metadata
from src.explicit_region.language import contains_han


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--all-data", action="store_true")
    args = parser.parse_args()
    counts = Counter()
    archives = set()
    for metadata in args.metadata:
        for row in iter_metadata(metadata, only_better_data=not args.all_data):
            counts["samples"] += 1
            counts[f"edit_type:{row.edit_type}"] += 1
            counts["han_instruction"] += int(contains_han(row.instruction_original))
            archives.update((row.source_archive, row.asset_archive))
    report = {"counts": dict(counts), "unique_archives": len(archives), "only_better_data": not args.all_data}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
