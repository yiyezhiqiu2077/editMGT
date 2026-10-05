#!/usr/bin/env python3
"""Register a selected formal SFT checkpoint as an immutable DiMO teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dimo.teacher_registration import register_teacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-checkpoint", required=True)
    parser.add_argument("--selected-checkpoint-record", required=True)
    parser.add_argument("--formal-assets", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = register_teacher(
        args.selected_checkpoint,
        args.selected_checkpoint_record,
        args.formal_assets,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
