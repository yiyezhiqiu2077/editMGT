#!/usr/bin/env python3
"""Freeze preregistration before runs; select from full DEV only. Never runs TEST."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.explicit_region.selection import (
    checkpoint_identity, freeze_preregistration, output_identity_name, select_candidates,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--config", default=str(ROOT / "configs/eval/formal.yaml"))
    freeze.add_argument("--corpus-ready", required=True)
    freeze.add_argument("--output", required=True)
    select = sub.add_parser("select")
    select.add_argument("--preregistration", required=True)
    select.add_argument("--stage", choices=("lr", "formal"), required=True)
    select.add_argument("--reports", nargs="+", required=True)
    select.add_argument("--output", required=True)
    name = sub.add_parser("output-name")
    name.add_argument("--experiment", required=True)
    name.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    if args.command == "output-name":
        print(output_identity_name(args.experiment, checkpoint_identity(args.checkpoint, args.experiment)))
        return
    if args.command == "freeze":
        result = freeze_preregistration(args.config, args.output, corpus_ready=args.corpus_ready)
    else:
        result = select_candidates(args.preregistration, args.reports, args.stage)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
