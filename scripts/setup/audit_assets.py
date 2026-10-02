#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explicit_region.contracts import audit_component_identity


def require_link(path: Path) -> Path:
    if not path.is_symlink():
        raise SystemExit(f"ASSET_NOT_SYMLINK: {path}")
    resolved = path.resolve(strict=True)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="local_assets/asset_audit.json")
    args = parser.parse_args()
    model = require_link(Path("local_assets/models/EditMGT"))
    train = require_link(Path("local_assets/datasets/magicbrush"))
    test = require_link(Path("local_assets/datasets/magicbrush-test"))
    report = {
        "model": audit_component_identity(model),
        "magicbrush_train": str(train),
        "magicbrush_test": str(test),
        "train_manifest": str((train / "manifest.jsonl").resolve(strict=True)),
        "test_manifest": str((test / "manifest.jsonl").resolve(strict=True)),
    }
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
