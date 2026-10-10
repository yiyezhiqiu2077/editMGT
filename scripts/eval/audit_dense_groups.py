#!/usr/bin/env python3
"""Read-only source/group train/DEV/TEST isolation audit."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.explicit_region.group_statistics import audit_split_isolation
from src.explicit_region.dense_checkpoint import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contract", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--dev", nargs="+", required=True)
    p.add_argument("--test", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = audit_split_isolation(a.train, a.dev, a.test, json.loads(Path(a.contract).read_text()))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(a.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
