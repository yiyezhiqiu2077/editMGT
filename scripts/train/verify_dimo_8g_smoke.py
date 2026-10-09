#!/usr/bin/env python3
"""Produce a smoke certificate only from real 8-GPU D200K evidence."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.dimo.smoke import verify_real_smoke
from src.explicit_region.formal_pipeline import write_json_atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh20', required=True)
    parser.add_argument('--fresh10', required=True)
    parser.add_argument('--resume20', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = verify_real_smoke(args.fresh20, args.fresh10, args.resume20)
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
