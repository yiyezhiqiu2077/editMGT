#!/usr/bin/env python3
"""Compare fresh and resumed checkpoints offline; no GPU allocation."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dimo.resume_comparison import compare_complete_checkpoints
from src.explicit_region.dense_checkpoint import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = compare_complete_checkpoints(args.fresh, args.resumed)
    write_json(output, result)
    print(result)


if __name__ == "__main__":
    main()
