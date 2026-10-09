#!/usr/bin/env python3
"""Export E3 step3125 as an immutable dev-only teacher, without selection."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.dimo.provisional_teacher import export_provisional_teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=os.environ.get('E3_CHECKPOINT_DIR'))
    parser.add_argument('--model-root', default=os.environ.get('EDITMGT_MODEL_ROOT'))
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if not args.checkpoint or not args.model_root:
        parser.error('provide --checkpoint/E3_CHECKPOINT_DIR and --model-root/EDITMGT_MODEL_ROOT')
    print(json.dumps(export_provisional_teacher(args.checkpoint, args.model_root, args.output_dir),
                     indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
