#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dimo.dense_teacher import export_dense_teacher, load_dense_teacher_contract


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    p.add_argument("--output-dir")
    p.add_argument("--status", choices=("provisional", "selected"))
    p.add_argument("--selected-record")
    p.add_argument("--verify")
    a = p.parse_args()
    if a.verify:
        result = load_dense_teacher_contract(a.verify).manifest
    else:
        if not all((a.checkpoint, a.output_dir, a.status)):
            p.error("--checkpoint, --output-dir and --status are required")
        result = export_dense_teacher(a.checkpoint, a.output_dir, status=a.status, selected_record=a.selected_record)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
