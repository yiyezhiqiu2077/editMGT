#!/usr/bin/env python3
"""Write READY only after all bound artifacts pass the shared strict validator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explicit_region.fixed_corpus import DATASET_PRIORITY, VALIDATION_NAMES, write_corpus_ready


def finalize(fixed_root, selection_config, translation_cache, git_sha):
    root = Path(fixed_root).resolve()
    marker = root / "CORPUS_READY.json"
    marker.unlink(missing_ok=True)
    required = {
        "train_200k": root / "train_200k.jsonl", "train_meta": root / "train_200k.meta.json",
        "candidate_selection": root / "candidate_selection.jsonl", "reserve_order": root / "reserve_order.jsonl",
        "final_selection": root / "final_selection.jsonl", "translation_rejections": root / "translation_rejections.jsonl",
        "backfill_history": root / "backfill_history.jsonl", "selection_report": root / "selection_report.json",
        "duplicate_report": root / "duplicate_report.json", "translation_report": root / "translation_report.json",
        "translation_cache": Path(translation_cache), "audit_report": root / "audit/audit_report.json",
        "audit_failures": root / "audit/integrity_failures.jsonl",
        "translation_manual_audit": root / "translation_manual_audit_500.jsonl",
        "translation_manual_audit_csv": root / "translation_manual_audit_500.csv",
        "selection_config": Path(selection_config), "validation_meta": root / "validation/validation.meta.json",
        "validation_translation_rejections": root / "validation/translation_rejections.jsonl",
    }
    for name in VALIDATION_NAMES:
        required[f"validation_{name}"] = root / "validation" / f"{name}.jsonl"
    for name in (*(f"{name}_100" for name in DATASET_PRIORITY), "final_200k_montage"):
        required[f"montage_{name}"] = root / "audit" / f"{name}.jpg"
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(f"required readiness artifact missing: {path}")
    metadata = json.loads(required["train_meta"].read_text(encoding="utf-8"))
    return write_corpus_ready(marker, files=required, metadata={
        "created_from_git_sha": git_sha, "total_rows": metadata["total_rows"],
        "dataset_counts": metadata["dataset_counts"],
        "unique_source_counts": {name: metadata["unique_source_counts"][name]
                                 for name in metadata["dataset_counts"]},
        "dataset_revisions": metadata["dataset_revisions"],
        "integrity_checks": {name: "PASS" for name in (
            "exact_rows", "unique_sample_uid", "contiguous_manifest_index", "train_validation_source_overlap",
            "train_validation_group_overlap", "required_reports", "required_montages", "exhaustive_integrity",
        )},
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-root", required=True)
    parser.add_argument("--selection-config", required=True)
    parser.add_argument("--translation-cache", required=True)
    parser.add_argument("--git-sha", required=True)
    args = parser.parse_args()
    payload = finalize(args.fixed_root, args.selection_config, args.translation_cache, args.git_sha)
    print(json.dumps({"status": payload["status"],
                      "marker": str(Path(args.fixed_root).resolve() / "CORPUS_READY.json")}, indent=2))


if __name__ == "__main__":
    main()
