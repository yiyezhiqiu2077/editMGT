#!/usr/bin/env python3
"""Audit local dataset structure without guessing a training adapter schema."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


ROLE_TOKENS = {
    "source": ("source", "input", "original", "before"),
    "target": ("target", "output", "edited", "after"),
    "mask": ("mask", "region", "segmentation"),
    "bbox": ("bbox", "box", "bounding"),
    "instruction": ("instruction", "prompt", "text", "edit"),
    "edit_type": ("task", "type", "category", "operation"),
    "group_id": ("source_id", "group", "session", "image_id", "img_id"),
}


def candidate_roles(columns: list[str]) -> dict[str, list[str]]:
    return {
        role: [column for column in columns if any(token in column.lower() for token in tokens)]
        for role, tokens in ROLE_TOKENS.items()
    }


def file_inventory(root: Path) -> tuple[Counter, str]:
    counter, digest = Counter(), hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        counter[path.suffix.lower() or "<none>"] += 1
        stat = path.stat()
        digest.update(relative.encode())
        digest.update(str(stat.st_size).encode())
    return counter, digest.hexdigest()


def audit_parquet(paths: list[Path]) -> dict:
    import pyarrow.parquet as pq
    rows, columns, nulls, presence = 0, set(), Counter(), Counter()
    schemas = set()
    categorical = {key: Counter() for key in ("edit_type", "type", "final_task", "better_data", "mask__qc_flag")}
    for path in paths:
        parquet = pq.ParquetFile(path)
        rows += parquet.metadata.num_rows
        names = parquet.schema_arrow.names
        for name in names:
            presence[name] += parquet.metadata.num_rows
        schemas.add(tuple(names))
        columns.update(names)
        for batch in parquet.iter_batches(batch_size=16):
            for name, array in zip(batch.schema.names, batch.columns):
                nulls[name] += array.null_count
                if name in categorical:
                    categorical[name].update(json.dumps(value, ensure_ascii=False, sort_keys=True) for value in array.to_pylist())
    columns = sorted(columns)
    return {
        "row_count": rows, "columns": columns,
        "schema_variants": [list(value) for value in sorted(schemas)],
        "null_counts": dict(nulls),
        "missing_counts": {name: rows - presence[name] for name in sorted(presence)},
        "categorical_counts": {key: dict(counts) for key, counts in categorical.items() if counts},
        "candidate_fields": candidate_roles(columns),
    }


def audit_json(paths: list[Path]) -> dict:
    rows, columns, nulls, presence = 0, set(), Counter(), Counter()
    file_rows, schemas = {}, Counter()
    categorical = {key: Counter() for key in ("edit_type", "type", "final_task", "better_data", "mask__qc_flag")}
    for path in paths:
        before = rows
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            if path.name.endswith((".jsonl", ".jsonl.gz")):
                values = (json.loads(line) for line in handle if line.strip())
            else:
                loaded = json.load(handle)
                values = loaded if isinstance(loaded, list) else [loaded]
            for row in values:
                if not isinstance(row, dict):
                    continue
                rows += 1
                columns.update(row)
                schemas[tuple(sorted(row))] += 1
                for key, counts in categorical.items():
                    if key in row:
                        counts[json.dumps(row[key], ensure_ascii=False, sort_keys=True)] += 1
                for key, value in row.items():
                    presence[key] += 1
                    if value is None or value == "":
                        nulls[key] += 1
        file_rows[str(path)] = rows - before
    names = sorted(columns)
    return {
        "row_count": rows, "columns": names, "null_counts": dict(nulls),
        "missing_counts": {name: rows - presence[name] for name in names},
        "file_row_counts": file_rows,
        "schema_variants": [{"columns": list(names), "rows": count} for names, count in sorted(schemas.items())],
        "categorical_counts": {key: dict(counts) for key, counts in categorical.items() if counts},
        "candidate_fields": candidate_roles(names),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"REAL_{args.dataset_name.upper()}_AUDIT=NOT_RUN root_missing={root}")
    inventory, identity = file_inventory(root)
    parquet = sorted(root.rglob("*.parquet"))
    json_paths = sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.jsonl.gz"))
    if not json_paths and not parquet:
        json_paths = sorted(root.rglob("*.json"))
    if not parquet and not json_paths:
        raise SystemExit(f"REAL_{args.dataset_name.upper()}_AUDIT=NOT_RUN no_supported_records={root}")
    result = {
        "dataset_name": args.dataset_name, "dataset_revision": args.revision,
        "root": str(root), "local_identity": identity,
        "storage_file_counts": dict(inventory),
        "parquet": audit_parquet(parquet) if parquet else None,
        "json": audit_json(json_paths) if json_paths else None,
        "tar_files": [path.relative_to(root).as_posix() for path in sorted(root.rglob("*.tar"))],
        "status": "REAL_SCHEMA_AUDITED",
        "adapter_schema_mapping": "NOT_INFERRED_AUTOMATICALLY",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output), "identity": identity}))


if __name__ == "__main__":
    main()
