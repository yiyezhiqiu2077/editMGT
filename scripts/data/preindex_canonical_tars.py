#!/usr/bin/env python3
"""Publish canonical tar indexes once, before corpus/DDP readers are started."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explicit_region.canonical import (
    TAR_INDEX_SCHEMA, _asset_identity, _build_tar_index, _require_asset_unchanged,
    _tar_index_path, validate_tar_index,
)


def _preindex_one(path_string):
    path = Path(path_string).resolve()
    identity = _asset_identity(path)
    shared = _tar_index_path(identity)
    if shared.exists():
        return {"archive": str(path), "index": str(shared), "action": "existing",
                **validate_tar_index(shared, identity)}
    # Completed process-private indexes are immutable after publication. Ignore
    # temporary files and validate every row's offsets/schema before promotion.
    private = sorted(shared.parent.glob(f"{shared.stem}.*.sqlite3"))
    if private:
        candidate = private[0]
        counts = validate_tar_index(candidate, identity)
        action = "promoted"
    else:
        staging = shared.with_name(f"{shared.stem}.preindex-{uuid.uuid4().hex}.sqlite3")
        candidate = _build_tar_index(path, identity, staging)
        counts = validate_tar_index(candidate, identity)
        action = "built"
    _require_asset_unchanged(path, identity)
    # The cache-wide mkdir lock guarantees one preindex command. Runtime cold
    # readers must be stopped by the operator; this command never replaces a
    # shared path, even if such an uncoordinated reader won the race.
    if shared.exists():
        return {"archive": str(path), "index": str(shared), "action": "existing",
                **validate_tar_index(shared, identity)}
    candidate.rename(shared)
    return {"archive": str(path), "index": str(shared), "action": action, **counts}


def preindex(root, *, workers=1, report_path=None):
    root = Path(root).resolve()
    files = sorted(root.rglob("*.tar"))
    if not files:
        raise ValueError(f"no uncompressed .tar archives under {root}")
    if workers < 1:
        raise ValueError("workers must be positive")
    cache = _tar_index_path(_asset_identity(files[0])).parent
    owner = cache / "PREINDEX_OWNER"
    try:
        owner.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"preindex owner lock exists: {owner}; confirm previous process stopped before manually removing stale lock") from exc
    identity_path = owner / "owner.json"
    identity_path.write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "root": str(root)}) + "\n")
    report = {"schema": "canonical-tar-preindex-v1", "cache_schema": TAR_INDEX_SCHEMA,
              "status": "RUNNING", "root": str(root), "archives": len(files), "workers": workers,
              "cache": str(cache), "results": []}
    output = Path(report_path) if report_path else cache / "preindex_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report["completed"] = len(report["results"])
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    save()
    try:
        if workers == 1:
            results = map(_preindex_one, map(str, files))
            for result in results:
                report["results"].append(result); save()
                print(json.dumps({"completed": len(report["results"]), **result}), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                # Unique resolved archives, one task each; no concurrent writer
                # for the same shared index even though distinct archives overlap.
                for result in pool.map(_preindex_one, map(str, files), chunksize=1):
                    report["results"].append(result); save()
                    print(json.dumps({"completed": len(report["results"]), **result}), flush=True)
        report["status"] = "PASS"; save()
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}"); save()
        raise
    finally:
        identity_path.unlink(missing_ok=True)
        owner.rmdir()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = preindex(args.root, workers=args.workers, report_path=args.report)
    print(json.dumps({"status": report["status"], "completed": report["completed"], "report": args.report}))


if __name__ == "__main__":
    main()
