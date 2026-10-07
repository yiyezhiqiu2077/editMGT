#!/usr/bin/env python3
"""Download only the exact large files admitted by a Mini download plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import hf_hub_download


def download_with_retry(*, attempts: int = 6, **kwargs: object) -> str:
    """Resume transiently interrupted multi-GB downloads with bounded retries."""
    for attempt in range(1, attempts + 1):
        try:
            return hf_hub_download(**kwargs)
        except Exception:
            if attempt == attempts:
                raise
            delay = min(60, 5 * (2 ** (attempt - 1)))
            print(json.dumps({"download_retry": attempt, "sleep_seconds": delay}), flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--asset-root", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    root = Path(args.asset_root).resolve() / "dev_e2e/four_dataset_512/assets"
    ledger = []
    for row in plan["files"]:
        destination = root / row["dataset"]
        path = Path(download_with_retry(
            repo_id=row["repo"], filename=row["file"], repo_type="dataset", revision=row["revision"],
            local_dir=destination,
        ))
        size = path.stat().st_size
        if size != row["remote_bytes"]:
            raise RuntimeError(f"download size mismatch: {row['file']}")
        ledger.append({**row, "local_path": str(path.resolve()), "bytes": size,
                       "sha256": sha256_file(path)})
        print(json.dumps({"downloaded": row["file"], "bytes": size}), flush=True)
    manifest = {
        "schema_version": "mini-asset-manifest-v1", "dev_only": True,
        "status": "PARTIAL_ASSETS_READY", "files": ledger,
        "network_bytes": sum(row["bytes"] for row in ledger),
    }
    output = root / "mini_asset_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "files": len(ledger)}, indent=2))


if __name__ == "__main__":
    main()
