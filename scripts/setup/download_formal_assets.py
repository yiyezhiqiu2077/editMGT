#!/usr/bin/env python3
"""Resolve and download every pinned public asset without runtime fallbacks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.explicit_region.formal_pipeline import (
    asset_path, load_formal_assets, validate_asset_structure,
    validate_resolved_revision, validated_identity, write_json_atomic,
)


def resolve(spec: dict, *, download: bool, target: Path) -> str:
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(endpoint=os.environ.get("HF_ENDPOINT"))
    info = api.repo_info(repo_id=spec["repo_id"], repo_type=spec["repo_type"], revision=spec["revision"])
    validate_resolved_revision(spec["revision"], info.sha)
    files = {sibling.rfilename for sibling in info.siblings}
    missing = [entry for entry in spec.get("expected_structure", [])
               if entry not in files and not any(name.startswith(entry.rstrip("/") + "/") for name in files)]
    if missing:
        raise RuntimeError(f"ASSET_REMOTE_STRUCTURE_INVALID: {spec['repo_id']} missing {missing}")
    if download:
        snapshot_download(
            repo_id=spec["repo_id"], repo_type=spec["repo_type"], revision=spec["revision"],
            local_dir=target, allow_patterns=spec.get("allow_patterns"),
        )
    return info.sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "configs/formal_assets.yaml"))
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--asset", action="append")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = load_formal_assets(args.manifest); selected = args.asset or list(config["assets"])
    unknown = sorted(set(selected) - set(config["assets"]))
    if unknown: raise SystemExit(f"unknown assets: {unknown}")
    plan = []
    for name in selected:
        spec = config["assets"][name]; target = asset_path(args.asset_root, spec)
        if spec["provider"] != "huggingface":
            plan.append({"asset": name, "action": "locked-package", "target": str(target), "revision": spec["revision"]})
            continue
        existing = validated_identity(target, spec)
        if existing:
            plan.append({"asset": name, "action": "skip-verified", "target": str(target), "revision": spec["revision"]})
            continue
        resolved = resolve(spec, download=args.run, target=target)
        action = "metadata-verified" if args.dry_run else "downloaded"
        if args.run:
            validate_asset_structure(target, spec)
            identity = {
                "schema_version": "formal-asset-identity-v1", "name": name,
                "repo_id": spec["repo_id"], "repo_type": spec["repo_type"],
                "requested_revision": spec["revision"], "resolved_revision": resolved,
                "download_location": str(target.resolve()), "transport_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                "snapshot_identity": f"hf:{spec['repo_type']}:{spec['repo_id']}@{resolved}",
            }
            write_json_atomic(target / "asset_identity.json", identity)
            (target / "_SUCCESS").write_text(resolved + "\n", encoding="utf-8")
        plan.append({"asset": name, "action": action, "target": str(target), "revision": resolved})
    print(json.dumps({"status": "PASS", "mode": "run" if args.run else "dry-run", "assets": plan}, indent=2))


if __name__ == "__main__": main()
