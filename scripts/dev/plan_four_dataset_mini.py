#!/usr/bin/env python3
"""Plan the bounded real-data download for the four-dataset Mini-512 run."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

import yaml
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.explicit_region.language import contains_han


def stable_key(seed: int, row: dict) -> str:
    identity = "\0".join(str(row[key]) for key in ("sample_id", "source_id", "edit_type", "instruction"))
    return hashlib.sha256(f"{seed}\0{identity}".encode()).hexdigest()


def remote_inventory(api: HfApi, asset: dict) -> tuple[str, dict[str, int]]:
    info = api.dataset_info(asset["repo_id"], revision=asset["revision"], files_metadata=True)
    if info.sha != asset["revision"]:
        raise RuntimeError(f"resolved revision mismatch for {asset['repo_id']}")
    sizes = {}
    for sibling in info.siblings:
        size = getattr(sibling, "size", None)
        if size is None and getattr(sibling, "lfs", None):
            size = sibling.lfs.get("size")
        sizes[sibling.rfilename] = int(size or 0)
    return info.sha, sizes


def small_download(asset: dict, filename: str, local_root: Path) -> Path:
    return Path(hf_hub_download(
        asset["repo_id"], filename, repo_type="dataset", revision=asset["revision"],
        local_dir=local_root,
    ))


def choose_parquets(manifest: dict, sizes: dict[str, int], *, dataset: str, target_rows: int) -> list[dict]:
    entries = []
    for item in manifest["shard_files"]:
        name = item["shard"]
        filename = f"shards/{item.get('category')}/{name}" if dataset == "crispedit" else f"shards/{name}"
        if filename not in sizes:
            continue
        entries.append((filename, int(item["rows"]), sizes[filename]))
    chosen, rows = [], 0
    for filename, count, size in sorted(entries):
        chosen.append({"file": filename, "remote_bytes": size, "manifest_rows": count})
        rows += count
        if rows >= target_rows:
            break
    if rows < target_rows:
        raise RuntimeError(f"{dataset} manifest cannot supply {target_rows} planned rows")
    return chosen


def choose_unique(rows: list[dict], count: int, seed: int, used_groups: set[str], predicate=lambda _row: True) -> list[dict]:
    selected = []
    for row in sorted(rows, key=lambda value: (stable_key(seed, value), int(value["sample_id"]))):
        group = str(row["source_id"])
        if group in used_groups or not predicate(row):
            continue
        selected.append(row); used_groups.add(group)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Inter-Edit candidate group cannot supply {count} unique rows")


def select_interedit(rows: list[dict], seed: int = 42) -> tuple[list[dict], dict]:
    eligible = [row for row in rows if type(row.get("better_data")) is bool and row["better_data"] is True]
    by_pair = defaultdict(list)
    for row in eligible:
        by_pair[(row["source_archive"], row["asset_archive"])].append(row)
    feasible = []
    for pair, members in by_pair.items():
        unique = {str(row["source_id"]) for row in members}
        languages = Counter("han" if contains_han(row["instruction"]) else "english" for row in members)
        types = Counter(row["edit_type"] for row in members)
        if len(unique) >= 296 and min(languages.get("han", 0), languages.get("english", 0)) >= 32 \
                and all(types.get(kind, 0) >= 18 for kind in ("Add", "Remove", "Local", "Texture")):
            feasible.append((pair, members))
    if not feasible:
        raise RuntimeError("no single deterministic Inter-Edit archive pair satisfies Mini coverage")
    pair, members = min(feasible, key=lambda item: hashlib.sha256(
        f"{seed}\0{item[0][0]}\0{item[0][1]}".encode()).hexdigest())
    used = set(); validation = []
    for kind in ("Add", "Remove", "Local", "Texture"):
        validation.extend(choose_unique(
            members, 2, seed, used, lambda row, kind=kind: row["edit_type"] == kind
        ))
    train = []
    for kind in ("Add", "Remove", "Local", "Texture"):
        train.extend(choose_unique(members, 8, seed, used, lambda row, kind=kind:
            row["edit_type"] == kind and contains_han(row["instruction"])))
        train.extend(choose_unique(members, 8, seed, used, lambda row, kind=kind:
            row["edit_type"] == kind and not contains_han(row["instruction"])))
    train.extend(choose_unique(members, 128 - len(train), seed, used))
    # Real geometry filtering can reject a substantial fraction of an archive.
    # Extra metadata candidates are effectively free once the two tar files are
    # fixed, and preserve deterministic backfill without another large download.
    reserve = []
    reserve_quotas = {"Add": 32, "Remove": 32, "Local": 32, "Texture": 40}
    for kind, reserve_count in reserve_quotas.items():
        reserve.extend(choose_unique(
            members, reserve_count, seed, used, lambda row, kind=kind: row["edit_type"] == kind
        ))
    reserve.extend(choose_unique(members, 24, seed, used))
    selected = validation + train + reserve
    report = {
        "source_archives": sorted({row["source_archive"] for row in selected}),
        "asset_archives": sorted({row["asset_archive"] for row in selected}),
        "validation_rows": len(validation), "train_rows": len(train), "reserve_rows": len(reserve),
        "train_language_counts": dict(Counter(
            "han" if contains_han(row["instruction"]) else "english" for row in train
        )),
        "train_edit_type_counts": dict(Counter(row["edit_type"] for row in train)),
        "validation_sample_ids": [row["sample_id"] for row in validation],
        "train_sample_ids": [row["sample_id"] for row in train],
        "reserve_sample_ids": [row["sample_id"] for row in reserve],
    }
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-assets", default="configs/formal_assets.yaml")
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--max-download-gb", type=float, default=60)
    parser.add_argument("--max-disk-gb", type=float, default=100)
    parser.add_argument("--crisp-plan-rows", type=int, default=256)
    parser.add_argument("--scale-plan-rows", type=int, default=256)
    args = parser.parse_args()
    root = Path(args.asset_root).resolve() / "dev_e2e/four_dataset_512"
    assets_root = root / "assets"; assets_root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.formal_assets).read_text(encoding="utf-8"))["assets"]
    api = HfApi()
    inventories = {}
    resolved = {}
    for name in ("crispedit", "scaleedit", "interedit"):
        resolved[name], inventories[name] = remote_inventory(api, config[name])
    manifests = {}
    for name in ("crispedit", "scaleedit"):
        local = assets_root / name
        path = small_download(config[name], "dataset_manifest.json", local)
        manifests[name] = json.loads(path.read_text(encoding="utf-8"))
    crisp = choose_parquets(manifests["crispedit"], inventories["crispedit"], dataset="crispedit", target_rows=args.crisp_plan_rows)
    scale = choose_parquets(manifests["scaleedit"], inventories["scaleedit"], dataset="scaleedit", target_rows=args.scale_plan_rows)
    inter_meta_name = "metadata/train-00000-of-00275.jsonl.gz"
    inter_meta = small_download(config["interedit"], inter_meta_name, assets_root / "interedit")
    with gzip.open(inter_meta, "rt", encoding="utf-8") as handle:
        inter_rows = [json.loads(line) for line in handle if line.strip()]
    selected, inter_report = select_interedit(inter_rows)
    filtered = assets_root / "interedit/interedit_mini_metadata.jsonl.gz"
    selected_ids = {int(row["sample_id"]) for row in selected}
    with gzip.open(filtered, "wt", encoding="utf-8") as handle:
        for index, row in enumerate(inter_rows):
            if int(row["sample_id"]) in selected_ids:
                row = dict(row, original_metadata_file=inter_meta_name, original_row_index=index)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    plan_rows = []
    for name, rows in (("crispedit", crisp), ("scaleedit", scale)):
        for row in rows:
            plan_rows.append({"dataset": name, "repo": config[name]["repo_id"],
                              "revision": config[name]["revision"], "reason": "candidate_pool", **row})
    plan_rows.append({"dataset": "interedit", "repo": config["interedit"]["repo_id"],
                      "revision": config["interedit"]["revision"], "file": inter_meta_name,
                      "remote_bytes": inventories["interedit"][inter_meta_name], "reason": "metadata_selection"})
    for filename in inter_report["source_archives"]:
        plan_rows.append({"dataset": "interedit", "repo": config["interedit"]["repo_id"],
                          "revision": config["interedit"]["revision"], "file": filename,
                          "remote_bytes": inventories["interedit"][filename], "reason": "selected_source_archive"})
    for filename in inter_report["asset_archives"]:
        plan_rows.append({"dataset": "interedit", "repo": config["interedit"]["repo_id"],
                          "revision": config["interedit"]["revision"], "file": filename,
                          "remote_bytes": inventories["interedit"][filename], "reason": "selected_asset_archive"})
    total = sum(row["remote_bytes"] for row in plan_rows)
    # Cache plus local-dir links/copies and derived canonical artifacts need headroom.
    estimated_disk = int(total * 1.35)
    plan = {
        "schema_version": "mini-download-plan-v1", "status": "PLANNED",
        "resolved_revisions": resolved, "files": plan_rows,
        "network_download_bytes": total, "network_download_gb": total / 10**9,
        "estimated_disk_bytes": estimated_disk, "estimated_disk_gb": estimated_disk / 10**9,
        "limits": {"download_gb": args.max_download_gb, "disk_gb": args.max_disk_gb},
        "interedit_selection": inter_report,
    }
    output = root / "mini_download_plan.json"
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(output), "network_gb": plan["network_download_gb"],
                      "estimated_disk_gb": plan["estimated_disk_gb"]}, indent=2))
    if plan["network_download_gb"] > args.max_download_gb or plan["estimated_disk_gb"] > args.max_disk_gb:
        raise SystemExit("MINI_DOWNLOAD_BUDGET_EXCEEDED")


if __name__ == "__main__":
    main()
