#!/usr/bin/env python3
"""Download a preregistered candidate subset, never the full InterEdit corpus."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.explicit_region.config import load_config
from src.explicit_region.formal_pipeline import write_json_atomic
from src.explicit_region.mirror_download import MirrorDownloader

REPOS = {
    "crispedit": ("TTangenty/CrispEdit-labeling-39k", "dcbd1c952e93e4361ad862b33f3acd1cc74bec5a"),
    "scaleedit": ("TTangenty/ScaleEdit-labeling-25k", "f97ffb061d4275bcbbff6500572139b6f8df1c4e"),
    "interedit": ("a1557811266/Inter-Edit-Train", "b319d9fdb45cc670ec263fe4aeaa962f1623d5dc"),
    "nllb": ("facebook/nllb-200-distilled-1.3B", "7be3e24664b38ce1cac29b8aeed6911aa0cf0576"),
}


def shard_order(name, revision, path):
    return hashlib.sha256(f"42\0{name}\0{revision}\0{path}".encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--config", default="configs/data/mixed20k_total.yaml")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    downloader = MirrorDownloader(reserve_bytes=config["mirror"]["minimum_free_disk_gib"] * 1024 ** 3)
    jobs, plan = [], {"schema": "mixed30k-mirror-download-v1", "endpoint": downloader.endpoint,
                      "trust_env": False, "selection_seed": 42, "datasets": {}}
    for name, (repo, revision) in REPOS.items():
        target = root / ("models/nllb" if name == "nllb" else f"datasets/{name}")
        metadata = downloader.metadata(repo, revision, dataset=name != "nllb")
        write_json_atomic(root / "metadata" / f"{name}_repo.json", metadata)
        entries = {entry["rfilename"]: entry for entry in metadata["siblings"]}
        if name == "nllb":
            # One weight representation only; tokenizer assets are small.
            names = [path for path in entries if path in (
                "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "sentencepiece.bpe.model", "special_tokens_map.json", "pytorch_model.bin")]
            if not {"config.json", "tokenizer_config.json", "pytorch_model.bin"} <= set(names):
                raise RuntimeError("pinned translator layout changed")
        elif name in ("crispedit", "scaleedit"):
            path = downloader.download(repo, revision, entries["dataset_manifest.json"], target)
            manifest = json.loads(path.read_text())
            ordered = sorted(manifest["shard_files"],
                             key=lambda row: shard_order(name, revision, row["shard"]))
            names, row_count = ["README.md", "dataset_manifest.json", "COMPLETE"], 0
            for shard in ordered:
                # The manifest also lists zero-row shards omitted from the export.
                if shard["rows"] == 0:
                    continue
                category = f"{shard['category']}/" if name == "crispedit" else ""
                names.append(f"shards/{category}{shard['shard']}")
                row_count += shard["rows"]
                if row_count >= config["mirror"]["initial_candidate_rows"][name]:
                    break
            plan["datasets"][name] = {"candidate_rows_before_qc": row_count}
        else:
            # Later asset shards reuse sources throughout the full 2 TB corpus.
            # A fixed early contiguous subset bounds required source archives.
            # This is explicitly a subset experiment, not a representative sample.
            assets = sorted(path for path in entries if path.startswith("asset_shards/"))
            assets = assets[:config["mirror"]["interedit_asset_shards"]]
            names = ["manifest.json", "README.md", *assets]
            sources, records = set(), []
            for asset in assets:
                index = asset.split("asset-")[1].split("-of-")[0]
                meta = f"metadata/train-{index}-of-00275.jsonl.gz"
                path = downloader.download(repo, revision, entries[meta], target)
                names.append(meta)
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        if row["better_data"] is True:
                            records.append(row)
                            sources.add(row["source_archive"])
            names.extend(sorted(sources))
            plan["datasets"][name] = {"candidate_rows_before_qc": len(records), "asset_shards": assets,
                                      "source_shards": sorted(sources), "only_better_data": True,
                                      "subset_policy": config["mirror"]["interedit_subset_order"],
                                      "globally_representative": False}
        selected = [entries[path] for path in dict.fromkeys(names)]
        plan["datasets"].setdefault(name, {}).update(repo=repo, revision=revision,
            root=str(target), files=selected, download_bytes=sum(entry["size"] for entry in selected))
        jobs.extend((name, repo, revision, entry, target) for entry in selected)
    plan["total_download_bytes"] = sum(value["download_bytes"] for value in plan["datasets"].values())
    if shutil.disk_usage(root).free < plan["total_download_bytes"] + downloader.reserve_bytes:
        raise RuntimeError("candidate download plan exceeds available disk budget")
    plan_path = root / "DOWNLOAD_PLAN.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("immutable download plan changed")
    write_json_atomic(plan_path, plan)
    print(json.dumps({"status": "DOWNLOAD_PLANNED", "bytes": plan["total_download_bytes"],
                      "files": len(jobs), "pools": {name: value.get("candidate_rows_before_qc")
                      for name, value in plan["datasets"].items()}}), flush=True)
    if args.plan_only:
        return
    priority=config["mirror"].get("download_priority",list(REPOS))
    jobs.sort(key=lambda job:priority.index(job[0]))

    def download(job):
        name, repo, revision, entry, target = job
        path = downloader.download(repo, revision, entry, target, dataset=name != "nllb")
        return {"dataset": name, "file": entry["rfilename"], "bytes": path.stat().st_size,
                "content_identity": entry.get("lfs", {}).get("sha256", entry.get("blobId"))}

    completed, byte_count = 0, 0
    with (root / "DOWNLOAD_LOG.jsonl").open("a") as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(download, job) for job in jobs]):
            record = future.result()
            log.write(json.dumps(record) + "\n");log.flush()
            completed += 1;byte_count += record["bytes"]
            state = {"status": "DOWNLOADING", "completed_files": completed, "total_files": len(jobs),
                     "verified_bytes": byte_count, "total_bytes": plan["total_download_bytes"]}
            write_json_atomic(root / "DOWNLOAD_STATUS.json", state)
            if completed % 20 == 0:
                print(json.dumps(state), flush=True)
    write_json_atomic(root / "DOWNLOAD_STATUS.json", dict(state, status="COMPLETE"))
    print(json.dumps(dict(state, status="COMPLETE")), flush=True)


if __name__ == "__main__":
    main()
