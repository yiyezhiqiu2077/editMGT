#!/usr/bin/env python3
"""Stream an audited better_data pool in deterministic metadata-shard order."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image

from src.explicit_region.canonical import (
    SCHEMA_VERSION, clear_locator_caches, read_locator_bytes, sample_uid_for,
    sha256_bytes, validate_record,
)
from src.explicit_region.contracts import sha256_file
from src.explicit_region.interedit import iter_metadata
from src.explicit_region.language import contains_han

MAPPING = {"Add": "add", "Remove": "remove", "Local": "local_attribute", "Texture": "texture"}
DEFAULT_WORKERS = 8
MAX_WORKERS = 8
REPORT_SCHEMA = "interedit-canonical-pool-build-v1"


class PoolBuildError(RuntimeError):
    pass


def _json_line(value):
    # Preserve the previous builder's exact serialization for every valid row.
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def _atomic_json(path: Path, value: dict) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def canonicalize(item, root: Path, revision: str) -> tuple[dict | None, str | None]:
    """Decode/qualify one record without changing asset bytes or canonical IDs.

    Empty masks/instructions were already ineligible. Geometry mirrors the
    existing aligned dataset's same-aspect policy; native sizes may differ.
    Missing, unreadable, malformed or corrupt assets are errors, never rejects.
    """
    locators = {
        "source": {"backend": "tar", "archive": item.source_archive, "member": item.source_file},
        "target": {"backend": "tar", "archive": item.asset_archive, "member": item.target_file},
        "region": {"backend": "tar", "archive": item.asset_archive, "member": item.mask_file},
    }
    values, sizes = {}, {}
    mask = None
    for role, locator in locators.items():
        try:
            value = read_locator_bytes(locator, root)
            # Image.open is lazy: load must happen while bytes and image handle
            # are alive, or truncated source/target data could enter the pool.
            with Image.open(io.BytesIO(value)) as image:
                image.load()
                sizes[role] = image.size
                if role == "region":
                    mask = np.asarray(image.convert("L")) > 0
            values[role] = value
        except Exception as exc:
            raise PoolBuildError(
                f"asset_integrity_error sample_id={item.sample_id} role={role} "
                f"locator={json.dumps(locator, sort_keys=True)}: {exc}"
            ) from exc
    if not mask.any():
        return None, "empty_mask"
    instruction = item.instruction_original.strip()
    if not instruction:
        return None, "empty_instruction"
    width, height = sizes["region"]
    # Equivalent to _align_to_mask_coordinates's guard without allocating two
    # unused resizes. Do not impose dimension, area, retention or quality cuts.
    if any(abs(sizes[role][0] / sizes[role][1] - width / height) > 1e-6 for role in ("source", "target")):
        return None, "unaligned_aspect_ratio"
    han = contains_han(instruction)
    row = {
        "schema_version": SCHEMA_VERSION, "sample_uid": "0" * 64, "manifest_index": None,
        "dataset_name": "interedit", "dataset_revision": revision, "group_id": str(item.source_id),
        **{f"{role}_locator": locator for role, locator in locators.items()},
        **{f"{role}_sha256": sha256_bytes(value) for role, value in values.items()},
        "instruction_original": instruction, "instruction_en": "" if han else instruction,
        "language_original": "zho_Hans" if han else "en", "edit_type_original": item.edit_type,
        "edit_type_canonical": MAPPING[item.edit_type], "mask_semantics": "user_guidance_region",
        "region_fraction": float(mask.mean()), "translation_status": "pending" if han else "passthrough_en",
        "translation_cache_key": None,
    }
    row["sample_uid"] = sample_uid_for(row)
    validate_record(row)
    return row, None


def build_shard(task: dict) -> dict:
    """Stream row outputs; retain only one record's decoded assets at a time.

    JSONL input is streaming too. Legacy JSON-array input is parsed one shard
    at a time by iter_metadata, rather than materializing the entire pool.
    """
    index = task["index"]
    directory = Path(task["directory"])
    metadata = Path(task["metadata"])
    output = directory / f"shard-{index:06d}.jsonl"
    rejected = directory / f"shard-{index:06d}.rejections.jsonl"
    report_path = directory / f"shard-{index:06d}.report.json"
    partial_output = output.with_suffix(".jsonl.partial")
    partial_rejected = rejected.with_suffix(".jsonl.partial")
    counts = Counter()
    reasons = Counter()
    report = {"index": index, "metadata": str(metadata), "status": "RUNNING"}
    item = None
    position = -1
    stage = "metadata"
    clear_locator_caches()
    try:
        metadata_sha256 = sha256_file(metadata)
        report["metadata_sha256"] = metadata_sha256
        with partial_output.open("w", encoding="utf-8") as rows, partial_rejected.open("w", encoding="utf-8") as rejects:
            for position, item in enumerate(iter_metadata(metadata, only_better_data=False)):
                counts["metadata_rows"] += 1
                if not item.better_data:
                    counts["not_better_data"] += 1
                    continue
                counts["better_data_rows"] += 1
                stage = "asset_integrity"
                row, reason = canonicalize(item, Path(task["root"]), task["revision"])
                if reason is not None:
                    reasons[reason] += 1
                    rejects.write(_json_line({
                        "metadata_index": index, "metadata_row": position,
                        "sample_id": item.sample_id, "source_id": item.source_id, "reason": reason,
                    }))
                else:
                    rows.write(_json_line(row))
                    counts["accepted_rows"] += 1
                stage = "metadata"
        if sha256_file(metadata) != metadata_sha256:
            raise PoolBuildError(f"metadata changed while building shard: {metadata}")
        partial_output.replace(output)
        partial_rejected.replace(rejected)
        report.update(status="COMPLETE", output=str(output), output_sha256=sha256_file(output),
                      rejections=str(rejected), rejections_sha256=sha256_file(rejected))
    except Exception as exc:
        report.update(status="FAILED", error={
            "reason": "asset_integrity_error" if stage == "asset_integrity" else "metadata_error",
            "type": type(exc).__name__, "message": str(exc),
            "metadata_row": position if stage == "asset_integrity" else position + 1,
            "sample_id": item.sample_id if item is not None and stage == "asset_integrity" else None,
        })
    finally:
        clear_locator_caches()
    report["counts"] = {key: counts[key] for key in ("metadata_rows", "better_data_rows", "not_better_data", "accepted_rows")}
    report["rejection_counts"] = dict(sorted(reasons.items()))
    report["rejected_rows"] = sum(reasons.values())
    _atomic_json(report_path, report)
    return report


def _concatenate(paths, destination: Path) -> None:
    # No full-pool Python list/string. Each worker file is copied in CLI shard
    # order into one same-filesystem temporary file before atomic replacement.
    with destination.open("wb") as output:
        for path in paths:
            with Path(path).open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def build_pool(metadata, *, interedit_root, revision, output, workers=DEFAULT_WORKERS) -> dict:
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be in [1,{MAX_WORKERS}]")
    if not metadata:
        raise ValueError("at least one metadata shard is required")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("immutable dataset revision is required")
    root = Path(interedit_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    metadata = [Path(path).resolve() for path in metadata]
    for path in metadata:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = Path(output).resolve()
    report_path = output.with_name(output.name + ".report.json")
    rejection_path = output.with_name(output.name + ".rejections.jsonl")
    if set(metadata) & {output, report_path, rejection_path}:
        raise ValueError("output artifacts must not overwrite input metadata")
    for artifact in (output, report_path, rejection_path):
        if artifact.is_relative_to(root):
            raise ValueError("output artifacts must not be written inside raw Inter-Edit assets")
    output.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f".{output.name}.build-", dir=output.parent))
    effective_workers = min(workers, len(metadata))
    tasks = [{"index": index, "metadata": str(path), "directory": str(directory),
              "root": str(root), "revision": revision} for index, path in enumerate(metadata)]
    report = {
        "schema": REPORT_SCHEMA, "status": "RUNNING", "dataset_name": "interedit",
        "dataset_revision": revision, "interedit_root": str(root), "output": str(output),
        "report_path": str(report_path), "rejections_path": str(rejection_path),
        "workers": effective_workers, "requested_workers": workers, "shard_directory": str(directory),
        "metadata_order": [str(path) for path in metadata], "only_better_data": True,
        "eligibility": "nonempty_mask_instruction_and_existing_same_aspect_alignment",
        "asset_error_policy": "hard_fail_no_pool_publication", "shards": [],
    }
    pool_published = False
    try:
        if effective_workers == 1:
            for task in tasks:
                shard = build_shard(task)
                report["shards"].append(shard)
                if shard["status"] != "COMPLETE":
                    raise PoolBuildError(shard["error"]["message"])
                print(json.dumps({"event": "shard_complete", "index": shard["index"],
                                  "counts": shard["counts"], "rejected_rows": shard["rejected_rows"]}), flush=True)
        else:
            # Spawn avoids inheriting locator handles/thread state. Only tiny
            # task/result dicts cross processes; image bytes stay in the worker.
            with ProcessPoolExecutor(max_workers=effective_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = [pool.submit(build_shard, task) for task in tasks]
                for future in futures:  # submission order, never completion order
                    shard = future.result()
                    report["shards"].append(shard)
                    if shard["status"] != "COMPLETE":
                        for pending in futures:
                            pending.cancel()
                        raise PoolBuildError(shard["error"]["message"])
                    print(json.dumps({"event": "shard_complete", "index": shard["index"],
                                      "counts": shard["counts"], "rejected_rows": shard["rejected_rows"]}), flush=True)
        counts, reasons = Counter(), Counter()
        for shard in report["shards"]:
            counts.update(shard["counts"])
            reasons.update(shard["rejection_counts"])
        report.update(counts=dict(counts), rejection_counts=dict(sorted(reasons.items())), rejected_rows=sum(reasons.values()))
        combined = directory / "combined.jsonl"
        rejected = directory / "rejections.jsonl"
        _concatenate([shard["output"] for shard in report["shards"]], combined)
        _concatenate([shard["rejections"] for shard in report["shards"]], rejected)
        report.update(rows=counts["accepted_rows"], output_sha256=sha256_file(combined),
                      rejections_sha256=sha256_file(rejected))
        # Neither a failed shard nor a partial concatenate can replace the pool.
        rejected.replace(rejection_path)
        combined.replace(output)
        pool_published = True
        report.update(status="COMPLETE", pool_published=True)
        _atomic_json(report_path, report)
        print(json.dumps({"rows": report["rows"], "output": str(output), "report": str(report_path),
                          "workers": effective_workers, "rejected_rows": report["rejected_rows"]}), flush=True)
        return report
    except Exception as exc:
        counts, reasons = Counter(), Counter()
        for shard in report["shards"]:
            counts.update(shard["counts"])
            reasons.update(shard["rejection_counts"])
        report.update(status="FAILED", error={"type": type(exc).__name__, "message": str(exc)},
                      pool_published=pool_published, counts=dict(counts),
                      rejection_counts=dict(sorted(reasons.items())), rejected_rows=sum(reasons.values()),
                      counts_scope="completed_or_failed_shards_in_cli_order_prefix")
        _atomic_json(report_path, report)
        raise PoolBuildError(f"Inter-Edit pool build failed; report={report_path}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", nargs="+", help="metadata shards, preserved in this exact order")
    parser.add_argument("--interedit-root", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, choices=range(1, MAX_WORKERS + 1))
    args = parser.parse_args()
    build_pool(args.metadata, interedit_root=args.interedit_root, revision=args.revision,
               output=args.output, workers=args.workers)


if __name__ == "__main__":
    main()
