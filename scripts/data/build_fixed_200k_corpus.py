#!/usr/bin/env python3
"""Freeze an exact canonical corpus from pre-audited eligible pool manifests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.explicit_region.canonical import image_from_locator, validate_record
from src.explicit_region.config import load_config
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dataset import _align_to_mask_coordinates
from src.explicit_region.fixed_corpus import (
    DATASET_PRIORITY, apply_translation, assert_frozen_corpus, canonical_freeze_order,
    collapse_exact_sample_duplicates, corpus_counts, deterministic_order,
    select_interedit, selection_hash,
)
from src.explicit_region.language import translation_cache_key
from src.explicit_region.geometry import SampleRejected, apply_geometry, sample_geometry


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sanitize_geometry(rows: list[dict], root: Path) -> tuple[list[dict], list[dict]]:
    kept, rejected = [], []
    for index, row in enumerate(rows):
        images = {}
        reason = None
        try:
            for role in ("source", "target", "region"):
                images[role] = image_from_locator(row[f"{role}_locator"], root)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            reason = {"reason": "decode_failure", "error": str(exc)}
        if reason is None:
            if not (np.asarray(images["region"].convert("L")) > 0).any():
                reason = {"reason": "empty_region"}
        if reason is None:
            aligned_source = aligned_target = None
            try:
                aligned_source, aligned_target, _ = _align_to_mask_coordinates(
                    images["source"], images["target"], images["region"], row["sample_uid"]
                )
                geometry = sample_geometry(
                    images["region"], resolution=1024, base_seed=42,
                    global_sample_index=index, sample_key=row["sample_uid"],
                    max_resample_attempts=8, minimum_mask_retention=.75,
                )
                if not (np.asarray(apply_geometry(images["region"], geometry, is_mask=True)) > 0).any():
                    reason = {"reason": "post_geometry_empty_region"}
            except SampleRejected as exc:
                reason = {"reason": "post_geometry_empty_region", "error": exc.reason}
            except ValueError as exc:
                if "unaligned aspect ratio" not in str(exc):
                    raise
                reason = {"reason": "unaligned_aspect_ratio", "error": str(exc)}
            finally:
                if aligned_source is not None and aligned_source is not images["source"]:
                    aligned_source.close()
                if aligned_target is not None and aligned_target is not images["target"]:
                    aligned_target.close()
        for image in images.values():
            image.close()
        if reason is None:
            kept.append(row)
        else:
            rejected.append({
                "dataset": row["dataset_name"], "sample_uid": row["sample_uid"],
                "group_id": row["group_id"], **reason,
            })
    return kept, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/fixed_200k.yaml")
    for name in DATASET_PRIORITY:
        parser.add_argument(f"--{name}-pool", required=True)
        parser.add_argument(f"--{name}-root")
    parser.add_argument("--validation", action="append", default=[])
    parser.add_argument("--translation-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-nonproduction-total", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config); seed = int(config["selection_seed"])
    translation_config = config["translation"]
    total = int(config["train_total"])
    if total != 200000 and not args.allow_nonproduction_total:
        raise SystemExit("production fixed corpus total must be exactly 200000")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "translation_required.jsonl").unlink(missing_ok=True)

    validation = [row for path in args.validation for row in read_jsonl(path)]
    validation_sources = {row["source_sha256"] for row in validation}
    validation_groups = {(row["dataset_name"], row["group_id"]) for row in validation}
    pools, exact_removed, geometry_rejections = {}, [], []
    for name in DATASET_PRIORITY:
        rows = read_jsonl(getattr(args, f"{name}_pool"))
        for row in rows:
            validate_record(row)
            if row["dataset_name"] != name:
                raise ValueError(f"{name} pool contains {row['dataset_name']} row")
        root_value = getattr(args, f"{name}_root")
        # Canonical builders validate every dataset. This independent freeze-time
        # backstop targets MagicBrush, whose legacy materialization predates that gate.
        if root_value and name == "magicbrush":
            rows, rejected = sanitize_geometry(rows, Path(root_value).resolve())
            geometry_rejections.extend(rejected)
        rows = [row for row in rows if row["source_sha256"] not in validation_sources
                and (row["dataset_name"], row["group_id"]) not in validation_groups]
        pools[name], removed = collapse_exact_sample_duplicates(rows, seed)
        exact_removed.extend({"reason": "same_dataset_exact_duplicate", "dataset": name,
                              "sample_uid": row["sample_uid"]} for row in removed)
    write_jsonl(output / "geometry_rejections.jsonl", geometry_rejections)

    cache_rows = read_jsonl(args.translation_cache)
    translations = defaultdict(list)
    for row in cache_rows:
        translations[row["source_text"]].append(row)

    class MissingTranslation(RuntimeError):
        pass

    def translate(text, _row):
        expected_key = translation_cache_key(
            text, translation_config["backend"], translation_config["immutable_revision"],
            translation_config["src_lang"], translation_config["tgt_lang"],
            translation_config["decoding"],
        )
        available = [row for row in translations.get(text, []) if row.get("cache_key") == expected_key]
        if not available:
            raise MissingTranslation(text)
        if len(available) != 1:
            raise RuntimeError(f"ambiguous translation cache entries for {text!r}")
        return available[0]["translated_text"], expected_key

    candidate_artifact, reserve_artifact = [], []
    translation_rejections, translation_required, backfill_history, duplicate_events = [], [], [], list(exact_removed)
    selected = {name: [] for name in DATASET_PRIORITY}; owned_source = {}

    def request_pending() -> None:
        if not translation_required:
            return
        unique_required = {row["instruction_original"]: row for row in translation_required}
        write_jsonl(output / "translation_required.jsonl", list(unique_required.values()))
        write_jsonl(output / "candidate_selection.jsonl", candidate_artifact)
        write_jsonl(output / "reserve_order.jsonl", reserve_artifact)
        (output / "provisional_candidate_uids.json").write_text(
            json.dumps([row["sample_uid"] for row in candidate_artifact], indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "status": "TRANSLATION_REQUIRED", "unique_prompts": len(unique_required),
            "path": str(output / "translation_required.jsonl"),
        }))
        raise SystemExit(42)

    magic_policy = config["dataset_policy"]["magicbrush"]
    magic_target = int(magic_policy.get("fixed_count", len(pools["magicbrush"])))
    requested_crisp = int(config["dataset_policy"]["crispedit"]["cap"])
    requested_scale = int(config["dataset_policy"]["scaleedit"]["cap"])
    strict_counts = bool(config.get("strict_dataset_counts", False))
    if strict_counts:
        requested = {
            "magicbrush": magic_target, "crispedit": requested_crisp,
            "scaleedit": requested_scale,
        }
        shortages = {
            name: count - len(pools[name]) for name, count in requested.items()
            if len(pools[name]) < count
        }
        if shortages:
            raise RuntimeError(f"INSUFFICIENT_FIXED_DATASET_ROWS: {shortages}")
    magic_target = min(magic_target, len(pools["magicbrush"]))
    crisp_target = min(requested_crisp, len(pools["crispedit"]))
    scale_target = min(requested_scale, len(pools["scaleedit"]))
    general_targets = {"magicbrush": magic_target, "crispedit": crisp_target, "scaleedit": scale_target}
    general_orders = {name: deterministic_order(pools[name], seed) for name in general_targets}
    # Duplicate ownership is frozen on provisional candidates before translation,
    # matching the v2 phase order. A winning candidate that later fails QA does
    # not make the lower-priority duplicate eligible again.
    provisional_owner = {}
    for name in ("magicbrush", "crispedit", "scaleedit"):
        for row in general_orders[name][:general_targets[name]]:
            provisional_owner.setdefault(row["source_sha256"], name)

    def select_general(name: str, cap: int) -> None:
        ordered = general_orders[name]
        candidate_artifact.extend(dict(row, selection_phase="provisional") for row in ordered[:cap])
        reserve_artifact.extend(dict(row, selection_phase="reserve") for row in ordered[cap:])
        occupied = 0
        for position, row in enumerate(ordered):
            owner = provisional_owner.get(row["source_sha256"], owned_source.get(row["source_sha256"]))
            if owner is not None and owner != name and DATASET_PRIORITY.index(owner) < DATASET_PRIORITY.index(name):
                duplicate_events.append({
                    "reason": "cross_dataset_source_duplicate", "source_sha256": row["source_sha256"],
                    "winning_dataset": owner, "removed_dataset": name,
                    "removed_sample_uid": row["sample_uid"],
                })
                continue
            if owner is not None and owner != name:
                duplicate_events.append({
                    "reason": "higher_priority_reserve_reclaims_source",
                    "source_sha256": row["source_sha256"], "winning_dataset": name,
                    "removed_dataset": owner, "refill_sample_uid": row["sample_uid"],
                })
                provisional_owner[row["source_sha256"]] = name
            try:
                translated, flags = apply_translation(row, translate)
            except MissingTranslation:
                translation_required.append({"dataset": name, "sample_uid": row["sample_uid"],
                                             "instruction_original": row["instruction_original"]})
                occupied += 1
                owned_source.setdefault(row["source_sha256"], name)
                provisional_owner.setdefault(row["source_sha256"], name)
                if occupied == cap:
                    return
                continue
            if translated is None:
                translation_rejections.append({"dataset": name, "sample_uid": row["sample_uid"], "qa_flags": flags})
                continue
            selected[name].append(translated); owned_source.setdefault(row["source_sha256"], name)
            provisional_owner.setdefault(row["source_sha256"], name)
            if position >= cap:
                backfill_history.append({"dataset": name, "reason": "general_reserve_backfill", "sample_uid": row["sample_uid"]})
            occupied += 1
            if occupied == cap:
                return

    for name, target in general_targets.items():
        select_general(name, target)
        request_pending()

    inter_quota = total - sum(len(selected[name]) for name in ("magicbrush", "crispedit", "scaleedit"))
    if inter_quota < 0:
        raise RuntimeError("non-InterEdit accepted rows exceed final corpus total")
    inter_pool = [row for row in pools["interedit"]
                  if row["source_sha256"] not in provisional_owner
                  and row["source_sha256"] not in owned_source]
    provisional, reserve, redistribution = select_interedit(inter_pool, inter_quota, seed)
    candidate_artifact.extend(dict(row, selection_phase="provisional") for row in provisional)
    reserve_artifact.extend(dict(row, selection_phase="reserve") for row in reserve)
    reserve_by_stratum = defaultdict(deque)
    for row in reserve:
        reserve_by_stratum[row["selection_stratum"]].append(row)
    global_reserve = deque(reserve)
    used_uids = {row["sample_uid"] for row in provisional}

    def next_reserve(stratum: str) -> dict | None:
        while reserve_by_stratum[stratum]:
            row = reserve_by_stratum[stratum].popleft()
            if row["sample_uid"] not in used_uids:
                used_uids.add(row["sample_uid"]); return row
        while global_reserve:
            row = global_reserve.popleft()
            if row["sample_uid"] not in used_uids:
                used_uids.add(row["sample_uid"]); return row
        return None

    queue = deque(provisional)
    while len(selected["interedit"]) < inter_quota:
        if not queue:
            break
        row = queue.popleft()
        owner = provisional_owner.get(row["source_sha256"], owned_source.get(row["source_sha256"]))
        if owner is not None and owner != "interedit":
            duplicate_events.append({"reason": "cross_dataset_source_duplicate", "source_sha256": row["source_sha256"],
                                     "winning_dataset": owner, "removed_dataset": "interedit",
                                     "removed_sample_uid": row["sample_uid"]})
            replacement = next_reserve(row["selection_stratum"])
            if replacement: queue.append(replacement)
            continue
        try:
            translated, flags = apply_translation(row, translate)
        except MissingTranslation:
            translation_required.append({"dataset": "interedit", "sample_uid": row["sample_uid"],
                                         "instruction_original": row["instruction_original"],
                                         "selection_stratum": row["selection_stratum"]})
            continue
        if translated is None:
            translation_rejections.append({"dataset": "interedit", "sample_uid": row["sample_uid"], "qa_flags": flags})
            replacement = next_reserve(row["selection_stratum"])
            if replacement:
                queue.append(replacement)
                backfill_history.append({"dataset": "interedit", "reason": "translation_qa",
                                         "rejected": row["sample_uid"], "replacement": replacement["sample_uid"],
                                         "requested_stratum": row["selection_stratum"],
                                         "replacement_stratum": replacement["selection_stratum"]})
            continue
        selected["interedit"].append(translated); owned_source.setdefault(row["source_sha256"], "interedit")

    request_pending()
    if len(selected["interedit"]) != inter_quota:
        raise RuntimeError("INSUFFICIENT_INTEREDIT_ELIGIBLE_DATA after translation QA")

    frozen = canonical_freeze_order([row for name in DATASET_PRIORITY for row in selected[name]], seed)
    assert_frozen_corpus(frozen, total)
    if {row["source_sha256"] for row in frozen} & validation_sources:
        raise RuntimeError("train/validation exact source leakage")
    if {(row["dataset_name"], row["group_id"]) for row in frozen} & validation_groups:
        raise RuntimeError("train/validation group leakage")
    train_path = output / ("train_200k.jsonl" if total == 200000 else f"train_{total}.jsonl")
    write_jsonl(train_path, frozen)
    write_jsonl(output / "candidate_selection.jsonl", candidate_artifact)
    write_jsonl(output / "reserve_order.jsonl", reserve_artifact)
    write_jsonl(output / "translation_rejections.jsonl", translation_rejections)
    write_jsonl(output / "backfill_history.jsonl", backfill_history)
    write_jsonl(output / "final_selection.jsonl", frozen)
    (output / "duplicate_report.json").write_text(json.dumps({"events": duplicate_events}, indent=2, sort_keys=True) + "\n")
    selection_report = {
        "schema_version": "fixed200k-selection-v2", "seed": seed,
        "dataset_counts": corpus_counts(frozen), "total_rows": len(frozen),
        "interedit_requested_quota": inter_quota, "interedit_redistribution": redistribution,
        "validation_source_count": len(validation_sources), "validation_group_count": len(validation_groups),
        "geometry_rejected_rows": len(geometry_rejections),
    }
    (output / "selection_report.json").write_text(json.dumps(selection_report, indent=2, sort_keys=True) + "\n")
    translation_report = {
        "rejections": len(translation_rejections),
        "qa_flag_counts": dict(Counter(flag for row in translation_rejections for flag in row["qa_flags"])),
        "backfills": len(backfill_history),
    }
    (output / "translation_report.json").write_text(json.dumps(translation_report, indent=2, sort_keys=True) + "\n")
    metadata = {
        "schema_version": "fixed200k-meta-v2", "seed": seed, "total_rows": len(frozen),
        "dataset_counts": corpus_counts(frozen),
        "dataset_fractions": {key: value / len(frozen) for key, value in corpus_counts(frozen).items()},
        "unique_source_counts": {name: len({row["source_sha256"] for row in selected[name]}) for name in DATASET_PRIORITY},
        "edit_type_counts": dict(Counter(row["edit_type_canonical"] for row in frozen)),
        "mask_semantics_counts": dict(Counter(row["mask_semantics"] for row in frozen)),
        "train_sha256": sha256_file(train_path), "selection_config_sha256": sha256_file(args.config),
        "source_pool_hashes": {name: sha256_file(getattr(args, f"{name}_pool")) for name in DATASET_PRIORITY},
        "dataset_revisions": {name: sorted({row["dataset_revision"] for row in pools[name]}) for name in DATASET_PRIORITY},
        "validation_hashes": {str(path): sha256_file(path) for path in args.validation},
        "translation_cache_sha256": sha256_file(args.translation_cache),
        "translation_contract": translation_config,
        "duplicate_policy": config["duplicate_policy"],
    }
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    metadata["created_from_git_sha"] = git_sha
    meta_path = output / ("train_200k.meta.json" if total == 200000 else f"train_{total}.meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "FINAL_SELECTION_FROZEN", "total": len(frozen), "counts": corpus_counts(frozen),
                      "train": str(train_path), "corpus_ready": "NOT_WRITTEN_UNTIL_AUDITS_COMPLETE"}, indent=2))


if __name__ == "__main__":
    main()
