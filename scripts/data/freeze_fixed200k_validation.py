#!/usr/bin/env python3
"""Freeze the v2 primary/probe/auxiliary validation manifests."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explicit_region.canonical import validate_record
from src.explicit_region.config import load_config
from src.explicit_region.contracts import sha256_file
from src.explicit_region.fixed_corpus import (
    INTEREDIT_TYPES, MissingTranslation, apply_translation, cached_translator,
    freeze_aux_groups, freeze_magicbrush_probe, selection_hash, validate_frozen_language,
)


def read(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def write(path, rows):
    Path(path).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def inter_type(row):
    value = str(row["edit_type_original"]).lower()
    return "local" if value == "local" else value


def freeze_interedit(rows, count=128, seed=42):
    groups_by_type = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = inter_type(row)
        if value in INTEREDIT_TYPES:
            groups_by_type[value][row["group_id"]].append(row)
    queues = {}
    for value in INTEREDIT_TYPES:
        ordered_groups = sorted(
            groups_by_type[value],
            key=lambda group: (hashlib.sha256(f"interedit\0aux-validation\0{seed}\0{group}".encode()).hexdigest(), group),
        )
        queues[value] = deque(
            min(groups_by_type[value][group], key=lambda row: (selection_hash(row, seed), row["sample_uid"]))
            for group in ordered_groups
        )
    target, remainder = divmod(count, len(INTEREDIT_TYPES))
    quotas = {value: target + (i < remainder) for i, value in enumerate(INTEREDIT_TYPES)}
    selected, used_groups = [], set()
    for value in INTEREDIT_TYPES:
        while queues[value] and sum(inter_type(row) == value for row in selected) < quotas[value]:
            row = queues[value].popleft()
            if row["group_id"] not in used_groups:
                selected.append(row); used_groups.add(row["group_id"])
    while len(selected) < count:
        progressed = False
        for value in INTEREDIT_TYPES:
            while queues[value]:
                row = queues[value].popleft()
                if row["group_id"] not in used_groups:
                    selected.append(row); used_groups.add(row["group_id"]); progressed = True
                    break
            if len(selected) == count:
                break
        if not progressed:
            raise RuntimeError("insufficient group-disjoint Inter-Edit auxiliary validation rows")
    return selected


def select_translated_aux(rows, name, count, seed, translate):
    """Reselect after QA only; an uncached candidate holds its exact group slot."""
    rejected, history = set(), []
    while True:
        eligible = [row for row in rows if row["sample_uid"] not in rejected]
        chosen = (freeze_interedit(eligible, count, seed) if name == "interedit"
                  else freeze_aux_groups(eligible, name, count, seed))
        accepted, pending, failed = [], [], []
        for row in chosen:
            try:
                translated, flags = apply_translation(row, translate)
            except MissingTranslation:
                pending.append({"dataset": name, "sample_uid": row["sample_uid"],
                                "instruction_original": row["instruction_original"]})
                continue
            if translated is None:
                failed.append(row["sample_uid"])
                history.append({"dataset": name, "sample_uid": row["sample_uid"], "qa_flags": flags})
            else:
                accepted.append(translated)
        # Emit pending before reselecting, including when known failures would
        # otherwise exhaust the eligible groups. No frozen files are published.
        if pending:
            return [], pending, history
        if not failed:
            return accepted, [], history
        rejected.update(failed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--magicbrush-dev", required=True)
    for name in ("crispedit", "scaleedit", "interedit"):
        parser.add_argument(f"--{name}-pool", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--config", default="configs/data/fixed_200k.yaml")
    parser.add_argument("--translation-cache", required=True)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("validation count must be positive")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "validation.meta.json").unlink(missing_ok=True)
    (out.parent / "CORPUS_READY.json").unlink(missing_ok=True)
    (out / "translation_required.jsonl").unlink(missing_ok=True)
    config = load_config(args.config)
    seed = int(config["selection_seed"])
    translate = cached_translator(args.translation_cache, config["translation"])

    def request(pending):
        if pending:
            unique = {row["instruction_original"]: row for row in pending}
            write(out / "translation_required.jsonl", list(unique.values()))
            print(json.dumps({"status": "TRANSLATION_REQUIRED", "unique_prompts": len(unique),
                              "path": str(out / "translation_required.jsonl")}))
            raise SystemExit(42)

    magic, pending = [], []
    # Official DEV must stay complete. Unlike auxiliary pools it cannot backfill
    # or silently discard a failed official record.
    for row in read(args.magicbrush_dev):
        validate_record(row)
        if row["dataset_name"] != "magicbrush":
            raise ValueError("official DEV contains a non-MagicBrush row")
        try:
            translated, flags = apply_translation(row, translate)
        except MissingTranslation:
            pending.append({"dataset": "magicbrush", "sample_uid": row["sample_uid"],
                            "instruction_original": row["instruction_original"]})
            continue
        if translated is None:
            raise RuntimeError(f"official DEV translation QA failed: {row['sample_uid']} {flags}")
        magic.append(translated)
    request(pending)
    outputs = {"magicbrush_official_dev.jsonl": magic,
               "magicbrush_probe128.jsonl": freeze_magicbrush_probe(magic, args.count, seed)}
    used_sources = {row["source_sha256"] for row in magic}
    rejections = []
    for name in ("crispedit", "scaleedit", "interedit"):
        pool = read(getattr(args, f"{name}_pool"))
        for row in pool:
            validate_record(row)
            if row["dataset_name"] != name:
                raise ValueError(f"{name} pool contains another dataset")
        # Validation-first source ownership is resolved deterministically in the
        # same dataset priority as training, before translating a candidate.
        eligible = [row for row in pool if row["source_sha256"] not in used_sources]
        rows, pending, rejected = select_translated_aux(eligible, name, args.count, seed, translate)
        request(pending)
        rejections.extend(rejected)
        outputs[f"{name}_aux128.jsonl"] = rows
        used_sources.update(row["source_sha256"] for row in rows)

    hashes = {}
    for name, rows in outputs.items():
        for row in rows:
            validate_frozen_language(row)
        if len({row["sample_uid"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate validation sample_uid: {name}")
        path = out / name; write(path, rows); hashes[name] = sha256_file(path)
    write(out / "translation_rejections.jsonl", rejections)
    report = {"schema": "fixed200k-validation-v2", "status": "FROZEN", "seed": seed,
              "counts": {name: len(rows) for name, rows in outputs.items()}, "sha256": hashes,
              "source_pool_hashes": {"magicbrush": sha256_file(args.magicbrush_dev),
                  **{name: sha256_file(getattr(args, f"{name}_pool"))
                     for name in ("crispedit", "scaleedit", "interedit")}},
              "selection_config_sha256": sha256_file(args.config),
              "translation_contract": config["translation"],
              "translation_rejections_sha256": sha256_file(out / "translation_rejections.jsonl"),
              "translation_qa_rejections": len(rejections)}
    (out / "validation.meta.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))

if __name__=="__main__":main()
