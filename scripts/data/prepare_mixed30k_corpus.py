#!/usr/bin/env python3
"""Freeze screened/translated 30K additions plus the existing MagicBrush train."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.explicit_region.canonical import read_locator_bytes, validate_record
from src.explicit_region.config import load_config
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dense_checkpoint import verify_dense_checkpoint
from src.explicit_region.fixed_corpus import assert_frozen_corpus
from src.explicit_region.fixed_dataset import FixedCorpusDataset
from src.explicit_region.formal_pipeline import write_json_atomic
from src.explicit_region.interedit import TarMemberReader
from src.explicit_region.language import contains_han
from src.explicit_region.mixed_experiment import COUNTS, MODE, ROWS


def read(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def exclude_connected_groups(rows, protected):
    """Remove full groups connected to protected source OR target content."""
    groups, blocked_images = {}, set()
    for row in protected:
        blocked_images.update((row["source_sha256"], row["target_sha256"]))
    for row in rows:
        key = (row["dataset_name"], row["group_id"])
        groups.setdefault(key, set()).update((row["source_sha256"], row["target_sha256"]))
    blocked_groups = {(row["dataset_name"].removesuffix("_dev"), row["group_id"]) for row in protected}
    while True:
        additional = {key for key, images in groups.items()
                      if key in blocked_groups or images & blocked_images} - blocked_groups
        if not additional:
            # Already blocked group labels must propagate all their image content too.
            before = len(blocked_images)
            for key in blocked_groups:
                blocked_images.update(groups.get(key, set()))
            if len(blocked_images) == before:
                break
        blocked_groups.update(additional)
        for key in additional:
            blocked_images.update(groups[key])
    return [row for row in rows if (row["dataset_name"], row["group_id"]) not in blocked_groups]


def materialize(rows, dataset_roots, output):
    """Preserve original bytes and sample IDs, change locators only."""
    import pyarrow.parquet as pq
    cache, readers, result = {}, {}, []
    try:
        for i, row in enumerate(rows):
            row = dict(row)
            name = row["dataset_name"]
            if name == "magicbrush":
                result.append(row); continue
            root = Path(dataset_roots[name]); target_root = output / name
            for role in ("source", "target", "region"):
                locator = row[f"{role}_locator"]
                digest = row[f"{role}_sha256"]
                path = target_root / "images" / digest[:2] / digest
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    if locator["backend"] == "parquet":
                        key = (name, locator["file"])
                        # At most one parquet cached per dataset; avoids re-reading per role.
                        if cache.get("key") != key:
                            columns = {r[f"{p}_locator"]["column"] for r in rows if r["dataset_name"] == name
                                       for p in ("source", "target", "region") if r[f"{p}_locator"]["file"] == locator["file"]}
                            cache = {"key": key, "table": pq.read_table(root / locator["file"], columns=sorted(columns))}
                        value = cache["table"][locator["column"]][locator["row_index"]].as_py()
                        value = value["bytes"] if isinstance(value, dict) else value
                    elif locator["backend"] == "tar":
                        reader = readers.setdefault(name, TarMemberReader(root))
                        value = reader.read(locator["archive"], locator["member"])
                    else:
                        value = read_locator_bytes(locator, root)
                    import hashlib
                    if hashlib.sha256(value).hexdigest() != digest:
                        raise RuntimeError("candidate bytes changed before materialization")
                    path.write_bytes(value)
                if sha256_file(path) != digest:
                    raise RuntimeError("materialized bytes mismatch")
                row[f"{role}_locator"] = {"backend": "file", "relative_path": str(path.relative_to(target_root))}
            validate_record(row); result.append(row)
            if (i + 1) % 500 == 0:
                print(json.dumps({"materialized": i + 1, "total": len(rows)}), flush=True)
    finally:
        for reader in readers.values():
            reader.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--previous-experiment", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root.resolve(); previous = args.previous_experiment.resolve()
    if json.loads((root / "DOWNLOAD_STATUS.json").read_text())["status"] != "COMPLETE":
        raise RuntimeError("mirror downloads not complete")
    data = root / "data"; pools = data / "pools"; pools.mkdir(parents=True, exist_ok=True)
    logs = root / "coordinator_logs"; logs.mkdir(exist_ok=True)
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OMP_NUM_THREADS"] = "2"
    for key in list(env):
        if key.lower().endswith("_proxy"):
            env.pop(key)

    def command(argv, label, pending=False):
        print(json.dumps({"phase": label}), flush=True)
        with (logs / f"{label}.log").open("a") as log:
            result = subprocess.run([sys.executable, *map(str, argv)], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode not in (0, 42) or (result.returncode == 42 and not pending):
            raise RuntimeError(f"{label} failed with code {result.returncode}")
        return result.returncode

    for name in ("crispedit", "scaleedit"):
        command(["scripts/data/build_schema_mapped_canonical.py", "--root", root / "datasets" / name,
                 "--mapping", f"configs/data/schema/{name}.yaml", "--edit-type-mapping",
                 "configs/data/edit_type_mapping.yaml", "--output", pools / f"{name}.raw.jsonl"], f"canonical_{name}")
    metadata = sorted((root / "datasets/interedit/metadata").glob("*.jsonl.gz"))
    # Only the preregistered subset; metadata from superseded download plans is excluded.
    plan = json.loads((root / "DOWNLOAD_PLAN.json").read_text())
    names = {entry["rfilename"] for entry in plan["datasets"]["interedit"]["files"]}
    metadata = [path for path in metadata if path.relative_to(root / "datasets/interedit").as_posix() in names]
    command(["scripts/data/build_interedit_canonical_pool.py", *metadata, "--interedit-root",
             root / "datasets/interedit", "--revision", plan["datasets"]["interedit"]["revision"],
             "--output", pools / "interedit.raw.jsonl"], "canonical_interedit")
    magic = read(previous / "data/train_8800.jsonl")
    if len(magic) != COUNTS["magicbrush"]:
        raise RuntimeError("original MagicBrush train budget changed")
    write(pools / "magicbrush.jsonl", magic)
    # Use native hashes for all 528 DEV rows, including the excluded geometry failure.
    original_audit = json.loads((previous / "admission/MAGICBRUSH_INPUT_AUDIT.json").read_text())
    protected_dev = [dict(row, dataset_name="magicbrush") for row in original_audit["records"] if row["split"] == "dev"]
    raw = [row for name in ("crispedit", "scaleedit", "interedit") for row in read(pools / f"{name}.raw.jsonl")]
    eligible = [row for row in raw if row["instruction_original"].strip()
                and row["source_sha256"] != row["target_sha256"]]
    eligible = exclude_connected_groups(eligible, protected_dev + magic)
    for name in ("crispedit", "scaleedit", "interedit"):
        write(pools / f"{name}.jsonl", [row for row in eligible if row["dataset_name"] == name])
    write_json_atomic(data / "POOL_SCREENING.json", {"raw_rows": len(raw), "eligible_rows": len(eligible),
        "rejected_empty_instruction_noedit_or_protected_lineage": len(raw) - len(eligible),
        "counts": dict(Counter(row["dataset_name"] for row in eligible)), "test_read": False,
        "quality_filter": "all schema quality/scene/grounding/mask eligibility flags",
        "protected_magicbrush_train_sha256": sha256_file(previous / "data/train_8800.jsonl")})
    # Existing MagicBrush evaluation geometry/pixels/order remain byte-for-byte fixed.
    for filename in ("dev_full527.jsonl", "dev128.jsonl", "dev32.jsonl"):
        (data / filename).write_bytes((previous / "data" / filename).read_bytes())
    cache = data / "translation_cache.jsonl"; cache.touch(exist_ok=True)
    config_path = ROOT / "configs/data/mixed30k_plus_magicbrush.yaml"
    config = load_config(config_path); revision = config["translation"]["immutable_revision"]

    def translate(required, label):
        command(["scripts/data/translate_canonical_candidates.py", "--required", required, "--cache", cache,
                 "--qa-output", data / f"{label}_translation_qa.jsonl", "--backend", "nllb", "--model-path",
                 root / "models/nllb", "--revision", revision, "--formal"], f"translate_{label}")

    # Official DEV needs a canonical record for generic freeze tooling; use only auxiliaries here.
    from scripts.data.freeze_fixed200k_validation import cached_translator, translate_selection, freeze_interedit
    from src.explicit_region.fixed_corpus import freeze_aux_groups
    validation = data / "validation"; validation.mkdir(exist_ok=True)
    for attempt in range(20):
        translator = cached_translator(cache, config["translation"])
        pending, selected = [], []
        for name in ("crispedit", "scaleedit", "interedit"):
            candidates = read(pools / f"{name}.jsonl")
            selector = (lambda rows: freeze_interedit(rows, 128)) if name == "interedit" else (
                lambda rows, name=name: freeze_aux_groups(rows, name, 128))
            values, missing, rejected = translate_selection(candidates, selector, translator, dataset=name)
            pending.extend(missing)
            if values is not None:
                selected.extend(values)
        if not pending:
            break
        required = validation / "translation_required.jsonl";write(required, pending)
        translate(required, "validation")
    else:
        raise RuntimeError("auxiliary translation backfill exhausted")
    # Auxiliaries win all connected source/target/group lineage, not just source hashes.
    for name in ("crispedit", "scaleedit", "interedit"):
        write(validation / f"{name}_aux128.jsonl", [row for row in selected if row["dataset_name"] == name])
        write(pools / f"{name}.jsonl", exclude_connected_groups(read(pools / f"{name}.jsonl"), selected))
    selection = data / "selection"
    argv = ["scripts/data/build_fixed_200k_corpus.py", "--config", config_path, "--allow-nonproduction-total",
            "--translation-cache", cache, "--output-dir", selection]
    for name in COUNTS:
        argv.extend([f"--{name}-pool", pools / f"{name}.jsonl"])
    for path in sorted(validation.glob("*_aux128.jsonl")):
        argv.extend(["--validation", path])
    for attempt in range(20):
        code = command(argv, f"selection_{attempt}", pending=True)
        if code == 0:
            break
        translate(selection / "translation_required.jsonl", f"selection_{attempt}")
    else:
        raise RuntimeError("training translation backfill exhausted")
    frozen = read(selection / f"train_{ROWS}.jsonl")
    if dict(Counter(row["dataset_name"] for row in frozen)) != COUNTS:
        raise RuntimeError("registered dataset quotas not satisfied")
    if {row["sample_uid"] for row in frozen if row["dataset_name"] == "magicbrush"} != {row["sample_uid"] for row in magic}:
        raise RuntimeError("MagicBrush train changed")
    if len(exclude_connected_groups(frozen, selected + protected_dev)) != len(frozen):
        raise RuntimeError("frozen train/DEV lineage leakage")
    dataset_roots = {name: str(root / "datasets" / name) for name in COUNTS if name != "magicbrush"}
    # Sort reads for locality, then restore the already frozen manifest order.
    combined = sorted(frozen + selected, key=lambda row: (row["dataset_name"], json.dumps(row["source_locator"], sort_keys=True)))
    materialized = materialize(combined, dataset_roots, data / "materialized")
    by_uid = {row["sample_uid"]: row for row in materialized}
    frozen = [by_uid[row["sample_uid"]] for row in frozen]
    for name in ("crispedit", "scaleedit", "interedit"):
        write(validation / f"{name}_aux128.jsonl", [by_uid[row["sample_uid"]] for row in selected if row["dataset_name"] == name])
    manifest = data / f"train_{ROWS}.jsonl";write(manifest, frozen);assert_frozen_corpus(frozen, ROWS)
    roots = {name: str(data / "materialized" / name) for name in COUNTS}
    roots["magicbrush"] = str(previous / "data/train")
    import torch
    torch.set_num_threads(2)
    dataset = FixedCorpusDataset(manifest, roots, resolution=1024, base_seed=42, verify_hashes=True)
    for index in range(len(dataset)):
        dataset[index]
        if (index + 1) % 500 == 0:
            print(json.dumps({"loader_verified": index + 1, "total": len(dataset)}), flush=True)
    command(["scripts/data/build_translation_manual_audit.py", "--manifest", manifest,
             "--output-jsonl", data / "translation_review500.jsonl", "--output-csv",
             data / "translation_review500.csv"], "translation_review")
    # FP32 VQ containment and fixed data montages, without changing selection from scores.
    model_root = "/data01/zhangyuyang/editmgt_eval_20261010/assets/editmgt"
    audit_argv = ["scripts/data/audit_fixed200k.py", "--manifest", manifest, "--model-root", model_root,
                  "--output-dir", data / "audit", "--containment-count", 128]
    for name in COUNTS:
        audit_argv.extend([f"--{name}-root", roots[name]])
    command(audit_argv, "data_vq_audit")
    files = [manifest, config_path, cache, data / "POOL_SCREENING.json", data / "audit/audit_report.json",
             previous / "data/MAGICBRUSH_READY.json", data / "translation_review500.jsonl",
             *sorted(validation.glob("*_aux128.jsonl")), *(data / file for file in ("dev_full527.jsonl", "dev128.jsonl", "dev32.jsonl"))]
    checkpoint = previous / "sft/train/checkpoint-3843"
    ready = {"schema": "mixed30k-training-ready-v1", "status": "VERIFIED", "selection_scope": MODE,
             "total_rows": ROWS, "dataset_counts": COUNTS, "train_manifest_sha256": sha256_file(manifest),
             "train_dev_leakage": "PASS", "test_isolation": "NOT_TESTED", "test_read": False,
             "loader_geometry": "PASS", "translation_qa": "PASS", "manual_translation_review": "PENDING",
             "magicbrush_preserved": True, "padding": False,
             "warm_start_checkpoint_sha256": verify_dense_checkpoint(checkpoint)["checkpoint_sha256"],
             "files": {str(path.resolve()): {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in files}}
    write_json_atomic(data / "MIXED_READY.json", ready)
    env.update(MIXED_EXPERIMENT_ROOT=str(root), PREVIOUS_EXPERIMENT_ROOT=str(previous), EDITMGT_MODEL_ROOT=model_root)
    os.environ.update({key: env[key] for key in ("MIXED_EXPERIMENT_ROOT", "PREVIOUS_EXPERIMENT_ROOT", "EDITMGT_MODEL_ROOT")})
    import yaml
    training = load_config(ROOT / "configs/train/dense_mixed30k_4a800_5epoch.yaml")
    (root / "configs").mkdir(exist_ok=True)
    (root / "configs/sft.yaml").write_text(yaml.safe_dump(training, sort_keys=False))
    print(json.dumps(ready), flush=True)


if __name__ == "__main__":
    main()
