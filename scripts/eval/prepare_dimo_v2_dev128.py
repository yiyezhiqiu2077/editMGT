#!/usr/bin/env python3
"""Freeze the existing DEV128 and old predictions without writing old assets."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dimo.dev_protocol import (COMMON, STUDENT_PROTOCOL, TEACHER_PROTOCOL, ROI_CROP_RULE,
    validate_native_manifest, validate_predictions, validate_plan)
from src.dimo.dense_teacher import load_dense_teacher_contract
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dense_checkpoint import write_json


def prepare(old_plan, old_results, output):
    old_plan, old_results, output = Path(old_plan), Path(old_results), Path(output)
    if output.exists():
        raise FileExistsError(output)
    previous = json.loads(old_plan.read_text())
    for key, value in COMMON.items():
        if previous["inference"].get(key) != value:
            raise RuntimeError(f"DIMO_DEV_PREVIOUS_PROTOCOL_MISMATCH: {key}")
    if previous["inference"].get("teacher_cfg") != 5. or previous["inference"].get("teacher_steps") != 12:
        raise RuntimeError("DIMO_DEV_PREVIOUS_TEACHER_PROTOCOL_MISMATCH")
    teacher = load_dense_teacher_contract(previous["teacher_checkpoint"])
    if teacher.manifest["dense_checkpoint_sha256"] != previous["teacher_dense_sha256"]:
        raise RuntimeError("DIMO_DEV_SELECTED_TEACHER_CHANGED")
    if sha256_file(previous["teacher12_plan"]) != previous["teacher12_plan_sha256"]:
        raise RuntimeError("DIMO_DEV_PREVIOUS_TEACHER_PLAN_CHANGED")
    teacher_plan = json.loads(Path(previous["teacher12_plan"]).read_text())
    if (teacher_plan["candidate_identity"]["checkpoint_sha256"] != previous["teacher_dense_sha256"]
            or teacher_plan["manifest_sha256"] != previous["manifest_sha256"]
            or Path(teacher_plan["candidate_checkpoint"]).resolve() != Path(teacher.manifest["created_from_checkpoint"]).resolve()
            or any(teacher_plan["inference"].get(k) != v for k, v in TEACHER_PROTOCOL.items())):
        raise RuntimeError("DIMO_DEV_HISTORICAL_TEACHER_IDENTITY_UNBOUND")
    identities = {}
    for role in ("student_step5000", "ema_step5000"):
        workers = sorted(old_results.glob(f"worker*/{role}/evaluation_identity.json"))
        if {path.parents[1].name for path in workers} != {f"worker{i}" for i in range(4)}:
            raise RuntimeError("DIMO_DEV_HISTORICAL_WORKER_IDENTITIES_INCOMPLETE")
        for path in workers:
            recorded = json.loads(path.read_text())
            checkpoint_identity = recorded["checkpoint_identity"]
            if (checkpoint_identity["teacher_bundle_fingerprint"]["bundle_sha256"] != teacher.checkpoint_hash
                    or checkpoint_identity["initialization"] != {"mask_ratio": 1., "random_tokens_inside_roi": False}):
                raise RuntimeError("DIMO_DEV_OLD_CHECKPOINT_TEACHER_OR_INIT_MISMATCH")
            identities[str(path.resolve())] = sha256_file(path)
    if not identities:
        raise RuntimeError("DIMO_DEV_HISTORICAL_CHECKPOINT_IDENTITIES_REQUIRED")
    rows = validate_native_manifest(previous["manifest"], previous["manifest_sha256"])
    historical = {}
    for name, folder, protocol in (("SFT Teacher", Path(previous["teacher12_source"]), TEACHER_PROTOCOL),
            ("Old Raw", old_results / "student_step5000", STUDENT_PROTOCOL),
            ("Old EMA", old_results / "ema_step5000", STUDENT_PROTOCOL)):
        path = folder / "predictions.jsonl"
        historical[name] = {"predictions": str(path.resolve()), "predictions_sha256": sha256_file(path),
            "output_sha256": validate_predictions(path, rows, previous["manifest_sha256"], protocol)}
    plan = {"schema": "dimo-v2-dev128-v1", "manifest": previous["manifest"],
        "manifest_sha256": previous["manifest_sha256"], "samples": 128,
        "model_root": previous["model_root"], "teacher_checkpoint": previous["teacher_checkpoint"],
        "teacher_bundle_sha256": teacher.checkpoint_hash,
        "teacher_source_checkpoint_sha256": teacher.manifest["dense_checkpoint_sha256"], "teacher_query_cfg": 4.,
        "student_protocol": STUDENT_PROTOCOL, "teacher_protocol": TEACHER_PROTOCOL,
        "roi_crop_rule": ROI_CROP_RULE, "embedding_models": previous["embedding_models"],
        "bootstrap": previous["bootstrap"], "steps": [0, 2500, 7500, 12500],
        "historical": historical, "historical_identity_files": identities,
        "teacher12_plan": previous["teacher12_plan"], "teacher12_plan_sha256": previous["teacher12_plan_sha256"],
        "source_plan": str(old_plan.resolve()), "source_plan_sha256": sha256_file(old_plan)}
    validate_plan(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, plan)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-plan", required=True)
    parser.add_argument("--old-results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.old_plan, args.old_results, args.output)
