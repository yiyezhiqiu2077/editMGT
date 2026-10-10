"""Content-bound provisional/selected Dense teachers, never fake adapters."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from safetensors import safe_open

from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dense_checkpoint import verify_dense_checkpoint, write_json

SCHEMA = "dimo-dense-teacher-v1"


def base_identity(payload):
    value = payload["model_identity"]
    return {"repo_id": value["repo_id"], "resolved_revision": value["resolved_revision"],
            "components": {n: r["config_sha256"] for n, r in value["components"].items()}}


def validate_selected_record(record, checkpoint, identity):
    if record.get("schema") != "selected-dense-checkpoint-v1" or record.get("status") != "READY" or record.get("dev_only") is not True:
        raise RuntimeError("DENSE_TEACHER_SELECTION_NOT_READY")
    if Path(record["checkpoint_path"]).resolve() != Path(checkpoint).resolve() or record.get("checkpoint_sha256") != identity["checkpoint_sha256"]:
        raise RuntimeError("DENSE_TEACHER_SELECTED_IDENTITY_MISMATCH")
    prereg = record.get("selection_preregistration", {})
    payload = {k: v for k, v in prereg.items() if k != "sha256"}
    if prereg.get("sha256") != recipe_fingerprint(payload) or prereg.get("schema") != "dense-selection-prereg-v1":
        raise RuntimeError("DENSE_TEACHER_PREREGISTRATION_MISSING")
    rules = prereg["plan"]["selection"]
    if rules.get("status") != "PREREGISTERED" or any(rules.get(k) is None for k in
            ("maximum_outside_lpips_degradation", "minimum_inside_lpips_improvement")):
        raise RuntimeError("DENSE_TEACHER_SELECTION_THRESHOLD_PENDING")
    metadata = json.loads((Path(checkpoint) / "metadata.json").read_text())
    if record.get("selected_step") != metadata["global_optimizer_step"] or record["selected_step"] not in prereg["plan"]["candidate_steps"]:
        raise RuntimeError("DENSE_TEACHER_SELECTED_STEP_MISMATCH")


def export_dense_teacher(checkpoint, output, *, status, selected_record=None):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    identity = verify_dense_checkpoint(checkpoint)
    payload = json.loads((checkpoint / "fingerprint.json").read_text())["payload"]
    if payload.get("training_mode") != "full_transformer" or payload.get("corruption", {}).get("persistent_conditioning") is not True:
        raise RuntimeError("DENSE_TEACHER_RECIPE_MISMATCH")
    git_sha = payload.get("git_sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", git_sha):
        raise RuntimeError("DENSE_TEACHER_GIT_IDENTITY_MISSING")
    record = None
    if status == "selected":
        if selected_record is None:
            raise RuntimeError("DENSE_TEACHER_SELECTED_RECORD_REQUIRED")
        record = json.loads(Path(selected_record).read_text())
        validate_selected_record(record, checkpoint, identity)
        if record["git_sha"] != git_sha:
            raise RuntimeError("DENSE_TEACHER_SELECTED_GIT_MISMATCH")
    elif status != "provisional":
        raise ValueError("teacher status must be provisional or selected")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    for name in identity["inference_files"]:
        shutil.copyfile(checkpoint / name, temporary / name)
    write_json(temporary / "identity.json", dict(identity, has_training_state=False))
    if record is not None:
        write_json(temporary / "selection_record.json", record)
    manifest = {"schema_version": SCHEMA, "teacher_backend": "dense", "status": status,
                "formal_teacher": status == "selected", "dev_only": True,
                "base_model_identity": base_identity(payload), "source_git_sha": git_sha,
                "dense_checkpoint_sha256": identity["checkpoint_sha256"],
                "selected_checkpoint_identity": identity["checkpoint_sha256"] if record else None,
                "dataset_identity": payload.get("data_content_hashes", {}),
                "region_embedding": {"key": "edit_region_embedding", "contained_in_full_weights": True},
                "training_dtype": "fp32", "created_from_checkpoint": str(checkpoint),
                "selection_record_sha256": sha256_file(temporary / "selection_record.json") if record else None}
    write_json(temporary / "dimo_teacher_manifest.json", manifest)
    ledger = {p.name: sha256_file(p) for p in temporary.iterdir()}
    write_json(temporary / "COMPLETED.json", {"schema": identity["schema"], "files": ledger})
    load_dense_teacher_contract(temporary)
    os.rename(temporary, output)
    return manifest


def load_dense_teacher_contract(root, *, expected_base_identity=None):
    from .contracts import TeacherCheckpointContract
    root = Path(root)
    identity = verify_dense_checkpoint(root, inference_only=False)
    manifest = json.loads((root / "dimo_teacher_manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA or manifest.get("teacher_backend") != "dense":
        raise RuntimeError("DENSE_TEACHER_SCHEMA_MISMATCH")
    if expected_base_identity is not None and manifest["base_model_identity"] != expected_base_identity:
        raise RuntimeError("DENSE_TEACHER_RELEASED_IDENTITY_MISMATCH")
    payload = json.loads((root / "fingerprint.json").read_text())["payload"]
    if manifest["base_model_identity"] != base_identity(payload) or manifest["source_git_sha"] != payload["git_sha"]:
        raise RuntimeError("DENSE_TEACHER_PROVENANCE_MISMATCH")
    if manifest["dense_checkpoint_sha256"] != identity["checkpoint_sha256"]:
        raise RuntimeError("DENSE_TEACHER_CHECKPOINT_IDENTITY_MISMATCH")
    if manifest.get("status") == "provisional":
        if manifest.get("formal_teacher") is not False or manifest.get("dev_only") is not True or manifest.get("selection_record_sha256") is not None:
            raise RuntimeError("PROVISIONAL_DENSE_TEACHER_CANNOT_BE_FORMAL")
    elif manifest.get("status") == "selected":
        if manifest.get("formal_teacher") is not True or not (root / "selection_record.json").is_file():
            raise RuntimeError("SELECTED_DENSE_TEACHER_RECORD_MISSING")
        if sha256_file(root / "selection_record.json") != manifest["selection_record_sha256"]:
            raise RuntimeError("SELECTED_DENSE_TEACHER_RECORD_CHANGED")
        selected = json.loads((root / "selection_record.json").read_text())
        # The export's source path is retained for identity checking, while
        # inference is portable and does not require that path to still exist.
        if selected["checkpoint_sha256"] != identity["checkpoint_sha256"] or selected["status"] != "READY":
            raise RuntimeError("SELECTED_DENSE_TEACHER_IDENTITY_MISMATCH")
        prereg = selected.get("selection_preregistration", {})
        if prereg.get("sha256") != recipe_fingerprint({k: v for k, v in prereg.items() if k != "sha256"}):
            raise RuntimeError("SELECTED_DENSE_TEACHER_PREREGISTRATION_MISMATCH")
    else:
        raise RuntimeError("DENSE_TEACHER_STATUS_INVALID")
    ledger = json.loads((root / "COMPLETED.json").read_text())["files"]
    if "dimo_teacher_manifest.json" not in ledger:
        raise RuntimeError("DENSE_TEACHER_MANIFEST_NOT_HASH_BOUND")
    fingerprint = {"teacher_backend": "dense", "base_model_identity": manifest["base_model_identity"],
                   "files": ledger}
    return TeacherCheckpointContract(root, manifest, recipe_fingerprint(fingerprint))


def dense_teacher_fingerprint(root, *, expected_base_identity, formal=False):
    contract = load_dense_teacher_contract(root, expected_base_identity=expected_base_identity)
    if formal and not contract.formal_teacher:
        raise RuntimeError("DENSE_TEACHER_NOT_SELECTED")
    return {"teacher_backend": "dense", "base_model_identity": contract.manifest["base_model_identity"],
            "bundle_sha256": contract.checkpoint_hash, "formal_teacher": contract.formal_teacher}
