"""Fail-closed handoff from a selected formal SFT checkpoint to DiMO."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from src.dimo.contracts import load_teacher_contract, teacher_bundle_fingerprint
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.formal_pipeline import sha256_file, stable_json_hash


MANDATORY_CHECKPOINT_FILES = (
    "adapter_model.safetensors",
    "mask_conditioning.safetensors",
    "trainable_config.json",
    "fingerprint.json",
)
TEACHER_MANIFEST = "dimo_teacher_manifest.json"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"DIMO_TEACHER_REGISTRATION_MISMATCH: invalid {label}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"DIMO_TEACHER_REGISTRATION_MISMATCH: invalid {label}")
    return value


def _component_hashes(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for name, record in value.items():
        if not isinstance(record, dict):
            continue
        digest = record.get("config_sha256") or record.get("sha256")
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
            result[str(name)] = digest
    return result


def _identity_fields(value: object) -> tuple[object, object, dict[str, str]]:
    if not isinstance(value, dict):
        return None, None, {}
    repo_id = value.get("repo_id") or value.get("repo_identity")
    revision = (
        value.get("resolved_revision")
        or value.get("revision")
        or value.get("revision_or_snapshot")
    )
    return repo_id, revision, _component_hashes(value.get("components"))


def normalize_base_model_identity(
    fingerprint: dict[str, Any], formal_assets: dict[str, Any]
) -> dict[str, Any]:
    """Resolve and cross-check the immutable released EditMGT identity."""
    payload = fingerprint.get("payload")
    checkpoint_identity = payload.get("model_identity") if isinstance(payload, dict) else None
    assets = formal_assets.get("assets")
    formal_identity = assets.get("editmgt") if isinstance(assets, dict) else None
    checkpoint_repo, checkpoint_revision, checkpoint_components = _identity_fields(
        checkpoint_identity
    )
    formal_repo, formal_revision, formal_components = _identity_fields(formal_identity)
    if (
        not isinstance(checkpoint_repo, str)
        or not isinstance(formal_repo, str)
        or not isinstance(checkpoint_revision, str)
        or not isinstance(formal_revision, str)
        or not _COMMIT.fullmatch(checkpoint_revision)
        or not _COMMIT.fullmatch(formal_revision)
    ):
        raise RuntimeError("DIMO_BASE_MODEL_IDENTITY_MISSING")
    if checkpoint_repo != formal_repo or checkpoint_revision != formal_revision:
        raise RuntimeError("DIMO_TEACHER_REGISTRATION_MISMATCH: base model identity")
    for name in sorted(set(checkpoint_components) & set(formal_components)):
        if checkpoint_components[name] != formal_components[name]:
            raise RuntimeError(
                f"DIMO_TEACHER_REGISTRATION_MISMATCH: base component {name}"
            )
    normalized: dict[str, Any] = {
        "repo_id": checkpoint_repo,
        "resolved_revision": checkpoint_revision,
    }
    if checkpoint_components:
        normalized["components"] = checkpoint_components
    return normalized


def _validated_inputs(
    selected_checkpoint: Path,
    selected_record_path: Path,
    formal_assets_path: Path,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    missing = [name for name in MANDATORY_CHECKPOINT_FILES if not (selected_checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"DIMO_TEACHER_BUNDLE_INCOMPLETE: {missing}")
    selected = _load_json(selected_record_path, "SELECTED_CHECKPOINT.json")
    formal = _load_json(formal_assets_path, "formal_assets.json")
    if selected.get("status") != "READY" or selected.get("schema_version") != "selected-checkpoint-v1":
        raise RuntimeError("DIMO_TEACHER_REGISTRATION_MISMATCH: selected record status")
    try:
        recorded_checkpoint = Path(selected["checkpoint_path"]).expanduser().resolve()
    except (KeyError, TypeError):
        raise RuntimeError("DIMO_TEACHER_REGISTRATION_MISMATCH: checkpoint path") from None
    if recorded_checkpoint != selected_checkpoint:
        raise RuntimeError("DIMO_TEACHER_REGISTRATION_MISMATCH: checkpoint path")
    recorded_files = selected.get("files")
    if not isinstance(recorded_files, dict) or not set(MANDATORY_CHECKPOINT_FILES) <= set(recorded_files):
        raise RuntimeError("DIMO_TEACHER_REGISTRATION_MISMATCH: selected file ledger")
    if any(Path(name).name != name for name in recorded_files):
        raise RuntimeError("DIMO_TEACHER_REGISTRATION_MISMATCH: unsafe selected file ledger")
    actual_files = {}
    for name in recorded_files:
        path = selected_checkpoint / name
        if not path.is_file():
            raise RuntimeError(f"DIMO_TEACHER_REGISTRATION_MISMATCH: missing recorded file {name}")
        actual_files[name] = sha256_file(path)
    if actual_files != recorded_files:
        raise RuntimeError("DIMO_TEACHER_REGISTRATION_MISMATCH: checkpoint file hash")
    if selected.get("checkpoint_identity_sha256") != stable_json_hash(recorded_files):
        raise RuntimeError("DIMO_TEACHER_REGISTRATION_MISMATCH: checkpoint identity")
    if selected.get("formal_assets_sha256") != sha256_file(formal_assets_path):
        raise RuntimeError("DIMO_TEACHER_REGISTRATION_MISMATCH: formal assets hash")
    fingerprint = _load_json(selected_checkpoint / "fingerprint.json", "fingerprint.json")
    payload = fingerprint.get("payload")
    if not isinstance(payload, dict) or fingerprint.get("sha256") != recipe_fingerprint(payload):
        raise RuntimeError("DIMO_TEACHER_REGISTRATION_MISMATCH: fingerprint identity")
    checkpoint_git = payload.get("git_sha") if isinstance(payload, dict) else None
    selected_git = selected.get("git_sha")
    formal_git = formal.get("git_sha")
    if (
        not isinstance(selected_git, str)
        or not _COMMIT.fullmatch(selected_git)
        or selected_git != formal_git
        or selected_git != checkpoint_git
    ):
        raise RuntimeError("DIMO_TEACHER_REGISTRATION_MISMATCH: Git SHA")
    normalized_identity = normalize_base_model_identity(fingerprint, formal)
    mandatory_hashes = {name: actual_files[name] for name in MANDATORY_CHECKPOINT_FILES}
    return selected, mandatory_hashes, normalized_identity


def _manifest(
    *,
    selected: dict[str, Any],
    hashes: dict[str, str],
    normalized_identity: dict[str, Any],
    selected_checkpoint: Path,
    selected_record_path: Path,
    formal_assets_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "dimo-formal-teacher-v1",
        "base_model_identity": normalized_identity,
        "lora_state": {
            "file": "adapter_model.safetensors",
            "sha256": hashes["adapter_model.safetensors"],
        },
        "edit_region_embedding_state": {
            "file": "mask_conditioning.safetensors",
            "sha256": hashes["mask_conditioning.safetensors"],
        },
        "training_config_identity": {
            "file": "trainable_config.json",
            "sha256": hashes["trainable_config.json"],
        },
        "fingerprint": {
            "file": "fingerprint.json",
            "sha256": hashes["fingerprint.json"],
        },
        "source_git_sha": selected["git_sha"],
        "selected_checkpoint_identity": selected["checkpoint_identity_sha256"],
        "formal_teacher": True,
        "formal_assets_sha256": sha256_file(formal_assets_path),
        "selected_checkpoint_record_sha256": sha256_file(selected_record_path),
        "created_from_checkpoint": str(selected_checkpoint),
    }


def _directory_hashes(root: Path) -> dict[str, str] | None:
    expected = set(MANDATORY_CHECKPOINT_FILES) | {TEACHER_MANIFEST}
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected:
        return None
    return {name: sha256_file(root / name) for name in sorted(expected)}


def register_teacher(
    selected_checkpoint: str | Path,
    selected_checkpoint_record: str | Path,
    formal_assets: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    checkpoint = Path(selected_checkpoint).expanduser().resolve()
    record_path = Path(selected_checkpoint_record).expanduser().resolve()
    assets_path = Path(formal_assets).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    selected, hashes, normalized_identity = _validated_inputs(checkpoint, record_path, assets_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent))
    renamed = False
    try:
        for name in MANDATORY_CHECKPOINT_FILES:
            shutil.copy2(checkpoint / name, temporary / name)
            if sha256_file(temporary / name) != hashes[name]:
                raise RuntimeError("DIMO_TEACHER_REGISTRATION_MISMATCH: copied file hash")
        manifest = _manifest(
            selected=selected,
            hashes=hashes,
            normalized_identity=normalized_identity,
            selected_checkpoint=checkpoint,
            selected_record_path=record_path,
            formal_assets_path=assets_path,
        )
        (temporary / TEACHER_MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        load_teacher_contract(temporary, expected_base_identity=normalized_identity)
        bundle = teacher_bundle_fingerprint(
            temporary, base_model_identity=normalized_identity, formal=True
        )
        prospective = _directory_hashes(temporary)
        if output.exists():
            if prospective is not None and _directory_hashes(output) == prospective:
                return {
                    "status": "ALREADY_REGISTERED_IDENTICAL",
                    "output_dir": str(output),
                    "bundle_sha256": bundle["bundle_sha256"],
                    "base_model_identity": normalized_identity,
                }
            raise RuntimeError("DIMO_TEACHER_OUTPUT_CONFLICT")
        os.rename(temporary, output)
        renamed = True
        contract = load_teacher_contract(output, expected_base_identity=normalized_identity)
        return {
            "status": "TEACHER_REGISTRATION_PASS",
            "output_dir": str(output),
            "bundle_sha256": contract.checkpoint_hash,
            "base_model_identity": normalized_identity,
        }
    finally:
        if not renamed and temporary.exists():
            shutil.rmtree(temporary)
