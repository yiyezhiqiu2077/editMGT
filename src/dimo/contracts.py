"""Teacher checkpoint contract and immutable formal-run gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


DIMO_EDIT_FORMAL_READY = False
DIMO_UPSTREAM_COMMIT = "24613741ec9ca730273a6ebc822327878d5a086b"
DIMO_MODEL_ROLES_V11 = {
    "backend": "shared_base_adapters",
    "student_trainable_scope": "lora_and_region",
    "auxiliary_trainable_scope": "lora_and_region",
    "lora_dropout_policy": "zero_for_dimo",
    "student_lora_dropout": 0.0,
    "auxiliary_lora_dropout": 0.0,
    "teacher_lora_dropout": "inherit",
}
TEACHER_MANIFEST = "dimo_teacher_manifest.json"
TEACHER_BUNDLE_FILES = {
    "adapter_sha256": "adapter_model.safetensors",
    "mask_conditioning_sha256": "mask_conditioning.safetensors",
    "trainable_config_sha256": "trainable_config.json",
    "fingerprint_sha256": "fingerprint.json",
    "teacher_manifest_sha256": TEACHER_MANIFEST,
}
REQUIRED_TEACHER_FIELDS = {
    "base_model_identity",
    "lora_state",
    "edit_region_embedding_state",
    "fingerprint",
    "training_config_identity",
    "source_git_sha",
    "selected_checkpoint_identity",
    "formal_teacher",
}


def released_base_model_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Use the same portable identity for registration, training, and inference.

    Prep-only snapshots created before the formal asset pipeline may not have a
    pinned revision. They retain the complete component report so the existing
    preparation workflow remains content-bound.
    """
    repo_id = identity.get("repo_id")
    revision = identity.get("resolved_revision")
    if not isinstance(repo_id, str) or not isinstance(revision, str) or not re.fullmatch(
        r"[0-9a-f]{40}", revision
    ):
        return identity
    components = {}
    for name, record in sorted(identity.get("components", {}).items()):
        if isinstance(record, dict) and isinstance(record.get("config_sha256"), str):
            components[name] = record["config_sha256"]
    result: dict[str, Any] = {"repo_id": repo_id, "resolved_revision": revision}
    if components:
        result["components"] = components
    return result


@dataclass(frozen=True)
class TeacherCheckpointContract:
    root: Path
    manifest: dict[str, Any]
    checkpoint_hash: str

    @property
    def formal_teacher(self) -> bool:
        return self.manifest.get("formal_teacher") is True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def teacher_bundle_fingerprint(
    checkpoint: str | Path,
    *,
    base_model_identity: object,
    formal: bool = False,
) -> dict[str, Any]:
    """Fingerprint every artifact that defines the teacher's behavior."""
    root = Path(checkpoint).expanduser().resolve()
    manifest_path = root / TEACHER_MANIFEST
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("teacher_backend") == "dense":
        from .dense_teacher import dense_teacher_fingerprint
        return dense_teacher_fingerprint(root, expected_base_identity=base_model_identity, formal=formal)
    result: dict[str, Any] = {"base_model_identity": base_model_identity}
    missing = []
    for field, filename in TEACHER_BUNDLE_FILES.items():
        path = root / filename
        result[field] = _sha256(path) if path.is_file() else None
        if not path.is_file():
            missing.append(filename)
    required = set(TEACHER_BUNDLE_FILES.values()) if formal else {
        "adapter_model.safetensors", "mask_conditioning.safetensors", "trainable_config.json"
    }
    required_missing = sorted(set(missing) & required)
    if base_model_identity in (None, "", {}):
        required_missing.append("base_model_identity")
    if required_missing:
        mode = "formal" if formal else "prep"
        raise RuntimeError(f"{mode} teacher bundle is incomplete: {sorted(required_missing)}")
    result["missing_optional_files"] = sorted(set(missing) - required)
    result["bundle_sha256"] = _canonical_hash(result)
    return result


def build_inference_fingerprint(
    *,
    teacher_bundle_sha256: str,
    base_model_identity: object,
    model_roles: dict[str, Any],
    upstream_commit: str,
) -> dict[str, Any]:
    payload = {
        "teacher_bundle_sha256": teacher_bundle_sha256,
        "base_model_identity": base_model_identity,
        "model_roles": model_roles,
        "dimo_upstream_reference_commit": upstream_commit,
    }
    return payload | {"sha256": _canonical_hash(payload)}


def load_teacher_contract(
    checkpoint: str | Path,
    *,
    expected_base_identity: object | None = None,
) -> TeacherCheckpointContract:
    root = Path(checkpoint).expanduser().resolve()
    manifest_path = root / TEACHER_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError(f"teacher checkpoint is missing {TEACHER_MANIFEST}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("teacher_backend") == "dense":
        from .dense_teacher import load_dense_teacher_contract
        return load_dense_teacher_contract(root, expected_base_identity=expected_base_identity)
    missing = sorted(REQUIRED_TEACHER_FIELDS - set(manifest))
    if missing:
        raise RuntimeError(f"teacher checkpoint contract is incomplete: {missing}")
    if expected_base_identity is not None and manifest["base_model_identity"] != expected_base_identity:
        raise RuntimeError("teacher checkpoint base snapshot is incompatible")
    bundle = teacher_bundle_fingerprint(
        root, base_model_identity=manifest["base_model_identity"], formal=True
    )
    manifest_hash_fields = {
        "lora_state": "adapter_sha256",
        "edit_region_embedding_state": "mask_conditioning_sha256",
        "training_config_identity": "trainable_config_sha256",
        "fingerprint": "fingerprint_sha256",
    }
    for manifest_field, bundle_field in manifest_hash_fields.items():
        value = manifest.get(manifest_field)
        if not isinstance(value, dict) or value.get("sha256") != bundle[bundle_field]:
            raise RuntimeError(
                f"teacher checkpoint contract identity mismatch: {manifest_field}"
            )
    return TeacherCheckpointContract(root, manifest, bundle["bundle_sha256"])


def resolve_teacher_checkpoint(value: str | None) -> str | None:
    if value in (None, "", "${DIMO_TEACHER_CHECKPOINT}"):
        return os.environ.get("DIMO_TEACHER_CHECKPOINT")
    return os.path.expandvars(value)


def enforce_run_guard(
    contract: TeacherCheckpointContract | None,
    *,
    prep_smoke: bool,
    max_optimizer_steps: int,
    formal_ready: bool,
) -> dict[str, bool]:
    if formal_ready or DIMO_EDIT_FORMAL_READY:
        raise RuntimeError("DIMO_EDIT_FORMAL_READY must remain false before teacher selection")
    if prep_smoke:
        if max_optimizer_steps > 2:
            raise RuntimeError("PREP_SMOKE_MAX_STEPS_EXCEEDED")
        return {"formal_teacher": False, "PREP_ONLY_TEACHER": True, "TEACHER_QUALITY_NOT_VALIDATED": True}
    if contract is None or not contract.formal_teacher:
        raise RuntimeError("DIMO_TEACHER_NOT_SELECTED")
    # This milestone cannot be turned into a formal run by editing a config.
    raise RuntimeError("DIMO_EDIT_FORMAL_READY=false")
