"""Teacher checkpoint contract and immutable formal-run gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DIMO_EDIT_FORMAL_READY = False
DIMO_UPSTREAM_COMMIT = "24613741ec9ca730273a6ebc822327878d5a086b"
TEACHER_MANIFEST = "dimo_teacher_manifest.json"
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
    missing = sorted(REQUIRED_TEACHER_FIELDS - set(manifest))
    if missing:
        raise RuntimeError(f"teacher checkpoint contract is incomplete: {missing}")
    if expected_base_identity is not None and manifest["base_model_identity"] != expected_base_identity:
        raise RuntimeError("teacher checkpoint base snapshot is incompatible")
    return TeacherCheckpointContract(root, manifest, _sha256(manifest_path))


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
