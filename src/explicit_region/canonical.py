"""Portable, content-addressed records for the fixed four-dataset corpus."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any

from PIL import Image

from .deterministic import canonical_json


SCHEMA_VERSION = "fixed200k-v2"
DATASET_NAMES = {"magicbrush", "crispedit", "scaleedit", "interedit"}
CANONICAL_EDIT_TYPES = {
    "add", "remove", "replace", "local_attribute", "texture",
    "background", "style", "motion", "other",
}
REQUIRED_FIELDS = {
    "schema_version", "sample_uid", "manifest_index", "dataset_name",
    "dataset_revision", "group_id", "source_locator", "target_locator",
    "region_locator", "source_sha256", "target_sha256", "region_sha256",
    "instruction_original", "instruction_en", "language_original",
    "edit_type_original", "edit_type_canonical", "mask_semantics",
    "region_fraction", "translation_status", "translation_cache_key",
}


class FrozenCorpusIntegrityError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalized_instruction(text: str) -> str:
    return " ".join(str(text).strip().split())


def sample_uid_for(record: dict[str, Any]) -> str:
    payload = "\0".join([
        str(record["dataset_name"]), str(record["dataset_revision"]),
        str(record["source_sha256"]), str(record["target_sha256"]),
        str(record["region_sha256"]),
        normalized_instruction(record["instruction_original"]),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_locator(locator: dict[str, Any]) -> None:
    backend = locator.get("backend")
    path_fields = {
        "file": ("relative_path",), "tar": ("archive", "member"),
        "parquet": ("file",), "derived_bbox": (),
    }
    if backend not in path_fields:
        raise ValueError(f"unsupported locator backend: {backend}")
    for field in path_fields[backend]:
        value = str(locator.get(field, ""))
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"locator {field} must be a safe relative path: {value!r}")
    if backend == "parquet":
        if not isinstance(locator.get("row_index"), int) or not locator.get("column"):
            raise ValueError("parquet locator requires integer row_index and column")
    if backend == "derived_bbox":
        bbox = locator.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("derived_bbox locator requires bbox=[x0,y0,x1,y1]")
        if min(int(locator.get("reference_width", 0)), int(locator.get("reference_height", 0))) <= 0:
            raise ValueError("derived_bbox locator requires positive reference dimensions")


def locator_identity_bytes(locator: dict[str, Any]) -> bytes:
    validate_locator(locator)
    return canonical_json(locator).encode("utf-8")


def _resolve(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise FrozenCorpusIntegrityError(f"asset escapes dataset root: {relative}") from exc
    return candidate


def read_locator_bytes(locator: dict[str, Any], root: str | Path) -> bytes:
    """Read an asset without allowing a manifest to escape its dataset root."""
    validate_locator(locator)
    root = Path(root)
    backend = locator["backend"]
    if backend == "file":
        return _resolve(root, locator["relative_path"]).read_bytes()
    if backend == "tar":
        with tarfile.open(_resolve(root, locator["archive"]), "r:*") as archive:
            handle = archive.extractfile(locator["member"])
            if handle is None:
                raise FrozenCorpusIntegrityError(f"tar member missing: {locator['member']}")
            return handle.read()
    if backend == "parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(
            _resolve(root, locator["file"]), columns=[locator["column"]],
        )
        value = table.column(locator["column"])[locator["row_index"]].as_py()
        if isinstance(value, dict) and "bytes" in value:
            value = value["bytes"]
        if isinstance(value, memoryview):
            value = value.tobytes()
        if not isinstance(value, bytes):
            raise FrozenCorpusIntegrityError("parquet image cell is not bytes")
        return value
    if backend == "derived_bbox":
        return locator_identity_bytes(locator)
    raise AssertionError(backend)


def image_from_locator(locator: dict[str, Any], root: str | Path) -> Image.Image:
    if locator["backend"] == "derived_bbox":
        width, height = int(locator["reference_width"]), int(locator["reference_height"])
        x0, y0, x1, y1 = map(int, locator["bbox"])
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise FrozenCorpusIntegrityError(f"invalid bbox coordinates: {locator['bbox']}")
        image = Image.new("L", (width, height), 0)
        image.paste(255, (x0, y0, x1, y1))
        return image
    return Image.open(io.BytesIO(read_locator_bytes(locator, root))).copy()


def expected_locator_hash(locator: dict[str, Any], root: str | Path) -> str:
    return sha256_bytes(
        locator_identity_bytes(locator)
        if locator["backend"] == "derived_bbox"
        else read_locator_bytes(locator, root)
    )


def validate_record(record: dict[str, Any], *, require_frozen_index: bool = False) -> None:
    missing = sorted(REQUIRED_FIELDS - record.keys())
    if missing:
        raise ValueError(f"canonical record missing fields: {missing}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported canonical schema: {record['schema_version']}")
    if record["dataset_name"] not in DATASET_NAMES:
        raise ValueError(f"invalid dataset_name: {record['dataset_name']}")
    if record["edit_type_canonical"] not in CANONICAL_EDIT_TYPES:
        raise ValueError(f"invalid canonical edit type: {record['edit_type_canonical']}")
    for field in ("source_locator", "target_locator", "region_locator"):
        validate_locator(record[field])
    for field in ("source_sha256", "target_sha256", "region_sha256", "sample_uid"):
        value = str(record[field])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{field} must be lowercase SHA256")
    if record["sample_uid"] != sample_uid_for(record):
        raise ValueError("sample_uid does not match stable content identity")
    if require_frozen_index and not isinstance(record["manifest_index"], int):
        raise ValueError("frozen record requires integer manifest_index")
    if not 0 <= float(record["region_fraction"]) <= 1:
        raise ValueError("region_fraction must be in [0,1]")


def verify_record_assets(record: dict[str, Any], roots: dict[str, str | Path]) -> None:
    validate_record(record)
    root = roots.get(record["dataset_name"])
    if root is None:
        raise FrozenCorpusIntegrityError(f"missing root for {record['dataset_name']}")
    for prefix in ("source", "target", "region"):
        actual = expected_locator_hash(record[f"{prefix}_locator"], root)
        if actual != record[f"{prefix}_sha256"]:
            raise FrozenCorpusIntegrityError(
                f"{prefix} hash mismatch for {record['sample_uid']}: "
                f"expected={record[f'{prefix}_sha256']} actual={actual}"
            )


def load_verified_record_images(
    record: dict[str, Any], root: str | Path, *, tar_reader: Any | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Read each locator once, verify its content identity, then decode it."""
    validate_record(record)
    images = []
    for prefix in ("source", "target", "region"):
        locator = record[f"{prefix}_locator"]
        if locator["backend"] == "derived_bbox":
            value = locator_identity_bytes(locator)
            image = image_from_locator(locator, root)
        elif locator["backend"] == "tar" and tar_reader is not None:
            value = tar_reader.read(locator["archive"], locator["member"])
        else:
            value = read_locator_bytes(locator, root)
        actual = sha256_bytes(value)
        if actual != record[f"{prefix}_sha256"]:
            raise FrozenCorpusIntegrityError(f"{prefix} hash mismatch for {record['sample_uid']}")
        if locator["backend"] != "derived_bbox":
            image = Image.open(io.BytesIO(value)).copy()
        images.append(image)
    return images[0], images[1], images[2]
