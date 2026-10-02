from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from src.explicit_region.canonical import SCHEMA_VERSION, sample_uid_for, sha256_bytes


def fixture_record(root: Path, dataset: str, index: int, *, edit_type="other", original_type="unknown") -> dict:
    root.mkdir(parents=True, exist_ok=True)
    source = np.full((24, 32, 3), 30 + index % 100, np.uint8)
    target = source.copy(); target[6:18, 10:24] = [200, 20, 40]
    mask = np.zeros((24, 32), np.uint8); mask[6:18, 10:24] = 255
    paths = {}
    for name, array in (("source", source), ("target", target), ("region", mask)):
        path = root / f"{dataset}_{index}_{name}.png"; Image.fromarray(array).save(path); paths[name] = path
    row = {
        "schema_version": SCHEMA_VERSION, "sample_uid": "0" * 64, "manifest_index": None,
        "dataset_name": dataset, "dataset_revision": "fixture-rev1", "group_id": f"g-{index}",
        **{f"{name}_locator": {"backend": "file", "relative_path": path.name} for name, path in paths.items()},
        **{f"{name}_sha256": sha256_bytes(path.read_bytes()) for name, path in paths.items()},
        "instruction_original": f"edit object {index}", "instruction_en": f"edit object {index}",
        "language_original": "en", "edit_type_original": original_type,
        "edit_type_canonical": edit_type, "mask_semantics": "edit_region",
        "region_fraction": float((mask > 0).mean()), "translation_status": "passthrough_en",
        "translation_cache_key": None,
    }
    row["sample_uid"] = sample_uid_for(row)
    return row


def synthetic_record(dataset: str, index: int, *, original_type="unknown", canonical_type="other", group=None) -> dict:
    def digest(label): return hashlib.sha256(f"{dataset}:{index}:{label}".encode()).hexdigest()
    row = {
        "schema_version": SCHEMA_VERSION, "sample_uid": "0" * 64, "manifest_index": None,
        "dataset_name": dataset, "dataset_revision": "fixture-rev1", "group_id": group or f"{dataset}-g-{index}",
        "source_locator": {"backend": "file", "relative_path": f"source/{index}.png"},
        "target_locator": {"backend": "file", "relative_path": f"target/{index}.png"},
        "region_locator": {"backend": "file", "relative_path": f"region/{index}.png"},
        "source_sha256": digest("source"), "target_sha256": digest("target"),
        "region_sha256": digest("region"), "instruction_original": f"edit {dataset} {index}",
        "instruction_en": f"edit {dataset} {index}", "language_original": "en",
        "edit_type_original": original_type, "edit_type_canonical": canonical_type,
        "mask_semantics": "user_guidance_region" if dataset == "interedit" else "edit_region",
        "region_fraction": ((index % 97) + 1) / 100, "translation_status": "passthrough_en",
        "translation_cache_key": None,
    }
    row["sample_uid"] = sample_uid_for(row)
    return row
