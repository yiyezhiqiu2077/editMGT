"""Fail-fast model identity and trainability reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Iterable


COMPONENT_FILES = {
    "transformer": ("editmgt", "config.json"),
    "text_encoder": ("text_encoder", "config.json"),
    "tokenizer": ("tokenizer", "tokenizer_config.json"),
    "llm_encoder": ("llm_encoder", "config.json"),
    "vqvae": ("vqvae", "config.json"),
    "scheduler": ("scheduler", "scheduler_config.json"),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_component_identity(model_root: str | Path, output_path: str | Path | None = None) -> dict:
    root = Path(model_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"model root not found: {root}")
    asset_marker = root / "asset_identity.json"
    asset_identity = json.loads(asset_marker.read_text(encoding="utf-8")) if asset_marker.is_file() else {}
    resolved_revision = asset_identity.get("resolved_revision")
    if resolved_revision is None and root.parent.name == "snapshots" and re.fullmatch(
        r"[0-9a-f]{40}", root.name
    ):
        resolved_revision = root.name
    if resolved_revision is not None and not re.fullmatch(r"[0-9a-f]{40}", str(resolved_revision)):
        raise RuntimeError("released model asset_identity.json has an invalid resolved_revision")
    repo_id = asset_identity.get("repo_id", "WeiChow/EditMGT")
    snapshot_identity = (
        str(resolved_revision)
        if resolved_revision is not None
        else root.name if root.parent.name == "snapshots"
        else sha256_file(root / "editmgt" / "config.json")
    )
    components = {}
    for name, (subfolder, marker) in COMPONENT_FILES.items():
        folder = (root / subfolder).resolve()
        try:
            folder.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"component {name} resolves outside model snapshot") from exc
        marker_path = folder / marker
        if not marker_path.is_file():
            raise FileNotFoundError(f"missing {name} marker: {marker_path}")
        components[name] = {
            "resolved_path": str(folder),
            "repo_identity": repo_id,
            "revision_or_snapshot": snapshot_identity,
            "subfolder": subfolder,
            "config_sha256": sha256_file(marker_path),
            "local_files_only": True,
        }
    report = {
        "model_root": str(root),
        "repo_id": repo_id,
        "resolved_revision": resolved_revision,
        "snapshot_identity": snapshot_identity,
        "components": components,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def audit_trainable_parameters(
    components: dict[str, object], optimizer, output_path: str | Path | None = None
) -> dict:
    rows = []
    all_ids = set()
    trainable_ids = set()
    for component, module in components.items():
        for name, parameter in module.named_parameters():
            pid = id(parameter)
            all_ids.add(pid)
            if parameter.requires_grad:
                trainable_ids.add(pid)
            rows.append(
                {
                    "name": name,
                    "component": component,
                    "requires_grad": parameter.requires_grad,
                    "shape": list(parameter.shape),
                    "numel": parameter.numel(),
                }
            )
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if optimizer_ids != trainable_ids:
        raise RuntimeError(
            f"optimizer/trainable mismatch: optimizer_only={len(optimizer_ids-trainable_ids)}, "
            f"trainable_only={len(trainable_ids-optimizer_ids)}"
        )
    edit_rows = [row for row in rows if row["name"].endswith("edit_region_embedding")]
    if len(edit_rows) != 1 or not edit_rows[0]["requires_grad"]:
        raise RuntimeError("exactly one trainable edit_region_embedding is required")
    total = sum(row["numel"] for row in rows)
    trainable = sum(row["numel"] for row in rows if row["requires_grad"])
    lora = sum(row["numel"] for row in rows if row["requires_grad"] and "lora_" in row["name"])
    report = {
        "parameters": rows,
        "total_params": total,
        "trainable_params": trainable,
        "trainable_ratio": trainable / total,
        "lora_params": lora,
        "edit_region_embedding_params": edit_rows[0]["numel"],
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def freeze_modules(modules: Iterable[object]) -> None:
    for module in modules:
        module.requires_grad_(False)
        module.eval()
