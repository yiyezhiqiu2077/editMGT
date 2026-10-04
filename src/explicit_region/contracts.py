"""Fail-fast model identity and trainability reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
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
    snapshot_identity = root.name if root.parent.name == "snapshots" else sha256_file(root / "editmgt" / "config.json")
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
            "repo_identity": "WeiChow/EditMGT",
            "revision_or_snapshot": snapshot_identity,
            "subfolder": subfolder,
            "config_sha256": sha256_file(marker_path),
            "local_files_only": True,
        }
    report = {"model_root": str(root), "snapshot_identity": snapshot_identity, "components": components}
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def region_embedding_is_active(corruption: dict) -> bool:
    """Whether this recipe ever uses the persistent region-conditioning vector."""
    if not corruption.get("persistent_conditioning", True):
        return False
    mode = corruption["mode"]
    return mode == "roi_hardlock" or (
        mode == "mixed" and float(corruption.get("roi_probability", 0.5)) > 0
    )


def configure_region_trainability(model, corruption: dict) -> bool:
    active = region_embedding_is_active(corruption)
    model.edit_region_embedding.requires_grad_(active)
    return active


def audit_trainable_parameters(
    components: dict[str, object], optimizer, output_path: str | Path | None = None,
    *, region_conditioning_active: bool = True,
) -> dict:
    rows = []
    trainable_ids = set()
    for component, module in components.items():
        for name, parameter in module.named_parameters():
            pid = id(parameter)
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
    optimizer_parameters = [p for group in optimizer.param_groups for p in group["params"]]
    optimizer_ids = {id(p) for p in optimizer_parameters}
    if len(optimizer_ids) != len(optimizer_parameters):
        raise RuntimeError("duplicate parameter in optimizer")
    if optimizer_ids != trainable_ids:
        raise RuntimeError(
            f"optimizer/trainable mismatch: optimizer_only={len(optimizer_ids-trainable_ids)}, "
            f"trainable_only={len(trainable_ids-optimizer_ids)}"
        )
    edit_rows = [row for row in rows if row["name"].endswith("edit_region_embedding")]
    if len(edit_rows) != 1 or edit_rows[0]["requires_grad"] != region_conditioning_active:
        raise RuntimeError(
            "exactly one edit_region_embedding with requires_grad="
            f"{region_conditioning_active} is required by this recipe"
        )
    total = sum(row["numel"] for row in rows)
    trainable = sum(row["numel"] for row in rows if row["requires_grad"])
    lora = sum(row["numel"] for row in rows if row["requires_grad"] and "lora_" in row["name"])
    report = {
        "parameters": rows,
        "total_params": total,
        "trainable_params": trainable,
        "trainable_ratio": trainable / total,
        "lora_params": lora,
        "region_conditioning_active": region_conditioning_active,
        "edit_region_embedding_params": edit_rows[0]["numel"],
        "trainable_edit_region_embedding_params": (
            edit_rows[0]["numel"] if region_conditioning_active else 0
        ),
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
