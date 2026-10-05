"""Strict runtime reader for an immutable canonical fixed corpus."""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset

from .canonical import (
    FrozenCorpusIntegrityError, image_from_locator, load_verified_record_images, validate_record,
    verify_record_assets,
)
from .dataset import _align_to_mask_coordinates
from .deterministic import stable_seed
from .geometry import apply_geometry, image_to_tensor, sample_geometry
from .language import contains_han


class CanonicalAlignedDataset(Dataset):
    """Read canonical records for one dataset with shared strict geometry."""

    expected_dataset_name: str | None = None

    def __init__(
        self, rows: list[dict], dataset_root: str | Path, *, resolution: int = 1024,
        base_seed: int = 42, epoch: int = 0, verify_hashes: bool = True,
        max_random_attempts: int = 8, minimum_mask_retention: float = 0.75,
    ):
        self.rows = rows
        self.root = Path(dataset_root).resolve()
        self.resolution = resolution
        self.base_seed = base_seed
        self.epoch = epoch
        self.verify_hashes = verify_hashes
        self.max_random_attempts = max_random_attempts
        self.minimum_mask_retention = minimum_mask_retention
        if not self.rows:
            raise ValueError("canonical dataset is empty")
        for row in self.rows:
            validate_record(row)
            if self.expected_dataset_name and row["dataset_name"] != self.expected_dataset_name:
                raise ValueError(
                    f"{self.__class__.__name__} received {row['dataset_name']} record"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict:
        return self.get_with_global_index(index, index)

    def get_with_global_index(self, index: int, global_index: int) -> dict:
        row = self.rows[index]
        try:
            if self.verify_hashes:
                source, target, mask = load_verified_record_images(row, self.root)
            if not row["instruction_en"] or contains_han(row["instruction_en"]):
                raise FrozenCorpusIntegrityError("invalid frozen instruction_en")
            if not self.verify_hashes:
                source = image_from_locator(row["source_locator"], self.root)
                target = image_from_locator(row["target_locator"], self.root)
                mask = image_from_locator(row["region_locator"], self.root)
            source, target, prealignment = _align_to_mask_coordinates(
                source, target, mask, row["sample_uid"]
            )
            geometry = sample_geometry(
                mask, resolution=self.resolution, base_seed=self.base_seed,
                global_sample_index=global_index, sample_key=row["sample_uid"],
                epoch=self.epoch, max_resample_attempts=self.max_random_attempts,
                minimum_mask_retention=self.minimum_mask_retention,
                deterministic_fallback=True,
            )
            source = apply_geometry(source, geometry, is_mask=False)
            target = apply_geometry(target, geometry, is_mask=False)
            mask_tensor = image_to_tensor(apply_geometry(mask, geometry, is_mask=True)).bool()
            if not mask_tensor.any():
                raise FrozenCorpusIntegrityError("empty frozen edit region after geometry")
        except FrozenCorpusIntegrityError:
            raise
        except Exception as exc:
            raise FrozenCorpusIntegrityError(
                f"FROZEN_CORPUS_INTEGRITY_ERROR sample_uid={row['sample_uid']}: {exc}"
            ) from exc
        return {
            "source_image": image_to_tensor(source),
            "target_image": image_to_tensor(target),
            "edit_region_mask": mask_tensor,
            "instruction_en": row["instruction_en"],
            "instruction_original": row["instruction_original"],
            "dataset_name": row["dataset_name"],
            "edit_type": row["edit_type_canonical"],
            "edit_type_original": row["edit_type_original"],
            "mask_semantics": row["mask_semantics"],
            "sample_key": row["sample_uid"],
            "sample_uid": row["sample_uid"],
            "manifest_index": row["manifest_index"],
            "session_id": row["group_id"],
            "group_id": row["group_id"],
            "global_sample_index": global_index,
            "geometry_seed_epoch": self.epoch,
            "geometry_seed": stable_seed(
                self.base_seed, self.epoch, row["sample_uid"], "geometry"
            ),
            "geometry": geometry.to_dict() | {"prealignment": prealignment},
        }


class FixedCorpusDataset(Dataset):
    """Manifest index is immutable; runtime rejection never selects another row."""

    def __init__(
        self, manifest: str | Path, roots: dict[str, str | Path], **dataset_kwargs,
    ):
        self.manifest = Path(manifest).resolve()
        self.rows = [
            json.loads(line) for line in self.manifest.open(encoding="utf-8") if line.strip()
        ]
        if not self.rows:
            raise ValueError(f"empty fixed corpus manifest: {self.manifest}")
        indices = [row.get("manifest_index") for row in self.rows]
        if indices != list(range(len(self.rows))):
            raise FrozenCorpusIntegrityError("manifest_index must be contiguous and ordered")
        uids = [row.get("sample_uid") for row in self.rows]
        if len(set(uids)) != len(uids):
            raise FrozenCorpusIntegrityError("duplicate sample_uid in fixed corpus")
        self.roots = {name: Path(value).resolve() for name, value in roots.items()}
        missing = sorted({row["dataset_name"] for row in self.rows} - self.roots.keys())
        if missing:
            raise ValueError(f"dataset roots missing: {missing}")
        self.backends = {}
        for dataset_name in sorted({row["dataset_name"] for row in self.rows}):
            positions = [i for i, row in enumerate(self.rows) if row["dataset_name"] == dataset_name]
            backend_rows = [self.rows[i] for i in positions]
            backend = CanonicalAlignedDataset(
                backend_rows, self.roots[dataset_name], **dataset_kwargs,
            )
            self.backends[dataset_name] = (backend, {position: j for j, position in enumerate(positions)})

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        for backend, _ in self.backends.values():
            backend.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict:
        return self.get_with_global_index(index, index)

    def get_with_global_index(self, index: int, global_index: int) -> dict:
        row = self.rows[index]
        backend, mapping = self.backends[row["dataset_name"]]
        return backend.get_with_global_index(mapping[index], global_index)
