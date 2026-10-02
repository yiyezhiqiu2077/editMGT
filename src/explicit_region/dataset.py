"""Map-style aligned datasets used by local correctness and formal sampling."""

from __future__ import annotations

import json
import io
from pathlib import Path

from PIL import Image
from PIL import UnidentifiedImageError
from torch.utils.data import Dataset

from .geometry import SampleRejected, apply_geometry, image_to_tensor, sample_geometry
from .deterministic import stable_seed
from .interedit import TarMemberReader
from .language import contains_han


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _align_to_mask_coordinates(
    source: Image.Image, target: Image.Image, mask: Image.Image, sample_key: str
) -> tuple[Image.Image, Image.Image, dict]:
    """Place all three inputs in the benchmark mask coordinate system.

    Some MagicBrush parquet rows store the source at 500x500 while the target
    and native edit mask are 1024x1024. They describe the same square canvas.
    This explicit, audited pre-alignment happens before the shared stochastic
    resize/crop/flip transform. A differing aspect ratio is rejected because a
    blind warp would destroy pixel correspondence.
    """
    original = {"source": list(source.size), "target": list(target.size), "mask": list(mask.size)}
    destination = mask.size
    for name, image in (("source", source), ("target", target)):
        if abs(image.width / image.height - destination[0] / destination[1]) > 1e-6:
            raise ValueError(
                f"unaligned aspect ratio for {sample_key}/{name}: {image.size} vs mask {destination}"
            )
    if target.size != destination:
        target = target.resize(destination, Image.Resampling.BICUBIC)
    if source.size != destination:
        source = source.resize(destination, Image.Resampling.BICUBIC)
    return source, target, {
        "policy": "resize_images_to_native_mask_coordinates_if_aspect_matches",
        "original_sizes": original,
        "aligned_size": list(destination),
    }


class MagicBrushAlignedDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        *,
        resolution: int = 1024,
        base_seed: int = 42,
        global_index_offset: int = 0,
        max_resample_attempts: int = 8,
        minimum_mask_retention: float = 0.75,
    ):
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        with self.manifest.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if not self.rows:
            raise ValueError(f"empty manifest: {self.manifest}")
        self.resolution = resolution
        self.base_seed = base_seed
        self.global_index_offset = global_index_offset
        self.max_resample_attempts = max_resample_attempts
        self.minimum_mask_retention = minimum_mask_retention

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.get_with_global_index(index, self.global_index_offset + index)

    def get_with_global_index(self, index: int, global_index: int) -> dict:
        row = self.rows[index]
        source_path = _resolve(self.root, row["source"])
        target_path = _resolve(self.root, row["target"])
        mask_path = _resolve(self.root, row["mask_edit"])
        with Image.open(source_path) as source_raw, Image.open(target_path) as target_raw, Image.open(mask_path) as mask_raw:
            source = source_raw.copy()
            target = target_raw.copy()
            mask = mask_raw.copy()
        source, target, prealignment = _align_to_mask_coordinates(
            source, target, mask, row["sample_key"]
        )
        geometry = sample_geometry(
            mask,
            resolution=self.resolution,
            base_seed=self.base_seed,
            global_sample_index=global_index,
            sample_key=row["sample_key"],
            max_resample_attempts=self.max_resample_attempts,
            minimum_mask_retention=self.minimum_mask_retention,
        )
        source = apply_geometry(source, geometry, is_mask=False)
        target = apply_geometry(target, geometry, is_mask=False)
        mask = apply_geometry(mask, geometry, is_mask=True)
        if source.size != target.size or source.size != mask.size:
            raise AssertionError("shared geometry produced different output sizes")
        mask_tensor = image_to_tensor(mask).bool()
        if not mask_tensor.any():
            raise SampleRejected("empty_region")
        return {
            "source_image": image_to_tensor(source),
            "target_image": image_to_tensor(target),
            "edit_region_mask": mask_tensor,
            "instruction_en": row["instruction"],
            "instruction_original": row["instruction"],
            "dataset_name": "magicbrush",
            "edit_type": "unknown",
            "mask_semantics": "edit_region",
            "sample_key": row["sample_key"],
            "session_id": row.get("img_id", row["sample_key"].split("_")[0]),
            "global_sample_index": global_index,
            "geometry": geometry.to_dict() | {"prealignment": prealignment},
        }


class InterEditArchiveDataset(Dataset):
    def __init__(
        self, manifest: str | Path, interedit_root: str | Path, *, resolution=1024,
        base_seed=42, max_resample_attempts=8, minimum_mask_retention=0.75,
    ):
        self.manifest = Path(manifest).resolve()
        self.root = Path(interedit_root).resolve()
        self.rows = [json.loads(line) for line in self.manifest.open(encoding="utf-8") if line.strip()]
        if not self.rows:
            raise ValueError(f"empty manifest: {self.manifest}")
        self.resolution = resolution
        self.base_seed = base_seed
        self.max_resample_attempts = max_resample_attempts
        self.minimum_mask_retention = minimum_mask_retention
        self._reader = None

    def __len__(self):
        return len(self.rows)

    @property
    def reader(self):
        if self._reader is None:
            self._reader = TarMemberReader(self.root)
        return self._reader

    def get_with_global_index(self, index: int, global_index: int) -> dict:
        row = self.rows[index]
        if not row.get("instruction_en") or contains_han(row["instruction_en"]):
            raise SampleRejected("invalid_translation")
        source = Image.open(io.BytesIO(self.reader.read(row["source_archive"], row["source_file"]))).copy()
        target = Image.open(io.BytesIO(self.reader.read(row["asset_archive"], row["target_file"]))).copy()
        mask = Image.open(io.BytesIO(self.reader.read(row["asset_archive"], row["mask_file"]))).copy()
        key = str(row["sample_id"])
        source, target, prealignment = _align_to_mask_coordinates(source, target, mask, key)
        geometry = sample_geometry(
            mask, resolution=self.resolution, base_seed=self.base_seed,
            global_sample_index=global_index, sample_key=key,
            max_resample_attempts=self.max_resample_attempts,
            minimum_mask_retention=self.minimum_mask_retention,
        )
        source = apply_geometry(source, geometry, is_mask=False)
        target = apply_geometry(target, geometry, is_mask=False)
        mask_tensor = image_to_tensor(apply_geometry(mask, geometry, is_mask=True)).bool()
        if not mask_tensor.any():
            raise SampleRejected("empty_region")
        return {
            "source_image": image_to_tensor(source), "target_image": image_to_tensor(target),
            "edit_region_mask": mask_tensor, "instruction_en": row["instruction_en"],
            "instruction_original": row["instruction_original"], "dataset_name": "interedit",
            "mask_semantics": "user_guidance_region", "sample_key": key,
            "edit_type": row["edit_type"],
            "session_id": str(row["source_id"]), "global_sample_index": global_index,
            "geometry": geometry.to_dict() | {"prealignment": prealignment},
        }

    def __getitem__(self, index):
        return self.get_with_global_index(index, index)


class DeterministicCoreDataset(Dataset):
    """Stateless weighted CORE sampler indexed only by global sample index."""

    def __init__(
        self, interedit, magicbrush, *, interedit_weight: float, length: int,
        base_seed: int, max_replacement_attempts: int = 32,
    ):
        if not 0 < interedit_weight < 1:
            raise ValueError("CORE requires both Inter-Edit and MagicBrush with nonzero weights")
        self.interedit, self.magicbrush = interedit, magicbrush
        self.interedit_weight, self.length, self.base_seed = interedit_weight, length, base_seed
        self.max_replacement_attempts = max_replacement_attempts

    def __len__(self):
        return self.length

    def __getitem__(self, global_index):
        rejects = []
        for attempt in range(self.max_replacement_attempts):
            draw = stable_seed(
                self.base_seed, global_index, "core", f"dataset-choice:{attempt}"
            ) / (2**63 - 2)
            dataset = self.interedit if draw < self.interedit_weight else self.magicbrush
            sample_index = stable_seed(
                self.base_seed, global_index, "core",
                f"sample-choice:{dataset.__class__.__name__}:{attempt}",
            ) % len(dataset)
            try:
                item = dataset.get_with_global_index(sample_index, global_index)
                item["replacement_attempt"] = attempt
                item["reject_count"] = len(rejects)
                item["reject_summary_json"] = json.dumps(rejects, sort_keys=True)
                return item
            except (SampleRejected, FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
                reason = getattr(exc, "reason", None) or (
                    "missing_asset" if isinstance(exc, FileNotFoundError)
                    else "decode_error" if isinstance(exc, (UnidentifiedImageError, OSError))
                    else "invalid_geometry"
                )
                rejects.append({
                    "reason": reason, "dataset": "interedit" if dataset is self.interedit else "magicbrush",
                    "sample_index": int(sample_index), "replacement_attempt": attempt,
                })
        raise RuntimeError(
            f"deterministic replacement exhausted for global_sample_index={global_index}: "
            + json.dumps(rejects, sort_keys=True)
        )
