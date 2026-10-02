"""Shared, mask-aware geometry for paired image editing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random

import numpy as np
from PIL import Image, ImageOps
import torch

from .deterministic import stable_seed


class SampleRejected(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Geometry:
    resized_width: int
    resized_height: int
    crop_left: int
    crop_top: int
    crop_width: int
    crop_height: int
    flip: bool
    mask_retention: float
    geometry_mode: str = "random_crop"
    pad_left: int = 0
    pad_top: int = 0
    output_size: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _resize_shape(width: int, height: int, resolution: int) -> tuple[int, int]:
    scale = resolution / min(width, height)
    return max(resolution, round(width * scale)), max(resolution, round(height * scale))


def _binary_mask(mask: Image.Image) -> Image.Image:
    array = np.asarray(mask.convert("L"))
    return Image.fromarray(np.where(array > 0, 255, 0).astype(np.uint8))


def sample_geometry(
    mask: Image.Image,
    *,
    resolution: int,
    base_seed: int,
    global_sample_index: int,
    sample_key: str,
    max_resample_attempts: int = 8,
    minimum_mask_retention: float = 0.75,
    random_flip: bool = True,
    epoch: int = 0,
    deterministic_fallback: bool = True,
) -> Geometry:
    mask = ImageOps.exif_transpose(_binary_mask(mask))
    resized_width, resized_height = _resize_shape(mask.width, mask.height, resolution)
    resized_mask = mask.resize((resized_width, resized_height), Image.Resampling.NEAREST)
    denominator = int((np.asarray(resized_mask) > 0).sum())
    if denominator == 0:
        raise SampleRejected("post_resize_empty_mask")
    if max_resample_attempts <= 0:
        raise ValueError("max_resample_attempts must be positive")
    for attempt in range(max_resample_attempts):
        rng = random.Random(
            stable_seed(base_seed, epoch, sample_key, f"geometry-{attempt}")
        )
        left = rng.randint(0, resized_width - resolution)
        top = rng.randint(0, resized_height - resolution)
        crop = resized_mask.crop((left, top, left + resolution, top + resolution))
        numerator = int((np.asarray(crop) > 0).sum())
        retention = numerator / denominator
        if numerator and retention >= minimum_mask_retention:
            return Geometry(
                resized_width=resized_width,
                resized_height=resized_height,
                crop_left=left,
                crop_top=top,
                crop_width=resolution,
                crop_height=resolution,
                flip=random_flip and rng.random() < 0.5,
                mask_retention=retention,
                geometry_mode="random_crop",
                output_size=resolution,
            )
    if not deterministic_fallback:
        raise SampleRejected("insufficient_mask_retention")
    # Unique whole-image contain resize followed by symmetric center padding.
    # Unlike a crop, this preserves every positive mask pixel.
    scale = resolution / max(mask.width, mask.height)
    contained_width = max(1, round(mask.width * scale))
    contained_height = max(1, round(mask.height * scale))
    resized = mask.resize((contained_width, contained_height), Image.Resampling.NEAREST)
    if not (np.asarray(resized) > 0).any():
        raise SampleRejected("fallback_empty_mask")
    return Geometry(
        resized_width=contained_width,
        resized_height=contained_height,
        crop_left=0,
        crop_top=0,
        crop_width=resolution,
        crop_height=resolution,
        flip=False,
        mask_retention=1.0,
        geometry_mode="contain_center_pad_fallback",
        pad_left=(resolution - contained_width) // 2,
        pad_top=(resolution - contained_height) // 2,
        output_size=resolution,
    )


def apply_geometry(image: Image.Image, geometry: Geometry, *, is_mask: bool) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    interpolation = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    if is_mask:
        image = _binary_mask(image)
    else:
        image = image.convert("RGB")
    image = image.resize((geometry.resized_width, geometry.resized_height), interpolation)
    if geometry.geometry_mode == "contain_center_pad_fallback":
        output_size = geometry.output_size or geometry.crop_width
        fill = 0 if is_mask else (127, 127, 127)
        canvas = Image.new("L" if is_mask else "RGB", (output_size, output_size), fill)
        canvas.paste(image, (geometry.pad_left, geometry.pad_top))
        return canvas
    image = image.crop(
        (
            geometry.crop_left,
            geometry.crop_top,
            geometry.crop_left + geometry.crop_width,
            geometry.crop_top + geometry.crop_height,
        )
    )
    if geometry.flip:
        image = ImageOps.mirror(image)
    return image


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        return torch.from_numpy((array > 0).astype(np.float32))
    return torch.from_numpy(array.transpose(2, 0, 1) / 255.0)
