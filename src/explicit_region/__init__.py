"""Explicit-region SFT primitives with deterministic, testable semantics."""

from .corruption import CorruptionBatch, prepare_corruption
from .losses import per_sample_cross_entropy
from .masks import pixel_mask_to_token_mask

__all__ = [
    "CorruptionBatch",
    "per_sample_cross_entropy",
    "pixel_mask_to_token_mask",
    "prepare_corruption",
]
