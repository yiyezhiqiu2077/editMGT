"""Stateless randomness keyed by committed sample identity."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_seed(base_seed: int, global_sample_index: int, sample_key: str, namespace: str) -> int:
    payload = canonical_json(
        {
            "base_seed": int(base_seed),
            "global_sample_index": int(global_sample_index),
            "sample_key": str(sample_key),
            "namespace": str(namespace),
        }
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def cpu_generator(base_seed: int, global_sample_index: int, sample_key: str, namespace: str) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(base_seed, global_sample_index, sample_key, namespace))
    return generator


def global_index_for_rank(
    committed_global_sample_count: int,
    micro_step: int,
    rank: int,
    world_size: int,
    per_device_batch: int,
    batch_offset: int = 0,
) -> int:
    if min(committed_global_sample_count, micro_step, rank, batch_offset) < 0:
        raise ValueError("sample-stream indices must be non-negative")
    if world_size <= 0 or per_device_batch <= 0 or rank >= world_size or batch_offset >= per_device_batch:
        raise ValueError("invalid distributed sample-stream geometry")
    return (
        committed_global_sample_count
        + micro_step * world_size * per_device_batch
        + rank * per_device_batch
        + batch_offset
    )
