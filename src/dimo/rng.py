"""Stateless, per-sample RNG streams for Region-DiMO."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import torch


RNG_NAMESPACES = (
    "dimo-init-mask",
    "dimo-init-token",
    "dimo-student-sample",
    "dimo-teacher-query-mask",
    "dimo-aux-query-mask",
    "dimo-embedding-noise",
)


def stable_seed(
    base_seed: int,
    epoch: int,
    sample_uid: str,
    global_dimo_step: int,
    namespace: str,
) -> int:
    """Return a stable 63-bit seed without consuming any global RNG stream."""
    if namespace not in RNG_NAMESPACES:
        raise ValueError(f"unknown DiMO RNG namespace: {namespace}")
    if min(int(base_seed), int(epoch), int(global_dimo_step)) < 0:
        raise ValueError("seed, epoch, and global_dimo_step must be non-negative")
    payload = json.dumps(
        {
            "base_seed": int(base_seed),
            "epoch": int(epoch),
            "sample_uid": str(sample_uid),
            "global_dimo_step": int(global_dimo_step),
            "namespace": namespace,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def sample_seeds(
    *,
    base_seed: int,
    epoch: int,
    sample_uids: Sequence[str],
    global_dimo_step: int,
    namespace: str,
) -> list[int]:
    return [
        stable_seed(base_seed, epoch, uid, global_dimo_step, namespace)
        for uid in sample_uids
    ]


def cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def normal_noise_per_sample(
    reference: torch.Tensor, seeds: Sequence[int]
) -> torch.Tensor:
    """Generate N(0,I) noise per sample without touching global torch RNG."""
    if reference.ndim < 1 or len(seeds) != reference.shape[0]:
        raise ValueError("one embedding-noise seed is required per batch sample")
    rows = [
        torch.randn(
            tuple(reference[i].shape), generator=cpu_generator(seed),
            dtype=torch.float32, device="cpu",
        )
        for i, seed in enumerate(seeds)
    ]
    return torch.stack(rows).to(device=reference.device, dtype=reference.dtype)


def stream_identity() -> dict[str, object]:
    return {"schema": "dimo-stateless-per-sample-v1", "namespaces": list(RNG_NAMESPACES)}
