"""Versioned deterministic no-replacement epoch permutation and DDP sharding."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Iterator

from torch.utils.data import Sampler


SAMPLER_SCHEMA_VERSION = "fixed200k-hash-sort-v1"


def epoch_seed(base_seed: int, epoch: int, train_manifest_sha256: str) -> str:
    payload = f"{int(base_seed)}\0epoch-permutation\0{int(epoch)}\0{train_manifest_sha256}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@lru_cache(maxsize=8)
def _cached_epoch_permutation(length: int, base_seed: int, epoch: int, train_manifest_sha256: str) -> tuple[int, ...]:
    """Hash-sort-v1 is independent of torch/numpy RNG implementation versions."""
    if length <= 0 or epoch < 0:
        raise ValueError("length must be positive and epoch non-negative")
    seed = bytes.fromhex(epoch_seed(base_seed, epoch, train_manifest_sha256))
    keyed = []
    for index in range(length):
        digest = hashlib.sha256(seed + index.to_bytes(8, "big")).digest()
        keyed.append((digest, index))
    keyed.sort()
    return tuple(index for _, index in keyed)


def epoch_permutation(length: int, base_seed: int, epoch: int, train_manifest_sha256: str) -> list[int]:
    return list(_cached_epoch_permutation(length, base_seed, epoch, train_manifest_sha256))


def permutation_sha256(permutation: list[int]) -> str:
    digest = hashlib.sha256()
    for index in permutation:
        digest.update(int(index).to_bytes(8, "big"))
    return digest.hexdigest()


class DeterministicEpochSampler(Sampler[int]):
    def __init__(
        self, length: int, *, base_seed: int, epoch: int,
        train_manifest_sha256: str, rank: int = 0, world_size: int = 1,
        samples_consumed_in_epoch: int = 0,
    ):
        if not 0 <= rank < world_size:
            raise ValueError("invalid rank/world_size")
        if length % world_size:
            raise ValueError("no-padding sampler requires length divisible by world_size")
        if samples_consumed_in_epoch < 0 or samples_consumed_in_epoch > length:
            raise ValueError("invalid committed sample cursor")
        if samples_consumed_in_epoch % world_size:
            raise ValueError("global committed cursor must be divisible by world_size")
        self.length = length
        self.base_seed = int(base_seed)
        self.epoch = int(epoch)
        self.train_manifest_sha256 = train_manifest_sha256
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.samples_consumed_in_epoch = int(samples_consumed_in_epoch)
        self.permutation = epoch_permutation(
            length, self.base_seed, self.epoch, self.train_manifest_sha256
        )
        self.epoch_permutation_sha256 = permutation_sha256(self.permutation)

    @property
    def local_skip(self) -> int:
        return self.samples_consumed_in_epoch // self.world_size

    def __iter__(self) -> Iterator[int]:
        return iter(self.permutation[self.rank :: self.world_size][self.local_skip :])

    def __len__(self) -> int:
        return self.length // self.world_size - self.local_skip

    def state_dict(self) -> dict:
        return {
            "sampler_schema_version": SAMPLER_SCHEMA_VERSION,
            "epoch": self.epoch,
            "samples_consumed_in_epoch": self.samples_consumed_in_epoch,
            "epoch_permutation_sha256": self.epoch_permutation_sha256,
            "train_200k_sha256": self.train_manifest_sha256,
            "world_size": self.world_size,
        }
