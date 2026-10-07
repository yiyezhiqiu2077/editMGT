"""Versioned deterministic no-replacement epoch permutation and DDP sharding."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Iterator

from torch.utils.data import Sampler
from torch.utils.data import DataLoader


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

    def reset_epoch(self, epoch: int, samples_consumed_in_epoch: int = 0) -> None:
        replacement = type(self)(
            self.length, base_seed=self.base_seed, epoch=epoch,
            train_manifest_sha256=self.train_manifest_sha256, rank=self.rank,
            world_size=self.world_size, samples_consumed_in_epoch=samples_consumed_in_epoch,
        )
        self.__dict__.update(replacement.__dict__)


class MultiEpochFixedLoader:
    """Iterate real epochs, retaining immutable row indices and sampler schema.

    The sampler reference is stable so checkpoints always observe the active
    epoch. Each new iterator replays from the construction-time resume cursor;
    this also allows DDP reducer priming without consuming committed samples.
    Geometry follows dataset.set_epoch; corruption's existing UID/index seed
    contract is deliberately unchanged.
    """

    def __init__(self, dataset, sampler, *, epochs: int, batch_size: int,
                 num_workers: int = 0, pin_memory: bool = True, generator=None):
        if epochs <= 0 or not 0 <= sampler.epoch < epochs:
            raise ValueError('invalid training epoch range')
        if sampler.length % (sampler.world_size * batch_size):
            raise ValueError('epoch cannot drop or pad incomplete batches')
        self.dataset, self.sampler = dataset, sampler
        self.epochs, self.batch_size = int(epochs), int(batch_size)
        self.initial_epoch = sampler.epoch
        self.initial_cursor = sampler.samples_consumed_in_epoch
        self.kwargs = dict(num_workers=num_workers, pin_memory=pin_memory, generator=generator)

    def __iter__(self):
        for epoch in range(self.initial_epoch, self.epochs):
            cursor = self.initial_cursor if epoch == self.initial_epoch else 0
            self.sampler.reset_epoch(epoch, cursor)
            self.dataset.set_epoch(epoch)
            yield from DataLoader(self.dataset, sampler=self.sampler,
                                  batch_size=self.batch_size, **self.kwargs)

    def __len__(self):
        remaining = (self.epochs - self.initial_epoch) * self.sampler.length - self.initial_cursor
        return remaining // (self.sampler.world_size * self.batch_size)


class EpochConsumptionTracker:
    """Audit each committed global update against the production permutation."""

    def __init__(self, rows, sampler, batch_per_gpu, accumulation, state=None):
        self.rows, self.sampler = rows, sampler
        self.local_update_samples = batch_per_gpu * accumulation
        self.epoch = sampler.epoch
        self.cursor = sampler.samples_consumed_in_epoch
        self.seen = {rows[index]['sample_uid'] for index in sampler.permutation[:self.cursor]}
        state = state or {}
        self.ce_sum = float(state.get('epoch_ce_sum', 0.0)) if state.get('epoch') == self.epoch else 0.0
        self.ce_count = int(state.get('epoch_ce_count', 0)) if state.get('epoch') == self.epoch else 0
        if self.ce_count != self.cursor:
            raise RuntimeError('resume epoch CE accounting does not match committed cursor')

    def commit(self, uids, sample_losses=None):
        if self.epoch != self.sampler.epoch:
            if self.cursor != len(self.rows) or len(self.seen) != len(self.rows):
                raise RuntimeError('incomplete epoch before reshuffle')
            self.epoch, self.cursor, self.seen = self.sampler.epoch, 0, set()
            self.ce_sum, self.ce_count = 0.0, 0
        expected = [self.rows[self.sampler.permutation[
            self.cursor + rank + slot * self.sampler.world_size]]['sample_uid']
            for rank in range(self.sampler.world_size)
            for slot in range(self.local_update_samples)]
        if list(uids) != expected or self.seen.intersection(uids) or len(set(uids)) != len(uids):
            raise RuntimeError('sampler duplication, omission or sequence mismatch')
        self.seen.update(uids)
        self.cursor += len(uids)
        if sample_losses is not None:
            if len(sample_losses) != len(uids):
                raise RuntimeError('epoch CE sample count mismatch')
            self.ce_sum += sum(sample_losses)
            self.ce_count += len(sample_losses)
        self.sampler.samples_consumed_in_epoch = self.cursor
        return {'epoch': self.epoch, 'within_epoch_cursor': self.cursor,
                'epoch_rows_consumed': self.cursor, 'epoch_unique_rows': len(self.seen),
                'epoch_train_ce_mean': self.ce_sum / self.ce_count if self.ce_count else None,
                'epoch_boundary': self.cursor == len(self.rows)}

    def state_dict(self):
        return {'epoch': self.epoch, 'epoch_ce_sum': self.ce_sum, 'epoch_ce_count': self.ce_count}
