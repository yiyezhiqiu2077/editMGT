"""Pairs from one no-replacement epoch sequence, with complete-step accounting."""
import torch.distributed as dist

from .transaction import collective_status


class PairedEpochLoader:
    def __init__(self, loader):
        self.loader = loader
        sampler = loader.sampler
        quantum = 2 * sampler.world_size * loader.batch_size
        if sampler.length % quantum or sampler.samples_consumed_in_epoch % quantum:
            raise RuntimeError("DIMO_FRESH_BATCH_PAIR_BOUNDARY_MISMATCH")

    def __iter__(self):
        iterator = iter(self.loader)
        for _ in range(len(self)):
            error = None
            try:
                a = next(iterator)
                epoch = self.loader.sampler.epoch
                b = next(iterator)
                if self.loader.sampler.epoch != epoch:
                    raise RuntimeError("DIMO_FRESH_BATCH_PAIR_CROSSED_EPOCH")
            except Exception as exc:
                error = f"DIMO_FRESH_BATCH_FETCH_FAILED: {type(exc).__name__}: {exc}"
            collective_status(error)
            yield a, b

    def __len__(self):
        return len(self.loader) // 2


def verify_batch_pair(a, b, *, dataset, sampler, cursor):
    local = {"a": list(a["sample_uids"]), "b": list(b["sample_uids"])}
    rows = [local]
    if dist.is_initialized():
        rows = [None] * dist.get_world_size()
        dist.all_gather_object(rows, local)
    batch = len(local["a"])
    world = sampler.world_size
    offset = cursor - sampler.epoch * len(dataset)
    error = None
    try:
        if len(rows) != world or offset < 0 or offset + 2 * world * batch > len(dataset):
            raise RuntimeError("DIMO_FRESH_BATCH_INVALID_GLOBAL_CURSOR")
        uids = []
        for role_offset, key in ((0, "a"), (world * batch, "b")):
            for rank, row in enumerate(rows):
                expected = [dataset.rows[sampler.permutation[offset + role_offset + rank + slot * world]]["sample_uid"]
                            for slot in range(batch)]
                if row[key] != expected:
                    raise RuntimeError("DIMO_FRESH_BATCH_GLOBAL_SEQUENCE_MISMATCH")
                uids.extend(row[key])
        if len(set(uids)) != len(uids):
            raise RuntimeError("DIMO_FRESH_BATCH_GLOBAL_OVERLAP")
    except Exception as exc:
        error = str(exc)
    collective_status(error)
    return 2 * world * batch
