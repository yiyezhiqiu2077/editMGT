"""Explicit, unvalidated alternate backend. Never selected by an OOM fallback.

    FSDP state uses separate collective full-state contexts and cannot read or
    publish a DDP v2 training checkpoint. Runtime validation is required before
    wiring this candidate into a production training entry.
    """
import torch
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig, FullOptimStateDictConfig


def wrap_trainable_full_shard(model, device):
    return FullyShardedDataParallel(model, device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True,
        mixed_precision=MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32))


def collect_candidate_state(model, optimizer):
    with FullyShardedDataParallel.state_dict_type(model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        return {"schema": "UNVALIDATED-full-dense-dimo-fsdp-candidate-v1",
            "model": model.state_dict(), "optimizer": FullyShardedDataParallel.optim_state_dict(model, optimizer)}
