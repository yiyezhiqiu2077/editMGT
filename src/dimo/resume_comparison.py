"""Offline full-state comparison, without loading additional Dense models."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .distributed import state_digest, tensor_digest
from .full_checkpoint import verify


def weight_digests(root, role):
    _, index = verify(root, weights=role)
    result = {}
    for filename in sorted(set(index.values())):
        shard = load_file(Path(root) / filename)
        if set(shard) != {name for name, file in index.items() if file == filename}:
            raise RuntimeError("RESUME_COMPARE_SHARD_COVERAGE_MISMATCH")
        result.update({name: tensor_digest(value) for name, value in shard.items()})
        del shard
    return result


def compare_complete_checkpoints(fresh, resumed):
    identities = [verify(root, full=True)[0] for root in (fresh, resumed)]
    # Output directories have different config hashes, but all scientific and
    # execution settings must match. Compare resolved configs independently.
    configs = []
    for root in (fresh, resumed):
        config = json.loads((Path(root).parents[1] / "resolved_config.json").read_text())
        config.pop("output_dir", None)
        configs.append(config)
    if configs[0] != configs[1]:
        raise RuntimeError("RESUME_COMPARE_CONFIG_MISMATCH")
    for identity in identities:
        identity.pop("config_sha256", None)
    if identities[0] != identities[1]:
        raise RuntimeError("RESUME_COMPARE_IDENTITY_MISMATCH")
    states = [torch.load(str(Path(root) / "training_state.pt"), map_location="cpu",
                         weights_only=False, mmap=True) for root in (fresh, resumed)]
    keys = ("step", "cursor", "epoch", "ema_decay", "ema_update_count", "optimizers",
            "schedulers", "rank_rng", "sampler_state", "loader_rng")
    checked = {key: state_digest(states[0][key]) == state_digest(states[1][key]) for key in keys}
    del states
    for role in ("student", "auxiliary", "ema"):
        checked[role] = weight_digests(fresh, role) == weight_digests(resumed, role)
    if not all(checked.values()):
        raise RuntimeError("RESUME_COMPARE_STATE_MISMATCH: " + str([k for k, v in checked.items() if not v]))
    identity = identities[0]
    if identity["formal"] and identity["world_size"] != 8:
        raise RuntimeError("RESUME_COMPARE_FORMAL_WORLD_SIZE_MISMATCH")
    return {"status": "PASS", "equivalence": "bitwise_exact", "checks": checked,
            "evidence": "REAL_8_GPU" if identity["formal"] else "CPU_SYNTHETIC",
            "world_size": identity["world_size"]}
