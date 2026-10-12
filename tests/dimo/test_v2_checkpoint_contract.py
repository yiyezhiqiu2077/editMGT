import json

import pytest
import torch

from src.dimo.full_checkpoint import save, load
from src.dimo.recipe import recipe_identity
from src.dimo.transaction import StepCommit
from src.explicit_region.contracts import sha256_file
from tests.dimo.test_v2_joint import bundle, config


def identity(roles):
    c = config()
    return {"recipe": "dimo-v2", "recipe_contract": recipe_identity(c, roles),
            "frozen_parameter_hashes": roles.frozen_parameter_hashes()}


def initial_checkpoint(root):
    roles, optimizers, schedulers, ema = bundle()
    contract = identity(roles)
    save(root, roles=roles, optimizers=optimizers, schedulers=schedulers, ema=ema, step=0, cursor=0,
         identity=contract, config_hash="v2", commit=StepCommit(), runtime_stats={"student_clip_count": 0},
         sampler_state={"epoch": 0, "samples_consumed_in_epoch": 0})
    return contract


@pytest.mark.parametrize("fault", ["optimizer_membership", "cursor", "clip_count", "fractional_clip_count", "recipe_schema", "missing_sampler"])
def test_resume_rejects_incompatible_state_even_with_rehashed_ledger(tmp_path, fault):
    root = tmp_path / "checkpoint"
    contract = initial_checkpoint(root)
    path = root / "training_state.pt"
    state = torch.load(path, weights_only=False)
    if fault == "optimizer_membership":
        state["optimizers"][0]["param_groups"][0]["params"].pop()
    elif fault == "cursor":
        state["cursor"] = 4
    elif fault == "clip_count":
        state["runtime_stats"]["student_clip_count"] = 1
    elif fault == "fractional_clip_count":
        state["runtime_stats"]["student_clip_count"] = 0.0
    elif fault == "missing_sampler":
        state["sampler_state"] = None
    else:
        state["schema"] = "full-dense-dimo-ddp-v2"
    torch.save(state, path)
    marker = json.loads((root / "COMPLETED.json").read_text())
    marker["files"][path.name] = sha256_file(path)
    (root / "COMPLETED.json").write_text(json.dumps(marker))
    roles, optimizers, schedulers, ema = bundle()
    with pytest.raises(RuntimeError, match="MISMATCH|MISSING"):
        load(root, roles=roles, optimizers=optimizers, schedulers=schedulers, ema=ema,
             expected_identity=contract, expected_config_hash="v2")


def test_old_checkpoint_cannot_be_resumed_as_v2(tmp_path):
    root = tmp_path / "old"
    roles, optimizers, schedulers, ema = bundle()
    save(root, roles=roles, optimizers=optimizers, schedulers=schedulers, ema=ema, step=0, cursor=0,
         identity={"recipe": "dimo-v1"}, config_hash="old", commit=StepCommit())
    with pytest.raises(RuntimeError, match="IDENTITY"):
        load(root, roles=roles, optimizers=optimizers, schedulers=schedulers, ema=ema,
             expected_identity=identity(roles), expected_config_hash="v2")
