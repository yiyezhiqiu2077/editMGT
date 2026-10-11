"""Formal all-MASK training and evaluation must share an explicit recipe."""
from pathlib import Path

import pytest
import torch

from src.explicit_region.config import _load_unexpanded
from src.dimo.evaluation import evaluation_r_init
from src.dimo.runtime import validate_full_config
from src.dimo.one_step import one_step_edit_tokens
from tests.dimo.test_one_step import CountingRoles


def test_formal_config_and_registered_manual_step500():
    training = _load_unexpanded(Path("configs/dimo/full_dense_d200k.yaml"))
    plan = _load_unexpanded(Path("configs/eval/full_dense_dimo_plan.yaml"))
    inference = _load_unexpanded(Path(plan["inference_config"]))
    validate_full_config(training)
    assert training["initialization"] == {"mask_ratio": 1., "random_tokens_inside_roi": False}
    assert evaluation_r_init(inference, training["initialization"]) == 1.
    # Disabling automatic DEV does not discard the recovery or manual-eval point.
    assert 500 in training["resume_checkpoint_steps"]
    assert 500 in training["evaluation_steps"] and 500 in plan["steps"]
    assert training["monitor"]["interval"] == 500


@pytest.mark.parametrize("initialization", [
    {"mask_ratio": .5, "random_tokens_inside_roi": True},
    {"mask_ratio": 1., "random_tokens_inside_roi": True},
])
def test_formal_gate_rejects_old_or_inconsistent_initialization(initialization):
    config = _load_unexpanded(Path("configs/dimo/full_dense_d200k.yaml"))
    config["initialization"] = initialization
    with pytest.raises(RuntimeError, match="INITIALIZATION_RECIPE_CHANGED"):
        validate_full_config(config)


def test_one_step_all_mask_forward_and_outside_source_lock():
    source = torch.arange(16).reshape(1, 4, 4) % 7
    region = torch.zeros_like(source, dtype=torch.bool)
    region[:, 1:3, 1:3] = True
    roles = CountingRoles(7)
    result = one_step_edit_tokens(roles, source_tokens=source, edit_region_mask=region,
        prompt_condition={"conditional": {}, "unconditional": {}}, timestep_model_kwargs={},
        mask_token_id=8, codebook_size=7, r_init=evaluation_r_init({"r_init": 1.}),
        init_mask_seeds=1, init_token_seeds=2, sample_seeds=3)
    assert result.initial_tokens[region].eq(8).all()
    assert torch.equal(result.initial_tokens[~region], source[~region])
    assert torch.equal(result.tokens[~region], source[~region])
    assert roles.calls == result.metadata["number_of_transformer_forwards"] == 1
    assert result.metadata["r_init"] == 1.


@pytest.mark.parametrize("training,inference", [(1., .5), (.5, 1.)])
def test_evaluation_rejects_checkpoint_initialization_mismatch(training, inference):
    with pytest.raises(RuntimeError, match="INITIALIZATION_MISMATCH"):
        evaluation_r_init({"r_init": inference}, {"mask_ratio": training})


def test_historical_evaluation_can_explicitly_use_matching_half_mask():
    assert evaluation_r_init({"r_init": .5}, {"mask_ratio": .5}) == .5
    with pytest.raises(RuntimeError, match="R_INIT_REQUIRED"):
        evaluation_r_init({})


@pytest.mark.parametrize("ratio", [float("nan"), -1., 1.1])
def test_evaluation_rejects_invalid_ratio(ratio):
    with pytest.raises(ValueError, match="OUT_OF_RANGE"):
        evaluation_r_init({"r_init": ratio})
