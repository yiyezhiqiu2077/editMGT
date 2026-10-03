import pytest
import torch

from src.dimo.forward_process import mask_student_prediction, schedule_mask_ratio


def test_pseudo_forward_is_roi_only_and_roi_relative():
    source = torch.arange(32).reshape(2, 4, 4)
    student = source + 100
    region = torch.zeros_like(source, dtype=torch.bool)
    region[0, :2, :2] = True       # four tokens
    region[1, :, :] = True         # sixteen tokens
    state = mask_student_prediction(
        student, source, region, [0.5, 0.5], 999, seeds=[1, 2]
    )
    assert torch.equal(state.pseudo_mask.sum((1, 2)), torch.tensor([2, 8]))
    assert torch.allclose(state.rho_actual, torch.tensor([0.5, 0.5]))
    assert not (state.pseudo_mask & ~region).any()
    assert torch.equal(state.pseudo_tokens[~region], source[~region])
    assert torch.equal(state.pseudo_tokens[region & ~state.pseudo_mask], student[region & ~state.pseudo_mask])


@pytest.mark.parametrize("mode", ["linear", "square", "cosine", "arccos"])
def test_ratio_schedulers_are_deterministic_and_bounded(mode):
    first = schedule_mask_ratio(seed=123, mode=mode)
    assert first == schedule_mask_ratio(seed=123, mode=mode)
    assert 0.02 <= first <= 0.98
