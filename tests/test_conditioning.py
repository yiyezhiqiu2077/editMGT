import pytest
import torch

from src.explicit_region.conditioning import (
    add_region_condition, duplicate_region_mask_for_cfg, initialize_hardlock_latents,
    region_scheduled_ratio,
)


def test_zero_init_and_inactive_are_legacy_equivalent():
    hidden = torch.randn(2, 4, 3, 3)
    mask = torch.randint(0, 2, (2, 3, 3), dtype=torch.bool)
    zero = torch.zeros(4, requires_grad=True)
    assert torch.equal(add_region_condition(hidden, zero, mask, active=True), hidden)
    learned = torch.ones(4)
    assert torch.equal(add_region_condition(hidden, learned, mask, active=False), hidden)
    assert torch.equal(add_region_condition(hidden, learned, None, active=True), hidden)


def test_outside_is_exactly_zero_and_vector_gets_gradient():
    hidden = torch.zeros(1, 2, 2, 2)
    vector = torch.ones(2, requires_grad=True)
    mask = torch.tensor([[[True, False], [False, False]]])
    output = add_region_condition(hidden, vector, mask, active=True)
    assert torch.equal(output[:, :, 0, 1], hidden[:, :, 0, 1])
    output.sum().backward()
    assert vector.grad.norm().item() > 0


def test_shape_mismatch_fails():
    with pytest.raises(ValueError, match="must have shape"):
        add_region_condition(torch.zeros(1, 2, 4, 4), torch.zeros(2), torch.zeros(1, 3, 3), active=True)


def test_cfg_explicitly_duplicates_region_mask_b_to_2b():
    mask = torch.tensor([[[True, False]], [[False, True]]])
    duplicated = duplicate_region_mask_for_cfg(mask, guidance_enabled=True)
    assert duplicated.shape[0] == 2 * mask.shape[0]
    assert torch.equal(duplicated[:2], mask)
    assert torch.equal(duplicated[2:], mask)
    assert duplicate_region_mask_for_cfg(mask, guidance_enabled=False) is mask


def test_hardlock_target_does_not_alias_clean_reference():
    reference = torch.arange(16).reshape(1, 4, 4)
    before = reference.clone()
    mask = torch.zeros_like(reference, dtype=torch.bool)
    mask[:, 1:3, 1:3] = True
    target = initialize_hardlock_latents(reference, mask, 999)
    assert torch.equal(reference, before)
    assert torch.all(target[mask] == 999)
    assert torch.equal(target[~mask], reference[~mask])


def test_roi_inference_timestep_matches_current_remaining_schedule():
    ratios = [region_scheduled_ratio(i, 4) for i in range(4)]
    assert ratios[0] == 1.0
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[1] == pytest.approx(torch.cos(torch.tensor(torch.pi / 8)).item())
