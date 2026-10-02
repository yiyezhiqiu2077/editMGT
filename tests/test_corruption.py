import torch

from src.explicit_region.corruption import prepare_corruption
from src.explicit_region.losses import per_sample_cross_entropy


def grids():
    source = torch.arange(32).reshape(2, 4, 4)
    target = source + 100
    region = torch.zeros_like(source, dtype=torch.bool)
    region[:, 1:3, 1:3] = True
    return source, target, region


def test_roi_hardlock_and_timestep_contract():
    source, target, region = grids()
    batch = prepare_corruption(
        source, target, region, 999, mode="roi_hardlock", base_seed=4,
        global_sample_indices=[10, 11], sample_keys=["a", "b"], full_roi_mask_probability=1.0,
    )
    assert torch.equal(batch.selected_mask, region)
    assert torch.equal(batch.input_tokens[~region], source[~region])
    assert torch.all(batch.labels[~region] == -100)
    assert torch.all(batch.scheduled_roi_ratio == 1)
    assert torch.all(batch.actual_roi_mask_fraction == 1)
    assert torch.all(batch.full_roi_branch)


def test_full_target_has_zero_condition_and_is_deterministic():
    source, target, region = grids()
    kwargs = dict(
        mode="full_target", base_seed=9, global_sample_indices=[0, 1], sample_keys=["a", "b"]
    )
    first = prepare_corruption(source, target, region, 999, **kwargs)
    second = prepare_corruption(source, target, region, 999, **kwargs)
    assert torch.equal(first.selected_mask, second.selected_mask)
    assert not first.edit_region_mask.any()
    assert first.conditioning_active is False
    unmasked = ~first.selected_mask
    assert torch.equal(first.input_tokens[unmasked], target[unmasked])


def test_mixed_selection_is_deterministic():
    source, target, region = grids()
    kwargs = dict(
        mode="mixed", roi_probability=0.5, base_seed=17,
        global_sample_indices=[20, 21], sample_keys=["a", "b"]
    )
    first = prepare_corruption(source, target, region, 999, **kwargs)
    second = prepare_corruption(source, target, region, 999, **kwargs)
    assert torch.equal(first.input_tokens, second.input_tokens)
    assert torch.equal(first.labels, second.labels)


def test_per_sample_ce_does_not_weight_large_masks_more():
    labels = torch.tensor([[[0, -100]], [[0, 0]]])
    logits = torch.zeros(2, 2, 1, 2)
    logits[0, 0, 0, 0] = 5
    logits[1, 1] = 5
    sample_mean, token_mean = per_sample_cross_entropy(logits, labels)
    assert not torch.isclose(sample_mean, token_mean)
