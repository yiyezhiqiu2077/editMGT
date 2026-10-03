import torch

from src.dimo.initialization import build_editing_initial_state, sample_student_tokens


def inputs():
    source = torch.arange(100).reshape(1, 10, 10) % 17
    region = torch.zeros_like(source, dtype=torch.bool)
    region[:, 1:9, 1:9] = True
    return source, region


def test_initialization_partition_exact_count_and_outside_lock():
    source, region = inputs()
    state = build_editing_initial_state(
        source, region, 99, 17, 0.5, mask_seeds=[11], token_seeds=[12]
    )
    assert torch.equal(state.initial_tokens[~region], source[~region])
    assert not (state.initial_mask & ~region).any()
    assert not (state.random_token_mask & ~region).any()
    assert not (state.initial_mask & state.random_token_mask).any()
    assert torch.equal(state.initial_mask | state.random_token_mask, region)
    assert state.initial_mask.sum() == 32
    assert torch.all(state.initial_tokens[state.initial_mask] == 99)
    assert torch.all((state.initial_tokens[state.random_token_mask] >= 0) & (state.initial_tokens[state.random_token_mask] < 17))


def test_r_init_one_masks_all_roi():
    source, region = inputs()
    state = build_editing_initial_state(
        source, region, 99, 17, 1.0, mask_seeds=2, token_seeds=3
    )
    assert torch.equal(state.initial_mask, region)
    assert not state.random_token_mask.any()
    assert state.initial_roi_mask_ratio.item() == 1.0


def test_initialization_is_seeded_per_stream():
    source, region = inputs()
    first = build_editing_initial_state(source, region, 99, 17, 0.5, mask_seeds=7, token_seeds=8)
    again = build_editing_initial_state(source, region, 99, 17, 0.5, mask_seeds=7, token_seeds=8)
    other = build_editing_initial_state(source, region, 99, 17, 0.5, mask_seeds=9, token_seeds=10)
    assert torch.equal(first.initial_tokens, again.initial_tokens)
    assert torch.equal(first.initial_mask, again.initial_mask)
    assert not torch.equal(first.initial_mask, other.initial_mask)


def test_student_sampling_detaches_and_only_writes_roi():
    source, region = inputs()
    logits = torch.randn(1, 10, 10, 17, requires_grad=True)
    sampled = sample_student_tokens(logits, source, region, seeds=[44])
    assert not sampled.requires_grad
    assert torch.equal(sampled[~region], source[~region])
    assert logits.grad is None
