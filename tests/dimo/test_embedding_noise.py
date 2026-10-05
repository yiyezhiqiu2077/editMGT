import torch

from src.dimo.forward_process import apply_target_embedding_perturbation
from src.dimo.rng import normal_noise_per_sample


def test_sigma_zero_is_exact_noop():
    hidden = torch.randn(2, 3, 4, 4)
    result = apply_target_embedding_perturbation(hidden, torch.randn_like(hidden), None, 0)
    assert result is hidden


def test_noise_changes_target_roi_only_and_never_reference():
    target = torch.randn(1, 3, 3, 3)
    reference = torch.randn_like(target)
    reference_before = reference.clone()
    region = torch.zeros(1, 3, 3, dtype=torch.bool)
    region[:, 1:, 1:] = True
    noise = torch.ones_like(target) * 7
    result = apply_target_embedding_perturbation(target, noise, region, 0.3)
    expanded = region[:, None].expand_as(target)
    assert torch.equal(result[~expanded], target[~expanded])
    assert not torch.equal(result[expanded], target[expanded])
    assert torch.equal(reference, reference_before)


def test_per_sample_noise_is_deterministic_without_global_rng():
    reference = torch.empty(2, 4, 3, 3)
    before = torch.get_rng_state().clone()
    first = normal_noise_per_sample(reference, [4, 9])
    again = normal_noise_per_sample(reference, [4, 9])
    assert torch.equal(first, again)
    assert torch.equal(before, torch.get_rng_state())


def test_transformer_default_optional_arguments_are_numerically_identical():
    from src.transformer import Transformer2DModel

    model = Transformer2DModel(
        in_channels=6, num_layers=0, num_single_layers=0,
        attention_head_dim=6, num_attention_heads=1,
        joint_attention_dim=6, pooled_projection_dim=6,
        axes_dims_rope=(2, 2, 2), vocab_size=6, codebook_size=5,
        text_encoder_architecture="CLIP", connector_type="none",
    ).eval()
    tokens = torch.tensor([[[0, 1], [2, 3]]])
    text = torch.randn(1, 1, 6)
    pooled = torch.randn(1, 6)
    common = dict(
        hidden_states=tokens, encoder_hidden_states=text, pooled_projections=pooled,
        timestep=torch.tensor([0.5]), img_ids=torch.zeros(4, 3), txt_ids=torch.zeros(1, 3),
        micro_conds=torch.zeros(1, 5), edit_region_mask=torch.ones(1, 2, 2, dtype=torch.bool),
    )
    with torch.no_grad():
        legacy = model(**common)
        explicit_defaults = model(
            **common, edit_region_embedding_override=None,
            target_embedding_noise=None, target_embedding_noise_sigma=0,
        )
    assert torch.equal(legacy, explicit_defaults)
