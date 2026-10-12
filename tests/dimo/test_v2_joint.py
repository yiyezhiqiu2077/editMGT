import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.dimo.auxiliary import auxiliary_loss
from src.dimo.conditioning import predict_role_logits
from src.dimo.dense_roles import DenseDiMOModelRoles
from src.dimo.ema import TrainableEMA, apply_ema_to_student_role
from src.dimo.gradient_stats import clip_student_gradients, snapshot_full_trainable, full_update_statistics
from src.dimo.recipe import consumption_budget, recipe_identity
from src.dimo.roles import audit_role_optimizers
from src.dimo.runtime import validate_full_config
from src.dimo.step import complete_dimo_step
from src.dimo.synthetic import SyntheticDenseTransformerV2, SyntheticDataset, synthetic_v2_batch
from src.explicit_region.config import _load_unexpanded


def config():
    return _load_unexpanded(Path("configs/dimo/dimo_v2_cpu_smoke.yaml"))


def bundle():
    torch.manual_seed(42)
    roles = DenseDiMOModelRoles(SyntheticDenseTransformerV2(), freeze_embed=True, freeze_mlm_head=True)
    optimizers = [torch.optim.AdamW([p for _, p in roles.role_named_parameters(r) if p.requires_grad],
                    lr=1e-6, betas=(.9, .999), weight_decay=0., foreach=False) for r in ("student", "auxiliary")]
    schedulers = [torch.optim.lr_scheduler.LambdaLR(o, lambda _: 1.) for o in optimizers]
    ema = TrainableEMA(roles.role_named_parameters("student"), decay=.9995)
    return roles, optimizers, schedulers, ema


def test_exact_freeze_and_optimizer_membership_and_head_backprop():
    roles, optimizers, _, _ = bundle()
    report = audit_role_optimizers(roles, *optimizers)
    assert 0 < report["student_trainable_ratio"] < 1
    hashes = roles.frozen_parameter_hashes()
    model = roles.model_for("student")
    for module in (model.embed, model.mlm_layer):
        assert all(not p.requires_grad for p in module.parameters())
    assert model.edit_region_embedding.requires_grad
    batch = synthetic_v2_batch([0], SyntheticDataset())
    model(hidden_states=batch["source_tokens"], edit_region_mask=batch["edit_region_mask"]).square().sum().backward()
    assert model.transformer_blocks[0].attn.weight.grad.norm() > 0
    assert all(p.grad is None for m in (model.embed, model.mlm_layer) for p in m.parameters())
    optimizers[0].step(); optimizers[0].zero_grad(set_to_none=True)
    roles.assert_frozen_unchanged(hashes)
    optimizers[0].add_param_group({"params": list(model.embed.parameters())})
    with pytest.raises(RuntimeError, match="membership"):
        audit_role_optimizers(roles, *optimizers)


def test_partial_ema_validates_exact_trainable_names_shapes_and_dtypes():
    roles, _, _, ema = bundle()
    frozen = roles.frozen_parameter_hashes()
    state = ema.state_dict()
    for value in state["shadow"].values():
        value.add_(.1)
    apply_ema_to_student_role(roles, state)
    roles.assert_frozen_unchanged(frozen)
    broken = copy.deepcopy(state); broken["shadow"].pop(next(iter(broken["shadow"])))
    with pytest.raises(RuntimeError, match="identity"):
        apply_ema_to_student_role(roles, broken)
    broken = copy.deepcopy(state); name = next(iter(broken["shadow"])); broken["shadow"][name] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="shape"):
        apply_ema_to_student_role(roles, broken)
    broken = copy.deepcopy(state); broken["shadow"][name] = broken["shadow"][name].double()
    with pytest.raises(RuntimeError, match="dtype"):
        apply_ema_to_student_role(roles, broken)


def test_global_clipping_scales_direction_and_leaves_other_role_untouched():
    a, b, auxiliary = (torch.nn.Parameter(torch.ones(2)) for _ in range(3))
    a.grad = torch.tensor([30., 40.]); b.grad = torch.tensor([0., 120.]); auxiliary.grad = torch.ones(2)
    old = torch.cat([a.grad.clone(), b.grad.clone()])
    frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    stats = clip_student_gradients([("transformer_blocks.0.attn.weight", a), ("transformer_blocks.0.ff.weight", b),
                                    ("embed.weight", frozen)], 10.)
    assert stats["student_grad_norm_preclip"] == pytest.approx(130.)
    assert stats["student_grad_norm_postclip"] == pytest.approx(10., abs=1e-5)
    torch.testing.assert_close(torch.cat([a.grad, b.grad]), old * (10. / 130.))
    assert stats["student_clip_triggered"] and stats["student_grad_norm_embed"] == 0
    assert auxiliary.grad.eq(1).all()
    a.grad.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="NONFINITE"):
        clip_student_gradients([("a", a)], 10.)


def test_full_update_norm_uses_all_trainable_coordinates():
    p = torch.nn.Parameter(torch.ones(100))
    before = snapshot_full_trainable([("p", p)])
    with torch.no_grad():
        p[99] += 2.
    actual = full_update_statistics(before, [("p", p)])
    assert actual["student_parameter_update_norm_full"] == pytest.approx(2.)
    assert actual["student_parameter_update_ratio_full"] == pytest.approx(.2)


@pytest.mark.parametrize("smoothing", [0., .1])
@pytest.mark.parametrize("vocab", [5, 8192])
def test_masked_smoothed_ce_value_and_gradient_and_legacy_equivalence(smoothing, vocab):
    torch.manual_seed(5)
    z = torch.randn(2, 3, 1, vocab, dtype=torch.float64, requires_grad=True)
    tokens = torch.randint(vocab, (2, 3, 1))
    mask = torch.tensor([[[1], [0], [0]], [[1], [1], [0]]], dtype=torch.bool)
    loss = auxiliary_loss(z, tokens, mask, label_smoothing=smoothing)
    log_p = z.log_softmax(-1)
    target = F.one_hot(tokens, vocab) * (1 - smoothing) + smoothing / vocab
    raw = -(target * log_p).sum(-1)
    expected = ((raw * mask).flatten(1).sum(1) / mask.flatten(1).sum(1)).mean()
    torch.testing.assert_close(loss, expected)
    gradient = torch.autograd.grad(loss, z)[0]
    expected_grad = (log_p.exp() - target) * mask[..., None] / mask.flatten(1).sum(1)[:, None, None, None] / 2
    torch.testing.assert_close(gradient, expected_grad)
    if smoothing == 0:
        assert torch.equal(loss, auxiliary_loss(z, tokens, mask))


def test_fresh_generation_is_after_update_no_grad_with_B_conditions_and_noise():
    roles, optimizers, schedulers, ema = bundle()
    dataset = SyntheticDataset()
    a, b = synthetic_v2_batch([0], dataset), synthetic_v2_batch([7], dataset)
    calls = []
    original = roles.forward_role
    def spy(role, **kwargs):
        calls.append((role, torch.is_grad_enabled(), kwargs,
                      roles.model_for("student").transformer_blocks[0].attn.weight.detach().clone()))
        return original(role, **kwargs)
    roles.forward_role = spy
    frozen = roles.frozen_parameter_hashes()
    teacher = copy.deepcopy(roles.model_for("teacher").state_dict())
    diagnostics, artifacts = complete_dimo_step(roles, **a, fresh_batch=b, student_optimizer=optimizers[0],
        auxiliary_optimizer=optimizers[1], student_scheduler=schedulers[0], auxiliary_scheduler=schedulers[1],
        student_ema=ema, base_seed=42, epoch=0, committed_dimo_step=0, mask_token_id=5, codebook_size=5, config=config())
    student_calls = [c for c in calls if c[0] == "student"]
    assert len(student_calls) == 2 and student_calls[0][1] and not student_calls[1][1]
    assert not torch.equal(student_calls[0][3], student_calls[1][3])
    second = student_calls[1][2]
    assert torch.equal(second["reference_image_hidden_states"], b["source_tokens"])
    assert torch.equal(second["encoder_hidden_states"], b["prompt_condition"]["conditional"]["encoder_hidden_states"])
    assert torch.equal(second["edit_region_mask"], b["edit_region_mask"])
    assert second["hidden_states"][b["edit_region_mask"]].eq(5).all()
    assert second["target_embedding_noise_sigma"] == .3
    assert not torch.equal(artifacts["embedding_noise"], artifacts["fresh_embedding_noise"])
    assert torch.equal(artifacts["fresh_student_tokens"][~b["edit_region_mask"]], b["source_tokens"][~b["edit_region_mask"]])
    roles.assert_frozen_unchanged(frozen)
    assert all(torch.equal(v, roles.model_for("teacher").state_dict()[n]) for n, v in teacher.items())
    assert diagnostics["student_parameter_update_ratio_full"] > 0
    assert diagnostics["student_grad_norm_embed"] == diagnostics["student_grad_norm_mlm_head"] == 0
    assert diagnostics["teacher_entropy"] > 0 and ema.update_count == 1
    teacher_call = [c for c in calls if c[0] == "teacher"][0][2]
    for key in ("hidden_states", "reference_image_hidden_states", "edit_region_mask", "timestep"):
        x, y = teacher_call[key].chunk(2)
        assert torch.equal(x, y)


def test_cfg_combination_promotes_before_bf16_arithmetic():
    class Roles:
        def forward_role(self, role, hidden_states, encoder_hidden_states, **kwargs):
            return encoder_hidden_states[:, None, None, :].expand(-1, 1, 1, -1).to(torch.bfloat16)
    u = torch.tensor([[10., -20.]], dtype=torch.bfloat16)
    c = torch.tensor([[10.0625, -19.875]], dtype=torch.bfloat16)
    tokens = torch.zeros(1, 1, 1, dtype=torch.long)
    actual = predict_role_logits(Roles(), "teacher", tokens, tokens,
        {"conditional": {"encoder_hidden_states": c}, "unconditional": {"encoder_hidden_states": u}},
        torch.ones_like(tokens), torch.ones(1), 4.)
    assert actual.dtype == torch.float32
    assert torch.equal(actual[0, 0, 0], (u.float() + 4 * (c.float() - u.float()))[0])


@pytest.mark.parametrize("section,key,value", [("model_roles", "freeze_embed", False),
    ("optimizer", "student_max_grad_norm", 1.), ("auxiliary", "label_smoothing", 0.),
    ("auxiliary", "batch_policy", "same_batch"), ("distillation", "teacher_cfg", 1.)])
def test_joint_recipe_cannot_silently_disable_one_mechanism(section, key, value):
    c = config(); validate_full_config(c, synthetic=True)
    c[section][key] = value
    with pytest.raises(RuntimeError, match="JOINT_RECIPE"):
        validate_full_config(c, synthetic=True)


def test_mixed20k_budget_and_identity():
    c = _load_unexpanded(Path("configs/dimo/dimo_v2_mixed20k.yaml"))
    validate_full_config(c)
    budget = consumption_budget(c)
    assert budget["optimizer_steps"] == 12500 and budget["total_sample_consumptions"] == 100000
    assert budget["student_update_sample_consumptions"] == budget["auxiliary_update_sample_consumptions"] == 50000
    roles, _, _, _ = bundle()
    assert recipe_identity(c, roles)["budget"] == budget
    c["max_optimizer_steps"] = 25000
    with pytest.raises(RuntimeError, match="BUDGET"):
        validate_full_config(c)
