import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.dimo.dense_roles import DenseDiMOModelRoles
from src.dimo.dense_distributed_checkpoint import save_checkpoint, load_checkpoint
from src.dimo.ema import TrainableEMA
from src.dimo.roles import audit_role_optimizers
from src.dimo.step import complete_dimo_step
from src.dimo.surrogate import linear_surrogate_logit_loss, surrogate_logit_loss, distribution_energy
from src.dimo.surrogate_audit import audit_surrogate_gradients
from src.dimo.transaction import StepCommit
from src.dimo.synthetic import SyntheticDenseTransformer
from src.dimo.divergence import dimo_divergence_gradient
from tests.dimo.toy import batch, step_config


def bundle():
    torch.manual_seed(17); random.seed(17); np.random.seed(17)
    roles = DenseDiMOModelRoles(SyntheticDenseTransformer())
    optimizers = [torch.optim.AdamW([p for _, p in roles.role_named_parameters(r)], lr=1e-6,
        betas=(.9, .999), weight_decay=0) for r in ("student", "auxiliary")]
    schedulers = [torch.optim.lr_scheduler.LambdaLR(o, lambda _: 1) for o in optimizers]
    ema = TrainableEMA(roles.role_named_parameters("student"), decay=.9995)
    return roles, optimizers, schedulers, ema


def run_step(parts, commit, **kwargs):
    roles, optimizers, schedulers, ema = parts
    config = step_config() | {"surrogate": {"implementation": "linear"}}
    return commit.execute(lambda: complete_dimo_step(roles, **batch(), student_optimizer=optimizers[0],
        auxiliary_optimizer=optimizers[1], student_scheduler=schedulers[0], auxiliary_scheduler=schedulers[1],
        student_ema=ema, base_seed=42, epoch=0, committed_dimo_step=commit.step,
        mask_token_id=5, codebook_size=5, config=config, **kwargs), samples=1)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("scale", [0., 1., 10., 1000.])
@pytest.mark.parametrize("magnitude", [1e-2, 1e-8, 1e-12])
def test_linear_preserves_analytical_gradient(dtype, scale, magnitude):
    z = torch.full((2, 3, 7), scale, dtype=dtype, requires_grad=True)
    g = torch.linspace(-magnitude, magnitude, 42, dtype=dtype).reshape_as(z).requires_grad_()
    mask = torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.bool)
    actual = torch.autograd.grad(linear_surrogate_logit_loss(z, g, mask), z)[0]
    expected = g.detach() * mask[..., None] / mask.sum(1)[:, None, None] / 2
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=0)
    assert g.grad is None


def test_squared_cancellation_reproducer_and_real_bf16_path():
    z = torch.full((1, 1, 2), 10., requires_grad=True)
    g = torch.tensor([[[1e-8, -1e-8]]])
    mask = torch.ones((1, 1), dtype=torch.bool)
    assert torch.autograd.grad(surrogate_logit_loss(z, g, mask), z)[0].eq(0).all()
    assert torch.equal(torch.autograd.grad(linear_surrogate_logit_loss(z, g, mask), z)[0], g)
    projection = torch.nn.Linear(5, 7)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = projection(torch.randn(2, 3, 5))
    assert logits.dtype == torch.bfloat16
    field = torch.randn_like(logits.float()) * 1e-8
    report = audit_surrogate_gradients(logits, field, torch.ones(2, 3, dtype=torch.bool))
    assert report["objectives"]["linear"]["matches_expected"]
    loss = linear_surrogate_logit_loss(logits, field, torch.ones(2, 3, dtype=torch.bool))
    loss.backward()
    assert projection.weight.grad.dtype == torch.float32 and projection.weight.grad.ne(0).any()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_divergence_never_sanitizes_nonfinite(bad):
    a = torch.tensor([[bad, 0.]])
    with pytest.raises(FloatingPointError):
        dimo_divergence_gradient(a, torch.zeros_like(a))


def test_full_roles_initial_zero_and_subsequent_signal():
    parts = bundle(); roles, optimizers, schedulers, ema = parts
    assert audit_role_optimizers(roles, *optimizers)["backend"] == "full_dense"
    initial = {r: copy.deepcopy(roles.model_for(r).state_dict()) for r in ("teacher", "student", "auxiliary")}
    assert all(torch.equal(initial["teacher"][n], initial["student"][n]) for n in initial["teacher"])
    commit = StepCommit()
    first, _ = run_step(parts, commit)
    assert first["student_grad_norm"] == 0 and first["aux_grad_norm"] > 0
    assert first["student_parameter_update_ratio"] == 0
    assert first["aux_parameter_update_ratio"] > 0
    second, _ = run_step(parts, commit)
    assert second["student_grad_norm"] > 0
    assert ema.update_count == commit.step == 2
    assert all(p.grad is None for p in roles.model_for("teacher").parameters())
    assert all(torch.equal(v, roles.model_for("teacher").state_dict()[n]) for n, v in initial["teacher"].items())


def test_ema_math_and_count():
    parts = bundle(); roles, _, _, ema = parts
    before = {n: p.clone() for n, p in ema.shadow.items()}
    with torch.no_grad():
        for _, p in roles.role_named_parameters("student"):
            p.add_(1)
    ema.update(roles.role_named_parameters("student"))
    for n, p in roles.role_named_parameters("student"):
        torch.testing.assert_close(ema.shadow[n], before[n]*.9995 + p*.0005)
    assert ema.update_count == 1


@pytest.mark.parametrize("phase", ["student_optimizer", "auxiliary_optimizer", "ema_complete", "ema_during", "before_commit"])
def test_partial_updates_cannot_publish_or_commit(tmp_path, phase):
    parts = bundle(); roles, optimizers, schedulers, ema = parts
    commit = StepCommit()
    run_step(parts, commit)
    root = tmp_path / "checkpoint-1"
    identity = {"test": "CPU_SYNTHETIC", "world_size": 1}
    save_checkpoint(root, roles=roles, optimizers=optimizers, schedulers=schedulers, ema=ema,
        step=commit.step, cursor=commit.cursor, identity=identity, config_hash="test", commit=commit)
    def fail(name):
        if name == phase:
            raise RuntimeError("injected failure")
    if phase == "ema_during":
        def broken_update(named, **kwargs):
            next(iter(ema.shadow.values())).add_(1)
            raise RuntimeError("partial EMA failure")
        ema.update = broken_update
    if phase == "before_commit":
        with pytest.raises(RuntimeError):
            commit.execute(lambda: run_operation(parts, commit), samples=1, failure_injector=fail)
    else:
        with pytest.raises(RuntimeError):
            run_step(parts, commit, failure_injector=fail)
    assert commit.failed and (commit.step, commit.cursor) == (1, 1)
    with pytest.raises(RuntimeError):
        save_checkpoint(tmp_path / "checkpoint-2", roles=roles, optimizers=optimizers, schedulers=schedulers,
            ema=ema, step=2, cursor=2, identity=identity, config_hash="test", commit=commit)
    assert not (tmp_path / "checkpoint-2").exists()
    restored = bundle()
    state = load_checkpoint(root, roles=restored[0], optimizers=restored[1], schedulers=restored[2], ema=restored[3],
        expected_identity=identity, expected_config_hash="test")
    assert state["cursor"] == state["step"] == restored[3].update_count == 1


def run_operation(parts, commit):
    roles, optimizers, schedulers, ema = parts
    return complete_dimo_step(roles, **batch(), student_optimizer=optimizers[0], auxiliary_optimizer=optimizers[1],
        student_scheduler=schedulers[0], auxiliary_scheduler=schedulers[1], student_ema=ema, base_seed=42,
        epoch=0, committed_dimo_step=commit.step, mask_token_id=5, codebook_size=5,
        config=step_config() | {"surrogate": {"implementation": "linear"}})


def test_auxiliary_fixed_batch_fitting():
    torch.manual_seed(42)
    model = SyntheticDenseTransformer()
    inputs = batch()["source_tokens"]
    region = batch()["edit_region_mask"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.05, weight_decay=0)
    losses = []
    for _ in range(80):
        optimizer.zero_grad()
        logits = model(hidden_states=inputs, edit_region_mask=region)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 5), inputs.reshape(-1))
        losses.append(float(loss)); loss.backward(); optimizer.step()
    assert losses[-1] < losses[0] * .1


@pytest.mark.parametrize("role,other", [("student", "auxiliary"), ("auxiliary", "student")])
def test_direct_role_update_storage_and_gradient_isolation(role, other):
    roles, optimizers, schedulers, ema = bundle()
    before = {r: copy.deepcopy(roles.model_for(r).state_dict()) for r in ("teacher", other)}
    payload = batch()
    loss = roles.forward_role(role, hidden_states=payload["source_tokens"], edit_region_mask=payload["edit_region_mask"]).square().mean()
    loss.backward()
    assert all(p.grad is None for p in roles.model_for(other).parameters())
    optimizers[0 if role == "student" else 1].step()
    for r in before:
        assert all(torch.equal(v, roles.model_for(r).state_dict()[n]) for n, v in before[r].items())
    roles.audit_storage()


def test_formal_configuration_rejects_recipe_drift():
    from src.explicit_region.config import _load_unexpanded
    from src.dimo.runtime import validate_full_config
    config = _load_unexpanded(Path("configs/dimo/full_dense_d200k.yaml"))
    validate_full_config(config)
    changed = copy.deepcopy(config)
    changed["component_dtypes"]["ema"] = "bf16"
    with pytest.raises(RuntimeError, match="PRECISION"):
        validate_full_config(changed)
    changed = copy.deepcopy(config)
    changed["sampling"]["temperature"] = .5
    with pytest.raises(RuntimeError, match="SAMPLING"):
        validate_full_config(changed)


def test_inference_only_does_not_create_other_roles(tmp_path, monkeypatch):
    parts = bundle(); roles, optimizers, schedulers, ema = parts
    identity = {"teacher_bundle_fingerprint": {"test": "CPU"}, "backend": "ddp",
                "surrogate": {"implementation": "linear", "version": "roi-vocabulary-sum-v1"}}
    save_checkpoint(tmp_path / "inference", roles=roles, optimizers=optimizers, schedulers=schedulers,
        ema=ema, step=0, cursor=0, identity=identity, config_hash="test", commit=StepCommit(), inference_only=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("inference must not copy Dense roles or load optimizer pickle")
    monkeypatch.setattr(copy, "deepcopy", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    from src.dimo.inference import load_dense_inference_role
    model = SyntheticDenseTransformer()
    inference, _ = load_dense_inference_role(model, tmp_path / "inference", weights="ema", teacher_bundle={"test": "CPU"})
    assert inference.base_model is model and not hasattr(inference, "models")
    assert all(not p.requires_grad for p in inference.parameters())
    with pytest.raises(ValueError):
        inference.model_for("teacher")


def test_resume_preserves_discarded_tail_and_removes_stale_exports(tmp_path):
    from src.dimo.runtime import prepare_resume_output, append_committed_log
    log = tmp_path / "metrics.jsonl"
    for step in (1, 2, 3):
        append_committed_log(log, {"committed": True, "global_step": step})
    export = tmp_path / "inference/checkpoint-3"
    export.mkdir(parents=True)
    (export / "COMPLETED.json").write_text(json.dumps({"committed_step": 3}))
    archive = Path(prepare_resume_output(tmp_path, 1))
    assert len(log.read_text().splitlines()) == 1
    assert len((archive / "metrics.jsonl").read_text().splitlines()) == 3
    assert (archive / "inference-checkpoint-3/COMPLETED.json").exists() and not export.exists()
    latest = tmp_path / "checkpoints/checkpoint-4"
    latest.mkdir(parents=True)
    (latest / "COMPLETED.json").write_text(json.dumps({"committed_step": 4}))
    with pytest.raises(RuntimeError, match="LATEST_COMPLETE"):
        prepare_resume_output(tmp_path, 1)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_real_transformer_full_roles_fp32_cpu_step(checkpointing):
    from src.transformer import Transformer2DModel
    torch.manual_seed(42)
    model = Transformer2DModel(in_channels=6, num_layers=1, num_single_layers=1,
        attention_head_dim=6, num_attention_heads=1, joint_attention_dim=6, pooled_projection_dim=6,
        axes_dims_rope=(2, 2, 2), vocab_size=6, codebook_size=5,
        text_encoder_architecture="CLIP", connector_type="none")
    roles = DenseDiMOModelRoles(model)
    if checkpointing:
        roles.enable_gradient_checkpointing()
    optimizers = [torch.optim.AdamW([p for _, p in roles.role_named_parameters(r)], lr=1e-6,
        betas=(.9, .999), weight_decay=0) for r in ("student", "auxiliary")]
    schedulers = [torch.optim.lr_scheduler.LambdaLR(o, lambda _: 1) for o in optimizers]
    ema = TrainableEMA(roles.role_named_parameters("student"))
    prepared = batch()
    prepared["prompt_condition"] = {"conditional": {
        "encoder_hidden_states": torch.randn(1, 1, 6), "pooled_projections": torch.randn(1, 6)}, "unconditional": {}}
    prepared["model_kwargs"] = {"img_ids": torch.zeros(16, 3), "reference_image_ids": torch.zeros(16, 3),
                                "txt_ids": torch.zeros(1, 3), "micro_conds": torch.zeros(1, 5)}
    commit = StepCommit()
    for _ in range(2):
        diagnostics, _ = commit.execute(lambda: complete_dimo_step(roles, **prepared,
            student_optimizer=optimizers[0], auxiliary_optimizer=optimizers[1],
            student_scheduler=schedulers[0], auxiliary_scheduler=schedulers[1], student_ema=ema,
            base_seed=42, epoch=0, committed_dimo_step=commit.step, mask_token_id=5, codebook_size=5,
            config=step_config() | {"surrogate": {"implementation": "linear"}}), samples=1)
        assert diagnostics["surrogate_gradient_audit"]["objectives"]["linear"]["matches_expected"]
        assert diagnostics["aux_grad_norm"] > 0
    assert ema.update_count == 2
