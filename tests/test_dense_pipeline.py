"""CPU evidence is explicitly synthetic; never an 8-GPU/model quality PASS."""
import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from diffusers.loaders.peft import PeftAdapterMixin

from src.explicit_region.dense_checkpoint import (save_dense_checkpoint, load_dense_weights,
    verify_dense_checkpoint, read_dense_resume, restore_dense_resume)
from src.explicit_region.dense_runtime import DenseQualityGate, validate_dense_config, trainable_report, UpdateSample
from src.explicit_region.config import _load_unexpanded
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.dense_evaluation import paired_comparison, selection_registration, select_checkpoint, CANDIDATES
from src.dimo.dense_teacher import export_dense_teacher, load_dense_teacher_contract
from src.dimo.dense_roles import initialize_dense_model_roles, initial_logits_parity
from src.dimo.roles import audit_role_optimizers
from src.dimo.step import complete_dimo_step
from src.dimo.ema import TrainableEMA, apply_ema_to_student_role
from src.dimo.one_step import one_step_edit_tokens
from tests.dimo.toy import step_config, batch


class ToyDense(nn.Module, PeftAdapterMixin):
    def __init__(self, checkpointing=True):
        super().__init__()
        self.config = {"vocab_size": 6, "codebook_size": 5, "class": "CPU_SYNTHETIC"}
        self.inner_dim = 5
        self.projection = nn.Linear(6, 5)
        self.edit_region_embedding = nn.Parameter(torch.linspace(0.01, .2, 5))
        self.checkpointing = checkpointing

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, hidden_states, edit_region_mask, edit_region_embedding_override=None, **kwargs):
        x = torch.nn.functional.one_hot(hidden_states, 6).float()
        region = self.edit_region_embedding if edit_region_embedding_override is None else edit_region_embedding_override
        def block(value, embedding):
            return self.projection(value) + edit_region_mask[..., None] * embedding
        return checkpoint(block, x, region, use_reentrant=False) if self.training and self.checkpointing else block(x, region)


def payload():
    return {"training_mode": "full_transformer", "git_sha": "a" * 40,
        "model_identity": {"repo_id": "CPU_SYNTHETIC", "resolved_revision": "b" * 40,
                           "components": {"transformer": {"config_sha256": "c" * 64}}},
        "data_content_hashes": {"CPU_SYNTHETIC": "d" * 64},
        "corruption": {"persistent_conditioning": True}}


def make_checkpoint(tmp_path, *, step=8, training=True):
    model = ToyDense()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: min(n / 200, 1))
    root = tmp_path / f"checkpoint-{step}"
    save_dense_checkpoint(root, model=model, fingerprint_payload=payload(),
        optimizer=optimizer if training else None, scheduler=scheduler,
        global_optimizer_step=step, committed_global_sample_count=step * 32,
        sampler_state={"epoch": 0, "samples_consumed_in_epoch": step * 32},
        quality_state={}, max_shard_bytes=64)
    return model, root


def test_dense_sharded_roundtrip_no_lora(tmp_path):
    model, root = make_checkpoint(tmp_path)
    assert len(list(root.glob("model-*.safetensors"))) >= 2
    target = ToyDense()
    load_dense_weights(target, root)
    assert all(torch.equal(v, target.state_dict()[k]) for k, v in model.state_dict().items())
    assert not (root / "adapter_model.safetensors").exists()


def test_inference_never_reads_optimizer(tmp_path, monkeypatch):
    _, root = make_checkpoint(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("inference called torch.load")
    monkeypatch.setattr(torch, "load", forbidden)
    load_dense_weights(ToyDense(), root)


@pytest.mark.parametrize("fault", ["missing_marker", "corrupted_shard", "wrong_architecture", "wrong_dtype", "wrong_fingerprint"])
def test_dense_rejects_incompatible_checkpoint(tmp_path, fault):
    _, root = make_checkpoint(tmp_path)
    target = ToyDense()
    if fault == "missing_marker":
        (root / "COMPLETED.json").unlink()
    elif fault == "corrupted_shard":
        with next(root.glob("model-*.safetensors")).open("ab") as handle:
            handle.write(b"corruption")
    elif fault == "wrong_architecture":
        target.config["vocab_size"] = 7
    elif fault == "wrong_dtype":
        target.bfloat16()
    with pytest.raises(RuntimeError):
        load_dense_weights(target, root, expected_fingerprint={"wrong": True} if fault == "wrong_fingerprint" else None)


def test_checkpoint_does_not_overwrite(tmp_path):
    model, root = make_checkpoint(tmp_path)
    with pytest.raises(RuntimeError, match="FileExistsError"):
        save_dense_checkpoint(root, model=model, fingerprint_payload=payload())


def test_dense_parameter_report_and_fp32_backward():
    model = ToyDense()
    frozen = [nn.Linear(2, 2).requires_grad_(False) for _ in range(3)]
    components = SimpleNamespace(transformer=model, text_encoder=frozen[0], llm_encoder=frozen[1], vqvae=frozen[2])
    optimizer = torch.optim.AdamW(model.parameters())
    report = trainable_report(components, optimizer)
    b = batch()
    logits = model(hidden_states=b["source_tokens"], edit_region_mask=b["edit_region_mask"])
    logits.square().mean().backward()
    assert model.edit_region_embedding.grad is not None
    assert all(p.grad is not None and p.grad.dtype == torch.float32 for p in model.parameters())
    assert all(p.grad is None for m in frozen for p in m.parameters())
    assert report["trainable_numel"] == report["optimizer_numel"]
    optimizer.param_groups[0]["params"] = list(model.parameters())[:-1]
    with pytest.raises(RuntimeError):
        trainable_report(components, optimizer)


def _training_run(root, stop, resume=None):
    torch.manual_seed(17); random.seed(17); np.random.seed(17)
    model = ToyDense()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, betas=(.9, .95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(s / 200, 1))
    generator = torch.Generator().manual_seed(91)
    gate = DenseQualityGate({"baseline_start": 1, "baseline_end": 8, "window_size": 4,
                              "max_loss_ratio": 100, "consecutive_bad_windows": 2})
    start = 0
    if resume:
        state = read_dense_resume(resume, fingerprint_payload=payload(), world_size=1)
        restore_dense_resume(model, optimizer, scheduler, state, resume, loader_generator=generator)
        gate = DenseQualityGate(gate.config, state["quality_state"])
        start = state["global_optimizer_step"]
    b = batch()
    for step in range(start, stop):
        logits = model(hidden_states=b["source_tokens"], edit_region_mask=b["edit_region_mask"])
        loss = logits.square().mean() + torch.rand(()) * 0.01
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        gate.update(step + 1, loss=float(loss), max_abs_logit=1, clip_applied=False)
        torch.rand((), generator=generator); random.random(); np.random.rand()
    save_dense_checkpoint(root, model=model, optimizer=optimizer, scheduler=scheduler,
        fingerprint_payload=payload(), global_optimizer_step=stop, committed_global_sample_count=stop,
        sampler_state={"epoch": 0, "samples_consumed_in_epoch": stop},
        quality_state=gate.state_dict(), loader_generator=generator, max_shard_bytes=64)
    return read_dense_resume(root, fingerprint_payload=payload(), world_size=1)


def assert_recursive_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_recursive_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_recursive_equal(x, y)
    else:
        assert a == b


def test_dense_cpu_fresh16_vs8_resume16_bitwise(tmp_path):
    fresh = _training_run(tmp_path / "fresh16", 16)
    _training_run(tmp_path / "fresh8", 8)
    resumed = _training_run(tmp_path / "resumed16", 16, tmp_path / "fresh8")
    assert_recursive_equal(fresh, resumed)
    a, b = ToyDense(), ToyDense()
    load_dense_weights(a, tmp_path / "fresh16"); load_dense_weights(b, tmp_path / "resumed16")
    assert_recursive_equal(a.state_dict(), b.state_dict())


def test_dense_unsampled_quality_is_explicit():
    gate = DenseQualityGate({"baseline_start": 1, "baseline_end": 2, "window_size": 1,
                             "max_loss_ratio": 2, "consecutive_bad_windows": 2})
    assert gate.update(1, loss=1, max_abs_logit=1, clip_applied=False)
    assert gate.state_dict()["last_diagnostic_step"] is None
    assert gate.state_dict()["full_model_update_ratio"] == "NOT_COLLECTED"
    assert not gate.update(2, loss=float("nan"), max_abs_logit=1, clip_applied=False)


def test_dense_config_exact_recipe():
    from pathlib import Path
    config = _load_unexpanded(Path("configs/train/dense_d200k_5epoch.yaml"))
    validate_dense_config(config, formal=True)
    for field, bad in (("precision", "fp32"), ("warmup_steps", 100), ("training_mode", "lora")):
        changed = dict(config, **{field: bad})
        with pytest.raises(RuntimeError):
            validate_dense_config(changed, formal=True)


@pytest.mark.parametrize("status", ["provisional", "selected"])
def test_teacher_export_gates(tmp_path, status):
    _, root = make_checkpoint(tmp_path, step=3125)
    if status == "selected":
        with pytest.raises(RuntimeError, match="SELECTED_RECORD_REQUIRED"):
            export_dense_teacher(root, tmp_path / "teacher", status=status)
        return
    export_dense_teacher(root, tmp_path / "teacher", status=status)
    contract = load_dense_teacher_contract(tmp_path / "teacher")
    assert not contract.formal_teacher
    from src.dimo.contracts import teacher_bundle_fingerprint, enforce_run_guard
    with pytest.raises(RuntimeError, match="NOT_SELECTED"):
        teacher_bundle_fingerprint(tmp_path / "teacher", base_model_identity=contract.manifest["base_model_identity"], formal=True)
    with pytest.raises(RuntimeError, match="NOT_SELECTED"):
        enforce_run_guard(contract, prep_smoke=False, max_optimizer_steps=20, formal_ready=False)
    assert not (tmp_path / "teacher" / "training_state.pt").exists()
    assert not (tmp_path / "teacher" / "adapter_model.safetensors").exists()


def test_dense_dimo_cpu_two_steps_and_one_step(tmp_path):
    _, root = make_checkpoint(tmp_path)
    export_dense_teacher(root, tmp_path / "teacher", status="provisional")
    roles = initialize_dense_model_roles(ToyDense(), tmp_path / "teacher",
        lora={"rank": 2, "alpha": 2, "target_modules": ["projection"]})
    b = batch()
    parity = initial_logits_parity(roles, {"hidden_states": b["source_tokens"], "edit_region_mask": b["edit_region_mask"]})
    assert parity["max_abs_error"] == {"student": 0, "auxiliary": 0}
    student = torch.optim.AdamW([p for _, p in roles.role_named_parameters("student")], lr=.01)
    auxiliary = torch.optim.AdamW([p for _, p in roles.role_named_parameters("auxiliary")], lr=.01)
    audit_role_optimizers(roles, student, auxiliary)
    ss = torch.optim.lr_scheduler.LambdaLR(student, lambda _: 1)
    sa = torch.optim.lr_scheduler.LambdaLR(auxiliary, lambda _: 1)
    ema = TrainableEMA(roles.role_named_parameters("student"), decay=.9)
    before = {n: p.detach().clone() for n, p in roles.base_model.named_parameters() if "lora_" not in n}
    for step in range(2):
        diagnostics, artifacts = complete_dimo_step(roles, **b, student_optimizer=student,
            auxiliary_optimizer=auxiliary, student_scheduler=ss, auxiliary_scheduler=sa, student_ema=ema,
            base_seed=42, epoch=0, committed_dimo_step=step, mask_token_id=5, codebook_size=5, config=step_config())
        assert diagnostics["nan_inf_count"] == 0
        assert torch.equal(artifacts["student_tokens"][~b["edit_region_mask"]], b["source_tokens"][~b["edit_region_mask"]])
    assert diagnostics["student_grad_norm"] > 0
    for n, p in roles.base_model.named_parameters():
        if n in before:
            assert torch.equal(p, before[n]) and p.grad is None
    apply_ema_to_student_role(roles, ema.state_dict())
    result = one_step_edit_tokens(roles, source_tokens=b["source_tokens"], edit_region_mask=b["edit_region_mask"],
        prompt_condition=b["prompt_condition"], timestep_model_kwargs={}, mask_token_id=5, codebook_size=5,
        r_init=.5, init_mask_seeds=1, init_token_seeds=2, sample_seeds=3)
    assert result.metadata["number_of_transformer_forwards"] == 1


def test_teacher_base_identity_rejected(tmp_path):
    _, root = make_checkpoint(tmp_path)
    export_dense_teacher(root, tmp_path / "teacher", status="provisional")
    with pytest.raises(RuntimeError, match="IDENTITY_MISMATCH"):
        load_dense_teacher_contract(tmp_path / "teacher", expected_base_identity={"wrong": True})


def test_paired_bootstrap_and_pending_selection(tmp_path):
    def result(offset):
        return {"per_sample_after_seed_mean": [{"dataset_name": "magicbrush", "sample_key": str(i),
            "metrics": {"inside_masked_lpips": .5 + offset}} for i in range(4)]}
    paired = paired_comparison(result(-.1), result(0), resamples=100)
    assert paired["magicbrush"]["inside_masked_lpips"]["mean_delta"] == pytest.approx(-.1)
    assert select_checkpoint({}, None, [], tmp_path / "selected.json")["status"] == "PENDING"
    with pytest.raises(RuntimeError, match="PENDING"):
        selection_registration({"selection": {"status": "PENDING"}}, tmp_path / "prereg.json")
