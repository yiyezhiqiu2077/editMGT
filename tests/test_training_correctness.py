"""CPU regressions for the actual training helpers, without model downloads."""
from copy import deepcopy
import json
import os
import random
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from accelerate.scheduler import AcceleratedScheduler
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader, Dataset

from scripts.train import train_explicit_region as training
from src.explicit_region import checkpoint
from src.explicit_region.contracts import (
    audit_trainable_parameters, configure_region_trainability,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.ones(4))
        self.edit_region_embedding = torch.nn.Parameter(torch.zeros(4))
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, value, active=True):
        result = self.dropout(value) * self.lora_weight
        if active:
            result = result + self.edit_region_embedding
        return result


@pytest.mark.parametrize("mode,persistent,probability,active", [
    ("full_target", False, 0.0, False), ("full_target", True, 0.0, False),
    ("roi_hardlock", False, 1.0, False), ("roi_hardlock", True, 1.0, True),
    ("mixed", True, 0.0, False), ("mixed", True, 0.5, True),
])
def test_recipe_controls_region_trainability_and_optimizer_audit(mode, persistent, probability, active):
    model = TinyModel()
    config = {"mode": mode, "persistent_conditioning": persistent, "roi_probability": probability}
    assert configure_region_trainability(model, config) is active
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    report = audit_trainable_parameters({"transformer": model}, optimizer, region_conditioning_active=active)
    assert report["region_conditioning_active"] is active
    assert report["trainable_edit_region_embedding_params"] == (4 if active else 0)
    model(torch.ones(4), active=active).sum().backward()
    assert (model.edit_region_embedding.grad is not None) is active
    with pytest.raises(RuntimeError, match="requires_grad"):
        audit_trainable_parameters({"transformer": model}, optimizer, region_conditioning_active=not active)


def test_inactive_embedding_cannot_leak_into_optimizer():
    model = TinyModel()
    configure_region_trainability(model, {"mode": "roi_hardlock", "persistent_conditioning": False})
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(RuntimeError, match="optimizer/trainable mismatch"):
        audit_trainable_parameters({"transformer": model}, optimizer, region_conditioning_active=False)


def test_scheduler_warmup_is_200_successful_updates_not_world8(monkeypatch):
    # Emulate the installed AcceleratedScheduler world8 wrapper. The default
    # step_with_optimizer=True would burn eight warmup steps per call.
    monkeypatch.setattr("accelerate.scheduler.AcceleratorState", lambda: SimpleNamespace(num_processes=8))
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=3e-5)
    raw = get_scheduler("constant_with_warmup", optimizer, num_warmup_steps=200, num_training_steps=6250)
    scheduler = AcceleratedScheduler(raw, optimizer, step_with_optimizer=False)
    accelerator = SimpleNamespace(sync_gradients=False, optimizer_step_was_skipped=False)
    for step in range(1, 202):
        for _ in range(3):
            assert training.step_update_scheduler(accelerator, scheduler) is False
        accelerator.sync_gradients = True
        expected_applied = 3e-5 * min(1, (step - 1) / 200)
        assert optimizer.param_groups[0]["lr"] == expected_applied
        optimizer.step()
        assert training.step_update_scheduler(accelerator, scheduler) is True
        assert raw.last_epoch == step
        assert scheduler.get_last_lr() == [3e-5 * min(1, step / 200)]
        accelerator.optimizer_step_was_skipped = True
        assert training.step_update_scheduler(accelerator, scheduler) is False
        assert raw.last_epoch == step
        accelerator.optimizer_step_was_skipped = False
        accelerator.sync_gradients = False


def rng_draws():
    return random.random(), np.random.random(4), torch.rand(4)


def assert_draws_equal(left, right):
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert torch.equal(left[2], right[2])


@pytest.fixture
def cpu_rng(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    random.seed(33)
    np.random.seed(34)
    torch.manual_seed(35)


def test_all_rng_families_roundtrip_and_validation_does_not_advance(cpu_rng):
    bundle = checkpoint.capture_rank_rng_state()
    expected = rng_draws()
    checkpoint.restore_rank_rng_state(bundle)
    checkpoint.validate_rank_rng_state(bundle, world_size=1)
    assert_draws_equal(expected, rng_draws())


@pytest.mark.parametrize("mutation,match", [
    (lambda b: b.update(world_size=2), "world_size"),
    (lambda b: b["states"].clear(), "RNG ranks"),
    (lambda b: b["states"].update({"1": b["states"]["0"]}), "RNG ranks"),
    (lambda b: b["states"]["0"].update(rank=1), "rank 0"),
    (lambda b: b["states"]["0"].pop("python"), "rank 0"),
    (lambda b: b["states"]["0"].update(torch_cpu=torch.ones(2)), "RNG state"),
    (lambda b: b["states"]["0"].update(python=(3, b["states"]["0"]["python"][1], float("nan"))), "RNG state"),
    (lambda b: b["states"]["0"].update(numpy=b["states"]["0"]["numpy"][:4] + (float("inf"),)), "RNG state"),
    (lambda b: b["states"]["0"].update(torch_cuda={"1": torch.ones(2, dtype=torch.uint8)}), "device keys"),
])
def test_rng_completeness_fails_closed(cpu_rng, mutation, match):
    bundle = checkpoint.capture_rank_rng_state()
    mutation(bundle)
    with pytest.raises(RuntimeError, match=match):
        checkpoint.restore_rank_rng_state(bundle)


def test_cuda_rng_restore_uses_own_rank_not_rank_zero(monkeypatch, cpu_rng):
    bundle = checkpoint.capture_rank_rng_state()
    bundle["world_size"] = 2
    bundle["states"]["1"] = deepcopy(bundle["states"]["0"])
    for rank in (0, 1):
        bundle["states"][str(rank)]["rank"] = rank
        bundle["states"][str(rank)]["torch_cuda"] = {"0": torch.tensor([rank, 13], dtype=torch.uint8)}
    received = []
    monkeypatch.setattr(checkpoint, "distributed_rank_world", lambda: (1, 2))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: received.extend(states))
    checkpoint.restore_rank_rng_state(bundle)
    assert torch.equal(received[0], torch.tensor([1, 13], dtype=torch.uint8))


class StatelessRows(Dataset):
    def __init__(self, start=0, stop=20):
        self.start, self.stop = start, stop

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        return torch.full((4,), (self.start + index + 1) / 20)


@pytest.fixture
def short_worker_tmp(tmp_path, monkeypatch):
    # Linux AF_UNIX addresses are <=108 bytes; RUN_ROOT's absolute path exceeds
    # that. The open-directory alias keeps all actual socket files in RUN_ROOT.
    descriptor = os.open(tmp_path, os.O_RDONLY)
    short_path = f"/proc/{os.getpid()}/fd/{descriptor}"
    monkeypatch.setenv("TMPDIR", short_path)
    monkeypatch.setattr(tempfile, "tempdir", short_path)
    yield
    os.close(descriptor)


@pytest.mark.parametrize("workers", [0, 2])
def test_resume_iterator_and_stochastic_optimizer_exact(cpu_rng, short_worker_tmp, workers):
    from scripts.train.verify_fixed200k_smoke import assert_exact
    torch.set_num_threads(1)

    def setup(start):
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, foreach=False)
        scheduler = get_scheduler("constant_with_warmup", optimizer, num_warmup_steps=4, num_training_steps=20)
        loader = DataLoader(StatelessRows(start), batch_size=1, num_workers=workers,
                            timeout=30 if workers else 0,
                            generator=training.loader_generator(42, 0, 0))
        return model, optimizer, scheduler, loader

    model, optimizer, scheduler, loader = setup(0)
    before = torch.get_rng_state()
    iterator = training.resume_training_iterator(loader)
    assert torch.equal(before, torch.get_rng_state())
    snapshot = None
    for index, batch in enumerate(iterator, 1):
        loss = model(batch).square().mean() * (1 + random.random() + np.random.random())
        loss.backward(); optimizer.step(); scheduler.step(); optimizer.zero_grad()
        if index == 10:
            snapshot = deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), checkpoint.capture_rank_rng_state()))
    expected = (deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), deepcopy(scheduler.state_dict()), checkpoint.capture_rank_rng_state())
    model, optimizer, scheduler, loader = setup(10)
    model.load_state_dict(snapshot[0]); optimizer.load_state_dict(snapshot[1]); scheduler.load_state_dict(snapshot[2])
    # Simulate resumed startup consuming every process RNG before iterator setup.
    rng_draws()
    iterator = training.resume_training_iterator(loader, {"rank_rng": snapshot[3]})
    for batch in iterator:
        loss = model(batch).square().mean() * (1 + random.random() + np.random.random())
        loss.backward(); optimizer.step(); scheduler.step(); optimizer.zero_grad()
    actual = (model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), checkpoint.capture_rank_rng_state())
    assert_exact(expected, actual)


def test_checkpoint_rng_schema_metadata_and_deferred_restore(cpu_rng, monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoint, "get_peft_model_state_dict", lambda model: {"lora_weight": model.lora_weight})
    def set_lora(model, state):
        with torch.no_grad():
            model.lora_weight.copy_(state["lora_weight"])
    monkeypatch.setattr(checkpoint, "set_peft_model_state_dict", set_lora)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = get_scheduler("constant_with_warmup", optimizer, num_warmup_steps=2, num_training_steps=20)
    model(torch.ones(4)).sum().backward(); optimizer.step(); scheduler.step(); optimizer.zero_grad()
    payload = {"recipe": "test", "world_size": 1}
    checkpoint.save_resume_state(tmp_path, model=model, optimizer=optimizer, scheduler=scheduler,
                                 global_optimizer_step=1, committed_global_sample_count=32,
                                 sampler_schema="test", fingerprint_payload=payload,
                                 resolved_config={"output_dir": "test"})
    expected = rng_draws()
    checkpoint.load_resume_state(tmp_path, model=model, optimizer=optimizer, scheduler=scheduler, fingerprint_payload=payload)
    assert_draws_equal(expected, rng_draws())
    before = checkpoint.capture_rank_rng_state()
    checkpoint.load_resume_state(tmp_path, model=model, optimizer=optimizer, scheduler=scheduler, fingerprint_payload=payload, restore_rng=False)
    from scripts.train.verify_fixed200k_smoke import assert_exact
    assert_exact(before, checkpoint.capture_rank_rng_state())
    metadata = json.loads((tmp_path / "checkpoint_metadata.json").read_text())
    assert metadata["global_optimizer_step"] == 1
    assert metadata["resolved_config_sha256"] == checkpoint.recipe_fingerprint({"output_dir": "test"})
    with pytest.raises(RuntimeError, match="fingerprint"):
        checkpoint.read_resume_state(tmp_path, fingerprint_payload={"changed": 1}, world_size=1)
    state = torch.load(tmp_path / "training_state.pt", weights_only=False)
    del state["rank_rng"]["states"]["0"]["numpy"]
    torch.save(state, tmp_path / "training_state.pt")
    with pytest.raises(RuntimeError, match="RNG"):
        checkpoint.read_resume_state(tmp_path, fingerprint_payload=payload, world_size=1)


def test_fingerprint_binds_missing_stochastic_settings(monkeypatch):
    monkeypatch.setattr(training, "repository_identity", lambda: {"uv_lock_sha256": "lock", "git_sha": "sha", "git_dirty_diff_sha256": "diff"})
    config = {
        "data": {"name": "magicbrush"}, "validation": {"enabled": False}, "model": {},
        "resolution": 1024, "seed": 42, "corruption": {}, "token_mask": {}, "lora": {},
        "optimizer": {}, "scheduler_horizon_steps": 6250, "warmup_steps": 200,
        "precision": "bf16", "component_dtypes": {}, "batch_per_gpu": 1,
        "gradient_accumulation": 4, "geometry": {"minimum_mask_retention": 0.75},
        "condition_dropout": {"probability": 0.1}, "num_workers": 4,
        "gradient_checkpointing": True, "quality_gate": {"enabled": True},
        "output_dir": "fresh10", "max_optimizer_steps": 10, "checkpoint_steps": [10],
    }
    identity = {"snapshot_identity": "revision", "components": {"vq": "config-hash"}}
    baseline = training.fingerprint_payload(config, identity)
    for key, value in (("condition_dropout", {"probability": 0.2}), ("geometry", {}),
                       ("num_workers", 0), ("gradient_checkpointing", False),
                       ("quality_gate", {"enabled": False})):
        changed = deepcopy(config); changed[key] = value
        assert training.fingerprint_payload(changed, identity) != baseline
    changed = deepcopy(config)
    changed.update(output_dir="resume20", max_optimizer_steps=20, checkpoint_steps=[20])
    assert training.fingerprint_payload(changed, identity) == baseline


def test_resume_rejects_uncommitted_or_inconsistent_cursor(tmp_path):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("fixture")
    from src.explicit_region.epoch_sampler import SAMPLER_SCHEMA_VERSION
    config = {"data": {"name": "fixed_200k", "manifest": str(manifest)}, "batch_per_gpu": 1,
              "gradient_accumulation": 4, "max_optimizer_steps": 20}
    state = {"global_optimizer_step": 10, "committed_global_sample_count": 320,
             "scheduler": {"last_epoch": 10}, "sampler_schema": SAMPLER_SCHEMA_VERSION,
             "sampler_state": {"sampler_schema_version": SAMPLER_SCHEMA_VERSION, "epoch": 0,
                               "samples_consumed_in_epoch": 320, "global_optimizer_step_in_epoch": 10,
                               "global_microbatch_in_epoch": 40, "world_size": 8,
                               "batch_per_gpu": 1, "gradient_accumulation": 4,
                               "train_200k_sha256": training.sha256_file(manifest)}}
    assert training.validate_resume_cursor(state, config, 8) == (320, 10)
    for mutation in (
        lambda s: s.update(committed_global_sample_count=319),
        lambda s: s.update(global_optimizer_step=10.5),
        lambda s: s["scheduler"].update(last_epoch=80),
        lambda s: s["sampler_state"].update(samples_consumed_in_epoch=352),
        lambda s: s["sampler_state"].update(train_200k_sha256="stale"),
        lambda s: s["sampler_state"].update(global_microbatch_in_epoch=44),
        lambda s: s["sampler_state"].update(epoch=1),
    ):
        bad = deepcopy(state); mutation(bad)
        with pytest.raises(RuntimeError):
            training.validate_resume_cursor(bad, config, 8)


def test_repository_digest_includes_staged_unstaged_and_untracked(monkeypatch, tmp_path):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    git("init")
    (tmp_path / "uv.lock").write_text("lock")
    (tmp_path / "tracked").write_text("base")
    git("add", ".")
    # Test fixture commit only, never commits the user's repository.
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    monkeypatch.setattr(training, "ROOT", tmp_path)
    clean = training.repository_identity()
    (tmp_path / "tracked").write_text("staged")
    git("add", "tracked")
    staged = training.repository_identity()
    assert staged["git_dirty_diff_sha256"] != clean["git_dirty_diff_sha256"]
    (tmp_path / "tracked").write_text("unstaged")
    unstaged = training.repository_identity()
    assert unstaged["git_dirty_diff_sha256"] != staged["git_dirty_diff_sha256"]
    (tmp_path / "before_p0_fix.patch").write_text("must not be exempt")
    assert training.repository_identity()["git_dirty_diff_sha256"] != unstaged["git_dirty_diff_sha256"]
