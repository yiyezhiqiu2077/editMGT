import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from tests.test_dense_pipeline import ToyDense, make_checkpoint, payload
from src.explicit_region.dense_checkpoint import architecture_of
from src.explicit_region.dense_evaluation import selection_registration, select_checkpoint, CANDIDATES, assert_matching_predictions
from src.explicit_region.dense_runtime import UpdateSample
from src.dimo.dense_teacher import export_dense_teacher, load_dense_teacher_contract
from src.dimo.dense_roles import initialize_dense_model_roles
from src.dimo.dense_distributed_checkpoint import save_checkpoint, load_checkpoint, load_inference
from src.dimo.ema import TrainableEMA


def test_zero_lr_warmup_step_not_false_failure():
    sample = UpdateSample(ToyDense())
    assert sample.finish(require_update=False)["dense_update_norm"] == 0


def test_selected_teacher_success(tmp_path):
    _, root = make_checkpoint(tmp_path, step=3125)
    manifest = tmp_path / "dev.jsonl"; manifest.write_text("{}\n")
    plan = {"candidate_steps": CANDIDATES, "datasets": [{"name": "magicbrush", "manifest": str(manifest)}],
        "selection": {"status": "PREREGISTERED", "maximum_outside_lpips_degradation": .01,
                      "minimum_inside_lpips_improvement": .01, "require_inside_ci_improvement": True}}
    prereg = selection_registration(plan, tmp_path / "prereg.json")
    from src.explicit_region.dense_checkpoint import verify_dense_checkpoint
    item = {"backend": "dense", "checkpoint": str(root), "step": 3125,
        "metrics": {"magicbrush": {"inside_masked_lpips": .3}},
        "paired_to_E0-region": {"magicbrush": {
            "inside_masked_lpips": {"mean_delta": -.2, "bootstrap_95_ci": [-.3, -.1]},
            "outside_masked_lpips": {"mean_delta": .001}}}}
    record = tmp_path / "selected.json"
    assert select_checkpoint(plan, prereg, [item], record)["status"] == "READY"
    export_dense_teacher(root, tmp_path / "teacher", status="selected", selected_record=record)
    assert load_dense_teacher_contract(tmp_path / "teacher").formal_teacher


def test_dense_dimo_checkpoint_inference_without_optimizer(tmp_path, monkeypatch):
    _, root = make_checkpoint(tmp_path)
    export_dense_teacher(root, tmp_path / "teacher", status="provisional")
    roles = initialize_dense_model_roles(ToyDense(), tmp_path / "teacher", lora={"rank": 2, "alpha": 2, "target_modules": ["projection"]})
    optimizers = [torch.optim.AdamW([p for _, p in roles.role_named_parameters(r)]) for r in ("student", "auxiliary")]
    schedulers = [torch.optim.lr_scheduler.LambdaLR(o, lambda _: 1) for o in optimizers]
    ema = TrainableEMA(roles.role_named_parameters("student"))
    identity = {"teacher_bundle_fingerprint": "CPU_SYNTHETIC"}
    save_checkpoint(tmp_path / "dimo", roles=roles, optimizers=optimizers, schedulers=schedulers, ema=ema,
                    step=2, cursor=2, identity=identity, config_hash="CPU_SYNTHETIC")
    state = load_checkpoint(tmp_path / "dimo", roles=roles, optimizers=optimizers, schedulers=schedulers, ema=ema,
                            expected_identity=identity, expected_config_hash="CPU_SYNTHETIC")
    assert state["step"] == 2
    monkeypatch.setattr(torch, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("optimizer loaded")))
    for weights in ("student", "ema"):
        load_inference(tmp_path / "dimo", roles=roles, expected_identity=identity, weights=weights)
    with pytest.raises(RuntimeError):
        load_inference(tmp_path / "dimo", roles=roles, expected_identity={"wrong": True})


def test_low_frequency_diagnostics_null_source_contract():
    source = Path("scripts/train/train_dense_region.py").read_text()
    assert '"dense_update_norm": None' in source
    assert 'before_update = [parameter.detach().clone()' not in source
    assert "save_trainable_state(" not in source
    assert "add_adapter(" not in source
    assert "load_trainable_state(" not in source


def test_old_lora_entry_unchanged():
    import subprocess
    from src.explicit_region.contracts import sha256_file
    import hashlib
    base = subprocess.check_output(["git", "show", "36dff5ff9ad10cee3d3910eb980932ec0a251b62:scripts/train/train_explicit_region.py"])
    assert sha256_file("scripts/train/train_explicit_region.py") == hashlib.sha256(base).hexdigest()


def test_dense_epoch_iterator_replays_generator_exactly():
    from src.explicit_region.dense_runtime import DenseEpochLoader
    from src.explicit_region.epoch_sampler import DeterministicEpochSampler
    class Dataset(torch.utils.data.Dataset):
        def __len__(self): return 64
        def __getitem__(self, index): return index
        def set_epoch(self, epoch): self.epoch = epoch
    def sampler(cursor):
        return DeterministicEpochSampler(64, base_seed=42, epoch=0, train_manifest_sha256="d" * 64,
                                         samples_consumed_in_epoch=cursor)
    generator = torch.Generator().manual_seed(99)
    fresh = DenseEpochLoader(Dataset(), sampler(0), epochs=1, batch_size=1, generator=generator, pin_memory=False)
    full = [int(x) for x in fresh]
    expected_rng = generator.get_state()
    resumed_generator = torch.Generator().manual_seed(99)
    resumed_generator.set_state(expected_rng)
    resumed = DenseEpochLoader(Dataset(), sampler(32), epochs=1, batch_size=1,
        generator=resumed_generator, pin_memory=False, resume_epoch_start_rng=fresh.epoch_start_rng)
    assert [int(x) for x in resumed] == full[32:]
    assert torch.equal(resumed_generator.get_state(), expected_rng)
