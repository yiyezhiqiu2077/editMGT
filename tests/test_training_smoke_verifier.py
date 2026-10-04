"""Synthetic fixtures exercise smoke fail-closed behavior; no GPU run is claimed."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from scripts.train import verify_fixed200k_smoke as smoke
from src.explicit_region.checkpoint import capture_local_rng_state, recipe_fingerprint
from src.explicit_region.epoch_sampler import SAMPLER_SCHEMA_VERSION


def write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True))


@pytest.fixture
def smoke_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    roots = [tmp_path / name for name in ("full", "first", "resume")]
    for root in roots:
        root.mkdir()
    identity = {
        "schema": "stage-gate-identity-v1", "corpus_ready_sha256": "ready", "git_sha": "code",
        "git_dirty_diff_sha256": "diff", "uv_lock_sha256": "lock", "smoke_configs": {},
    }
    configs = []
    for index, (root, relative) in enumerate(zip(roots, smoke.SMOKE_CONFIGS)):
        config = {
            "data": {"corpus_ready": str(tmp_path / "READY.json")}, "output_dir": str(root),
            "batch_per_gpu": 1, "gradient_accumulation": 4, "resolution": 1024,
            "precision": "bf16", "torch_compile": False, "component_dtypes": {"vqvae": "fp32"},
            "warmup_steps": 200, "scheduler_horizon_steps": 6250,
            "corruption": {"mode": "roi_hardlock", "persistent_conditioning": True},
            "lora": {"scope": "both"}, "optimizer": {"learning_rate": 3e-5},
            "max_optimizer_steps": 10 if index == 1 else 20,
        }
        configs.append(config)
        identity["smoke_configs"][relative] = {
            "path": str(smoke.ROOT / relative), "sha256": f"raw{index}",
            "resolved_sha256": recipe_fingerprint(config),
        }
    monkeypatch.setattr(smoke, "build_gate_identity", lambda *args, **kwargs: deepcopy(identity))
    payload = {
        "schema": "explicit-region-sft-v2",
        "world_size": 8, "git_sha": "code", "git_dirty_diff_sha256": "diff", "uv_lock_sha256": "lock",
        "data_content_hashes": {"data.corpus_ready": {"sha256": "ready"}, "data.manifest": {"sha256": "manifest"}},
        "training_config": {key: value for key, value in configs[0].items()
                            if key not in {"output_dir", "max_optimizer_steps", "checkpoint_steps", "run_type"}},
    }
    fingerprint = recipe_fingerprint(payload)
    rank_rng = {"schema_version": 1, "world_size": 8, "states": {}}
    for rank in range(8):
        # A CPU fixture has synthetic CUDA byte arrays solely for schema testing.
        state = capture_local_rng_state(rank)
        state["torch_cuda"] = {"0": torch.tensor([rank, 12, 42], dtype=torch.uint8)}
        rank_rng["states"][str(rank)] = state
    def quality(step):
        return {"losses": [(i, 1.0) for i in range(1, step + 1)], "clips": [False] * step,
                "bad_windows": 0, "status": "INSUFFICIENT_BASELINE", "reason": None}
    for index, (root, config, relative) in enumerate(zip(roots, configs, smoke.SMOKE_CONFIGS)):
        provenance = {key: identity[key] for key in ("git_sha", "git_dirty_diff_sha256", "uv_lock_sha256", "corpus_ready_sha256")}
        provenance.update(world_size=8, deterministic_algorithms=True, cudnn_deterministic=True,
                          cudnn_benchmark=False, allow_tf32=False, torch_compile=False,
                          gate_identity=identity, resolved_config=config,
                          resolved_config_sha256=recipe_fingerprint(config), config_sha256=f"raw{index}",
                          config_path=str(smoke.ROOT / relative))
        write_json(root / "run_provenance.json", provenance)
        steps = range(1, 11) if index == 1 else (range(11, 21) if index == 2 else range(1, 21))
        metrics = []
        for step in steps:
            metrics.append({
                "global_step": step, "committed_global_sample_count": step * 32,
                "committed_sample_uids": [f"sample-{i}" for i in range((step - 1) * 32, step * 32)],
                "committed_geometry_seeds": list(range((step - 1) * 32, step * 32)),
                "committed_corruption_seeds": list(range((step - 1) * 32 + 7, step * 32 + 7)),
                "learning_rate": 3e-5 * ((step - 1) / 200), "next_learning_rate": 3e-5 * (step / 200),
                "finite": True, "finite_parameters": True, "sample_mean_ce": 1.0, "token_mean_ce": 1.1,
                "quality_status": "INSUFFICIENT_BASELINE", "grad_norm_pre_clip": 0.1,
                "grad_norm_post_clip": 0.1, "parameter_norm": 1.0, "optimizer_update_norm": 0.01,
                "update_ratio": 0.01, "max_abs_logit": 1.0,
                "peak_vram_bytes": index * 1024,
            })
        (root / "train_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))
        for step in ([10, 20] if index == 0 else [10 if index == 1 else 20]):
            directory = root / f"checkpoint-{step}"
            directory.mkdir()
            save_file({"adapter": torch.ones(4)}, directory / "adapter_model.safetensors")
            save_file({"edit_region_embedding": torch.zeros(4)}, directory / "mask_conditioning.safetensors")
            lr = 3e-5 * (step / 200)
            state = {
                "schema_version": 2,
                "optimizer": {"state": {0: {"step": torch.tensor(float(step)), "exp_avg": torch.ones(4), "exp_avg_sq": torch.ones(4)}},
                              "param_groups": [{"lr": lr, "params": [0]}]},
                "scheduler": {"last_epoch": step, "_last_lr": [lr]},
                "rank_rng": rank_rng, "global_optimizer_step": step,
                "committed_global_sample_count": step * 32, "sampler_schema": SAMPLER_SCHEMA_VERSION,
                "quality_state": quality(step), "metrics_state": {}, "fingerprint": fingerprint,
                "sampler_state": {
                    "sampler_schema_version": SAMPLER_SCHEMA_VERSION, "epoch": 0,
                    "samples_consumed_in_epoch": step * 32, "world_size": 8, "batch_per_gpu": 1,
                    "gradient_accumulation": 4, "global_optimizer_step_in_epoch": step,
                    "global_microbatch_in_epoch": step * 4, "epoch_permutation_sha256": "a" * 64,
                    "train_200k_sha256": "manifest",
                },
            }
            torch.save(state, directory / "training_state.pt")
            write_json(directory / "fingerprint.json", {"sha256": fingerprint, "payload": payload})
            write_json(directory / "trainable_config.json", payload)
            write_json(directory / "checkpoint_metadata.json", {
                "schema_version": 2, "global_optimizer_step": step, "committed_global_sample_count": step * 32,
                "fingerprint": fingerprint, "resolved_config_sha256": recipe_fingerprint(config),
            })
        write_json(root / "quality_status.json", quality(10 if index == 1 else 20))
    return roots


def test_exact_finite_smoke_fixture_passes(smoke_runs):
    report = smoke.verify_smoke(*smoke_runs)
    assert report["status"] == "PASS", report
    assert report["samples"] == report["unique_samples"] == 640
    assert report["rank_rng_match"] and report["optimizer_match"]


@pytest.mark.parametrize("filename,key,value", [
    ("adapter_model.safetensors", "adapter", 1.01),
    ("mask_conditioning.safetensors", "edit_region_embedding", 0.01),
    ("adapter_model.safetensors", "adapter", float("nan")),
    ("mask_conditioning.safetensors", "edit_region_embedding", float("inf")),
])
def test_any_weight_difference_or_nonfinite_fails(smoke_runs, filename, key, value):
    save_file({key: torch.full((4,), value)}, smoke_runs[2] / "checkpoint-20" / filename)
    assert smoke.verify_smoke(*smoke_runs)["status"] == "FAIL"


@pytest.mark.parametrize("mutation", [
    lambda s: s["optimizer"]["state"][0]["exp_avg"].add_(1),
    lambda s: s["optimizer"]["state"][0]["exp_avg_sq"].fill_(float("nan")),
    lambda s: s["scheduler"].update(last_epoch=160),
    lambda s: s["quality_state"].update(bad_windows=1),
    lambda s: s["quality_state"].pop("clips"),
    lambda s: s["sampler_state"].update(epoch_permutation_sha256="b" * 64),
    lambda s: s.update(committed_global_sample_count=639),
    lambda s: s["rank_rng"]["states"].pop("7"),
    lambda s: s["rank_rng"]["states"]["7"].pop("numpy"),
    lambda s: s["rank_rng"]["states"]["7"]["torch_cpu"].fill_(0),
    lambda s: s["rank_rng"]["states"]["7"]["torch_cuda"]["0"].add_(1),
    lambda s: s["rank_rng"].update(world_size=1),
    lambda s: s.pop("optimizer"),
])
def test_state_difference_missing_or_nonfinite_fails(smoke_runs, mutation):
    path = smoke_runs[2] / "checkpoint-20" / "training_state.pt"
    state = torch.load(path, weights_only=False)
    mutation(state)
    torch.save(state, path)
    assert smoke.verify_smoke(*smoke_runs)["status"] == "FAIL"


@pytest.mark.parametrize("mutation", [
    lambda rows: rows[0]["committed_sample_uids"].__setitem__(0, "sample-0"),
    lambda rows: rows[0]["committed_geometry_seeds"].__setitem__(0, None),
    lambda rows: rows[0].update(sample_mean_ce=float("nan")),
    lambda rows: rows[0].update(finite=False),
    lambda rows: rows[0].update(learning_rate=0.0),
    lambda rows: rows[0].update(committed_global_sample_count=1),
    lambda rows: rows.pop(),
])
def test_metric_sequence_and_finiteness_fails(smoke_runs, mutation):
    path = smoke_runs[2] / "train_metrics.jsonl"
    values = smoke.rows(smoke_runs[2]); mutation(values)
    path.write_text("".join(json.dumps(row) + "\n" for row in values))
    assert smoke.verify_smoke(*smoke_runs)["status"] == "FAIL"


def test_stale_provenance_cannot_receive_current_gate_identity(smoke_runs):
    path = smoke_runs[1] / "run_provenance.json"
    provenance = json.loads(path.read_text()); provenance["git_sha"] = "stale"
    write_json(path, provenance)
    report = smoke.verify_smoke(*smoke_runs)
    assert report["status"] == "FAIL" and "gate_identity" not in report


def test_missing_artifact_fails_instead_of_reusing_pass(smoke_runs):
    (smoke_runs[1] / "checkpoint-10" / "adapter_model.safetensors").unlink()
    assert smoke.verify_smoke(*smoke_runs)["status"] == "FAIL"


def test_nested_comparison_checks_types_and_finiteness():
    for left, right in (({"a": 1}, {"b": 1}), ([1], (1,)),
                        (np.array([np.nan]), np.array([np.nan])),
                        (torch.tensor([1.0]), torch.tensor([1.0], dtype=torch.float64)),
                        ({"x": float("inf")}, {"x": float("inf")})):
        with pytest.raises(smoke.SmokeMismatch):
            smoke.assert_exact(left, right)
