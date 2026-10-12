import json
import sys

import pytest

from scripts.train import run_dimo_v2 as runner
from src.explicit_region.config import _load_unexpanded


def smoke_fixture(output):
    output.mkdir()
    rows = [{"global_step": step, "sampler_cursor": step * 8, "committed": True,
             "nan_inf_count": 0, "outside_mismatch_count": 0, "fresh_outside_mismatch_count": 0,
             "student_grad_norm_postclip": 2., "student_grad_norm_embed": 0., "student_grad_norm_mlm_head": 0.,
             "student_signal_missing": False, "auxiliary_signal_missing": False,
             "student_clip_triggered": False, "student_clip_fraction": 0.,
             "student_parameter_update_ratio_full": 1e-6 if step == 20 else None,
             "aux_parameter_update_ratio": 1e-6} for step in range(1, 21)]
    (output / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for rank in range(4):
        (output / f"v2_acceptance_rank{rank}.json").write_text(json.dumps(
            {"status": "PASS", "evidence": "REAL_GPU", "frozen_weights": "bitwise_hash_equal"}))
    return rows


def test_smoke_pass_does_not_require_natural_clipping(tmp_path):
    rows = smoke_fixture(tmp_path / "smoke")
    assert runner.validate_smoke_logs(tmp_path / "smoke") == rows[-1]


def test_smoke_singleton_proof_requires_finite_one_forward_locked_inference(tmp_path):
    path = tmp_path / "image.png.json"
    values = {"nan_inf_count": 0, "number_of_transformer_forwards": 1, "outside_mismatch_count": 0,
              "r_init": 1., "cfg_scale": 1., "embedding_noise_sigma": .3}
    path.write_text(json.dumps(values))
    runner.validate_singleton_metadata(path)
    for key in values:
        path.write_text(json.dumps(values | {key: 99}))
        with pytest.raises(RuntimeError, match="INFERENCE_FAILURE"):
            runner.validate_singleton_metadata(path)


@pytest.mark.parametrize("field,value", [("student_grad_norm_postclip", 10.1),
                                       ("fresh_outside_mismatch_count", 1), ("committed", False)])
def test_smoke_rejects_invalid_step(tmp_path, field, value):
    output = tmp_path / "smoke"
    rows = smoke_fixture(output)
    rows[3][field] = value
    (output / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(RuntimeError, match="FAILURE"):
        runner.validate_smoke_logs(output)


def setup_cli(monkeypatch, output, mode):
    config = _load_unexpanded(runner.ROOT / "configs/dimo/dimo_v2_mixed20k.yaml")
    monkeypatch.setattr(runner, "load_config", lambda _: config)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "a" * 40)
    monkeypatch.setattr(sys, "argv", ["run_dimo_v2.py", "--mode", mode, "--output-dir", str(output)])
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("unexpected process launch"))


def test_long_run_requires_confirmation_and_gpu_proof_before_admission(tmp_path, monkeypatch):
    setup_cli(monkeypatch, tmp_path / "train", "train")
    monkeypatch.setattr(runner, "gpu_admission", lambda **k: pytest.fail("GPU admission reached"))
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not (tmp_path / "train").exists()


def test_busy_gpu_refuses_before_launch_or_output_creation(tmp_path, monkeypatch):
    setup_cli(monkeypatch, tmp_path / "smoke", "smoke")
    def busy(**kwargs):
        assert kwargs == {"magicbrush": True, "allow_other_jobs": False}
        raise RuntimeError("FULL_DIMO_GPU_HAS_OTHER_JOB_DO_NOT_PREEMPT")
    monkeypatch.setattr(runner, "gpu_admission", busy)
    with pytest.raises(RuntimeError, match="DO_NOT_PREEMPT"):
        runner.main()
    assert not (tmp_path / "smoke").exists()
