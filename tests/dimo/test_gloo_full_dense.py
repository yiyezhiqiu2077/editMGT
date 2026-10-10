"""Real two-process Gloo tests; these do not claim NCCL/GPU acceptance."""
import json
import os
from pathlib import Path
import subprocess
import sys
import signal

import pytest
import torch
from src.dimo.full_checkpoint import verify
from src.dimo.distributed import state_digest

ROOT = Path(__file__).resolve().parents[2]


def launch(output, *args):
    environment = os.environ | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
        "scripts/train/train_full_dense_dimo.py", "--config", "configs/dimo/full_dense_local_smoke.yaml",
        "--synthetic", "--output-dir", str(output), *args]
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail("Gloo test timed out; workers stopped: " + stdout + stderr)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def test_two_rank_fresh_vs_resume_full_state(tmp_path):
    fresh, resumed = tmp_path / "fresh", tmp_path / "resumed"
    result = launch(fresh, "--stop-at-global-step", "3")
    assert result.returncode == 0, result.stdout + result.stderr
    result = launch(resumed, "--stop-at-global-step", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = resumed / "checkpoints/checkpoint-1"
    result = launch(resumed, "--stop-at-global-step", "3", "--resume", str(checkpoint))
    assert result.returncode == 0, result.stdout + result.stderr
    def loaded(folder):
        root = folder / "checkpoints/checkpoint-3"
        verify(root, full=True)
        state = torch.load(root / "training_state.pt", weights_only=False)
        return root, state
    a_root, a = loaded(fresh); b_root, b = loaded(resumed)
    from src.dimo.resume_comparison import compare_complete_checkpoints
    comparison = compare_complete_checkpoints(a_root, b_root)
    assert comparison["status"] == "PASS" and comparison["evidence"] == "CPU_SYNTHETIC"
    assert a["step"] == b["step"] == a["ema_update_count"] == b["ema_update_count"] == 3
    assert a["cursor"] == b["cursor"] == 6
    for key in ("optimizers", "schedulers", "rank_rng", "sampler_state", "loader_rng"):
        assert state_digest(a[key]) == state_digest(b[key]), key
    for role in ("student", "auxiliary", "ema"):
        from safetensors.torch import load_file
        def weights(root):
            index = json.loads((root / f"{role}.index.json").read_text())["weight_map"]
            state = {}
            for path in set(index.values()): state.update(load_file(root / path))
            return state
        assert state_digest(weights(a_root)) == state_digest(weights(b_root)), role
    for rank in (0, 1):
        acceptance = json.loads((resumed / f"runtime_acceptance_rank{rank}.json").read_text())
        assert acceptance["gpu_acceptance"] == "NOT_RUN_RESOURCE_UNAVAILABLE"


@pytest.mark.parametrize("phase", ["student_optimizer", "auxiliary_optimizer", "ema_during", "ema_complete", "before_commit"])
def test_single_rank_failure_ends_both_workers_without_commit(tmp_path, phase):
    output = tmp_path / phase
    result = launch(output, "--stop-at-global-step", "2", "--inject-failure", phase, "--failure-rank", "0")
    assert result.returncode != 0
    assert "INJECTED_FAILURE" in result.stdout + result.stderr
    assert not (output / "checkpoints/checkpoint-1").exists()
    assert not (output / "metrics.jsonl").exists()
    for rank in (0, 1):
        failure = json.loads((output / f"failure_rank{rank}.json").read_text())
        assert failure["last_committed_step"] == 0
        assert failure["last_committed_cursor"] == 0
