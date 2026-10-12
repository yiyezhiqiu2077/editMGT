"""Real two-process v2 transactions, epoch boundaries, and fault replay on CPU."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch

from src.dimo.full_checkpoint import verify
from src.dimo.inference import load_dense_inference_role
from src.dimo.resume_comparison import compare_complete_checkpoints
from src.dimo.synthetic import SyntheticDenseTransformerV2

ROOT = Path(__file__).resolve().parents[2]


def launch(output, *args):
    environment = os.environ | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
        "scripts/train/train_full_dense_dimo.py", "--config", "configs/dimo/dimo_v2_cpu_smoke.yaml",
        "--synthetic", "--output-dir", str(output), *args]
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail("v2 Gloo timeout: " + stdout + stderr)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def require_success(result):
    assert result.returncode == 0, result.stdout + result.stderr


def test_epoch_boundary_resume_full_state_and_all_rank_data_order(tmp_path):
    fresh, resumed = tmp_path / "fresh", tmp_path / "resumed"
    require_success(launch(fresh))
    require_success(launch(resumed, "--stop-at-global-step", "4"))
    require_success(launch(resumed, "--stop-at-global-step", "5", "--resume", str(resumed / "checkpoints/checkpoint-4")))
    require_success(launch(resumed, "--resume", str(resumed / "checkpoints/checkpoint-5")))
    comparison = compare_complete_checkpoints(fresh / "checkpoints/checkpoint-12", resumed / "checkpoints/checkpoint-12")
    assert comparison["status"] == "PASS" and comparison["checks"]["runtime_stats"]
    traces = []
    for rank in range(2):
        a = [json.loads(l) for l in (fresh / f"trace_rank{rank}.jsonl").read_text().splitlines()]
        b = [json.loads(l) for l in (resumed / f"trace_rank{rank}.jsonl").read_text().splitlines()]
        assert a == b
        assert a[-1]["global_cursor"] == 48
        traces.append(a)
        report = json.loads((resumed / f"v2_acceptance_rank{rank}.json").read_text())
        assert report["frozen_weights"] == "bitwise_hash_equal" and report["completed_updates_per_role"] == 12
    for epoch in range(3):
        uids = [uid for trace in traces for row in trace if row["epoch"] == epoch
                for key in ("sample_uids_a", "sample_uids_b") for uid in row[key]]
        assert len(uids) == len(set(uids)) == 16
    for weights in ("student", "ema"):
        model = SyntheticDenseTransformerV2()
        role, identity = load_dense_inference_role(model, resumed / "inference/checkpoint-12", weights=weights,
                                                   teacher_bundle={"identity": "CPU_SYNTHETIC", "formal_teacher": False})
        assert identity["recipe"] == "dimo-v2" and not role.base_model.training
        assert all(not p.requires_grad for p in role.parameters())
    metrics = [json.loads(l) for l in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["total_sample_consumptions"] == 48
    assert metrics[-1]["student_grad_norm_embed"] == metrics[-1]["student_grad_norm_mlm_head"] == 0


@pytest.mark.parametrize("phase", ["student_optimizer", "fresh_student_forward", "auxiliary_optimizer",
                                    "ema_during", "ema_complete", "before_commit"])
def test_failure_after_partial_update_reloads_and_replays(tmp_path, phase):
    output = tmp_path / phase
    require_success(launch(output, "--stop-at-global-step", "1"))
    checkpoint = output / "checkpoints/checkpoint-1"
    result = launch(output, "--stop-at-global-step", "4", "--resume", str(checkpoint),
                    "--inject-failure", phase, "--failure-rank", "0")
    assert result.returncode != 0 and "INJECTED_FAILURE" in result.stdout + result.stderr
    assert not (output / "checkpoints/checkpoint-2").exists()
    for rank in range(2):
        failure = json.loads((output / f"failure_rank{rank}.json").read_text())
        assert failure["last_committed_step"] == 1 and failure["last_committed_cursor"] == 4
    require_success(launch(output, "--stop-at-global-step", "4", "--resume", str(checkpoint)))
    fresh = tmp_path / "continuous"
    require_success(launch(fresh, "--stop-at-global-step", "4"))
    assert compare_complete_checkpoints(output / "checkpoints/checkpoint-4", fresh / "checkpoints/checkpoint-4")["status"] == "PASS"
    identity, index = verify(output / "checkpoints/checkpoint-4", full=True)
    state = torch.load(output / "checkpoints/checkpoint-4/training_state.pt", weights_only=False)
    assert state["schema"] == "full-dense-dimo-ddp-v3" and state["cursor"] == 16
    assert set(identity["recipe_contract"]["ema_parameter_names"]) < set(index)
