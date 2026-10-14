import hashlib
import json
from pathlib import Path
import tarfile
import os
import subprocess
import sys

import pytest

from src.dimo.monitoring import Monitor, GitHubPNGPublisher, read_committed, plot_curves
from scripts.report.build_experiment_report import build_report


def log(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [{"global_step": i, "committed": True, "objective": "linear", "loss_dimo": -1e-5*i,
             "loss_aux": 9., "distribution_energy": 1e-12, "student_grad_norm": None if i % 100 else .01,
             "aux_grad_norm": .2, "dimo_gradient_norm": .001} for i in range(1, 501)]
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    return path


@pytest.mark.parametrize("error", [PermissionError, TimeoutError, ConnectionError, FileNotFoundError])
def test_upload_failure_preserves_png_and_never_controls_training(tmp_path, error):
    class FailedPublisher:
        def publish(self, paths):
            raise error("injected upload fault")
    path = log(tmp_path)
    monitor = Monitor(path, tmp_path / "curves", stage="dimo", publisher=FailedPublisher())
    monitor.poll()
    assert monitor.state["pending"] == [500]
    assert monitor.state["last_error_class"] == error.__name__
    assert len(list((tmp_path / "curves").glob("*.png"))) == 2
    with path.open("a") as handle:
        handle.write(json.dumps({"global_step": 501, "committed": True})+"\n")
    assert read_committed(path, "dimo")[-1]["global_step"] == 501
    class RecoveredPublisher:
        def publish(self, paths): pass
    monitor.publisher = RecoveredPublisher()
    monitor.poll()
    assert monitor.state["uploaded"] == [500] and not monitor.state["pending"]


def test_incomplete_logs_and_noncommitted_rows_are_ignored(tmp_path):
    path = log(tmp_path)
    with path.open("a") as handle:
        handle.write('{"global_step": 501, "committed": false}\n')
        handle.write('{"global_step": 502')
    assert read_committed(path, "dimo")[-1]["global_step"] == 500


def test_assets_are_exactly_matching_png_pair(tmp_path):
    rows = read_committed(log(tmp_path), "dimo")
    paths = [tmp_path / f"{kind}_step_000500.png" for kind in ("loss", "gradient")]
    for kind, path in zip(("loss", "gradient"), paths):
        plot_curves(rows, path, stage="dimo", step=500, kind=kind)
    publisher = GitHubPNGPublisher("owner/repository", "test-run", "a"*40)
    with pytest.raises(ValueError, match="WHITELIST"):
        publisher.publish([paths[0], tmp_path / "metrics.jsonl"])


def test_report_is_partial_local_and_integrity_checked(tmp_path):
    root = tmp_path / "run"; root.mkdir()
    log(root)
    (root / "experiment_manifest.json").write_text(json.dumps({"evidence": "CPU_SYNTHETIC"}))
    (root / "model.safetensors").write_bytes(b"excluded model")
    archive = build_report({"dimo": root}, tmp_path / "report.tar.gz", stage="dimo")
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        assert "dimo/metrics.jsonl" in names and "dimo/loss.png" in names
        assert not any("safetensors" in name for name in names)
        index = json.load(bundle.extractfile("archive_manifest.json"))
        for name, digest in index.items():
            assert hashlib.sha256(bundle.extractfile(name).read()).hexdigest() == digest
    assert archive.stat().st_size < 1024*1024
    assert "NOT_AVAILABLE" in tarfile.open(archive).extractfile("summary.md").read().decode()


def test_monitor_process_crash_does_not_end_training(tmp_path):
    environment = os.environ | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    output = tmp_path / "train"
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
        "scripts/train/train_full_dense_dimo.py", "--config", "configs/dimo/full_dense_local_smoke.yaml",
        "--synthetic", "--output-dir", str(output), "--stop-at-global-step", "3"]
    training = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    monitor = subprocess.Popen([sys.executable, "scripts/monitor/live_monitor.py", "--stage", "dimo",
        "--log", str(output / "metrics.jsonl"), "--output", str(tmp_path / "curves"),
        "--run-id", "crash-test", "--git-sha", "a"*40], env=environment,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        monitor.kill(); monitor.wait(timeout=10)
        stdout, stderr = training.communicate(timeout=60)
        assert monitor.returncode != 0
        assert training.returncode == 0, stdout + stderr
        assert read_committed(output / "metrics.jsonl", "dimo")[-1]["global_step"] == 3
    finally:
        if training.poll() is None:
            training.kill(); training.wait(timeout=10)
