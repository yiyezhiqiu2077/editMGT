#!/usr/bin/env python3
"""Run separately from torchrun. Its exit status never controls training."""
import argparse
import os
from pathlib import Path
import sys
import time
import resource

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dimo.monitoring import Monitor, GitHubPNGPublisher


def main():
    # Independent process bounds; never applied to a training worker.
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16*1024**2, 16*1024**2))
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("e3", "dimo"), required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--git-sha", required=True)
    p.add_argument("--repository", default="yiyezhiqiu2077/editMGT")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--once", action="store_true")
    p.add_argument("--retry-pending", action="store_true")
    p.add_argument("--max-pending", type=int, default=128)
    a = p.parse_args()
    if a.max_pending <= 0:
        p.error("max-pending must be positive")
    publisher = GitHubPNGPublisher(a.repository, f"monitor-{a.stage}-dense-{a.run_id}", a.git_sha) if a.upload else None
    monitor = Monitor(a.log, a.output, stage=a.stage, publisher=publisher, max_pending=a.max_pending)
    if a.retry_pending:
        monitor.state["attempts"] = {}
    while True:
        try:
            monitor.poll()
        except Exception as exc:
            print(f"MONITOR_LOCAL_DIAGNOSTIC: {type(exc).__name__}", file=sys.stderr, flush=True)
        if a.once:
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
