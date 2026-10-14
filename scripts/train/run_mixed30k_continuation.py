#!/usr/bin/env python3
"""Screen/translate/freeze -> real 4-GPU SFT acceptance -> five epochs -> DEV."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.train.run_magicbrush_two_stage import Run, memory_peaks
from scripts.train.compare_dense_resume import compare
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.config import load_config
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dense_checkpoint import verify_dense_checkpoint, write_json
from src.explicit_region.dense_evaluation import assert_matching_predictions, comparison_visual, paired_comparison
from src.explicit_region.formal_pipeline import write_json_atomic
from src.explicit_region.mixed_experiment import COUNTS, MODE, ROWS, verify_mixed_ready
from src.dimo.runtime import gpu_admission


def choose(candidates):
    eligible = [row for row in candidates if row["paired_inside"]["mean_delta"] <= -.005
                and row["paired_inside"]["bootstrap_95_ci"][1] < 0
                and row["paired_outside"]["mean_delta"] <= .01]
    return min(eligible, key=lambda row: (row["inside_mean"], row["step"])) if eligible else None


def recover_pretraining(root):
    """Explicit continuation only; never replay accepted or trained experiments."""
    marker = root / "PIPELINE_STARTED.json"
    state = json.loads((root / "STATUS.json").read_text())
    if (not marker.exists() or state.get("status") != "STOPPED_HARD_ERROR"
            or state.get("phase") not in ("parallel_download_screen_translate", "mirror_downloads", "screen_translate_freeze")
            or (root / "data/MIXED_READY.json").exists()
            or any((root / "sft").rglob("checkpoint-*"))):
        raise RuntimeError("MIXED_RECOVERY_REQUIRES_FAILED_PRETRAINING_ONLY")
    proc = Path(f"/proc/{state.get('pid', 0)}")
    if proc.exists() and not proc.joinpath("stat").read_text().split(") ")[1].startswith("Z"):
        raise RuntimeError("MIXED_PREVIOUS_COORDINATOR_STILL_RUNNING")
    archive = root / "recovery" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(root / "STATUS.json", archive / "STATUS.json")
    marker.replace(archive / marker.name)
    write_json_atomic(archive / "RECOVERY.json", {"explicit_pretraining_recovery": True,
        "previous_failure": state, "total_rows": ROWS, "dataset_counts": COUNTS,
        "reason": "authorized continuation after repairing failed pretraining tooling; current quotas recorded above",
        "training_checkpoints_present": False, "old_dimo_restart": False})


def verify_skip_sft_acceptance(root):
    path = root / 'USER_SKIP_SFT_GPU_ACCEPTANCE.json'
    record = json.loads(path.read_text())
    if (record.get('schema') != 'mixed20k-user-skip-sft-gpu-acceptance-v1'
            or not record.get('user_instruction', '').strip()
            or record.get('skip_sft_gpu_acceptance') is not True
            or (record.get('sft_epochs'), record.get('sft_steps'), record.get('data_rows')) != (5, 6250, ROWS)):
        raise RuntimeError('MIXED_SFT_ACCEPTANCE_SKIP_REQUIRES_EXPLICIT_USER_REQUEST')
    return sha256_file(path)


def recover_ready_training(root):
    """Explicit switch from stopped smoke/acceptance to untouched long training."""
    authorization = verify_skip_sft_acceptance(root)
    if not (root / 'data/MIXED_READY.json').is_file():
        raise RuntimeError('MIXED_READY_DIRECT_START_REQUIRES_VERIFIED_DATA')
    train = root / 'sft/train'
    if train.exists() and any(train.iterdir()):
        raise RuntimeError('MIXED_DIRECT_START_CANNOT_REPLAY_EXISTING_LONG_TRAINING')
    marker = root / 'PIPELINE_STARTED.json'
    if not marker.exists(): return
    state = json.loads((root / 'STATUS.json').read_text())
    if state.get('status') not in ('STOPPED_HARD_ERROR', 'USER_PAUSED'):
        raise RuntimeError('MIXED_DIRECT_START_REQUIRES_STOPPED_PREDECESSOR')
    proc = Path(f"/proc/{state.get('pid', 0)}")
    if proc.exists() and not proc.joinpath('stat').read_text().split(') ')[1].startswith('Z'):
        raise RuntimeError('MIXED_PREVIOUS_COORDINATOR_STILL_RUNNING')
    verify_mixed_ready(load_config(root / 'configs/sft.yaml'))
    archive = root / 'recovery' / time.strftime('%Y%m%dT%H%M%SZ_direct_sft', time.gmtime())
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(root / 'STATUS.json', archive / 'STATUS.json')
    marker.replace(archive / marker.name)
    write_json_atomic(archive / 'RECOVERY.json', {'reason': 'explicit user skip of SFT GPU acceptance',
        'authorization_sha256': authorization, 'previous_state': state,
        'verified_data_reused': True, 'existing_smoke_outputs_retained': True,
        'long_training_replayed': False, 'old_dimo_restart': False})


class MixedRun(Run):
    def __init__(self, args):
        super().__init__(args)
        for key in list(self.env):
            if key.lower().endswith("_proxy"):
                self.env.pop(key)
        self.env.update(HF_ENDPOINT="https://hf-mirror.com", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if getattr(args, 'allow_shared_gpus', False): self.env['EDITMGT_ALLOW_SHARED_A800'] = '1'

    def train(self, stage, output, stop=None, resume=None):
        self.phase = "waiting_for_idle_four_gpus"
        while True:
            self.assert_code()
            try:
                gpu_admission(magicbrush=True,
                    allow_other_jobs=getattr(getattr(self, 'a', None), 'allow_shared_gpus', False))
                break
            except RuntimeError as exc:
                if "GPU_HAS_OTHER_JOB" not in str(exc): raise
                self.state("WAITING_FOR_IDLE_GPUS", reason=str(exc), next_stage=stage,
                           next_output=str(output), next_stop=stop)
                time.sleep(30)
        super().train(stage, output, stop, resume)

    def assert_code(self):
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if sha != self.sha or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
            raise RuntimeError("MIXED_PIPELINE_CODE_CHANGED")

    def wait_downloads(self):
        self.phase = "mirror_downloads"; self.state("WAITING_FOR_DOWNLOADS")
        while True:
            path = self.root / "DOWNLOAD_STATUS.json"
            if path.exists() and json.loads(path.read_text())["status"] == "COMPLETE":
                return
            pid = int((self.root / "DOWNLOAD_PID").read_text())
            proc = Path(f"/proc/{pid}")
            if not proc.exists() or proc.joinpath("stat").read_text().split(") ")[1].startswith("Z"):
                raise RuntimeError("MIXED_MIRROR_DOWNLOADER_STOPPED; inspect coordinator_logs/download.log")
            if b"download_mixed30k_mirror.py" not in proc.joinpath("cmdline").read_bytes():
                raise RuntimeError("MIXED_MIRROR_PID_IDENTITY_CHANGED")
            time.sleep(30)

    def execute(self):
        self.assert_code()
        if subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() != "exp/e3-dense-fp32-v1":
            raise RuntimeError("MIXED_EXISTING_MAINTENANCE_BRANCH_REQUIRED")
        cpu = json.loads((self.root / "admission/CPU_ACCEPTANCE.json").read_text())
        if cpu.get("status") != "PASS" or cpu.get("git_sha") != self.sha:
            raise RuntimeError("MIXED_CURRENT_CPU_ACCEPTANCE_REQUIRED")
        if (self.root / "PIPELINE_STARTED.json").exists():
            raise RuntimeError("MIXED_PIPELINE_ALREADY_STARTED")
        write_json(self.root / "PIPELINE_STARTED.json", {"git_sha": self.sha, "scope": MODE, "total_rows": ROWS,
                   "epoch7_checkpoint": str(self.a.previous_experiment / "sft/train/checkpoint-3843"),
                   "additional_epochs": 5, "release_upload": False, "old_dimo_restart": False})
        if self.a.start_from_ready:
            self.phase = 'reusing_verified_frozen_20k'; self.state('RUNNING')
        else:
            self.phase = "parallel_download_screen_translate"; self.state("RUNNING")
            self.command([sys.executable, ROOT / "scripts/data/incremental_mixed_preparation.py",
                          "--experiment-root", self.root], "incremental_preparation")
            self.wait_downloads(); self.assert_code()
            self.phase = "screen_translate_freeze"; self.state("RUNNING")
            self.command([sys.executable, ROOT / "scripts/data/prepare_mixed30k_corpus.py", "--experiment-root",
                          self.root, "--previous-experiment", self.a.previous_experiment], "prepare_corpus")
        self.assert_code()
        sft = load_config(self.root / "configs/sft.yaml"); ready = verify_mixed_ready(sft)
        # Four smoke/resume states + five epoch candidates + one atomic save + metrics/images margin.
        parameters = 1009307648
        required = int((4 * 12 + 5 * 12 + 12) * parameters * 1.12) + 16 * 1024 ** 3
        free = shutil.disk_usage(self.root).free
        write_json_atomic(self.root / "admission/SFT_DISK_BUDGET.json", {"status": "PASS" if free >= required else "FAIL",
                          "free_bytes": free, "required_bytes": required, "retention": "all newly created SFT checkpoints"})
        if free < required:
            raise RuntimeError("MIXED_SFT_DISK_BUDGET_FAILED")
        baseline = verify_dense_checkpoint(sft["initial_dense_checkpoint"])
        old = Path("/data01/zhangyuyang/editmgt_eval_20261010")
        old_plan = json.loads((self.a.previous_experiment / "evaluation_plan.json").read_text())
        plan = dict(old_plan, schema="mixed20k-sft-evaluation-v1", selection_scope=MODE,
                    baseline_checkpoint=sft["initial_dense_checkpoint"], teacher_checkpoint=str(self.root / "selected_teacher"),
                    train_manifest=sft["data"]["manifest"], dev_manifest=str(self.root / "data/dev_full527.jsonl"),
                    dev32_manifest=str(self.root / "data/dev32.jsonl"), candidate_steps=sft["checkpoint_steps"],
                    candidate_checkpoints={str(step): str(self.root / "sft/train" / f"checkpoint-{step}") for step in sft["checkpoint_steps"]})
        rules = {"candidate_epochs": [1, 2, 3, 4, 5], "baseline": "previous_dense_sft_epoch7",
                 "minimum_inside_lpips_improvement": .005, "inside_ci_upper_strictly_negative": True,
                 "maximum_outside_lpips_degradation": .01, "tie_break": "earlier_step",
                 "no_eligible_checkpoint": "STOP_BEFORE_DIMO", "bootstrap_unit": "source_group_cluster",
                 "semantic_quality": "requires case review; LPIPS alone does not establish instruction success",
                 "statistical_scope": "exploratory; five candidates; individual 95% CIs"}
        plan["selection_rules"] = rules
        write_json(self.root / "evaluation_plan.json", plan)
        prereg = {"schema": "mixed20k-selection-prereg-v1", "git_sha": self.sha, "rules": rules,
                  "baseline_checkpoint_sha256": baseline["checkpoint_sha256"], "plan_sha256": recipe_fingerprint(plan),
                  "train_manifest_sha256": ready["train_manifest_sha256"],
                  "dev_manifest_sha256": sha256_file(plan["dev_manifest"])}
        write_json(self.root / "selection_preregistration.json", prereg)
        smoke = self.root / "sft/smoke2";fresh = self.root / "sft/fresh20";resumed = self.root / "sft/resume20"
        self.assert_code()
        if self.a.skip_sft_gpu_acceptance:
            authorization = verify_skip_sft_acceptance(self.root)
            acceptance = {'status': 'SKIPPED_BY_EXPLICIT_USER_REQUEST', 'evidence': 'NOT_TESTED',
                          'authorization_sha256': authorization, 'bitwise_resume_equivalence': 'NOT_TESTED',
                          'existing_smoke_outputs_retained': True}
        else:
            self.train("sft", smoke, 2);self.train("sft", fresh, 20);self.train("sft", resumed, 10)
            self.train("sft", resumed, 20, resumed / "checkpoint-10")
            acceptance = compare(fresh / "checkpoint-20", resumed / "checkpoint-20", fresh_log=fresh / "train_metrics.jsonl",
                                 interrupted_logs=[resumed / "train_metrics.jsonl"])
            acceptance.update(evidence="REAL_4_A800", gpu_peak_allocated_bytes={"smoke2": memory_peaks(smoke, "sft"),
                "fresh20": memory_peaks(fresh, "sft"), "resume20": memory_peaks(resumed, "sft")})
        write_json(self.root / "admission/SFT_CUDA_ACCEPTANCE.json", acceptance)
        train = self.root / "sft/train"
        self.monitor("sft", train); self.assert_code(); self.train("sft", train)
        self.phase = "four_gpu_full_dev_evaluation";self.state("RUNNING")
        jobs = [("baseline_epoch7", sft["initial_dense_checkpoint"])] + [(str(step), plan["candidate_checkpoints"][str(step)])
                for step in sft["checkpoint_steps"]]
        def queue(gpu):
            for label, checkpoint in jobs[gpu::4]:
                self.eval_teacher(checkpoint, self.root / "sft/dev_evaluation" / label, gpu)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(queue, range(4)))
        evaluation = self.root / "sft/dev_evaluation"
        baseline_metrics = json.loads((evaluation / "baseline_epoch7/metrics.json").read_text())
        candidates = []
        for step in sft["checkpoint_steps"]:
            output = evaluation / str(step)
            paths = [evaluation / "baseline_epoch7/predictions.jsonl", output / "predictions.jsonl"]
            assert_matching_predictions(paths, expected_seeds=[0, 1])
            metric = json.loads((output / "metrics.json").read_text())
            pairs = paired_comparison(metric, baseline_metrics, strict_groups=True)["magicbrush_dev"]
            identity = verify_dense_checkpoint(plan["candidate_checkpoints"][str(step)])
            row = {"step": step, "checkpoint": plan["candidate_checkpoints"][str(step)],
                   "checkpoint_sha256": identity["checkpoint_sha256"],
                   "inside_mean": sum(r["metrics"]["inside_masked_lpips"] for r in metric["per_sample_after_seed_mean"]) / 527,
                   "paired_inside": pairs["inside_masked_lpips"], "paired_outside": pairs["outside_masked_lpips"],
                   "all_paired_metrics": pairs}
            candidates.append(row)
            write_json(output / "paired_to_epoch7.json", pairs)
            comparison_visual({"Epoch7": paths[0], "Continued SFT": paths[1]}, output / "comparison.jpg", count=8)
        best = choose(candidates)
        result = {"schema": "mixed20k-sft-quality-result-v1", "git_sha": self.sha, "candidates": candidates,
                  "selection_preregistration": prereg, "selected": best, "formal_teacher": False,
                  "test_read": False, "instruction_success": "NOT_CONFIRMED_BY_LPIPS_ALONE"}
        write_json(self.root / "SFT_QUALITY_RESULT.json", result)
        report = ["# Total20K including MagicBrush: epoch7 continuation", "", f"Code: {self.sha}",
                  f"Train: {ROWS} total; dataset counts: {COUNTS}.",
                  "Five additional epochs; new optimizer; LR 2e-6; FP32 parameters/gradients/moments, BF16 forward.",
                  "DEV: unchanged 527 samples / 266 groups, seeds [0,1], 12 steps / CFG5.",
                  "InterEdit: fixed early one asset shard; guidance masks; not a global representative sample.",
                  "Translations: pinned NLLB, automatic QA/backfill; human semantic review pending; TEST unused.", "",
                  "| Added epoch | Inside LPIPS | Paired delta | 95% group CI | Outside delta |",
                  "|---|---:|---:|---|---:|"]
        for epoch, row in enumerate(candidates, 1):
            report.append(f"| {epoch} | {row['inside_mean']:.6f} | {row['paired_inside']['mean_delta']:.6f} | "
                          f"{row['paired_inside']['bootstrap_95_ci']} | {row['paired_outside']['mean_delta']:.6f} |")
        report.extend(["", "PSNR/SSIM, DINO/CLIP, no-edit diagnostics, paired statistics and fixed cases remain in each evaluation directory.",
                       "Loss and LPIPS alone do not establish semantic quality improvement.",
                       "Selected candidate: " + (best["checkpoint"] if best else "NONE; stop before DiMO")])
        (self.root / "REPORT.md").write_text("\n".join(report) + "\n")
        self.phase = "sft_quality_review_before_new_dimo"
        self.state("SFT_EVALUATED_DIMO_PENDING" if best else "STOPPED_NO_QUALITY_IMPROVEMENT",
                   selected_checkpoint=best["checkpoint"] if best else None, old_dimo_still_paused=True,
                   new_dimo_training_started=False, report=str(self.root / "REPORT.md"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--previous-experiment", type=Path, required=True)
    parser.add_argument("--recover-pretraining", action="store_true")
    parser.add_argument('--start-from-ready', action='store_true', help='reuse the verified immutable 20K without preparing it again')
    parser.add_argument('--skip-sft-gpu-acceptance', action='store_true', help='explicit user skip of smoke/fresh/resume acceptance')
    parser.add_argument("--allow-shared-gpus", action="store_true",
                        help="explicit user authorization to coexist with current GPU jobs")
    args = parser.parse_args();args.upload_curves = False;args.monitor_token_file = None
    args.experiment_root = args.experiment_root.resolve();args.previous_experiment = args.previous_experiment.resolve()
    lock = (args.experiment_root / ".coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = MixedRun(args)
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("mixed pipeline interrupted")))
    try:
        if args.recover_pretraining:
            recover_pretraining(args.experiment_root)
        if args.start_from_ready:
            recover_ready_training(args.experiment_root)
        run.execute()
    except BaseException as exc:
        run.failed.set();run.stop_children()
        run.state("STOPPED_HARD_ERROR", error_class=type(exc).__name__, error=str(exc),
                  automatic_failure_retry=False, automatic_recipe_changes=False)
        raise
    finally:
        for monitor in run.monitors:
            monitor.terminate()
        lock.close()


if __name__ == "__main__":
    main()
