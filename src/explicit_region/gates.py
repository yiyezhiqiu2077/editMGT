"""One fail-closed, stage-aware gate shared by training/evaluation launchers.

READY attests data integrity, not human approval. A plan approval is not a corpus
review. This module only reads external review evidence; it never creates it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .config import load_config
from .contracts import sha256_file
from .fixed_corpus import verify_corpus_ready
from .selection import committed_config_files, config_identity, file_identity, object_sha256, verify_preregistration

ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_CONFIGS = (
    "configs/train/cluster_8g_diagnostic_e1.yaml", "configs/train/cluster_8g_diagnostic_e2.yaml",
)
SMOKE_CONFIGS = (
    "configs/train/cluster_8g_smoke.yaml", "configs/train/cluster_8g_smoke_fresh10.yaml",
    "configs/train/cluster_8g_smoke_resume20.yaml",
)


def repository_identity(repo_root=ROOT) -> dict:
    root = Path(repo_root)
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout
    digest = hashlib.sha256(git("diff", "HEAD", "--binary"))
    for relative in sorted(p.decode() for p in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if p):
        path = root / relative
        if path.is_file():
            digest.update(relative.encode())
            digest.update(path.read_bytes())
    return {"git_sha": git("rev-parse", "HEAD").decode().strip(),
            "git_dirty": bool(git("status", "--porcelain")),
            "git_dirty_diff_sha256": digest.hexdigest(), "uv_lock_sha256": sha256_file(root / "uv.lock")}


def build_gate_identity(corpus_ready, *, repo_root=ROOT) -> dict:
    repo = repository_identity(repo_root)
    return {"schema": "stage-gate-identity-v1", "corpus_ready_sha256": sha256_file(corpus_ready),
            **{key: repo[key] for key in ("git_sha", "git_dirty_diff_sha256", "uv_lock_sha256")},
            "smoke_configs": {relative: config_identity(Path(repo_root) / relative) for relative in SMOKE_CONFIGS}}


def verify_human_approval(path, corpus_ready) -> dict:
    if not path:
        raise ValueError("CORPUS_APPROVAL is required: a real external human corpus review, not plan approval")
    approval = json.loads(Path(path).read_text(encoding="utf-8"))
    if (approval.get("schema") != "fixed200k-human-approval-v1"
            or approval.get("actor_type") != "human" or approval.get("decision") != "go"):
        raise ValueError("corpus requires explicit external human go approval")
    for field in ("approver", "approved_at", "evidence"):
        if not isinstance(approval.get(field), str) or not approval[field].strip():
            raise ValueError(f"human approval missing {field}")
    if approval.get("corpus_ready_sha256") != sha256_file(corpus_ready):
        raise ValueError("human approval is for a different READY corpus")
    return file_identity(path) | {"approver": approval["approver"], "approved_at": approval["approved_at"]}


def diagnostic_recipe_identity(config, *, repo_root=ROOT) -> dict:
    """Only the two exact committed 2-update branch checks, never arbitrary short runs."""
    if not isinstance(config, dict):
        raise ValueError("diagnostic stage requires --train-config with an exact whitelisted recipe")
    for index, relative in enumerate(DIAGNOSTIC_CONFIGS, start=1):
        path = Path(repo_root) / relative
        if config != load_config(path):
            continue
        corruption = {"mode": "full_target", "roi_probability": 0.0, "full_roi_mask_probability": 0.0,
                      "persistent_conditioning": False} if index == 1 else {
                      "mode": "roi_hardlock", "roi_probability": 1.0, "full_roi_mask_probability": .15,
                      "persistent_conditioning": False}
        expected_output = (Path(os.environ["EDITMGT_OUTPUT_ROOT"]) / f"branch-preflight-e{index}-2updates").absolute()
        actual_output = Path(config["output_dir"]).absolute()
        if (config.get("run_type") != "diagnostic" or config.get("max_optimizer_steps") != 2
                or config.get("checkpoint_steps") != [2] or config.get("scheduler_horizon_steps") != 6250
                or config.get("warmup_steps") != 200 or config.get("resolution") != 1024
                or config.get("batch_per_gpu") != 1 or config.get("gradient_accumulation") != 4
                or config.get("data", {}).get("name") != "fixed_200k"
                or config.get("component_dtypes", {}).get("vqvae") != "fp32"
                or config.get("torch_compile") is not False or config.get("validation", {}).get("enabled") is not False
                or config.get("corruption") != corruption or config.get("lora", {}).get("scope") != "both"
                or config.get("optimizer", {}).get("learning_rate") != .00003
                or actual_output != expected_output or actual_output.resolve() != actual_output):
            raise ValueError("diagnostic recipe no longer matches approved two-update E1/E2 contract")
        for name in ("e1", "e2", "e3", "e4", "lr_1e-5", "lr_3e-5", "lr_5e-5"):
            candidate_output = Path(load_config(Path(repo_root) / f"configs/train/cluster_8g_{name}.yaml")["output_dir"]).resolve()
            if actual_output.resolve() == candidate_output or candidate_output in actual_output.resolve().parents:
                raise ValueError("diagnostic output overlaps a formal/LR candidate")
        return config_identity(path) | {"committed_files": committed_config_files(path, repo_root),
                                      "successful_optimizer_updates": 2, "ranking_eligible": False}
    raise ValueError("diagnostic stage requires an exact whitelisted recipe")


def enforce_stage_gate(stage, *, corpus_ready, corpus_approval, smoke_report=None,
                       preregistration=None, repo_root=ROOT, diagnostic_config=None) -> dict:
    if stage not in ("diagnostic", "smoke", "probe", "formal", "eval"):
        raise ValueError(f"unsupported gated stage: {stage}")
    verify_corpus_ready(corpus_ready)
    approval = verify_human_approval(corpus_approval, corpus_ready)
    identity = build_gate_identity(corpus_ready, repo_root=repo_root)
    evidence = {"stage": stage, "status": "PASS", "gate_identity": identity, "corpus_approval": approval}
    if stage == "diagnostic":
        evidence["diagnostic_recipe"] = diagnostic_recipe_identity(diagnostic_config, repo_root=repo_root)
        if diagnostic_config["data"]["corpus_ready"] != str(corpus_ready):
            raise ValueError("diagnostic READY path mismatch")
    if stage not in ("smoke", "diagnostic"):
        if not smoke_report:
            raise ValueError("matching fixed200k smoke PASS is required before probes/formal/eval")
        report = json.loads(Path(smoke_report).read_text(encoding="utf-8"))
        required_checks = (
            "sample_sequence_match", "geometry_seed_sequence_match", "corruption_seed_sequence_match",
            "learning_rate_sequence_match", "finite_state", "exact_weights_match", "optimizer_match",
            "scheduler_match", "quality_state_match", "sampler_cursor_match", "rank_rng_match", "exact_metrics_match",
        )
        if (report.get("schema") != "fixed200k-smoke-verification-v2" or report.get("status") != "PASS"
                or report.get("gate_identity") != identity or report.get("samples") != 640
                or report.get("unique_samples") != 640
                or any(report.get(key) is not True for key in required_checks)
                or set(report.get("checkpoint_evidence", {})) != {"10", "20"}):
            raise ValueError("smoke PASS identity/state evidence is missing/stale/mismatched")
        if not preregistration:
            raise ValueError("committed frozen preregistration is required before probes/formal/eval")
        frozen = verify_preregistration(preregistration, repo_root=repo_root, corpus_ready=corpus_ready)
        evidence.update(smoke_report=file_identity(smoke_report),
                        preregistration_sha256=frozen["identity_sha256"])
    return evidence


def training_stage(config, *, repo_root=ROOT) -> str:
    """Only exact known resolved smoke recipes can bypass the smoke PASS gate."""
    for relative in SMOKE_CONFIGS:
        if config == load_config(Path(repo_root) / relative):
            return "smoke"
    for relative in DIAGNOSTIC_CONFIGS:
        if config == load_config(Path(repo_root) / relative):
            return "diagnostic"
    name = Path(config.get("output_dir", "")).name
    if name.startswith("lr-probe-"):
        return "probe"
    # Unrecognized long/short configs never become smoke by reducing step count.
    return "formal"


def enforce_training_gate(config, *, repo_root=ROOT) -> dict:
    if config.get("data", {}).get("name") != "fixed_200k":
        raise ValueError("training gate is defined only for fixed_200k")
    stage = training_stage(config, repo_root=repo_root)
    ready = config["data"]["corpus_ready"]
    evidence = enforce_stage_gate(
        stage, corpus_ready=ready, corpus_approval=os.environ.get("CORPUS_APPROVAL"),
        smoke_report=os.environ.get("FIXED200K_SMOKE_REPORT"),
        preregistration=os.environ.get("EDITMGT_PREREGISTRATION"), repo_root=repo_root,
        diagnostic_config=config if stage == "diagnostic" else None,
    )
    if stage not in ("smoke", "diagnostic"):
        frozen = verify_preregistration(os.environ["EDITMGT_PREREGISTRATION"], repo_root=repo_root, corpus_ready=ready)
        digest = object_sha256(config)
        group = frozen["candidate_sets"]["lr" if stage == "probe" else "formal"]
        if not any(candidate["train_config"]["resolved_sha256"] == digest for candidate in group):
            raise ValueError("training config is not a frozen candidate recipe; review/commit/preregister explicitly")
    return evidence


def main():
    parser = argparse.ArgumentParser(description="Inspect/enforce stage evidence without starting training")
    parser.add_argument("--stage", choices=("diagnostic", "smoke", "probe", "formal", "eval"))
    parser.add_argument("--train-config")
    parser.add_argument("--corpus-ready")
    parser.add_argument("--corpus-approval", default=os.environ.get("CORPUS_APPROVAL"))
    parser.add_argument("--smoke-report", default=os.environ.get("FIXED200K_SMOKE_REPORT"))
    parser.add_argument("--preregistration", default=os.environ.get("EDITMGT_PREREGISTRATION"))
    parser.add_argument("--repo-root", default=str(ROOT))
    args = parser.parse_args()
    try:
        if args.train_config:
            result = enforce_training_gate(load_config(args.train_config), repo_root=args.repo_root)
        else:
            if not args.stage or not args.corpus_ready:
                parser.error("--stage and --corpus-ready are required without --train-config")
            result = enforce_stage_gate(args.stage, corpus_ready=args.corpus_ready, corpus_approval=args.corpus_approval,
                                        smoke_report=args.smoke_report, preregistration=args.preregistration,
                                        repo_root=args.repo_root)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, sort_keys=True))
        raise SystemExit(3)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
