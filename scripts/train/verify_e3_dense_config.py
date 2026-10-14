#!/usr/bin/env python3
"""Separate Dense readiness; shared LoRA attestations are never rewritten."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.explicit_region.config import load_config, _load_unexpanded
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.contracts import sha256_file, audit_component_identity
from src.explicit_region.dense_runtime import validate_dense_config, readonly_reattest, audit_released_fp32
from src.explicit_region.dense_checkpoint import verify_dense_checkpoint

BASE_SHA = "36dff5ff9ad10cee3d3910eb980932ec0a251b62"


def current_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def gpu_preflight(minimum_free_gib):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    if len(visible) != 8 or len(set(visible)) != 8:
        raise RuntimeError("DENSE_REQUIRES_EIGHT_DISTINCT_VISIBLE_GPUS")
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    records = [line.split(", ") for line in raw.strip().splitlines()]
    chosen = [r for r in records if r[0] in visible or r[1] in visible]
    if len(chosen) != 8 or any("A100" not in r[2] for r in chosen):
        raise RuntimeError("DENSE_REQUIRES_EIGHT_A100_GPUS")
    # Memory and utilization are checked; an active compute process also fails
    # even if it currently has no allocated memory or low utilization.
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    used = {line.split(",")[0].strip() for line in apps.splitlines() if "," in line}
    if any(r[1] in used or int(r[5]) > 0 or int(r[4]) > 512 or
           (int(r[3]) - int(r[4])) / 1024 < minimum_free_gib for r in chosen):
        raise RuntimeError("DENSE_GPU_BUSY_OR_INSUFFICIENT_VRAM: do not preempt existing jobs")
    return chosen


def verify_training_admission(config, resume=None, *, check_gpu=True):
    validate_dense_config(config, formal=True)
    sha = current_sha()
    if os.environ.get("DENSE_EXPECTED_GIT_SHA") != sha:
        raise RuntimeError("DENSE_EXPECTED_GIT_SHA_REQUIRED_OR_MISMATCH")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, sha], cwd=ROOT, check=True)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("DENSE_WORKTREE_NOT_CLEAN")
    from src.explicit_region.fixed_corpus import verify_corpus_ready
    ready = verify_corpus_ready(config["data"]["corpus_ready"])
    if ready.get("total_rows") != 200000:
        raise RuntimeError("DENSE_D200K_NOT_READY")
    proof_path = Path(os.environ["DENSE_OUTPUT_ROOT"]) / "dense_reattestation.json"
    proof = json.loads(proof_path.read_text())
    if proof.get("status") != "PASS" or proof.get("git_sha") != sha or proof.get("config_sha256") != recipe_fingerprint(config):
        raise RuntimeError("DENSE_INDEPENDENT_REATTESTATION_MISSING_OR_STALE")
    if proof["frozen"]["train_sha256"] != sha256_file(config["data"]["manifest"]) or proof["shared_ready_sha256"] != sha256_file(config["data"]["corpus_ready"]):
        raise RuntimeError("DENSE_ATTESTED_DATA_CHANGED")
    for path, digest in proof["validation"].items():
        if sha256_file(path) != digest:
            raise RuntimeError("DENSE_VALIDATION_CHANGED")
    probe = config["validation"]["manifest"]
    if sha256_file(probe) not in proof["validation"].values():
        raise RuntimeError("DENSE_PERIODIC_PROBE_NOT_ATTESTED")
    audit_component_identity(config["model"]["repo_or_root"])
    audit = audit_released_fp32(config["model"]["repo_or_root"])
    if audit["files"] != proof.get("released_weight_hashes"):
        raise RuntimeError("DENSE_ATTESTED_RELEASED_WEIGHTS_CHANGED")
    output = Path(config["output_dir"]).resolve()
    protected = [Path(config["model"]["repo_or_root"]).resolve(), Path(config["data"]["manifest"]).parent.resolve(),
                 *(Path(p).resolve() for p in config["data"]["roots"].values())]
    if any(output == p or p in output.parents for p in protected):
        raise RuntimeError("DENSE_OUTPUT_OVERLAPS_PROTECTED_ASSETS")
    if output.exists() and any(output.iterdir()) and not resume:
        raise RuntimeError("DENSE_OUTPUT_NOT_EMPTY")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not os.access(output.parent, os.W_OK):
        raise RuntimeError("DENSE_OUTPUT_NOT_WRITABLE")
    if resume:
        verify_dense_checkpoint(resume, inference_only=False)
    if check_gpu:
        gpu_preflight(config["dense"]["minimum_free_vram_gib"])
    return {"status": "READY_TO_RUN", "git_sha": sha, "training_mode": "full_transformer",
            "gpu_smoke": "NOT_RUN", "early_checks": [1, 4, 16]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/dense_d200k_5epoch.yaml")
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--reattest", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.static:
        result = validate_dense_config(_load_unexpanded(Path(args.config)), formal=True)
    elif args.reattest:
        config = load_config(args.config)
        validate_dense_config(config, formal=True)
        sha = current_sha()
        if sha != os.environ.get("DENSE_EXPECTED_GIT_SHA"):
            raise RuntimeError("DENSE_EXPECTED_GIT_SHA_MISMATCH")
        path = Path(os.environ["DENSE_OUTPUT_ROOT"]) / "dense_reattestation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        result = readonly_reattest(config, path, git_sha=sha)
        result["released_weight_hashes"] = audit_released_fp32(config["model"]["repo_or_root"])["files"]
        from src.explicit_region.dense_checkpoint import write_json
        write_json(path, result)
    else:
        result = verify_training_admission(load_config(args.config), args.resume)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
