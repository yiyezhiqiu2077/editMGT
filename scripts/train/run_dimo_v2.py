#!/usr/bin/env python3
"""Resource-checked v2 smoke; a long run additionally needs explicit confirmation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dimo.recipe import is_v2
from src.dimo.runtime import validate_full_config, validate_output, gpu_admission, disk_budget
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.config import load_config
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dense_checkpoint import write_json

ROOT = Path(__file__).resolve().parents[2]


def validate_smoke_logs(output):
    rows = [json.loads(l) for l in (output / "metrics.jsonl").read_text().splitlines() if l.strip()]
    if len(rows) != 20 or rows[-1]["global_step"] != 20 or rows[-1]["sampler_cursor"] != 160:
        raise RuntimeError("DIMO_V2_SMOKE_BUDGET_MISMATCH")
    for row in rows:
        if (not row["committed"] or row["nan_inf_count"] != 0 or row["outside_mismatch_count"] != 0
                or row["fresh_outside_mismatch_count"] != 0 or row["student_grad_norm_postclip"] > 10.0001
                or row["student_grad_norm_embed"] != 0 or row["student_grad_norm_mlm_head"] != 0
                or row["student_signal_missing"] or row["auxiliary_signal_missing"]):
            raise RuntimeError("DIMO_V2_SMOKE_NUMERICAL_OR_ROLE_FAILURE")
    if not any((r.get("student_parameter_update_ratio_full") or 0) > 0 for r in rows):
        raise RuntimeError("DIMO_V2_SMOKE_STUDENT_DID_NOT_UPDATE")
    if not any((r.get("aux_parameter_update_ratio") or 0) > 0 for r in rows):
        raise RuntimeError("DIMO_V2_SMOKE_AUXILIARY_DID_NOT_UPDATE")
    for rank in range(4):
        report = json.loads((output / f"v2_acceptance_rank{rank}.json").read_text())
        if report["status"] != "PASS" or report["evidence"] != "REAL_GPU" or report["frozen_weights"] != "bitwise_hash_equal":
            raise RuntimeError("DIMO_V2_SMOKE_FREEZE_OR_EVIDENCE_FAILURE")
    return rows[-1]


def validate_singleton_metadata(path):
    metadata = json.loads(path.read_text())
    expected = {"nan_inf_count": 0, "number_of_transformer_forwards": 1, "outside_mismatch_count": 0,
                "r_init": 1., "cfg_scale": 1., "embedding_noise_sigma": .3}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("DIMO_V2_SMOKE_SINGLE_IMAGE_INFERENCE_FAILURE")


def prepare_inference_output(output, inference_output, *, resume=False):
    """Route full Raw/EMA exports to a separately budgeted filesystem."""
    output, inference_output = Path(output), Path(inference_output).resolve()
    if (inference_output == output.resolve() or output.resolve() in inference_output.parents
            or inference_output in output.resolve().parents):
        raise RuntimeError("DIMO_V2_INFERENCE_OUTPUT_MUST_BE_SEPARATE")
    link = output / "inference"
    if link.is_symlink():
        if link.resolve() != inference_output:
            raise RuntimeError("DIMO_V2_INFERENCE_OUTPUT_CHANGED")
        return
    if link.exists() or (inference_output.exists() and (not inference_output.is_dir()
            or (any(inference_output.iterdir()) and not resume))):
        raise RuntimeError("DIMO_V2_INFERENCE_OUTPUT_EXISTS")
    inference_output.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    link.symlink_to(inference_output, target_is_directory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dimo/dimo_v2_mixed20k.yaml")
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inference-output-dir", type=Path)
    parser.add_argument("--smoke-proof", type=Path)
    parser.add_argument("--confirm-long-run", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_full_config(config)
    if not is_v2(config):
        raise RuntimeError("DIMO_V2_RUNNER_REQUIRES_JOINT_RECIPE")
    output = Path(args.output_dir).resolve()
    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    signature = recipe_fingerprint({k: v for k, v in config.items() if k != "output_dir"})
    if args.mode == "train":
        if not args.confirm_long_run or args.smoke_proof is None:
            parser.error("a long run requires --confirm-long-run and --smoke-proof")
        proof = json.loads(args.smoke_proof.read_text())
        if proof.get("status") != "PASS" or proof.get("config_without_output_sha256") != signature or proof.get("git_sha") != code_sha:
            raise RuntimeError("DIMO_V2_GPU_SMOKE_PROOF_MISMATCH")
        proof_root = Path(proof["output_dir"])
        for path, digest in proof["evidence_files"].items():
            if sha256_file(proof_root / path) != digest:
                raise RuntimeError("DIMO_V2_GPU_SMOKE_EVIDENCE_CHANGED")
        if output == proof_root.resolve():
            raise RuntimeError("DIMO_V2_TRAIN_MUST_USE_SEPARATE_SMOKE_OUTPUT")
    prepared_only = (args.inference_output_dir is not None and output.is_dir()
        and list(output.iterdir()) == [output / "inference"] and (output / "inference").is_symlink()
        and (output / "inference").resolve() == args.inference_output_dir.resolve()
        and not any((output / "inference").iterdir()))
    if output.exists() and any(output.iterdir()) and args.resume is None and not prepared_only:
        raise RuntimeError("DIMO_V2_OUTPUT_EXISTS_USE_EXPLICIT_RESUME")
    # This read-only check happens before torchrun creates any GPU contexts.
    gpu_admission(magicbrush=True, allow_other_jobs=False)
    if args.inference_output_dir:
        validate_output(config | {"output_dir": str(output)})
        validate_output(config | {"output_dir": str(args.inference_output_dir)})
        prepare_inference_output(output, args.inference_output_dir, resume=args.resume is not None)
    from safetensors import safe_open
    import math
    teacher = Path(config["teacher_checkpoint"])
    index = json.loads((teacher / "model.safetensors.index.json").read_text())["weight_map"]
    total = trainable = 0
    for filename in set(index.values()):
        with safe_open(teacher / filename, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                size = math.prod(handle.get_slice(name).get_shape())
                total += size
                if not name.startswith(("embed.", "mlm_layer.")):
                    trainable += size
    disk_budget(total, config, output, trainable_parameters=trainable,
                stop_at=20 if args.mode == "smoke" else config["max_optimizer_steps"])
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4",
        str(ROOT / "scripts/train/train_full_dense_dimo.py"), "--config", args.config, "--output-dir", str(output)]
    if args.mode == "smoke":
        command += ["--stop-at-global-step", "20"]
    if args.resume:
        command += ["--resume", str(args.resume)]
    subprocess.run(command, cwd=ROOT, check=True)
    if args.mode != "smoke":
        return
    last = validate_smoke_logs(output)
    from src.dimo.full_checkpoint import verify
    checkpoint = output / "checkpoints/checkpoint-20"
    for weights in ("student", "ema"):
        verify(checkpoint, full=True, weights=weights)
    # Prove the GPU checkpoint can be resumed, without another optimizer update.
    subprocess.run(command + ["--resume", str(checkpoint)], cwd=ROOT, check=True)
    from src.dimo.dev_protocol import validate_plan
    plan = load_config(config["evaluation"]["plan"])
    row = validate_plan(plan)[0]
    environment = os.environ | {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")[0]}
    for weights in ("student", "ema"):
        subprocess.run([sys.executable, str(ROOT / "scripts/eval/sample_dimo_editing.py"),
            "--source-image", row["source"], "--instruction", row["instruction_en"], "--edit-region-mask", row["mask"],
            "--student-checkpoint", str(checkpoint), "--teacher-checkpoint", config["teacher_checkpoint"],
            "--model-root", config["model"]["repo_or_root"], "--weights", weights,
            "--output", str(output / f"smoke_{weights}.png")], cwd=ROOT, env=environment, check=True)
        validate_singleton_metadata(output / f"smoke_{weights}.png.json")
    files = ["metrics.jsonl", "checkpoints/checkpoint-20/COMPLETED.json", "smoke_student.png", "smoke_ema.png"]
    files += ["smoke_student.png.json", "smoke_ema.png.json"]
    files += [f"v2_acceptance_rank{i}.json" for i in range(4)]
    write_json(output / "GPU_SMOKE_PASS.json", {"status": "PASS", "git_sha": code_sha,
        "config_without_output_sha256": signature, "output_dir": str(output), "steps": 20,
        "sample_consumptions": 160, "clip_fraction": last["student_clip_fraction"],
        "resume": "checkpoint_loaded_without_additional_update", "raw_ema_single_image_inference": "PASS",
        "evidence_files": {path: sha256_file(output / path) for path in files}})


if __name__ == "__main__":
    main()
