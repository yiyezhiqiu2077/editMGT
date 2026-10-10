"""Frozen configuration validation, stage memory evidence and cheap diagnostics."""
import json
import math
from pathlib import Path
import shutil

import torch


def validate_full_config(config, *, synthetic=False):
    if config["model_roles"]["backend"] != "full_dense" or config["distributed"]["backend"] != "ddp":
        raise RuntimeError("FULL_DIMO_REQUIRES_FULL_DENSE_DDP")
    expected = {"student_learning_rate": 1e-6, "auxiliary_learning_rate": 1e-6,
                "betas": [.9, .999], "weight_decay": 0.0, "scheduler": "constant", "warmup_steps": 0}
    if any(config["optimizer"].get(k) != v for k, v in expected.items()):
        raise RuntimeError("FULL_DIMO_OPTIMIZER_RECIPE_CHANGED")
    if config["ema"]["decay"] != .9995 or not config["ema"]["enabled"]:
        raise RuntimeError("FULL_DIMO_EMA_RECIPE_CHANGED")
    if config["surrogate"] != {"implementation": "linear", "version": "roi-vocabulary-sum-v1"}:
        raise RuntimeError("FULL_DIMO_SURROGATE_NOT_FROZEN")
    if (config["batch_per_gpu"], config["gradient_accumulation"], config["epochs"]) != (1, 1, 1):
        raise RuntimeError("FULL_DIMO_BATCH_RECIPE_CHANGED")
    if not synthetic and (config["max_optimizer_steps"], config["distributed"]["world_size"], config["data"]["expected_rows"], config["formal_ready"]) != (25000, 8, 200000, True):
        raise RuntimeError("FULL_DIMO_FORMAL_HORIZON_OR_DATA_CHANGED")
    if synthetic and (config["formal_ready"] or config["data"]["name"] != "synthetic"):
        raise RuntimeError("SYNTHETIC_CANNOT_BE_FORMAL")
    if config["distillation"]["mode"] != "FKL" or config["auxiliary"]["updates_per_student"] != 1:
        raise RuntimeError("FULL_DIMO_MATH_RECIPE_CHANGED")
    if config["component_dtypes"] != {"teacher": "fp32", "student": "fp32", "auxiliary": "fp32",
            "gradients": "fp32", "adamw_moments": "fp32", "ema": "fp32", "forward": "bf16_autocast"}:
        raise RuntimeError("FULL_DIMO_PRECISION_RECIPE_CHANGED")
    if config["initialization"] != {"mask_ratio": .5, "random_tokens_inside_roi": True}:
        raise RuntimeError("FULL_DIMO_INITIALIZATION_RECIPE_CHANGED")
    if config["sampling"] != {"temperature": 1., "top_k": 0, "top_p": 0.}:
        raise RuntimeError("FULL_DIMO_SAMPLING_RECIPE_CHANGED")
    if config["pseudo_forward"] != {"teacher_ratio_mode": "cosine", "auxiliary_ratio_mode": "cosine", "ratio_min": .02, "ratio_max": .98}:
        raise RuntimeError("FULL_DIMO_PSEUDO_MASK_RECIPE_CHANGED")
    for key in ("teacher_temperature", "auxiliary_temperature", "teacher_cfg", "auxiliary_cfg", "student_cfg", "auxiliary_train_cfg"):
        if config["distillation"].get(key, 1.) != 1.:
            raise RuntimeError(f"FULL_DIMO_DISTRIBUTION_RECIPE_CHANGED: {key}")
    if config["auxiliary"]["soft_target_weight"] != 0 or config["auxiliary"]["batch_policy"] != "same_batch" or config["gt_anchor"]["weight"] != 0:
        raise RuntimeError("FULL_DIMO_TARGET_RECIPE_CHANGED")
    if not synthetic and config["embedding_perturbation"] != {"enabled": True, "sigma": .3, "scope": "target_roi_only"}:
        raise RuntimeError("FULL_DIMO_EMBEDDING_NOISE_RECIPE_CHANGED")


def validate_output(config, *, synthetic=False):
    output = Path(config["output_dir"]).resolve()
    if synthetic:
        return output
    protected = [Path(config["teacher_checkpoint"]).resolve(), Path(config["model"]["repo_or_root"]).resolve(),
                 Path(config["data"]["manifest"]).parent.resolve(), *(Path(root).resolve() for root in config["data"]["roots"].values())]
    if any(output == p or p in output.parents or output in p.parents for p in protected):
        raise RuntimeError("FULL_DIMO_OUTPUT_OVERLAPS_PROTECTED_ASSETS")
    return output


def gpu_admission():
    import csv
    import os
    import subprocess
    import psutil
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    if len(visible) != 8 or len(set(visible)) != 8:
        raise RuntimeError("FULL_DIMO_REQUIRES_EIGHT_DISTINCT_VISIBLE_GPUS")
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used",
        "--format=csv,noheader,nounits"], text=True)
    records = [[cell.strip() for cell in row] for row in csv.reader(raw.splitlines())]
    selected = [row for row in records if row[0] in visible or row[1] in visible]
    if len(selected) != 8:
        raise RuntimeError("FULL_DIMO_GPU_INVENTORY_MISMATCH")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    for row in csv.reader(apps.splitlines()):
        if len(row) != 2 or row[0].strip() not in {r[1] for r in selected}:
            continue
        pid = int(row[1].strip())
        try:
            owned = pid == os.getpid() or psutil.Process(pid).ppid() == os.getppid()
        except psutil.NoSuchProcess:
            continue
        if not owned:
            raise RuntimeError("FULL_DIMO_GPU_HAS_OTHER_JOB_DO_NOT_PREEMPT")
    return {"status": "READY_FOR_MEASURED_DDP_STEP", "gpus": selected, "smoke": "NOT_RUN"}


class MemoryRecorder:
    def __init__(self, device, output, rank):
        self.device, self.output, self.rank = device, Path(output), rank
        self.last_phase = "preflight"

    def record(self, phase, step):
        self.last_phase = phase
        row = {"rank": self.rank, "step": step, "phase": phase,
               "gpu_status": "NOT_RUN_RESOURCE_UNAVAILABLE"}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            row.update(gpu_status="MEASURED", allocated_bytes=torch.cuda.memory_allocated(self.device),
                reserved_bytes=torch.cuda.memory_reserved(self.device), peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        with self.output.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


def disk_budget(parameters, config, output):
    # Two FP32 roles+EMA (12P bytes), two AdamW pairs (16P bytes).
    resume_bytes = 28 * parameters
    inference_bytes = 8 * parameters
    required = (len(config["resume_checkpoint_steps"]) + 1) * resume_bytes + len(config["evaluation_steps"]) * inference_bytes
    required = int(required * 1.15)
    free = shutil.disk_usage(output).free
    if free < required:
        raise RuntimeError(f"FULL_DIMO_DISK_BUDGET_FAILED: required={required} free={free}")
    return {"parameters_per_role": parameters, "resume_checkpoint_estimate_bytes": resume_bytes,
            "inference_checkpoint_estimate_bytes": inference_bytes, "required_with_atomic_and_margin_bytes": required,
            "free_bytes": free, "retention": "fixed_six_resume_checkpoints_no_automatic_deletion"}


def append_committed_log(path, row):
    import os
    with Path(path).open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def prepare_resume_output(output, step):
    """Preserve discarded tail evidence and expose only the restored timeline."""
    import os
    import tempfile
    output = Path(output)
    for marker in (output / "checkpoints").glob("checkpoint-*/COMPLETED.json"):
        if json.loads(marker.read_text())["committed_step"] > step:
            raise RuntimeError("FULL_DIMO_RESUME_REQUIRES_LATEST_COMPLETE_CHECKPOINT")
    future_exports = [marker.parent for marker in (output / "inference").glob("checkpoint-*/COMPLETED.json")
                      if json.loads(marker.read_text())["committed_step"] > step]
    path = output / "metrics.jsonl"
    retained = []
    discard = False
    if path.exists():
        for line in path.read_text().splitlines(keepends=True):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                discard = True
                continue
            if line.endswith("\n") and row.get("committed") and row["global_step"] <= step:
                retained.append(line)
            else:
                discard = True
    if not discard and not future_exports:
        return None
    archive = Path(tempfile.mkdtemp(prefix=f"recovery-from-{step}-", dir=output))
    if discard:
        path.rename(archive / path.name)
        temporary = output / ".metrics.recovered.tmp"
        with temporary.open("x") as handle:
            handle.writelines(retained)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    for export in future_exports:
        export.rename(archive / ("inference-" + export.name))
    return str(archive)
