"""Independent total-20K SFT admission; formal D200K is unchanged."""
from collections import Counter
import json
from pathlib import Path

from .contracts import sha256_file

MODE = "mixed20k_epoch7_continuation"
COUNTS = {"magicbrush": 8784, "crispedit": 2000, "scaleedit": 7000, "interedit": 2216}
ROWS = sum(COUNTS.values())


def is_mixed(config):
    return config.get("experiment_mode") == MODE


def validate_mixed_budget(config):
    if not is_mixed(config):
        raise RuntimeError("MIXED_EXPLICIT_MODE_REQUIRED")
    if (config.get("world_size"), config["epochs"], config["batch_per_gpu"],
            config["gradient_accumulation"], config["data"]["name"],
            config["data"]["expected_rows"], config["max_optimizer_steps"],
            config["scheduler_horizon_steps"]) != (4, 5, 1, 4, "fixed_mixed20k_total",
                                                    ROWS, 5 * ROWS // 16, 5 * ROWS // 16):
        raise RuntimeError("MIXED_REGISTERED_SFT_BUDGET_CHANGED")
    if config["checkpoint_steps"] != [epoch * (ROWS // 16) for epoch in range(1, 6)]:
        raise RuntimeError("MIXED_REGISTERED_EPOCH_CHECKPOINTS_CHANGED")
    if not config.get("initial_dense_checkpoint"):
        raise RuntimeError("MIXED_VERIFIED_EPOCH7_WARM_START_REQUIRED")
    if config.get("continuation", {}) != {"weights_only": True, "optimizer_inherited": False,
                                          "previous_sft_epoch": 7, "additional_epochs": 5}:
        raise RuntimeError("MIXED_CONTINUATION_IDENTITY_REQUIRED")


def verify_mixed_ready(config):
    validate_mixed_budget(config)
    data = config["data"]
    record = json.loads(Path(data["corpus_ready"]).read_text())
    expected = {"schema": "mixed-training-ready-v2", "status": "VERIFIED",
                "selection_scope": MODE, "total_rows": ROWS, "dataset_counts": COUNTS,
                "train_dev_leakage": "PASS", "loader_geometry": "PASS",
                "translation_qa": "PASS", "magicbrush_preserved": True, "padding": False}
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("MIXED_REAL_DATA_NOT_VERIFIED")
    if record["train_manifest_sha256"] != sha256_file(data["manifest"]):
        raise RuntimeError("MIXED_FROZEN_MANIFEST_CHANGED")
    for entry in record["files"].values():
        if sha256_file(entry["path"]) != entry["sha256"]:
            raise RuntimeError("MIXED_FROZEN_EVIDENCE_CHANGED")
    from .dense_checkpoint import verify_dense_checkpoint
    if verify_dense_checkpoint(config["initial_dense_checkpoint"])["checkpoint_sha256"] != record["warm_start_checkpoint_sha256"]:
        raise RuntimeError("MIXED_EPOCH7_WEIGHTS_CHANGED")
    return record
