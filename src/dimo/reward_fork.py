"""Explicit weights-only fork; old optimizer, sampler and RNG are never loaded."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from src.explicit_region.contracts import sha256_file
from .full_checkpoint import verify, load_weights


def fork_manifest(source):
    root = Path(source).resolve()
    marker = json.loads((root / "COMPLETED.json").read_text())
    identity = json.loads((root / "inference_identity.json").read_text())
    if (marker.get("schema") != "full-dense-dimo-ddp-v3" or identity.get("recipe") != "dimo-v2"
            or identity.get("reward_extension") or "training_state.pt" not in marker["files"]
            or any(f"{r}.index.json" not in marker["files"] for r in ("student", "auxiliary", "ema"))):
        raise RuntimeError("SOFT_REWARD_FORK_REQUIRES_COMPLETE_ORIGINAL_DIMO_V2")
    return {"source": str(root), "source_step": marker["committed_step"],
            "completed_sha256": sha256_file(root / "COMPLETED.json"),
            "identity_sha256": sha256_file(root / "inference_identity.json"),
            "student_weights": "raw", "auxiliary_weights": "same_checkpoint",
            "ema_values": "carry_same_checkpoint_shadow_reset_update_count",
            "optimizer_policy": "reset_both_no_state_transfer", "scheduler_policy": "reset",
            "sampler_policy": "new_experiment_cursor_zero", "rng_policy": "new_experiment_seed"}


def protect_outputs(output, source):
    output, source = Path(output).resolve(), Path(source).resolve()
    # Protect the whole original run, not just the selected checkpoint folder.
    protected = source.parent.parent
    if output == protected or protected in output.parents or output in protected.parents:
        raise RuntimeError("SOFT_REWARD_OUTPUT_OVERLAPS_ORIGINAL_RUN")


@torch.no_grad()
def fork_weights(source, roles, optimizers, ema, expected_manifest, new_identity):
    if fork_manifest(source) != expected_manifest:
        raise RuntimeError("SOFT_REWARD_FORK_SOURCE_CHANGED")
    parent, _ = verify(source, full=True)
    for key in ("teacher_bundle_fingerprint", "data_sha256"):
        if parent.get(key) != new_identity.get(key):
            raise RuntimeError("SOFT_REWARD_FORK_TEACHER_OR_DATA_MISMATCH")
    if any(o.state for o in optimizers) or ema.update_count:
        raise RuntimeError("SOFT_REWARD_FORK_REQUIRES_FRESH_OPTIMIZERS_AND_COUNTERS")
    for role in ("student", "auxiliary"):
        load_weights(source, roles.model_for(role), weights=role, expected_identity=parent)
    roles.assert_frozen_unchanged(parent["frozen_parameter_hashes"])
    _, index = verify(source, weights="ema", expected_identity=parent)
    if not set(ema.shadow) <= set(index):
        raise RuntimeError("SOFT_REWARD_PARENT_EMA_COVERAGE_MISMATCH")
    for filename in sorted(set(index.values())):
        for name, value in load_file(Path(source) / filename).items():
            if name in ema.shadow:
                if value.dtype != torch.float32 or value.shape != ema.shadow[name].shape:
                    raise RuntimeError("SOFT_REWARD_PARENT_EMA_TENSOR_MISMATCH")
                ema.shadow[name].copy_(value.to(ema.device))
    ema.update_count = 0


def validate_reward_config(config):
    reward = config.get("soft_reward")
    if reward is None:
        return
    if (not reward.get("enabled") or reward.get("extension") != "soft-dimo-editing-reward-v1"
            or config.get("recipe") != "dimo-v2" or reward.get("max_new_steps") != 500
            or reward.get("clip_text_enabled") is not False
            or reward.get("lpips_resolution") != 512 or reward.get("gap_interval") != 20
            or reward.get("resize_backend") != "deterministic-separable-antialias-v1"
            or reward.get("calibration") != {"batches": 4, "relative_scale": .1, "max_weight": 100.}):
        raise RuntimeError("SOFT_REWARD_UNREGISTERED_RECIPE")
    manifest = fork_manifest(reward["fork_checkpoint"])
    if manifest != reward["fork_identity"]:
        raise RuntimeError("SOFT_REWARD_CONFIG_SOURCE_IDENTITY_CHANGED")
    protect_outputs(config["output_dir"], reward["fork_checkpoint"])
    protect_outputs(reward["inference_output_dir"], reward["fork_checkpoint"])
    inference = Path(reward["inference_output_dir"]).resolve()
    if inference == Path(config["output_dir"]).resolve() or Path(config["output_dir"]).resolve() in inference.parents:
        raise RuntimeError("SOFT_REWARD_SPLIT_STORAGE_REQUIRES_INDEPENDENT_INFERENCE_ROOT")
    # Content-bind each frozen reward model and LPIPS learned weights. Paths
    # alone are insufficient for correct resume/gradient calibration identity.
    for path, expected_hash in reward["asset_sha256"].items():
        if sha256_file(path) != expected_hash:
            raise RuntimeError("SOFT_REWARD_FROZEN_ASSET_CHANGED")
    from importlib.metadata import version
    if any(version(package) != expected for package,expected in reward["library_versions"].items()):
        raise RuntimeError("SOFT_REWARD_LIBRARY_VERSION_CHANGED")
    if config.get('soft_reward_continuation'):
        from .reward_continuation import validate_continuation
        validate_continuation(config)
