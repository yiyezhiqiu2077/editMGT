#!/usr/bin/env python3
"""Generate immutable, identity-bound evaluation rows. Never opens a TEST split."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from src.explicit_region.config import load_config
from src.explicit_region.contracts import audit_component_identity, sha256_file
from src.explicit_region.gates import enforce_stage_gate
from src.explicit_region.selection import (
    baseline_evaluation_identity, checkpoint_identity, evaluation_identity, file_identity, inference_settings,
    object_sha256, verify_preregistration,
)


def pil(tensor):
    return Image.fromarray((tensor.numpy().transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8))


def prepare_evaluation(experiment, checkpoint, config_path, manifest, dataset, preregistration=None):
    """All trainable evaluations require current gates, including diagnostics."""
    prereg_path = preregistration or os.environ.get("EDITMGT_PREREGISTRATION")
    if not prereg_path:
        raise ValueError("EDITMGT_PREREGISTRATION is required")
    frozen = verify_preregistration(prereg_path)
    enforce_stage_gate("eval", corpus_ready=frozen["corpus_ready"]["path"],
                       corpus_approval=os.environ.get("CORPUS_APPROVAL"),
                       smoke_report=os.environ.get("FIXED200K_SMOKE_REPORT"), preregistration=prereg_path)
    identity = evaluation_identity(experiment=experiment, checkpoint=checkpoint, config_path=config_path,
                                   manifest=manifest, dataset=dataset, preregistration=frozen)
    if identity["eval_config"] != frozen["eval_config"]:
        raise ValueError("evaluation config differs from preregistration")
    candidates = [candidate for group in frozen["candidate_sets"].values() for candidate in group
                  if candidate["experiment"] == experiment and candidate["step"] == identity["checkpoint"]["step"]]
    if not candidates and experiment in frozen["selection"]["candidate_sets"]["lr"]["experiments"]:
        step = identity["checkpoint"]["step"]
        if step in frozen["selection"]["candidate_sets"]["lr"]["diagnostic_steps"]:
            parent = next(candidate for candidate in frozen["candidate_sets"]["lr"] if candidate["experiment"] == experiment)
            candidates = [parent | {"step": step, "checkpoint_path": str(Path(parent["checkpoint_path"]).parent / f"checkpoint-{step}")}]
    if len(candidates) != 1:
        raise ValueError("checkpoint is neither a preregistered candidate nor an LR diagnostic")
    checkpoint_identity(checkpoint, experiment, expected=candidates[0])
    saved = json.loads((Path(checkpoint) / "trainable_config.json").read_text())
    if saved.get("data_content_hashes", {}).get("data.corpus_ready") != {
        "resolved_path": frozen["corpus_ready"]["path"], "sha256": frozen["corpus_ready"]["sha256"]
    }:
        raise ValueError("checkpoint training corpus differs from preregistered READY identity")
    if dataset == "magicbrush" and identity["manifest"] != frozen["primary_manifest"]:
        raise ValueError("formal primary evaluation must use full official DEV, never probe128")
    if dataset != "magicbrush":
        ready = json.loads(Path(frozen["corpus_ready"]["path"]).read_text())
        attested = ready["files"][f"validation_{dataset}_aux128"]
        if file_identity(manifest) != file_identity(attested["path"]):
            raise ValueError("auxiliary manifest is not READY-attested")
    return load_config(config_path), identity


def prepare_baseline_evaluation(experiment, model_root, release_metadata, config_path, manifest,
                                dataset, timestep_mode, preregistration=None):
    prereg_path = preregistration or os.environ.get("EDITMGT_PREREGISTRATION")
    if not prereg_path:
        raise ValueError("canonical E0 requires frozen preregistration and current evaluation gates")
    frozen = verify_preregistration(prereg_path)
    enforce_stage_gate("eval", corpus_ready=frozen["corpus_ready"]["path"],
                       corpus_approval=os.environ.get("CORPUS_APPROVAL"),
                       smoke_report=os.environ.get("FIXED200K_SMOKE_REPORT"), preregistration=prereg_path)
    identity = baseline_evaluation_identity(experiment=experiment, model_root=model_root, release_metadata=release_metadata,
                  config_path=config_path, manifest=manifest, dataset=dataset, timestep_mode=timestep_mode, preregistration=frozen)
    return load_config(config_path), identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("magicbrush", "crispedit", "scaleedit", "interedit"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--interedit-root")
    parser.add_argument("--canonical-root")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--baseline-canonical", action="store_true", help="Identity-verified released E0 full DEV, never ranked")
    parser.add_argument("--release-metadata", help="Pinned official model metadata with LFS/blob hashes")
    parser.add_argument("--experiment")
    parser.add_argument("--preregistration")
    parser.add_argument("--config", required=True)
    parser.add_argument("--timestep-mode", choices=("roi_relative", "official_upstream_timestep"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    if "test" in Path(args.manifest).name.lower():
        parser.error("TEST evaluation is not authorized by this entry point")
    if args.baseline_canonical:
        if args.checkpoint or not args.experiment or not args.canonical_root or not args.release_metadata or args.count is not None:
            parser.error("canonical E0 requires experiment/canonical-root/release-metadata and forbids checkpoint/partial count")
        config, identity = prepare_baseline_evaluation(args.experiment, args.model_root, args.release_metadata,
                                args.config, args.manifest, args.dataset, args.timestep_mode, args.preregistration)
        settings = identity["inference_settings"]
    elif args.checkpoint:
        if not args.experiment or not args.canonical_root:
            parser.error("trainable evaluation requires --experiment and --canonical-root")
        config, identity = prepare_evaluation(args.experiment, args.checkpoint, args.config, args.manifest,
                                              args.dataset, args.preregistration)
        if args.count is not None:
            parser.error("partial counts are not permitted in frozen full evaluation")
        if args.timestep_mode != config["inference_timestep_mode"]:
            parser.error("timestep protocol differs from frozen config")
        settings = inference_settings(args.checkpoint)
        trainable = json.loads((Path(args.checkpoint) / "trainable_config.json").read_text())
        if trainable["model_snapshot"] != audit_component_identity(args.model_root)["snapshot_identity"]:
            parser.error("evaluation model snapshot differs from checkpoint training snapshot")
    else:
        # E0 remains explicitly legacy/diagnostic and can never enter the selector.
        config = load_config(args.config)
        identity = None
        settings = {"lora_scope": "both", "persistent_conditioning": False}
    out = Path(args.output_dir).resolve()
    if args.baseline_canonical:
        out = out / f"{args.experiment}-{identity['identity_sha256']}"
    out.mkdir(parents=True, exist_ok=False)
    from src.explicit_region.fixed_dataset import CanonicalAlignedDataset
    from src.explicit_region.dataset import MagicBrushAlignedDataset, InterEditArchiveDataset
    from src.editmgt import init_edit_mgt
    kwargs = dict(resolution=config["resolution"], base_seed=42, minimum_mask_retention=.75)
    if args.canonical_root:
        records = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
        dataset = CanonicalAlignedDataset(records, args.canonical_root, max_random_attempts=8, **kwargs)
    elif args.dataset == "magicbrush":
        dataset = MagicBrushAlignedDataset(args.manifest, max_resample_attempts=8, **kwargs)
    elif args.dataset == "interedit":
        dataset = InterEditArchiveDataset(args.manifest, args.interedit_root, max_resample_attempts=8, **kwargs)
    else:
        parser.error("CrispEdit/ScaleEdit require --canonical-root")
    pipe = init_edit_mgt("cuda", enable_bf16=True, base_model_path=args.model_root,
                         local_files_only=True, trainable_state_path=args.checkpoint)
    pipe.set_progress_bar_config(disable=True)
    count = min(len(dataset), args.count if args.count is not None else len(dataset))
    seen = set()
    with (out / "predictions.jsonl").open("x", encoding="utf-8") as manifest_handle:
        for index in range(count):
            item = dataset[index]
            key = item["sample_key"]
            if key in seen:
                raise ValueError("duplicate/resampled evaluation sample identity")
            seen.add(key)
            if identity is not None and key != records[index].get("sample_uid", records[index].get("sample_key")):
                raise ValueError("evaluation dataset substituted a frozen manifest sample")
            source, target = pil(item["source_image"]), pil(item["target_image"])
            mask = Image.fromarray(item["edit_region_mask"].numpy().astype(np.uint8) * 255)
            stem = f"{args.dataset}_{index:06d}_{object_sha256(key)[:16]}"
            common_paths = {}
            for name, image in (("source", source), ("target", target), ("mask", mask)):
                path = out / f"{stem}_{name}.png"
                image.save(path)
                common_paths[name] = str(path)
            common_hashes = {name: sha256_file(path) for name, path in common_paths.items()}
            for seed in config["generation_seeds"]:
                generator = torch.Generator(device="cuda").manual_seed(seed)
                started = time.perf_counter()
                result = pipe(prompt=item["instruction_en"], reference_image=source, mask_image=mask,
                              height=config["resolution"], width=config["resolution"],
                              num_inference_steps=config["steps"], guidance_scale=config["guidance_scale"],
                              reference_strength=config["reference_strength"], generator=generator,
                              inference_timestep_mode=args.timestep_mode, **settings).images[0]
                runtime = time.perf_counter() - started
                output = out / f"{stem}_seed{seed}_output.png"
                result.save(output)
                row = common_paths | {"output": str(output), "identity": identity,
                      "image_sha256": common_hashes | {"output": sha256_file(output)},
                      "sample_key": key, "dataset_name": args.dataset, "edit_type": item["edit_type"],
                      "seed": seed, "runtime_seconds": runtime, "timestep_mode": args.timestep_mode,
                      "inference_settings": settings, "timestep_diagnostics": pipe.last_inference_diagnostics}
                manifest_handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                manifest_handle.flush()
    print(json.dumps({"generations": count * len(config["generation_seeds"]), "manifest": str(out / "predictions.jsonl")}))


if __name__ == "__main__":
    main()
