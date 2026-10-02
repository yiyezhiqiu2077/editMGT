#!/usr/bin/env python3
"""Local MagicBrush TEST pipeline-only smoke; never a metric evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from src.editmgt import init_edit_mgt
from src.explicit_region.dataset import MagicBrushAlignedDataset


STATUS = "PIPELINE_ONLY_NOT_METRIC_EVAL"


def image_from_tensor(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.cpu().numpy().transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--timestep-mode", choices=("roi_relative", "official_upstream_timestep"),
        default="roi_relative",
    )
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("inference smoke requires exactly one visible GPU")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = MagicBrushAlignedDataset(
        args.manifest, resolution=1024, base_seed=args.seed,
        max_resample_attempts=8, minimum_mask_retention=0.75,
    )
    pipe = init_edit_mgt(
        "cuda", enable_bf16=True, base_model_path=args.model_root,
        local_files_only=True, trainable_state_path=args.checkpoint,
    )
    pipe.set_progress_bar_config(disable=True)
    rows = []
    for index in range(min(args.count, len(dataset))):
        item = dataset[index]
        source = image_from_tensor(item["source_image"])
        target = image_from_tensor(item["target_image"])
        mask = Image.fromarray(item["edit_region_mask"].numpy().astype(np.uint8) * 255)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        result = pipe(
            prompt=item["instruction_en"], reference_image=source, mask_image=mask,
            height=1024, width=1024, num_inference_steps=args.steps,
            guidance_scale=10.0, generator=generator, lora_scope="both",
            inference_timestep_mode=args.timestep_mode,
        ).images[0]
        stem = f"{index:03d}_{item['sample_key']}"
        for name, image in (("source", source), ("mask", mask), ("target", target), ("output", result)):
            image.save(output / f"{stem}_{name}.png")
        rows.append({
            "index": index, "sample_key": item["sample_key"],
            "instruction": item["instruction_en"], "seed": args.seed,
            "steps": args.steps, "resolution": 1024,
            "evaluation_status": STATUS,
            "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else "released-base",
            "inference_timestep_mode": args.timestep_mode,
            "timestep_diagnostics": pipe.last_inference_diagnostics,
            "output": str((output / f"{stem}_output.png").resolve()),
        })
    metadata = {
        "evaluation_status": STATUS,
        "eligible_for_checkpoint_selection": False,
        "eligible_for_formal_results_csv": False,
        "samples": rows,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"evaluation_status": STATUS, "samples": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
