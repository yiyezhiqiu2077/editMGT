#!/usr/bin/env python3
"""Audit token-region conversion and corruption invariants on real masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from diffusers import VQModel

from src.explicit_region.corruption import prepare_corruption
from src.explicit_region.dataset import MagicBrushAlignedDataset
from src.explicit_region.masks import pixel_mask_to_token_mask
from src.v2_utils import prepare_cond_token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--model-root")
    args = parser.parse_args()
    dataset = MagicBrushAlignedDataset(
        args.manifest, resolution=args.resolution, base_seed=args.seed
    )
    records = []
    grid = args.resolution // 16
    vq = None
    if args.model_root:
        vq = VQModel.from_pretrained(
            args.model_root, subfolder="vqvae", local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to("cuda").eval()
    for index in range(min(args.count, len(dataset))):
        item = dataset[index]
        region = pixel_mask_to_token_mask(
            item["edit_region_mask"][None], (grid, grid),
            mode="any_overlap", dilation_tokens=0, minimum_edit_tokens=1,
        )
        if vq is not None:
            with torch.no_grad():
                source = prepare_cond_token(None, item["source_image"][None].to("cuda",dtype=vq.dtype), vq).reshape(1,grid,grid)
                target = prepare_cond_token(None, item["target_image"][None].to("cuda",dtype=vq.dtype), vq).reshape(1,grid,grid)
            region = region.to("cuda")
        else:
            source = torch.arange(grid * grid).reshape(1, grid, grid)
            target = source + grid * grid
        corruption = prepare_corruption(
            source, target, region, 8255,
            mode="roi_hardlock", base_seed=args.seed,
            global_sample_indices=[index], sample_keys=[item["sample_key"]],
            full_roi_mask_probability=0.15,
        )
        selected = corruption.selected_mask
        assert not (selected & ~region).any()
        assert torch.equal(corruption.input_tokens[~region], source[~region])
        assert (corruption.labels[~region] == -100).all()
        records.append({
            "global_sample_index": index,
            "sample_key": item["sample_key"],
            "pixel_mask_fraction": float(item["edit_region_mask"].float().mean()),
            "token_mask_fraction": float(region.float().mean()),
            "token_mask_count": int(region.sum()),
            "scheduled_roi_ratio": float(corruption.scheduled_roi_ratio[0]),
            "actual_roi_mask_fraction": float(corruption.actual_roi_mask_fraction[0]),
            "actual_global_mask_fraction": float(corruption.actual_global_mask_fraction[0]),
            "full_roi_branch": bool(corruption.full_roi_branch[0]),
            "source_token_shape": list(source.shape),
            "target_token_shape": list(target.shape),
            "region_token_shape": list(region.shape),
            "corruption_mode": corruption.resolved_modes[0],
            "outside_region_source_locked": True,
            "outside_region_labels_ignored": True,
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(records), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
