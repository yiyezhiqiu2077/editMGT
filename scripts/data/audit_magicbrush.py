#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image, ImageDraw

from src.explicit_region.dataset import MagicBrushAlignedDataset
from src.explicit_region.masks import pixel_mask_to_token_mask


def to_pil(tensor) -> Image.Image:
    array = (tensor.numpy().transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=1024)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = MagicBrushAlignedDataset(args.manifest, resolution=args.resolution, base_seed=args.seed)
    indices = random.Random(args.seed).sample(range(len(dataset)), min(args.count, len(dataset)))
    tile = 256
    montage = Image.new("RGB", (tile * 4, tile * len(indices)), "white")
    records = []
    for row_index, index in enumerate(indices):
        item = dataset[index]
        source = to_pil(item["source_image"])
        target = to_pil(item["target_image"])
        mask = Image.fromarray((item["edit_region_mask"].numpy() * 255).astype(np.uint8))
        overlay = source.copy()
        red = Image.new("RGB", source.size, (255, 0, 0))
        overlay = Image.blend(source, Image.composite(red, overlay, mask), 0.55)
        token_mask = pixel_mask_to_token_mask(item["edit_region_mask"][None], (64, 64))[0]
        panels = [source, mask.convert("RGB"), overlay, target]
        for column, panel in enumerate(panels):
            montage.paste(panel.resize((tile, tile), Image.Resampling.BICUBIC), (column * tile, row_index * tile))
        record = {
            "index": index,
            "sample_key": item["sample_key"],
            "instruction": item["instruction_en"],
            "mask_semantics": item["mask_semantics"],
            "pixel_mask_fraction": float(item["edit_region_mask"].float().mean()),
            "token_mask_fraction": float(token_mask.float().mean()),
            "token_mask_count": int(token_mask.sum()),
            "mask_retention": item["geometry"]["mask_retention"],
            "geometry": item["geometry"],
        }
        records.append(record)
    montage.save(output / "magicbrush_alignment_32.jpg", quality=92)
    with (output / "magicbrush_alignment_32.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "samples": len(records),
        "post_geometry_empty_masks": 0,
        "minimum_retention": min(row["mask_retention"] for row in records),
        "mean_pixel_mask_fraction": sum(row["pixel_mask_fraction"] for row in records) / len(records),
        "mean_token_mask_fraction": sum(row["token_mask_fraction"] for row in records) / len(records),
        "montage": str(output / "magicbrush_alignment_32.jpg"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
