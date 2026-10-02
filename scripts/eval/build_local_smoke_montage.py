#!/usr/bin/env python3
"""Build the required six-column local pipeline-only smoke montage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


STATUS = "PIPELINE_ONLY_NOT_METRIC_EVAL"
HEADERS = ("Input", "Region", "GT", "E0-official", "E0-region", "Smoke")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--smoke", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    roots = {name: Path(value) for name, value in (
        ("official", args.official), ("region", args.region), ("smoke", args.smoke)
    )}
    metadata = {name: json.loads((root / "metadata.json").read_text(encoding="utf-8"))
                for name, root in roots.items()}
    if any(value["evaluation_status"] != STATUS for value in metadata.values()):
        raise RuntimeError("all inputs must be pipeline-only diagnostics")

    tile = 256
    label_height = 32
    count = min(args.count, *(len(value["samples"]) for value in metadata.values()))
    canvas = Image.new("RGB", (tile * len(HEADERS), label_height + tile * count), "white")
    draw = ImageDraw.Draw(canvas)
    for column, header in enumerate(HEADERS):
        draw.text((column * tile + 8, 9), header, fill="black")
    for row in range(count):
        official = metadata["official"]["samples"][row]
        stem = f"{row:03d}_{official['sample_key']}"
        paths = (
            roots["official"] / f"{stem}_source.png",
            roots["official"] / f"{stem}_mask.png",
            roots["official"] / f"{stem}_target.png",
            Path(official["output"]),
            Path(metadata["region"]["samples"][row]["output"]),
            Path(metadata["smoke"]["samples"][row]["output"]),
        )
        for column, path in enumerate(paths):
            image = Image.open(path).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(image, (column * tile, label_height + row * tile))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=92)
    sidecar = {
        "evaluation_status": STATUS,
        "eligible_for_checkpoint_selection": False,
        "eligible_for_formal_results_csv": False,
        "rows": count,
        "columns": list(HEADERS),
        "montage": str(destination.resolve()),
    }
    destination.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
