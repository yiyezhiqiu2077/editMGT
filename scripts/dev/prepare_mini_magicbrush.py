#!/usr/bin/env python3
"""Hard-link existing images and derive exact inverse-alpha masks for Mini rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data.prepare_magicbrush_assets import edit_mask_from_raw
from src.explicit_region.dataset import _align_to_mask_coordinates


def key(row: dict) -> str:
    return f"{row['img_id']}_{int(row['turn_index']):02d}"


def geometry_eligible(base: Path, row: dict) -> bool:
    try:
        with Image.open(base / row["source"]) as source, Image.open(base / row["target"]) as target, Image.open(base / row["mask_edit"]) as mask:
            if not mask.getbbox(): return False
            aligned_source, aligned_target, _ = _align_to_mask_coordinates(source, target, mask, row["sample_key"])
            if aligned_source is not source: aligned_source.close()
            if aligned_target is not target: aligned_target.close()
        return True
    except (OSError, ValueError):
        return False


def select(existing: Path, split: str, count: int) -> list[dict]:
    base = existing / split
    rows = [json.loads(line) for line in (base / "manifest.jsonl").open(encoding="utf-8") if line.strip()]
    rows.sort(key=lambda row: (hashlib.sha256(f"42\0magicbrush\0{row['sample_key']}".encode()).hexdigest(), row["sample_key"]))
    eligible = []
    for row in rows:
        if geometry_eligible(base, row):
            eligible.append(row)
            if len(eligible) == count:
                break
    if len(eligible) < count: raise RuntimeError(f"insufficient eligible MagicBrush {split} rows")
    return eligible[:count]


def raw_masks(parquet_root: Path, split: str, wanted: set[str]) -> dict[str, Image.Image]:
    patterns = [parquet_root / "data"] if (parquet_root / "data").is_dir() else [parquet_root]
    files = sorted(path for base in patterns for path in base.glob(f"{split}-*.parquet"))
    found = {}
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["img_id", "turn_index", "mask_img"], batch_size=32, use_threads=False):
            for row in batch.to_pylist():
                identity = key(row)
                if identity not in wanted: continue
                value = row["mask_img"]; value = value.get("bytes") if isinstance(value, dict) else value
                from io import BytesIO
                with Image.open(BytesIO(value)) as image:
                    found[identity] = edit_mask_from_raw(image).copy()
        if len(found) == len(wanted): break
    if set(found) != wanted: raise RuntimeError(f"raw MagicBrush masks missing: {sorted(wanted-set(found))[:5]}")
    return found


def materialize(existing: Path, raw_root: Path, output: Path, split: str, selected: list[dict]) -> None:
    target = output / split; target.mkdir(parents=True, exist_ok=True)
    masks = raw_masks(raw_root, split, {row["sample_key"] for row in selected})
    result = []
    for index, row in enumerate(selected):
        sample = row["sample_key"]; folder = target / "images" / sample; folder.mkdir(parents=True, exist_ok=True)
        paths = {"source": folder / "source.png", "target": folder / "target.png", "mask_edit": folder / "mask_edit.png"}
        for role in ("source", "target"):
            source = existing / split / row[role]
            if paths[role].exists():
                if paths[role].stat().st_size != source.stat().st_size:
                    raise RuntimeError(f"existing mini asset size mismatch: {paths[role]}")
            else:
                try: os.link(source, paths[role])
                except OSError:
                    temporary = paths[role].with_suffix(".tmp")
                    shutil.copy2(source, temporary)
                    os.replace(temporary, paths[role])
        if paths["mask_edit"].exists():
            with Image.open(paths["mask_edit"]) as current:
                if list(current.getdata()) != list(masks[sample].getdata()):
                    raise RuntimeError(f"existing mini mask mismatch: {sample}")
        else:
            masks[sample].save(paths["mask_edit"])
        result.append({"sample_key": sample, "img_id": str(row["img_id"]),
                       "turn_index": int(row["turn_index"]), "instruction": row["instruction"],
                       **{role: path.relative_to(target).as_posix() for role, path in paths.items()}})
    (target / "manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in result), encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--existing-root",required=True)
    parser.add_argument("--raw-train-root",required=True);parser.add_argument("--raw-dev-root",required=True)
    parser.add_argument("--output-root",required=True);args=parser.parse_args()
    existing=Path(args.existing_root).resolve();output=Path(args.output_root).resolve()
    train=select(existing,"train",320);dev=select(existing,"dev",32)
    materialize(existing,Path(args.raw_train_root).resolve(),output,"train",train)
    materialize(existing,Path(args.raw_dev_root).resolve(),output,"dev",dev)
    print(json.dumps({"train":len(train),"dev":len(dev),"output":str(output)}))


if __name__=="__main__":main()
