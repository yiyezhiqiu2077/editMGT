#!/usr/bin/env python3
"""Materialize pinned MagicBrush parquet and TEST mirror as portable manifests."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from PIL import Image, ImageChops, ImageOps


REQUIRED=("img_id","turn_index","instruction","source_img","target_img","mask_img")

def save(value,path,mode):
    image=value if isinstance(value,Image.Image) else value.convert(mode)
    image.convert(mode).save(path)


def edit_mask_from_raw(value) -> Image.Image:
    """Convert the official erased-source mask to white=editable semantics."""
    image = value if isinstance(value, Image.Image) else value.convert("RGBA")
    bands = image.getbands()
    if "A" in bands:
        return ImageOps.invert(image.getchannel("A"))
    if image.mode == "RGB":
        red, green, blue = image.split()
        if ImageChops.difference(red, green).getbbox() or ImageChops.difference(red, blue).getbbox():
            raise ValueError("ambiguous colored MagicBrush mask without alpha")
    if image.mode not in ("1", "L", "RGB"):
        raise ValueError(f"unsupported MagicBrush mask mode: {image.mode}")
    return image.convert("L")

def prepare_split(snapshot:Path,split:str,output:Path,expected:int):
    from datasets import load_dataset
    files=sorted((snapshot/"data").glob(f"{split}-*.parquet"))
    if not files: raise RuntimeError(f"SCHEMA_CONTRACT_MISMATCH: no MagicBrush {split} parquet")
    ds=load_dataset("parquet",data_files={split:[str(p) for p in files]},split=split)
    output.mkdir(parents=True,exist_ok=True);images=output/"images";images.mkdir(exist_ok=True);rows=[]
    for index,item in enumerate(ds):
        missing=[key for key in REQUIRED if key not in item or item[key] is None]
        if missing: raise RuntimeError(f"SCHEMA_CONTRACT_MISMATCH: MagicBrush row {index} missing {missing}")
        img_id=str(item["img_id"]);turn=int(item["turn_index"]);stem=f"{index:05d}_{img_id}_turn{turn}"
        paths={"source":images/f"{stem}_source.png","target":images/f"{stem}_target.png","mask_raw":images/f"{stem}_mask_raw.png","mask_edit":images/f"{stem}_mask_edit.png"}
        save(item["source_img"],paths["source"],"RGB");save(item["target_img"],paths["target"],"RGB")
        item["mask_img"].save(paths["mask_raw"])
        edit_mask_from_raw(item["mask_img"]).save(paths["mask_edit"])
        rows.append({"sample_key":f"magicbrush/{img_id}_turn{turn}","img_id":img_id,"turn_index":turn,"instruction":str(item["instruction"]).strip(),**{k:v.relative_to(output).as_posix() for k,v in paths.items()}})
    if len(rows)!=expected: raise RuntimeError(f"MagicBrush {split} expected {expected}, got {len(rows)}")
    if len({r["sample_key"] for r in rows})!=len(rows) or any(not r["instruction"] for r in rows): raise RuntimeError("MagicBrush identity/instruction contract failed")
    (output/"manifest.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows),encoding="utf-8")

def prepare_test(root:Path,output:Path):
    sessions=json.loads((root/"edit_sessions.json").read_text());turns=[]
    for session_id,items in sessions.items():
        for index,item in enumerate(items): turns.append((str(session_id),index,item))
    flat=json.loads((root/"edit_turns.json").read_text())
    if len(sessions)!=535 or len(turns)!=1053 or len(flat)!=1053: raise RuntimeError("MAGICBRUSH_TEST_IDENTITY_MISMATCH: expected 535 sessions/1053 turns")
    if [item for _,_,item in turns] != flat: raise RuntimeError("MAGICBRUSH_TEST_IDENTITY_MISMATCH: session/turn annotations disagree")
    output.mkdir(parents=True,exist_ok=True);rows=[]
    for session_id,index,item in turns:
        paths={key:root/"images"/session_id/item[field] for key,field in (("source","input"),("target","output"),("mask_edit","mask"))}
        if not all(path.is_file() for path in paths.values()): raise RuntimeError(f"MAGICBRUSH_TEST_IDENTITY_MISMATCH: missing file for {session_id}/{index}")
        rows.append({"sample_key":f"magicbrush-test/{session_id}_turn{index}","img_id":session_id,"session_id":session_id,"turn_index":index,"instruction":item["instruction"],**{k:os.path.relpath(v,output) for k,v in paths.items()}})
    (output/"manifest.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows),encoding="utf-8")
    (output/"dataset_meta.json").write_text(json.dumps({"split":"test","sessions":535,"turns":1053,"source":"pinned_magicbrush_test_mirror"},indent=2,sort_keys=True)+"\n")

def main():
    p=argparse.ArgumentParser();p.add_argument("--snapshot",required=True);p.add_argument("--train-output",required=True);p.add_argument("--dev-output",required=True);p.add_argument("--test-root",required=True);p.add_argument("--test-output",required=True);a=p.parse_args()
    prepare_split(Path(a.snapshot),"train",Path(a.train_output),8807);prepare_split(Path(a.snapshot),"dev",Path(a.dev_output),528);prepare_test(Path(a.test_root),Path(a.test_output));print(json.dumps({"status":"PASS","train":8807,"dev":528,"test_sessions":535,"test_turns":1053}))
if __name__=="__main__":main()
