#!/usr/bin/env python3
"""Freeze the v2 primary/probe/auxiliary validation manifests."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explicit_region.fixed_corpus import (
    INTEREDIT_TYPES, apply_translation, freeze_aux_groups, freeze_magicbrush_probe,
    selection_hash,
)
from src.explicit_region.config import load_config
from src.explicit_region.language import contains_han, translation_cache_key


def read(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def write(path, rows):
    Path(path).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def inter_type(row):
    value = str(row["edit_type_original"]).lower()
    return "local" if value == "local" else value


def freeze_interedit(rows, count=128, seed=42):
    groups_by_type = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = inter_type(row)
        if value in INTEREDIT_TYPES:
            groups_by_type[value][row["group_id"]].append(row)
    queues = {}
    for value in INTEREDIT_TYPES:
        ordered_groups = sorted(
            groups_by_type[value],
            key=lambda group: (hashlib.sha256(f"interedit\0aux-validation\0{seed}\0{group}".encode()).hexdigest(), group),
        )
        queues[value] = deque(
            min(groups_by_type[value][group], key=lambda row: (selection_hash(row, seed), row["sample_uid"]))
            for group in ordered_groups
        )
    target, remainder = divmod(count, len(INTEREDIT_TYPES))
    quotas = {value: target + (i < remainder) for i, value in enumerate(INTEREDIT_TYPES)}
    selected, used_groups = [], set()
    for value in INTEREDIT_TYPES:
        while queues[value] and sum(inter_type(row) == value for row in selected) < quotas[value]:
            row = queues[value].popleft()
            if row["group_id"] not in used_groups:
                selected.append(row); used_groups.add(row["group_id"])
    while len(selected) < count:
        progressed = False
        for value in INTEREDIT_TYPES:
            while queues[value]:
                row = queues[value].popleft()
                if row["group_id"] not in used_groups:
                    selected.append(row); used_groups.add(row["group_id"]); progressed = True
                    break
            if len(selected) == count:
                break
        if not progressed:
            raise RuntimeError("insufficient group-disjoint Inter-Edit auxiliary validation rows")
    return selected


class MissingTranslation(RuntimeError):
    pass


def cached_translator(cache_path, translation_config):
    translations = defaultdict(list)
    for row in read(cache_path):
        translations[row["source_text"]].append(row)

    def translate(text, _row):
        expected = translation_cache_key(
            text, translation_config["backend"], translation_config["immutable_revision"],
            translation_config["src_lang"], translation_config["tgt_lang"],
            translation_config["decoding"],
        )
        available = [row for row in translations.get(text, []) if row.get("cache_key") == expected]
        if not available:
            raise MissingTranslation(text)
        if len(available) != 1:
            raise RuntimeError(f"ambiguous translation cache entries for {text!r}")
        return available[0]["translated_text"], expected
    return translate


def translate_selection(rows, selector, translate, *, dataset):
    rejected = set()
    rejection_rows = []
    while True:
        selected = selector([row for row in rows if row["sample_uid"] not in rejected])
        translated, pending, failed = [], [], []
        for row in selected:
            try:
                value, flags = apply_translation(row, translate)
            except MissingTranslation:
                pending.append({
                    "dataset": dataset, "sample_uid": row["sample_uid"],
                    "instruction_original": row["instruction_original"],
                })
                continue
            if value is None:
                failed.append((row, flags))
            else:
                translated.append(value)
        if pending:
            return None, pending, rejection_rows
        if not failed:
            return translated, [], rejection_rows
        for row, flags in failed:
            rejected.add(row["sample_uid"])
            rejection_rows.append({
                "dataset": dataset, "sample_uid": row["sample_uid"],
                "reason": "translation_qa", "qa_flags": flags,
            })


def translate_all(rows, translate, *, dataset):
    values, pending = [], []
    for row in rows:
        try:
            value, flags = apply_translation(row, translate)
        except MissingTranslation:
            pending.append({
                "dataset": dataset, "sample_uid": row["sample_uid"],
                "instruction_original": row["instruction_original"],
            })
            continue
        if value is None:
            raise RuntimeError(f"official validation translation QA failure: {flags}")
        values.append(value)
    return values, pending


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--magicbrush-dev",required=True)
    parser.add_argument("--crispedit-pool",required=True)
    parser.add_argument("--scaleedit-pool",required=True)
    parser.add_argument("--interedit-pool",required=True)
    parser.add_argument("--translation-cache",required=True)
    parser.add_argument("--config",default="configs/data/fixed_200k.yaml")
    parser.add_argument("--output-dir",required=True);parser.add_argument("--count",type=int,default=128)
    args=parser.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    magic=read(args.magicbrush_dev);crisp=read(args.crispedit_pool);scale=read(args.scaleedit_pool);inter=read(args.interedit_pool)
    translation=load_config(args.config)["translation"]
    translate=cached_translator(args.translation_cache,translation)
    magic,pending_magic=translate_all(magic,translate,dataset="magicbrush")
    pending=list(pending_magic);rejections=[]
    crisp_selected,crisp_pending,crisp_rejections=translate_selection(
        crisp,lambda rows:freeze_aux_groups(rows,"crispedit",args.count),translate,dataset="crispedit")
    scale_selected,scale_pending,scale_rejections=translate_selection(
        scale,lambda rows:freeze_aux_groups(rows,"scaleedit",args.count),translate,dataset="scaleedit")
    inter_selected,inter_pending,inter_rejections=translate_selection(
        inter,lambda rows:freeze_interedit(rows,args.count),translate,dataset="interedit")
    pending.extend(crisp_pending+scale_pending+inter_pending)
    rejections.extend(crisp_rejections+scale_rejections+inter_rejections)
    if pending:
        unique={row["instruction_original"]:row for row in pending}
        write(out/"translation_required.jsonl",list(unique.values()))
        print(json.dumps({"status":"TRANSLATION_REQUIRED","unique_prompts":len(unique)}))
        raise SystemExit(42)
    suffix=str(args.count)
    outputs={
      "magicbrush_official_dev.jsonl":magic,
      f"magicbrush_probe{suffix}.jsonl":freeze_magicbrush_probe(magic,args.count),
      f"crispedit_aux{suffix}.jsonl":crisp_selected,
      f"scaleedit_aux{suffix}.jsonl":scale_selected,
      f"interedit_aux{suffix}.jsonl":inter_selected,
    }
    hashes={}
    for name,rows in outputs.items():
        path=out/name;write(path,rows);hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    sources={name:{row["source_sha256"] for row in rows} for name,rows in outputs.items()}
    aux_names=[name for name in outputs if "_aux" in name]
    for i,name in enumerate(aux_names):
        for other in aux_names[i+1:]:
            if sources[name]&sources[other]: raise RuntimeError(f"cross-aux source leak: {name}/{other}")
    write(out/"translation_rejections.jsonl",rejections)
    report={"schema":"fixed200k-validation-v2","counts":{name:len(rows) for name,rows in outputs.items()},"sha256":hashes,"translation_rejections":len(rejections),"all_instruction_en_valid":all(row.get("instruction_en") and not contains_han(row["instruction_en"]) for rows in outputs.values() for row in rows)}
    if not report["all_instruction_en_valid"]:raise RuntimeError("validation contains invalid instruction_en")
    (out/"validation.meta.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
