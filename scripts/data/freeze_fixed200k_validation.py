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
    INTEREDIT_TYPES, freeze_aux_groups, freeze_magicbrush_probe, selection_hash,
)


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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--magicbrush-dev",required=True)
    parser.add_argument("--crispedit-pool",required=True)
    parser.add_argument("--scaleedit-pool",required=True)
    parser.add_argument("--interedit-pool",required=True)
    parser.add_argument("--output-dir",required=True);parser.add_argument("--count",type=int,default=128)
    args=parser.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    magic=read(args.magicbrush_dev);crisp=read(args.crispedit_pool);scale=read(args.scaleedit_pool);inter=read(args.interedit_pool)
    outputs={
      "magicbrush_official_dev.jsonl":magic,
      "magicbrush_probe128.jsonl":freeze_magicbrush_probe(magic,args.count),
      "crispedit_aux128.jsonl":freeze_aux_groups(crisp,"crispedit",args.count),
      "scaleedit_aux128.jsonl":freeze_aux_groups(scale,"scaleedit",args.count),
      "interedit_aux128.jsonl":freeze_interedit(inter,args.count),
    }
    hashes={}
    for name,rows in outputs.items():
        path=out/name;write(path,rows);hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    sources={name:{row["source_sha256"] for row in rows} for name,rows in outputs.items()}
    aux_names=[name for name in outputs if "aux128" in name]
    for i,name in enumerate(aux_names):
        for other in aux_names[i+1:]:
            if sources[name]&sources[other]: raise RuntimeError(f"cross-aux source leak: {name}/{other}")
    report={"schema":"fixed200k-validation-v2","counts":{name:len(rows) for name,rows in outputs.items()},"sha256":hashes}
    (out/"validation.meta.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
