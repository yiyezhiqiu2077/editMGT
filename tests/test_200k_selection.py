import json
from pathlib import Path
import subprocess
import sys
import os

import yaml

from src.explicit_region.canonical import sample_uid_for
from src.explicit_region.fixed_corpus import (
    deterministic_order, interedit_strata, select_interedit, selection_hash,
)
from src.explicit_region.language import translation_cache_key
from tests.fixed200k_helpers import synthetic_record


def write_jsonl(path,rows):
    Path(path).write_text("".join(json.dumps(row)+"\n" for row in rows))


def test_candidate_order_region_thirds_and_source_round_robin():
    rows=[]
    for i in range(48):
        kind=("Add","Remove","Local","Texture")[i%4]
        canonical={"Add":"add","Remove":"remove","Local":"local_attribute","Texture":"texture"}[kind]
        rows.append(synthetic_record("interedit",i,original_type=kind,canonical_type=canonical,group=f"g-{i//2}"))
    assert deterministic_order(rows)==deterministic_order(rows)
    strata=interedit_strata(rows)
    assert set(strata)=={(kind,bucket) for kind in ("add","remove","local","texture") for bucket in ("small","medium","large")}
    selected,reserve,events=select_interedit(rows,36)
    assert len(selected)==36 and len({row["sample_uid"] for row in selected})==36
    assert len(reserve)==12
    # Round-robin must consume one sample per source before a second from that source.
    for members in strata.values():
        groups=[row["group_id"] for row in members]
        first_repeat=next((i for i,g in enumerate(groups) if g in groups[:i]),len(groups))
        assert len(set(groups[:first_repeat]))==first_repeat


def test_synthetic_four_dataset_fixed_200_with_translation_backfill(tmp_path):
    pools={
        "magicbrush":[synthetic_record("magicbrush",i) for i in range(8)],
        "crispedit":[synthetic_record("crispedit",i) for i in range(42)],
        "scaleedit":[synthetic_record("scaleedit",i) for i in range(30)],
    }
    inter=[]
    mapping={"Add":"add","Remove":"remove","Local":"local_attribute","Texture":"texture"}
    for i in range(140):
        kind=tuple(mapping)[i%4]
        inter.append(synthetic_record("interedit",i,original_type=kind,canonical_type=mapping[kind]))
    # Force several provisionally selected rows through translation QA failure;
    # deterministic reserves must refill them without runtime replacement.
    for iteration in range(5):
        selected,_,_=select_interedit(inter,128)
        chosen=next(row for row in selected if "测试" not in row["instruction_original"])
        target=next(row for row in inter if row["sample_uid"]==chosen["sample_uid"])
        target["instruction_original"]=f"测试编辑 {iteration}"
        target["instruction_en"]="";target["language_original"]="zho_Hans";target["translation_status"]="pending"
        target["sample_uid"]=sample_uid_for(target)
    pools["interedit"]=inter
    paths={}
    for name,rows in pools.items():
        paths[name]=tmp_path/f"{name}.jsonl";write_jsonl(paths[name],rows)
    cache=tmp_path/"translations.jsonl"
    cache.write_text("".join(json.dumps({"source_text":row["instruction_original"],
        "translated_text":row["instruction_original"],"cache_key":translation_cache_key(
            row["instruction_original"],"nllb","fixture-translation-rev","zho_Hans","eng_Latn",
            {"do_sample":False,"num_beams":1,"max_new_tokens":128}),
        "status":"qa_flagged"})+"\n" for row in inter if "测试" in row["instruction_original"]))
    config=yaml.safe_load(Path("configs/data/fixed_200k.yaml").read_text())
    config["train_total"]=200;config["dataset_policy"]["crispedit"]["cap"]=39;config["dataset_policy"]["scaleedit"]["cap"]=25
    config_path=tmp_path/"config.yaml";config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    output=tmp_path/"fixed"
    command=[sys.executable,"scripts/data/build_fixed_200k_corpus.py","--config",str(config_path),
             "--magicbrush-pool",str(paths["magicbrush"]),"--crispedit-pool",str(paths["crispedit"]),
             "--scaleedit-pool",str(paths["scaleedit"]),"--interedit-pool",str(paths["interedit"]),
             "--translation-cache",str(cache),"--output-dir",str(output),"--allow-nonproduction-total"]
    environment=os.environ.copy();environment["TRANSLATOR_REVISION"]="fixture-translation-rev"
    subprocess.run(command,check=True,cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,env=environment)
    final=[json.loads(line) for line in (output/"train_200.jsonl").read_text().splitlines()]
    assert len(final)==len({row["sample_uid"] for row in final})==200
    assert [row["manifest_index"] for row in final]==list(range(200))
    counts={name:sum(row["dataset_name"]==name for row in final) for name in pools}
    assert counts=={"magicbrush":8,"crispedit":39,"scaleedit":25,"interedit":128}
    rejections=[json.loads(line) for line in (output/"translation_rejections.jsonl").read_text().splitlines()]
    backfills=[json.loads(line) for line in (output/"backfill_history.jsonl").read_text().splitlines()]
    assert rejections and any("copy_output" in row["qa_flags"] for row in rejections)
    assert any(row["reason"]=="translation_qa" for row in backfills)


def test_missing_translation_holds_provisional_slot_without_consuming_reserve(tmp_path):
    pools = {
        "magicbrush": [synthetic_record("magicbrush", 0)],
        "crispedit": [synthetic_record("crispedit", i) for i in range(3)],
        "scaleedit": [synthetic_record("scaleedit", 0)],
        "interedit": [synthetic_record(
            "interedit", i, original_type=("Add", "Remove", "Local", "Texture")[i % 4],
            canonical_type={"Add":"add","Remove":"remove","Local":"local_attribute","Texture":"texture"}[("Add", "Remove", "Local", "Texture")[i % 4]],
        ) for i in range(12)],
    }
    for row in pools["crispedit"]:
        row["instruction_original"] = f"添加一只狗 {row['group_id']}"
        row["instruction_en"] = ""
        row["language_original"] = "zho_Hans"
        row["translation_status"] = "pending"
        row["sample_uid"] = sample_uid_for(row)
    pending_uid = deterministic_order(pools["crispedit"])[0]["sample_uid"]
    paths = {}
    for name, rows in pools.items():
        paths[name] = tmp_path / f"{name}.jsonl"; write_jsonl(paths[name], rows)
    config = yaml.safe_load(Path("configs/data/fixed_200k.yaml").read_text())
    config["train_total"] = 8; config["strict_dataset_counts"] = True
    config["dataset_policy"]["magicbrush"] = {"fixed_count": 1}
    config["dataset_policy"]["crispedit"]["cap"] = 1
    config["dataset_policy"]["scaleedit"]["cap"] = 1
    config_path = tmp_path / "config.yaml"; config_path.write_text(yaml.safe_dump(config))
    cache = tmp_path / "cache.jsonl"; cache.write_text("")
    output = tmp_path / "fixed"
    command = [sys.executable, "scripts/data/build_fixed_200k_corpus.py", "--config", str(config_path),
        "--magicbrush-pool", str(paths["magicbrush"]), "--crispedit-pool", str(paths["crispedit"]),
        "--scaleedit-pool", str(paths["scaleedit"]), "--interedit-pool", str(paths["interedit"]),
        "--translation-cache", str(cache), "--output-dir", str(output), "--allow-nonproduction-total"]
    environment = os.environ.copy(); environment["TRANSLATOR_REVISION"] = "fixture-translation-rev"
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=environment)
    assert result.returncode == 42
    required = [json.loads(line) for line in (output / "translation_required.jsonl").read_text().splitlines()]
    assert [row["sample_uid"] for row in required] == [pending_uid]
    reserve = [json.loads(line) for line in (output / "reserve_order.jsonl").read_text().splitlines()]
    assert pending_uid not in {row["sample_uid"] for row in reserve}
