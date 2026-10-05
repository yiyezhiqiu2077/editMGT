import json
from pathlib import Path
import subprocess
import sys
import os

import yaml
from PIL import Image

from src.explicit_region.canonical import sample_uid_for, sha256_bytes
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


def test_cold_cache_pending_preserves_quota_and_requests_before_exhaustion(tmp_path):
    # Exactly enough Inter-Edit rows: treating missing translation as rejection
    # used to consume reserve and raise before writing any translation requests.
    pools = {"magicbrush": [synthetic_record("magicbrush", 0)],
             "crispedit": [], "scaleedit": [], "interedit": []}
    for i, kind in enumerate(("Add", "Remove", "Local", "Texture")):
        row = synthetic_record("interedit", i, original_type=kind,
                               canonical_type="local_attribute" if kind == "Local" else kind.lower())
        row.update(instruction_original=f"把物体改成红色 {i}", instruction_en="",
                   language_original="zho_Hans", translation_status="pending")
        row["sample_uid"] = sample_uid_for(row)
        pools["interedit"].append(row)
    config = yaml.safe_load(Path("configs/data/fixed_200k.yaml").read_text())
    config["train_total"] = 5
    config["translation"]["immutable_revision"] = "fixture-translation-rev"
    config_path = tmp_path / "config.yaml"; config_path.write_text(yaml.safe_dump(config))
    cache = tmp_path / "cache.jsonl"; cache.write_text("")
    output = tmp_path / "fixed"
    command = [sys.executable, "scripts/data/build_fixed_200k_corpus.py", "--config", str(config_path),
               "--translation-cache", str(cache), "--output-dir", str(output), "--allow-nonproduction-total"]
    for name, rows in pools.items():
        path = tmp_path / f"{name}.jsonl"; write_jsonl(path, rows)
        command.extend([f"--{name}-pool", str(path)])
    cold = subprocess.run(command, capture_output=True, text=True)
    assert cold.returncode == 42, cold.stderr
    requested = [json.loads(line) for line in (output / "translation_required.jsonl").read_text().splitlines()]
    assert len(requested) == 4
    assert {r["sample_uid"] for r in requested} == {r["sample_uid"] for r in pools["interedit"]}
    assert not (output / "train_5.jsonl").exists()
    cache_rows = []
    for i, row in enumerate(pools["interedit"]):
        text = row["instruction_original"]
        cache_rows.append({"source_text": text, "translated_text": f"Make it red {i}",
                           "cache_key": translation_cache_key(text, "nllb", "fixture-translation-rev",
                               "zho_Hans", "eng_Latn", config["translation"]["decoding"])})
    write_jsonl(cache, cache_rows)
    warm = subprocess.run(command, capture_output=True, text=True)
    assert warm.returncode == 0, warm.stderr
    frozen = (output / "train_5.jsonl").read_bytes()
    subprocess.run(command, check=True, capture_output=True)
    assert (output / "train_5.jsonl").read_bytes() == frozen
    assert not (output / "translation_required.jsonl").exists()
    report = json.loads((output / "selection_report.json").read_text())
    assert report["dataset_counts"] == {"magicbrush": 1, "interedit": 4}
    assert report["interedit_requested_quota"] == 4


def test_general_pending_does_not_transfer_quota_to_interedit(tmp_path):
    pools = {"magicbrush": [synthetic_record("magicbrush", i) for i in range(2)],
             "crispedit": [], "scaleedit": [], "interedit": []}
    for row in pools["magicbrush"]:
        row.update(instruction_original="把物体改成红色", instruction_en="", language_original="zho_Hans",
                   translation_status="pending")
        row["sample_uid"] = sample_uid_for(row)
    config = yaml.safe_load(Path("configs/data/fixed_200k.yaml").read_text())
    config["train_total"] = 2; config["translation"]["immutable_revision"] = "fixture-rev"
    config_path = tmp_path / "config.yaml"; config_path.write_text(yaml.safe_dump(config))
    command = [sys.executable, "scripts/data/build_fixed_200k_corpus.py", "--config", str(config_path),
               "--translation-cache", str(tmp_path / "absent-cache.jsonl"), "--output-dir", str(tmp_path / "out"),
               "--allow-nonproduction-total"]
    for name, rows in pools.items():
        path = tmp_path / f"{name}.jsonl"; write_jsonl(path, rows); command.extend([f"--{name}-pool", str(path)])
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 42, result.stderr
    assert json.loads((tmp_path / "out/translation_required.jsonl").read_text())["dataset"] == "magicbrush"


def test_magicbrush_geometry_sanitation_filters_bad_rows_before_freeze(tmp_path):
    roots = {name: tmp_path / name for name in ("magicbrush", "crispedit", "scaleedit")}
    for root in roots.values():
        (root / "source").mkdir(parents=True)
        (root / "target").mkdir(parents=True)
        (root / "region").mkdir(parents=True)

    def materialize_magicbrush(index: int, *, source_size=(32, 32), target_size=(32, 32), mask_size=(32, 32)):
        row = synthetic_record("magicbrush", index)
        Image.new("RGB", source_size, (10 + index, 20, 30)).save(roots["magicbrush"] / row["source_locator"]["relative_path"])
        Image.new("RGB", target_size, (40, 50 + index, 60)).save(roots["magicbrush"] / row["target_locator"]["relative_path"])
        Image.new("L", mask_size, 255).save(roots["magicbrush"] / row["region_locator"]["relative_path"])
        row["source_sha256"] = sha256_bytes((roots["magicbrush"] / row["source_locator"]["relative_path"]).read_bytes())
        row["target_sha256"] = sha256_bytes((roots["magicbrush"] / row["target_locator"]["relative_path"]).read_bytes())
        row["region_sha256"] = sha256_bytes((roots["magicbrush"] / row["region_locator"]["relative_path"]).read_bytes())
        row["sample_uid"] = sample_uid_for(row)
        return row

    magicbrush_rows = [
        materialize_magicbrush(0),
        materialize_magicbrush(1, source_size=(1024, 1023), target_size=(1024, 1024), mask_size=(1024, 1024)),
    ]
    pools = {
        "magicbrush": magicbrush_rows,
        "crispedit": [synthetic_record("crispedit", 0)],
        "scaleedit": [],
        "interedit": [synthetic_record("interedit", i, original_type="Add", canonical_type="add") for i in range(2)],
    }
    config = yaml.safe_load(Path("configs/data/fixed_200k.yaml").read_text())
    config["train_total"] = 4
    config["dataset_policy"]["crispedit"]["cap"] = 1
    config["dataset_policy"]["scaleedit"]["cap"] = 0
    config["translation"]["immutable_revision"] = "fixture-rev"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    output = tmp_path / "fixed"
    cache = tmp_path / "cache.jsonl"
    cache.write_text("")
    command = [
        sys.executable, "scripts/data/build_fixed_200k_corpus.py", "--config", str(config_path),
        "--translation-cache", str(cache), "--output-dir", str(output), "--allow-nonproduction-total",
        "--magicbrush-root", str(roots["magicbrush"]),
    ]
    for name, rows in pools.items():
        path = tmp_path / f"{name}.jsonl"
        write_jsonl(path, rows)
        command.extend([f"--{name}-pool", str(path)])
    environment = os.environ.copy()
    environment["TRANSLATOR_REVISION"] = "fixture-rev"
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, env=environment)
    final = [json.loads(line) for line in (output / "train_4.jsonl").read_text().splitlines()]
    assert len(final) == 4
    assert {row["sample_uid"] for row in final}.isdisjoint({magicbrush_rows[1]["sample_uid"]})
    counts = {name: sum(row["dataset_name"] == name for row in final) for name in pools}
    assert counts == {"magicbrush": 1, "crispedit": 1, "scaleedit": 0, "interedit": 2}
    geometry_rejections = [json.loads(line) for line in (output / "geometry_rejections.jsonl").read_text().splitlines()]
    assert len(geometry_rejections) == 1
    assert geometry_rejections[0]["sample_uid"] == magicbrush_rows[1]["sample_uid"]
    assert geometry_rejections[0]["reason"] == "unaligned_aspect_ratio"
    selection_report = json.loads((output / "selection_report.json").read_text())
    assert selection_report["geometry_sanitation"]["datasets"]["magicbrush"]["rejected"] == 1
    assert selection_report["dataset_counts"] == {"magicbrush": 1, "crispedit": 1, "interedit": 2}
