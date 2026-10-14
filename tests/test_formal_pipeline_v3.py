from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import yaml

from src.explicit_region.fixed_corpus import verify_corpus_ready, write_corpus_ready
from src.explicit_region.formal_pipeline import (
    HF_ASSETS, STAGES, asset_path, invalidate_from, load_formal_assets,
    match_schema_contract, require_selection_ready, sha256_file,
    stage_fingerprint, stable_json_hash, validate_resolved_revision,
    valid_stage_marker, verify_formal_ready, verify_magicbrush_test_counts, write_stage_marker,
)
from src.explicit_region.language import translation_qa_flags


ROOT = Path(__file__).resolve().parents[1]


def test_formal_asset_manifest_is_complete_and_pinned(tmp_path):
    config = load_formal_assets(ROOT / "configs/formal_assets.yaml")
    assert HF_ASSETS <= set(config["assets"])
    assert all(len(config["assets"][name]["revision"]) == 40 for name in HF_ASSETS)
    assert asset_path(tmp_path, config["assets"]["editmgt"]) == tmp_path / "models/editmgt"
    with pytest.raises(RuntimeError, match="ASSET_REVISION_MISMATCH"):
        validate_resolved_revision("a" * 40, "b" * 40)


def test_schema_contract_and_edit_type_coverage():
    mappings = yaml.safe_load((ROOT / "configs/data/edit_type_mapping.yaml").read_text())["datasets"]
    for name in ("crispedit", "scaleedit"):
        contract = yaml.safe_load((ROOT / f"configs/data/schema/{name}.yaml").read_text())
        assert set(contract["known_edit_types"]) == set(mappings[name])
        match_schema_contract(contract, contract["required_columns"])
        bad = dict(contract["required_columns"]); bad.pop(next(iter(bad)))
        with pytest.raises(RuntimeError, match="SCHEMA_CONTRACT_MISMATCH"):
            match_schema_contract(contract, bad)


def test_stage_marker_invalidation(tmp_path):
    config = tmp_path / "config.yaml"; config.write_text("value: 1\n")
    first = stage_fingerprint(stage=STAGES[0], git="a" * 40, config_files=[config])
    output = tmp_path / "output.json"; output.write_text("ok\n")
    first_marker = tmp_path / STAGES[0] / "_SUCCESS.json"
    write_stage_marker(first_marker, first, {"asset": "b" * 40}, [output])
    assert valid_stage_marker(first_marker, first, [output])
    output.write_text("mutated\n")
    assert not valid_stage_marker(first_marker, first, [output])
    later = stage_fingerprint(stage=STAGES[1], git="a" * 40, config_files=[config])
    write_stage_marker(tmp_path / STAGES[1] / "_SUCCESS.json", later, {})
    config.write_text("value: 2\n")
    changed = stage_fingerprint(stage=STAGES[0], git="a" * 40, config_files=[config])
    assert changed["fingerprint"] != first["fingerprint"]
    assert invalidate_from(tmp_path, STAGES[0]) == [STAGES[0], STAGES[1]]


def test_machine_translation_and_selection_gate():
    assert "han_remaining" in translation_qa_flags("请添加猫", "add 猫")
    with pytest.raises(RuntimeError, match="SELECTION_RULE_NOT_PREREGISTERED"):
        require_selection_ready(ROOT / "configs/eval/formal.yaml")


def test_v3_corpus_and_formal_ready_contract(tmp_path):
    train = tmp_path / "train.jsonl"; train.write_text("frozen\n")
    git = "a" * 40
    identity = stable_json_hash({"assets": "pinned"})
    corpus_path = tmp_path / "CORPUS_READY.json"
    metadata = {
        "schema_version": "fixed200k-ready-v3", "git_sha": git,
        "formal_assets_sha256": identity, "selection_config_sha256": "b" * 64,
        "train_manifest_sha256": sha256_file(train), "total_rows": 200000,
        "dataset_counts": {"magicbrush": 1}, "unique_source_counts": {"magicbrush": 1},
        **{key: "PASS" for key in ("schema_contract", "asset_integrity", "translation", "dedup", "geometry", "train_dev_leakage", "exact_200k", "vq_audit")},
    }
    write_corpus_ready(corpus_path, files={"train_200k": train}, metadata=metadata)
    corpus_payload = verify_corpus_ready(corpus_path)
    assert corpus_payload["schema_version"] == "fixed200k-ready-v3"
    assert not ({"human_approved", "reviewer", "manual_go"} & set(corpus_payload))
    assets_path = tmp_path / "formal_assets.json"
    assets = {"git_sha": git, "formal_identity_sha256": identity,
              "corpus_ready": {"sha256": sha256_file(corpus_path)}}
    assets_path.write_text(json.dumps(assets))
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_text(json.dumps({"status": "PASS", "nan_inf_count": 0, "no_oom": True, "no_nccl_error": True}))
    ready_path = tmp_path / "FORMAL_READY.json"
    ready_path.write_text(json.dumps({"status": "READY", "world_size": 8, "git_sha": git,
        "formal_assets_sha256": sha256_file(assets_path), "corpus_ready_sha256": sha256_file(corpus_path),
        "smoke_verification": {"path": str(smoke_path), "sha256": sha256_file(smoke_path)}}))
    assert verify_formal_ready(assets_path, corpus_path, ready_path, git)["status"] == "READY"
    ready_path.write_text(json.dumps({"status": "READY", "world_size": 4}))
    with pytest.raises(RuntimeError, match="FORMAL_PIPELINE_NOT_READY"):
        verify_formal_ready(assets_path, corpus_path, ready_path, git)


def test_magicbrush_test_contract_rejects_wrong_identity():
    with pytest.raises(RuntimeError, match="MAGICBRUSH_TEST_IDENTITY_MISMATCH"):
        verify_magicbrush_test_counts({"sessions": 535, "turns": 1052}, [])


def test_run_formal_refuses_missing_formal_ready(tmp_path):
    env = {"ASSET_ROOT": str(tmp_path), "DERIVED_ROOT": str(tmp_path / "derived"),
           "CONFIRM_FORMAL_RUN": "YES"}
    # Preserve the real environment (notably uv's location), then override the gate inputs.
    import os
    completed = subprocess.run(["bash", "scripts/train/run_formal.sh", "configs/train/cluster_8g_e1.yaml", "--run"], cwd=ROOT, env=os.environ | env, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "FORMAL_PIPELINE_NOT_READY" in completed.stderr + completed.stdout


def test_train_verifier_does_not_require_selection_ready(tmp_path):
    import os
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    fixed = tmp_path / "derived/fixed200k"; fixed.mkdir(parents=True)
    artifacts = tmp_path / "artifacts"; artifacts.mkdir()
    train = fixed / "train_200k.jsonl"; train.write_text("frozen\n")
    identity = stable_json_hash({"assets": "pinned"})
    corpus = fixed / "CORPUS_READY.json"
    metadata = {"schema_version": "fixed200k-ready-v3", "git_sha": git,
        "formal_assets_sha256": identity, "selection_config_sha256": "b" * 64,
        "train_manifest_sha256": sha256_file(train), "total_rows": 200000,
        "dataset_counts": {}, "unique_source_counts": {},
        **{key: "PASS" for key in ("schema_contract", "asset_integrity", "translation", "dedup", "geometry", "train_dev_leakage", "exact_200k", "vq_audit")}}
    write_corpus_ready(corpus, files={"train_200k": train}, metadata=metadata)
    formal = artifacts / "formal_assets.json"
    formal.write_text(json.dumps({"git_sha": git, "formal_identity_sha256": identity,
                                  "corpus_ready": {"sha256": sha256_file(corpus)}}))
    smoke = artifacts / "smoke.json"
    smoke.write_text(json.dumps({"status": "PASS", "nan_inf_count": 0, "no_oom": True, "no_nccl_error": True}))
    ready = artifacts / "FORMAL_READY.json"
    ready.write_text(json.dumps({"status": "READY", "world_size": 8, "git_sha": git,
        "formal_assets_sha256": sha256_file(formal), "corpus_ready_sha256": sha256_file(corpus),
        "smoke_verification": {"path": str(smoke), "sha256": sha256_file(smoke)}}))
    env = os.environ | {"ASSET_ROOT": str(tmp_path), "DERIVED_ROOT": str(tmp_path / "derived"), "CONFIRM_FORMAL_RUN": "YES"}
    completed = subprocess.run(["uv", "run", "python", "scripts/setup/verify_formal_pipeline.py",
        "--mode", "train", "--formal-assets", str(formal), "--corpus-ready", str(corpus),
        "--formal-ready", str(ready), "--selection-config", "configs/eval/formal.yaml"],
        cwd=ROOT, env=env, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)['status'] == 'READY'
    completed = subprocess.run(["uv", "run", "python", "scripts/setup/verify_formal_pipeline.py",
        "--mode", "select", "--formal-assets", str(formal), "--corpus-ready", str(corpus),
        "--formal-ready", str(ready), "--selection-config", "configs/eval/formal.yaml"],
        cwd=ROOT, env=env, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "SELECTION_RULE_NOT_PREREGISTERED" in completed.stderr
