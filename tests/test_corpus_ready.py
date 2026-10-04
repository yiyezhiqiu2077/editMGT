import json
from pathlib import Path

import pytest
import yaml

from scripts.data.finalize_corpus_ready import finalize
from src.explicit_region.contracts import sha256_file
from src.explicit_region.fixed_corpus import (
    INTEGRITY_CHECK_NAMES, READY_REQUIRED_FILES, VALIDATION_NAMES,
    canonical_freeze_order, verify_corpus_ready, write_corpus_ready,
)
from tests.fixed200k_helpers import synthetic_record


def write_json(path, value):
    Path(path).write_text(json.dumps(value))


def fixture_artifacts(tmp_path, train_count=1):
    files = {name: tmp_path / f"{name}.json" for name in READY_REQUIRED_FILES}
    for path in files.values():
        path.write_text("")
    for name in ("duplicate_report", "translation_report"):
        write_json(files[name], {})
    from PIL import Image
    for name in files:
        if name.startswith("montage_"):
            Image.new("RGB", (8, 8)).save(files[name], format="JPEG")
    rows = canonical_freeze_order([synthetic_record("magicbrush", i) for i in range(train_count)])
    files["train_200k"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    files["final_selection"].write_bytes(files["train_200k"].read_bytes())
    val_rows = {}
    for name in VALIDATION_NAMES:
        dataset = name.split("_", 1)[0]
        val_rows[name] = [synthetic_record(dataset, 1000000 + i) for i in range(128)]
        files[f"validation_{name}"].write_text("".join(json.dumps(row) + "\n" for row in val_rows[name]))
    contract = {"backend": "nllb", "immutable_revision": "fixture-rev", "src_lang": "zho_Hans",
                "tgt_lang": "eng_Latn", "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": 128}}
    files["selection_config"].write_text(yaml.safe_dump({"schema_version": "fixed200k-v2", "train_total": 200000}))
    write_json(files["validation_meta"], {
        "schema": "fixed200k-validation-v2", "status": "FROZEN",
        "counts": {f"{name}.jsonl": 128 for name in VALIDATION_NAMES},
        "sha256": {f"{name}.jsonl": sha256_file(files[f"validation_{name}"]) for name in VALIDATION_NAMES},
        "translation_contract": contract, "selection_config_sha256": sha256_file(files["selection_config"]),
        "translation_rejections_sha256": sha256_file(files["validation_translation_rejections"]),
    })
    metadata = {"total_rows": 200000, "dataset_counts": {"magicbrush": 200000},
                "dataset_revisions": {"magicbrush": ["fixture-rev1"]},
                "unique_source_counts": {"magicbrush": 200000}, "created_from_git_sha": "fixture-code"}
    write_json(files["train_meta"], {
        **metadata, "schema_version": "fixed200k-meta-v2", "translation_contract": contract,
        "train_sha256": sha256_file(files["train_200k"]),
        "selection_config_sha256": sha256_file(files["selection_config"]),
        "translation_cache_sha256": sha256_file(files["translation_cache"]),
        "validation_hashes": {str(files[f"validation_{name}"]): sha256_file(files[f"validation_{name}"])
                              for name in VALIDATION_NAMES if name != "magicbrush_probe128"},
    })
    write_json(files["selection_report"], {**metadata, "schema_version": "fixed200k-selection-v2"})
    counts = {"train_200k": 200000, **{f"validation_{name}": 128 for name in VALIDATION_NAMES}}
    write_json(files["audit_report"], {
        "schema": "fixed200k-audit-v2", "status": "PASS", "total": 200000,
        "integrity": {"mode": "exhaustive", "status": "PASS", "resolution": 1024,
            "total_rows": sum(counts.values()), "manifests": {
                name: {"sha256": sha256_file(files[name]), "status": "PASS", "mode": "exhaustive",
                       "total_rows": count, "attempted_rows": count,
                       "completed_checks": {check: count for check in INTEGRITY_CHECK_NAMES}, "failure_counts": {}}
                for name, count in counts.items()}},
        "vq_containment": {"status": "PASS", "mode": "sampled", "dtype": "float32", "datasets": {
            "magicbrush": {"status": "PASS", "mode": "sampled", "samples": min(500, len(rows)),
                           "sample_uids": [row["sample_uid"] for row in rows[:500]]}}},
        "montages": {files[name].name: sha256_file(files[name]) for name in files if name.startswith("montage_")},
    })
    return files, metadata


def marker_payload(files, metadata):
    return {**metadata, "status": "READY", "schema_version": "fixed200k-ready-v2",
            "files": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in files.items()}}


@pytest.mark.parametrize("payload", [
    {"status": "READY"}, {"status": "READY", "schema_version": "other", "total_rows": 200000, "files": {}},
    {"status": "READY", "schema_version": "fixed200k-ready-v2", "total_rows": 2, "files": {}},
])
def test_corpus_ready_rejects_incomplete_markers(tmp_path, payload):
    marker = tmp_path / "CORPUS_READY.json"; write_json(marker, payload)
    with pytest.raises(RuntimeError, match="CORPUS_NOT_READY"):
        verify_corpus_ready(marker)


def test_write_corpus_ready_cannot_override_reserved_schema(tmp_path):
    with pytest.raises(ValueError, match="reserved READY"):
        write_corpus_ready(tmp_path / "marker.json", files={}, metadata={"status": "READY"})


def test_ready_counts_actual_rows_not_claimed_total(tmp_path):
    files, metadata = fixture_artifacts(tmp_path)
    marker = tmp_path / "CORPUS_READY.json"; write_json(marker, marker_payload(files, metadata))
    with pytest.raises(RuntimeError, match="exactly 200000 unique rows, got 1"):
        verify_corpus_ready(marker)
    with pytest.raises(ValueError, match="exactly 200000 unique rows"):
        write_corpus_ready(marker, files=files, metadata=metadata)


def test_ready_requires_all_artifacts_and_current_hashes(tmp_path):
    files, metadata = fixture_artifacts(tmp_path)
    payload = marker_payload(files, metadata)
    marker = tmp_path / "CORPUS_READY.json"
    payload["files"].pop("validation_interedit_aux128")
    write_json(marker, payload)
    with pytest.raises(RuntimeError, match="required readiness artifacts missing"):
        verify_corpus_ready(marker)
    payload = marker_payload(files, metadata)
    write_json(marker, payload)
    files["train_200k"].write_text("mutated\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_corpus_ready(marker)


def test_ready_requires_frozen_english_validation(tmp_path):
    files, metadata = fixture_artifacts(tmp_path)
    rows = [json.loads(line) for line in files["validation_crispedit_aux128"].read_text().splitlines()]
    rows[-1]["instruction_en"] = "   "
    files["validation_crispedit_aux128"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    marker = tmp_path / "CORPUS_READY.json"; write_json(marker, marker_payload(files, metadata))
    with pytest.raises(RuntimeError, match="invalid frozen instruction_en"):
        verify_corpus_ready(marker)


def test_finalizer_removes_stale_marker_on_missing_artifact(tmp_path):
    marker = tmp_path / "CORPUS_READY.json"; marker.write_text("stale")
    with pytest.raises(FileNotFoundError, match="required readiness artifact"):
        finalize(tmp_path, tmp_path / "config", tmp_path / "cache", "git-id")
    assert not marker.exists()


def test_ready_exhaustive_audit_and_real_200k_stream(tmp_path):
    # CPU-only synthetic canonical rows, no asset/model download or image corpus.
    # The production row invariant is not monkeypatched down for this test.
    files, metadata = fixture_artifacts(tmp_path, train_count=200000)
    marker = tmp_path / "CORPUS_READY.json"
    write_corpus_ready(marker, files=files, metadata=metadata)
    assert verify_corpus_ready(marker)["total_rows"] == 200000
    audit = json.loads(files["audit_report"].read_text())
    audit["integrity"]["manifests"]["train_200k"]["completed_checks"]["decode"] = 500
    write_json(files["audit_report"], audit)
    # Even with the new report hash rebound, sampled/partial coverage cannot pass.
    write_json(marker, marker_payload(files, metadata))
    with pytest.raises(RuntimeError, match="audit checks incomplete"):
        verify_corpus_ready(marker)
