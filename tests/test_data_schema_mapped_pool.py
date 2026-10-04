import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image
import yaml

from scripts.data.build_schema_mapped_canonical import build_pool, iter_rows
from src.explicit_region.canonical import expected_locator_hash, image_from_locator
from src.explicit_region.contracts import sha256_file


def png(mode="RGB", size=(20, 20), fill=100):
    stream = io.BytesIO(); Image.new(mode, size, fill).save(stream, format="PNG")
    return stream.getvalue()


def fixture(tmp_path):
    root = tmp_path / "raw"; root.mkdir()
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "REAL_SCHEMA_AUDITED", "dataset_name": "scaleedit",
                                "dataset_revision": "revision1", "root": str(root)}))
    mapping = {"schema_audit_status": "REAL_SCHEMA_AUDITED", "dataset_name": "scaleedit",
               "dataset_revision": "revision1", "format": "parquet", "files_glob": "*.parquet",
               "mask_semantics": "edit_region", "schema_audit": {
                   "path": str(audit), "sha256": sha256_file(audit), "dataset_revision": "revision1"},
               "fields": {"source": {"storage": "embedded_parquet", "column": "source"},
                          "target": {"storage": "embedded_parquet", "column": "target"},
                          "region": {"storage": "embedded_parquet", "column": "mask"},
                          "instruction": "instruction", "edit_type": "edit_type",
                          "group_id_fields": ["source_shard", "sample_id"]}}
    mapping_path = tmp_path / "mapping.yaml"; mapping_path.write_text(yaml.safe_dump(mapping))
    taxonomy = tmp_path / "taxonomy.yaml"
    taxonomy.write_text(yaml.safe_dump({"datasets": {"scaleedit": {"color": "local_attribute"}}}))
    return root, mapping, mapping_path, taxonomy


def raw(index, **overrides):
    return {"source": png(size=(20, 20)), "target": png(size=(40, 40)),
            "mask": png("L", (40, 40), 255), "instruction": f"Make object {index} red",
            "edit_type": "color", "source_shard": "shard0", "sample_id": index,
            "qc": "MASK_REVIEW", "unused_image": b"unrelated", **overrides}


def test_streamed_mapping_projects_only_relevant_columns_and_global_indices(tmp_path, monkeypatch):
    root, mapping, _, _ = fixture(tmp_path)
    pq.write_table(pa.Table.from_pylist([raw(i) for i in range(35)]), root / "data.parquet", row_group_size=9)
    original = pq.ParquetFile
    observed = []

    class Projected:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.inner.close()
        def __getattr__(self, name):
            return getattr(self.inner, name)
        def iter_batches(self, **kwargs):
            observed.append(kwargs)
            yield from self.inner.iter_batches(**kwargs)

    monkeypatch.setattr(pq, "ParquetFile", Projected)
    monkeypatch.setattr(pq, "read_table", lambda *args, **kwargs: pytest.fail("whole table read forbidden"))
    rows = list(iter_rows(root, mapping))
    assert [index for _, index, _ in rows] == list(range(35))
    assert observed[0]["batch_size"] == 16
    assert not {"qc", "unused_image"} & set(observed[0]["columns"])


def test_pool_eligibility_and_composite_group_retains_mask_review(tmp_path):
    root, _, mapping, taxonomy = fixture(tmp_path)
    rows = [raw(0), raw(0, source_shard="shard1"), raw(2, instruction="   "),
            raw(3, mask=png("L", (40, 40), 0)), raw(4, source=png(size=(30, 20)))]
    pq.write_table(pa.Table.from_pylist(rows), root / "data.parquet", row_group_size=2)
    output = tmp_path / "pool.jsonl"
    report = build_pool(root, mapping, taxonomy, output)
    assert report["status"] == "PASS"
    assert report["counts"] == {"input_rows": 5, "decoded_rows": 5, "eligible_rows": 2}
    assert report["rejection_counts"] == {"empty_instruction": 1, "empty_region": 1, "unaligned_aspect_ratio": 1}
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[0]["group_id"] != records[1]["group_id"]
    # Same-aspect different native resolutions are valid, no artificial QC filter.
    assert image_from_locator(records[0]["source_locator"], root).size == (20, 20)
    assert image_from_locator(records[0]["region_locator"], root).size == (40, 40)
    for row in records:
        for role in ("source", "target", "region"):
            assert expected_locator_hash(row[f"{role}_locator"], root) == row[f"{role}_sha256"]
    assert output.with_suffix(".report.json").is_file()
    assert len(output.with_suffix(".rejections.jsonl").read_text().splitlines()) == 3


def test_corrupt_embedded_asset_hardfails_without_partial_output(tmp_path):
    root, _, mapping, taxonomy = fixture(tmp_path)
    pq.write_table(pa.Table.from_pylist([raw(0), raw(1, source=b"invalid PNG")]), root / "data.parquet")
    output = tmp_path / "pool.jsonl"
    with pytest.raises(Exception, match="cannot identify image"):
        build_pool(root, mapping, taxonomy, output)
    assert not output.exists() and not output.with_suffix(".jsonl.tmp").exists()
    assert json.loads(output.with_suffix(".report.json").read_text())["status"] == "FAIL"


@pytest.mark.parametrize("case", ["valid", "parse_error", "no_policy", "bad_annotation", "bad_status", "corrupt_source", "nonempty_corrupt"])
def test_explicit_empty_annotation_policy_never_hides_corruption(tmp_path, case):
    root, value, mapping, taxonomy = fixture(tmp_path)
    policy = {
        "action": "reject_annotated_empty_bytes",
        "zero_columns": ["mask_width", "mask_height", "mask_sum", "mask_area"],
        "status_requirements": [
            {"column": "mask_status", "values": ["GROUND_FAIL", "PARSE_ERROR"]},
            {"column": "ground_status", "values": ["GROUND_FAIL", "PARSE_ERROR"]},
            {"column": "mask_qc", "values": ["GROUND_FAIL"]},
            {"column": "ground_qc", "values": ["GROUND_FAIL"]},
        ],
        "empty_list_columns": ["instances"],
    }
    if case != "no_policy":
        value["empty_region_policy"] = policy
    mapping.write_text(yaml.safe_dump(value))
    row = raw(
        1, mask=b"", mask_width=0, mask_height=0, mask_sum=0, mask_area=0.0,
        mask_status="GROUND_FAIL", ground_status="GROUND_FAIL",
        mask_qc="GROUND_FAIL", ground_qc="GROUND_FAIL", instances=[])
    if case == "parse_error":
        row["mask_status"] = row["ground_status"] = "PARSE_ERROR"
    if case == "bad_annotation":
        row["mask_sum"] = 10
    if case == "bad_status":
        row["mask_status"] = "OK"
    if case == "corrupt_source":
        row["source"] = b"invalid image"
    if case == "nonempty_corrupt":
        row["mask"] = b"invalid PNG"
    pq.write_table(pa.Table.from_pylist([row]), root / "data.parquet")
    output = tmp_path / "pool.jsonl"
    if case in {"valid", "parse_error"}:
        report = build_pool(root, mapping, taxonomy, output)
        assert report["status"] == "PASS"
        assert report["rejection_counts"] == {"annotated_empty_region": 1}
        assert report["counts"] == {"input_rows": 1}
        assert output.read_text() == ""
        rejection = json.loads(output.with_suffix(".rejections.jsonl").read_text())
        assert rejection["annotation_status"]["mask_status"] == row["mask_status"]
        assert rejection["annotation_status"]["ground_status"] == row["ground_status"]
        assert rejection["annotation_lists"] == {"instances": []}
        assert rejection["asset_hashes"]["region"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    else:
        with pytest.raises(Exception, match="disagree|cannot identify image"):
            build_pool(root, mapping, taxonomy, output)
        assert not output.exists()


@pytest.mark.parametrize("change", ["hash", "revision", "root"])
def test_mapping_requires_verified_audit_binding(tmp_path, change):
    root, value, mapping, taxonomy = fixture(tmp_path)
    pq.write_table(pa.Table.from_pylist([raw(0)]), root / "data.parquet")
    if change == "hash":
        value["schema_audit"]["sha256"] = "0" * 64
    elif change == "revision":
        value["dataset_revision"] = "wrong"
    else:
        audit_path = Path(value["schema_audit"]["path"])
        audit = json.loads(audit_path.read_text()); audit["root"] = str(tmp_path / "another")
        audit_path.write_text(json.dumps(audit)); value["schema_audit"]["sha256"] = sha256_file(audit_path)
    mapping.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="schema audit"):
        build_pool(root, mapping, taxonomy, tmp_path / "pool.jsonl")
