"""InterEdit shard builder tests use only tiny local CPU tar/metadata fixtures."""
import gzip
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
import pytest

from scripts.data import build_interedit_canonical_pool as builder
from src.explicit_region import canonical
from src.explicit_region.interedit import iter_metadata, parse_record


@pytest.fixture(autouse=True)
def isolated_locator_cache(tmp_path, monkeypatch):
    canonical.clear_locator_caches()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    yield
    canonical.clear_locator_caches()


def png(size=(16, 16), *, mask=False, empty=False, color=32):
    image = Image.new("L" if mask else "RGB", size, 0 if mask else (color, color, color))
    if mask and not empty:
        image.paste(255, (1, 1, size[0] - 1, size[1] - 1))
    stream = io.BytesIO(); image.save(stream, format="PNG")
    return stream.getvalue()


def metadata_row(sample_id, **changes):
    row = {
        "sample_id": sample_id, "source_id": 100 + sample_id, "edit_type": "Add",
        "instruction": f"  Add object {sample_id}  ", "better_data": True,
        "source_archive": "images.tar", "source_file": "source.png",
        "asset_archive": "images.tar", "target_file": "target.png", "mask_file": "mask.png",
    }
    row.update(changes)
    return row


def write_tar(root, members):
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(root / "images.tar", "w") as archive:
        for name, value in members.items():
            info = tarfile.TarInfo(name); info.size = len(value)
            archive.addfile(info, io.BytesIO(value))


def write_metadata(path, rows):
    value = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(value)
    else:
        path.write_text(value, encoding="utf-8")
    return path


def fixture_data(tmp_path):
    root = tmp_path / "raw"
    values = {"source.png": png(), "target.png": png((32, 32), color=72),
              "mask.png": png(mask=True), "empty.png": png(mask=True, empty=True),
              "wide.png": png((32, 16))}
    write_tar(root, values)
    first = write_metadata(tmp_path / "z-first.jsonl.gz", [
        metadata_row(7, instruction="添加 7 个物体", edit_type="Local"),
        metadata_row(8, mask_file="empty.png"),
        metadata_row(9, instruction=" \t "),
        metadata_row(10, source_file="wide.png"),
        metadata_row(11, better_data=False, source_file="does-not-exist.png"),
    ])
    second = write_metadata(tmp_path / "a-second.jsonl", [metadata_row(1), metadata_row(2, edit_type="Texture")])
    return root, [first, second], values


def legacy_row(item, values, revision):
    # Independent reproduction of the old valid-row serialization contract.
    mask = np.asarray(Image.open(io.BytesIO(values["mask.png"])).convert("L")) > 0
    instruction = item.instruction_original.strip()
    han = builder.contains_han(instruction)
    row = {
        "schema_version": canonical.SCHEMA_VERSION, "sample_uid": "0" * 64,
        "manifest_index": None, "dataset_name": "interedit", "dataset_revision": revision,
        "group_id": str(item.source_id),
        "source_locator": {"backend": "tar", "archive": item.source_archive, "member": item.source_file},
        "target_locator": {"backend": "tar", "archive": item.asset_archive, "member": item.target_file},
        "region_locator": {"backend": "tar", "archive": item.asset_archive, "member": item.mask_file},
        "source_sha256": canonical.sha256_bytes(values["source.png"]),
        "target_sha256": canonical.sha256_bytes(values["target.png"]),
        "region_sha256": canonical.sha256_bytes(values["mask.png"]),
        "instruction_original": instruction, "instruction_en": "" if han else instruction,
        "language_original": "zho_Hans" if han else "en", "edit_type_original": item.edit_type,
        "edit_type_canonical": builder.MAPPING[item.edit_type], "mask_semantics": "user_guidance_region",
        "region_fraction": float(mask.mean()), "translation_status": "pending" if han else "passthrough_en",
        "translation_cache_key": None,
    }
    row["sample_uid"] = canonical.sample_uid_for(row)
    return row


def test_serial_parallel_pool_is_identical_in_exact_cli_order(tmp_path):
    root, shards, values = fixture_data(tmp_path)
    first_output = tmp_path / "outputs" / "serial.jsonl"
    second_output = tmp_path / "outputs" / "parallel.jsonl"
    # Parallel comes first so both workers may encounter the same cold tar index.
    parallel = builder.build_pool(shards, interedit_root=root, revision="rev", output=second_output, workers=2)
    serial = builder.build_pool(shards, interedit_root=root, revision="rev", output=first_output, workers=1)
    assert first_output.read_bytes() == second_output.read_bytes()
    accepted_items = [parse_record(metadata_row(7, instruction="添加 7 个物体", edit_type="Local")),
                      parse_record(metadata_row(1)), parse_record(metadata_row(2, edit_type="Texture"))]
    expected = "".join(json.dumps(legacy_row(item, values, "rev"), ensure_ascii=False, sort_keys=True) + "\n" for item in accepted_items)
    assert first_output.read_text() == expected
    assert serial["rows"] == parallel["rows"] == 3
    assert parallel["workers"] == 2
    assert serial["metadata_order"] == [str(path) for path in shards]
    assert serial["counts"] == {"metadata_rows": 7, "better_data_rows": 6, "not_better_data": 1, "accepted_rows": 3}
    assert serial["rejection_counts"] == {"empty_instruction": 1, "empty_mask": 1, "unaligned_aspect_ratio": 1}
    assert serial["output_sha256"] == parallel["output_sha256"]
    assert Path(serial["shard_directory"]).parent == first_output.parent
    assert len(list(Path(serial["shard_directory"]).glob("shard-*.jsonl"))) == 4  # rows + rejections per shard
    rejected = [json.loads(line) for line in Path(serial["rejections_path"]).read_text().splitlines()]
    assert [row["sample_id"] for row in rejected] == [8, 9, 10]
    assert Path(serial["rejections_path"]).read_bytes() == Path(parallel["rejections_path"]).read_bytes()


@pytest.mark.parametrize("role", ["source", "target", "mask"])
def test_corrupt_any_asset_hardfails_and_preserves_previous_pool(tmp_path, role):
    root = tmp_path / "raw"
    members = {"source.png": png(), "target.png": png(), "mask.png": png(mask=True)}
    members[f"{role}.png"] = b"not an image"
    write_tar(root, members)
    metadata = write_metadata(tmp_path / "metadata.jsonl", [metadata_row(1)])
    output = tmp_path / "output" / "pool.jsonl"
    output.parent.mkdir(); output.write_text("previous immutable pool\n")
    with pytest.raises(builder.PoolBuildError, match="asset_integrity_error"):
        builder.build_pool([metadata], interedit_root=root, revision="rev", output=output, workers=1)
    assert output.read_text() == "previous immutable pool\n"
    report = json.loads(output.with_name(output.name + ".report.json").read_text())
    assert report["status"] == "FAILED" and report["pool_published"] is False
    assert report["shards"][0]["error"]["reason"] == "asset_integrity_error"
    assert report["shards"][0]["error"]["sample_id"] == 1


def test_truncated_lazy_source_decode_cannot_pass(tmp_path):
    root = tmp_path / "raw"
    # Preserve enough PNG header for Image.open but remove image payload/IEND.
    truncated = png((128, 128))[:55]
    with Image.open(io.BytesIO(truncated)) as image:
        assert image.size == (128, 128)
    write_tar(root, {"source.png": truncated, "target.png": png(), "mask.png": png(mask=True)})
    metadata = write_metadata(tmp_path / "metadata.jsonl", [metadata_row(1)])
    output = tmp_path / "out" / "pool.jsonl"
    with pytest.raises(builder.PoolBuildError, match="asset_integrity_error"):
        builder.build_pool([metadata], interedit_root=root, revision="rev", output=output, workers=1)
    assert not output.exists()


def test_missing_member_in_parallel_shard_never_publishes_partial_pool(tmp_path):
    root, shards, _ = fixture_data(tmp_path)
    write_metadata(shards[1], [metadata_row(1, target_file="missing.png")])
    output = tmp_path / "out" / "pool.jsonl"
    with pytest.raises(builder.PoolBuildError, match="tar member missing: missing.png"):
        builder.build_pool(shards, interedit_root=root, revision="rev", output=output, workers=2)
    assert not output.exists()
    report = json.loads(output.with_name(output.name + ".report.json").read_text())
    assert report["status"] == "FAILED" and report["counts"]["accepted_rows"] == 1
    assert report["rejection_counts"]["empty_mask"] == 1


@pytest.mark.parametrize("value", ["false", "true", "0", 0, 1, None, [], {}])
def test_better_data_requires_real_json_bool(value):
    with pytest.raises(ValueError, match="JSON boolean"):
        parse_record(metadata_row(1, better_data=value))


def test_actual_false_remains_excluded_true_preserved(tmp_path):
    path = write_metadata(tmp_path / "metadata.jsonl.gz", [metadata_row(1, better_data=False), metadata_row(2)])
    assert [row.sample_id for row in iter_metadata(path)] == [2]
    assert [row.better_data for row in iter_metadata(path, only_better_data=False)] == [False, True]


def test_invalid_boolean_fails_build_with_metadata_report(tmp_path):
    root = tmp_path / "raw"; root.mkdir()
    metadata = write_metadata(tmp_path / "metadata.jsonl", [metadata_row(1, better_data="false")])
    output = tmp_path / "out" / "pool.jsonl"
    with pytest.raises(builder.PoolBuildError, match="JSON boolean"):
        builder.build_pool([metadata], interedit_root=root, revision="rev", output=output, workers=1)
    report = json.loads(output.with_name(output.name + ".report.json").read_text())
    assert report["shards"][0]["error"]["reason"] == "metadata_error"
    assert report["shards"][0]["error"]["metadata_row"] == 0
    assert not output.exists()


def test_canonicalization_is_streamed_before_next_metadata_row(tmp_path, monkeypatch):
    root = tmp_path / "raw"; root.mkdir()
    metadata = tmp_path / "metadata.jsonl"; metadata.write_text("fixture")
    seen = []
    def records(*args, **kwargs):
        yield parse_record(metadata_row(1))
        assert seen == [1], "builder materialized metadata before processing first item"
        yield parse_record(metadata_row(2))
    def canonicalize(item, *args):
        seen.append(item.sample_id)
        return {"sample_id": item.sample_id}, None
    monkeypatch.setattr(builder, "iter_metadata", records)
    monkeypatch.setattr(builder, "canonicalize", canonicalize)
    output = tmp_path / "out" / "pool.jsonl"
    report = builder.build_pool([metadata], interedit_root=root, revision="rev", output=output, workers=1)
    assert report["rows"] == 2 and seen == [1, 2]


@pytest.mark.parametrize("changes,match", [
    ({"source_archive": "missing.tar"}, "missing.tar"),
    ({"source_file": "../outside.png"}, "safe relative"),
    ({"asset_archive": "../outside.tar"}, "safe relative"),
])
def test_missing_archive_or_unsafe_asset_is_not_silently_filtered(tmp_path, changes, match):
    root, _, _ = fixture_data(tmp_path)
    path = write_metadata(tmp_path / "bad.jsonl", [metadata_row(1, **changes)])
    output = tmp_path / "out" / "pool.jsonl"
    with pytest.raises(builder.PoolBuildError, match=match):
        builder.build_pool([path], interedit_root=root, revision="rev", output=output, workers=1)
    assert not output.exists()


def test_metadata_mutation_during_build_is_detected(tmp_path, monkeypatch):
    root, shards, _ = fixture_data(tmp_path)
    path = shards[1]
    original = builder.canonicalize
    def mutate(item, *args):
        result = original(item, *args)
        with path.open("a") as handle:
            handle.write("\n")
        return result
    monkeypatch.setattr(builder, "canonicalize", mutate)
    output = tmp_path / "out" / "pool.jsonl"
    with pytest.raises(builder.PoolBuildError, match="metadata changed"):
        builder.build_pool([path], interedit_root=root, revision="rev", output=output, workers=1)
    assert not output.exists()


def test_failed_concatenation_preserves_previous_output(tmp_path, monkeypatch):
    root, shards, _ = fixture_data(tmp_path)
    output = tmp_path / "out" / "pool.jsonl"
    output.parent.mkdir(); output.write_text("original\n")
    def fail(paths, destination):
        destination.write_text("partial bytes")
        raise OSError("injected interrupted copy")
    monkeypatch.setattr(builder, "_concatenate", fail)
    with pytest.raises(builder.PoolBuildError, match="interrupted copy"):
        builder.build_pool(shards, interedit_root=root, revision="rev", output=output, workers=1)
    assert output.read_text() == "original\n"


def test_report_failure_after_pool_publication_is_not_misreported(tmp_path, monkeypatch):
    root, shards, _ = fixture_data(tmp_path)
    output = tmp_path / "out" / "pool.jsonl"
    original = builder._atomic_json
    def fail_complete_report(path, value):
        if value.get("schema") == builder.REPORT_SCHEMA and value["status"] == "COMPLETE":
            raise OSError("injected final report failure")
        return original(path, value)
    monkeypatch.setattr(builder, "_atomic_json", fail_complete_report)
    with pytest.raises(builder.PoolBuildError, match="final report failure"):
        builder.build_pool(shards, interedit_root=root, revision="rev", output=output, workers=1)
    report = json.loads(output.with_name(output.name + ".report.json").read_text())
    assert report["status"] == "FAILED" and report["pool_published"] is True
    assert canonical.sha256_bytes(output.read_bytes()) == report["output_sha256"]


def test_default_workers_bounded_by_shard_count_and_reject_unbounded(tmp_path):
    root, shards, _ = fixture_data(tmp_path)
    report = builder.build_pool(shards[:1], interedit_root=root, revision="rev", output=tmp_path / "out" / "pool.jsonl")
    assert report["requested_workers"] == 8 and report["workers"] == 1
    for value in (0, 9, -1, True):
        with pytest.raises(ValueError, match="workers"):
            builder.build_pool(shards, interedit_root=root, revision="rev", output=tmp_path / "out" / "bad.jsonl", workers=value)


def test_outputs_cannot_overwrite_raw_inputs(tmp_path):
    root, shards, _ = fixture_data(tmp_path)
    with pytest.raises(ValueError, match="raw Inter-Edit"):
        builder.build_pool(shards, interedit_root=root, revision="rev", output=root / "generated" / "pool.jsonl", workers=1)
    assert not (root / "generated").exists()
    with pytest.raises(ValueError, match="input metadata"):
        builder.build_pool(shards, interedit_root=root, revision="rev", output=shards[0], workers=1)
