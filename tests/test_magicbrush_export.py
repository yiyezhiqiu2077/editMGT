from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image, PngImagePlugin

from scripts.data import export_magicbrush_official as export


def image_bytes(mode="RGB", *, fmt="PNG", size=(3, 2), color=None):
    image = Image.new(mode, size, color or ((13, 44, 91) if mode == "RGB" else 0))
    buffer = BytesIO()
    if fmt == "PNG":
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("preserve", "original release bytes, not a re-encoding")
        image.save(buffer, format=fmt, pnginfo=metadata)
    else:
        image.save(buffer, format=fmt)
    return buffer.getvalue()


def raw_row(img_id="0017", turn=1):
    # Deliberately colored RGB and soft alpha: .convert('L') is NOT an edit mask.
    mask = Image.new("RGBA", (3, 2), (99, 30, 10, 255))
    mask.putalpha(Image.frombytes("L", (3, 2), bytes([0, 1, 127, 128, 254, 255])))
    buffer = BytesIO()
    mask.save(buffer, format="PNG")
    return {"img_id": img_id, "turn_index": turn,
            "source_img": {"bytes": image_bytes(size=(2, 2)), "path": "name-lies.jpg"},
            "mask_img": {"bytes": buffer.getvalue(), "path": "mask.png"},
            "instruction": "  Change the café sign.\nKeep its shape.  ",
            "target_img": {"bytes": image_bytes(fmt="JPEG"), "path": "name-lies.png"}}


@pytest.fixture
def release(tmp_path, monkeypatch):
    snapshot = tmp_path / export.REVISION
    (snapshot / "data").mkdir(parents=True)
    card = export.MASK_CARD_TEXT.encode()
    (snapshot / "README.md").write_bytes(card)
    siblings = [{"rfilename": "README.md", "size": len(card),
                 "blobId": hashlib.sha1(b"blob " + str(len(card)).encode() + b"\0" + card).hexdigest()}]
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    schema = pa.schema([("img_id", pa.string()), ("turn_index", pa.int32()),
                        ("source_img", image_type), ("mask_img", image_type),
                        ("instruction", pa.string()), ("target_img", image_type)])
    rows = [raw_row(), raw_row(turn=2)]
    # Lexical shard order, not directory discovery order.
    for index in [1, 0]:
        name = f"data/train-{index:05d}-of-00002-deadbeef.parquet"
        pq.write_table(pa.Table.from_pylist([rows[index]], schema=schema), snapshot / name)
        size = (snapshot / name).stat().st_size
        siblings.append({"rfilename": name, "size": size, "lfs": {"sha256": export.file_sha256(snapshot / name), "size": size}})
    # The exporter must never discover, read or validate this unrelated content.
    (snapshot / "data" / "test-00000.parquet").write_bytes(b"DO NOT OPEN")
    siblings.append({"rfilename": "data/test-00000.parquet", "size": 11})
    metadata = tmp_path / "remote.json"
    metadata.write_bytes(export.json_bytes({"id": export.REPO_ID, "sha": export.REVISION,
                                           "cardData": {"dataset_info": {"splits": [{"name": "train", "num_examples": 2}]}},
                                           "siblings": siblings}))
    monkeypatch.setitem(export.ROW_COUNTS, "train", 2)
    monkeypatch.setitem(export.SHARD_COUNTS, "train", 2)
    return snapshot, metadata, tmp_path / "derived", rows


def run(release, **kwargs):
    snapshot, metadata, output, _ = release
    return export.export_split(snapshot, metadata, output, "train", **kwargs)


def manifest_rows(output):
    return [json.loads(line) for line in (output / "train/manifest.jsonl").read_text().splitlines()]


def test_full_export_preserves_raw_bytes_and_derives_inverse_alpha(release):
    summary = run(release, workers=2)
    snapshot, _, output, raw = release
    records = manifest_rows(output)
    split_root = output / "train"
    assert [record["sample_key"] for record in records] == ["0017_1", "0017_2"]
    assert summary["rows"] == 2
    assert summary["groups"] == 1
    assert summary["raw_mask_modes"] == {"RGBA": 2}
    assert summary["source_target_size_mismatches"] == 2
    assert summary["raw_image_formats"] == {"PNG": 4, "JPEG": 2}
    for index, row in enumerate(records):
        assert row["instruction"] == raw[index]["instruction"]
        assert row["img_id"] == "0017"
        assert row["turn_index"] == index + 1
        assert (split_root / row["source"]).read_bytes() == raw[index]["source_img"]["bytes"]
        assert (split_root / row["mask_raw"]).read_bytes() == raw[index]["mask_img"]["bytes"]
        assert (split_root / row["target"]).read_bytes().startswith(export.PNG_SIGNATURE)
        for name in ["source", "target", "mask_raw", "mask_edit"]:
            assert not Path(row[name]).is_absolute()
            assert ".." not in Path(row[name]).parts
            assert export.file_sha256(split_root / row[name]) == row[f"{name}_sha256"]
        provenance = row["raw_provenance"]
        assert export.file_sha256(snapshot / provenance["parquet"]) == provenance["parquet_sha256"]
        assert provenance["split_row_index"] == index
        assert provenance["row_in_shard"] == 0
        assert provenance["assets"]["target_img"]["operation"] == "decode_to_png_no_geometry_change"
        assert provenance["assets"]["source_img"]["operation"] == "preserve_raw_png"
        with Image.open(split_root / row["mask_edit"]) as mask:
            assert mask.mode == "L"
            assert list(mask.getdata()) == [255, 254, 128, 127, 1, 0]
        assert row["mask_provenance"]["alpha_partial_pixels"] == 4
        assert row["mask_provenance"]["region_fraction_gt0"] == 5 / 6
    assert export.file_sha256(output / "train/manifest.jsonl") == summary["manifest_sha256"]
    assert json.loads((output / "train/export_provenance.json").read_text())["release"]["revision"] == export.REVISION


def test_byte_identical_rerun_and_worker_count_independent(release):
    first = run(release, workers=1)
    output = release[2]
    before = {str(path.relative_to(output)): (path.read_bytes(), path.stat().st_mtime_ns)
              for path in output.rglob("*") if path.is_file() and path.name != ".export.lock"}
    assert run(release, workers=2) == first
    after = {str(path.relative_to(output)): (path.read_bytes(), path.stat().st_mtime_ns)
             for path in output.rglob("*") if path.is_file() and path.name != ".export.lock"}
    assert after == before
    other = output.parent / "second-root"
    assert export.export_split(release[0], release[1], other, "train", workers=2) == first
    assert (other / "train/manifest.jsonl").read_bytes() == (output / "train/manifest.jsonl").read_bytes()


def test_interruption_resumes_without_overwriting_valid_assets(release, monkeypatch):
    original = export.export_row

    def interrupt(row, **kwargs):
        if row["turn_index"] == 2:
            raise RuntimeError("simulated interruption")
        return original(row, **kwargs)

    monkeypatch.setattr(export, "export_row", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        run(release, workers=1)
    output = release[2]
    assert not (output / "train/manifest.jsonl").exists()
    assert not (output / "train/export_summary.json").exists()
    source = output / "train/images/0017_1/source.png"
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    monkeypatch.setattr(export, "export_row", original)
    run(release)
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before


@pytest.mark.parametrize("relative", ["train/images/0017_1/source.png", "train/images/0017_2/mask_edit.png",
                                      "train/manifest.jsonl", "train/export_provenance.json", "train/export_summary.json"])
def test_existing_corruption_is_never_silently_overwritten(release, relative):
    run(release)
    path = release[2] / relative
    path.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="refusing overwrite"):
        run(release)
    assert path.read_bytes() == b"corruption"


def test_release_digest_mismatch_rejected_before_assets(release):
    path = sorted((release[0] / "data").glob("train-*.parquet"))[1]
    content = bytearray(path.read_bytes())
    content[10] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run(release)
    assert not (release[2] / "train/images").exists()


def test_wrong_revision_or_card_hash_rejected(release):
    release[0].joinpath("README.md").write_bytes(b"unverified card")
    with pytest.raises(ValueError, match="card release hash mismatch"):
        run(release)
    metadata = json.loads(release[1].read_text())
    metadata["sha"] = "0" * 40
    release[1].write_bytes(export.json_bytes(metadata))
    with pytest.raises(ValueError, match="pinned official"):
        run(release)


def test_official_row_and_shard_counts_are_enforced(release, monkeypatch):
    monkeypatch.setitem(export.ROW_COUNTS, "train", 3)
    with pytest.raises(ValueError, match="row count mismatch"):
        run(release)
    monkeypatch.setitem(export.ROW_COUNTS, "train", 2)
    monkeypatch.setitem(export.SHARD_COUNTS, "train", 3)
    with pytest.raises(ValueError, match="release shard list"):
        run(release)


def test_parquet_row_count_is_not_just_trusted_metadata(release, monkeypatch):
    monkeypatch.setitem(export.ROW_COUNTS, "train", 3)
    metadata = json.loads(release[1].read_text())
    metadata["cardData"]["dataset_info"]["splits"][0]["num_examples"] = 3
    release[1].write_bytes(export.json_bytes(metadata))
    with pytest.raises(ValueError, match="Parquet row count mismatch"):
        run(release)


def test_duplicate_identity_rejected_before_assets(release):
    metadata = json.loads(release[1].read_text())
    name = "data/train-00001-of-00002-deadbeef.parquet"
    path = release[0] / name
    table = pq.read_table(path)
    pq.write_table(pa.Table.from_pylist([release[3][0]], schema=table.schema), path)
    for entry in metadata["siblings"]:
        if entry["rfilename"] == name:
            entry["size"] = path.stat().st_size
            entry["lfs"] = {"size": entry["size"], "sha256": export.file_sha256(path)}
    release[1].write_bytes(export.json_bytes(metadata))
    with pytest.raises(ValueError, match="Duplicate sample_key"):
        run(release)
    assert not (release[2] / "train/images").exists()


@pytest.mark.parametrize("split", ["test", "validation", "../train"])
def test_only_train_dev_allowed_without_reading_inputs(tmp_path, split):
    with pytest.raises(ValueError, match="never test"):
        export.export_split(tmp_path / "does-not-exist", tmp_path / "absent", tmp_path / "output", split)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("change", [{"img_id": "../../escape"}, {"img_id": 17}, {"turn_index": 0},
                                    {"turn_index": True}, {"instruction": " "}])
def test_invalid_row_identity(change):
    row = raw_row()
    row.update(change)
    with pytest.raises(ValueError):
        export.identity(row)


def test_grayscale_white_masks_keep_png_bytes_and_ambiguous_color_fails():
    raw = image_bytes("L", color=255)
    png, mask, _ = export.decode_asset({"bytes": raw, "path": "mask.png"})
    output, stats = export.edit_mask(mask, png)
    assert output == raw
    assert stats["region_fraction_gt0"] == 1
    assert stats["operation"] == "preserve_white_edit_grayscale_png"
    with pytest.raises(ValueError, match="Ambiguous colored"):
        export.edit_mask(Image.new("RGB", (2, 2), (8, 90, 120)), b"unused")


def test_path_only_and_corrupt_images_fail_closed():
    with pytest.raises(ValueError, match="path fallback is forbidden"):
        export.decode_asset({"bytes": None, "path": "/untrusted/image.png"})
    with pytest.raises(OSError):
        export.decode_asset({"bytes": b"not an image", "path": "image.png"})


def test_output_symlinks_and_raw_snapshot_output_refused(release):
    outside = release[2].parent / "outside"
    outside.mkdir()
    release[2].symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        run(release)
    assert not list(outside.iterdir())
    with pytest.raises(ValueError, match="raw snapshot"):
        export.export_split(release[0], release[1], release[0] / "derived", "train")
    assert not (release[0] / "derived").exists()


def test_symlink_file_not_followed(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"expected")
    path = tmp_path / "asset.png"
    path.symlink_to(target)
    with pytest.raises(ValueError, match="refusing overwrite"):
        export.publish_bytes(path, b"expected")
    assert target.read_bytes() == b"expected"


def test_concurrent_export_lock(tmp_path):
    with export.export_lock(tmp_path):
        with pytest.raises(ValueError, match="Another exporter"):
            with export.export_lock(tmp_path):
                pass


def test_explicit_legacy_migration_keeps_evidence_and_png_bytes(release):
    run(release)
    output = release[2]
    split_root = output / "train"
    image = split_root / "images/0017_1/source.png"
    before = (image.read_bytes(), image.stat().st_mtime_ns)
    # Reconstruct v1's common-root contract without changing its actual images.
    provenance_path = split_root / "export_provenance.json"
    provenance = json.loads(provenance_path.read_bytes())
    provenance["schema_version"] = 1
    provenance["manifest_path_base"] = "output_root"
    provenance_path.write_bytes(export.json_bytes(provenance))
    records = manifest_rows(output)
    for row in records:
        for key in ("source", "target", "mask_raw", "mask_edit"):
            row[key] = "train/" + row[key]
    manifest = b"".join(export.json_bytes(row) for row in records)
    (split_root / "manifest.jsonl").write_bytes(manifest)
    summary_path = split_root / "export_summary.json"
    summary = json.loads(summary_path.read_bytes())
    summary.update({"manifest": "train/manifest.jsonl", "manifest_sha256": export.sha256(manifest),
                    "provenance_sha256": export.file_sha256(provenance_path)})
    summary_path.write_bytes(export.json_bytes(summary))
    legacy = {name: (split_root / name).read_bytes() for name in
              ("manifest.jsonl", "export_summary.json", "export_provenance.json")}
    with pytest.raises(ValueError, match="refusing overwrite"):
        run(release)
    # Simulate interruption after archiving and removing just the completion marker.
    archive = output.parent / "legacy-archive" / "train"
    archive.mkdir(parents=True)
    for name, content in legacy.items():
        (archive / name).write_bytes(content)
    (split_root / "export_summary.json").unlink()
    run(release, migrate_common_root_v1=output.parent / "legacy-archive")
    for name, content in legacy.items():
        assert (output.parent / "legacy-archive" / "train" / name).read_bytes() == content
    assert manifest_rows(output)[0]["source"] == "images/0017_1/source.png"
    assert (image.read_bytes(), image.stat().st_mtime_ns) == before
    assert json.loads(provenance_path.read_bytes())["schema_version"] == 2
    assert list(split_root.rglob("*.jsonl")) == [split_root / "manifest.jsonl"]
    run(release, migrate_common_root_v1=output.parent / "legacy-archive")
    run(release)  # ordinary subsequent resumes need no migration flag


def test_manifest_is_accepted_by_existing_canonical_builder(release):
    run(release)
    output = release[2]
    canonical = output / "canonical.jsonl"
    script = Path(export.__file__).with_name("build_magicbrush_canonical.py")
    subprocess.run([sys.executable, str(script), "--manifest", str(output / "train/manifest.jsonl"),
                    "--dataset-root", str(output / "train"), "--revision", export.REVISION, "--output", str(canonical)], check=True)
    rows = [json.loads(line) for line in canonical.read_text().splitlines()]
    assert rows == []
    rejections = [json.loads(line) for line in canonical.with_suffix(".rejections.jsonl").read_text().splitlines()]
    assert len(rejections) == 2
    assert {item["reason"] for item in rejections} == {"unaligned_aspect_ratio"}
    report = json.loads(canonical.with_suffix(".report.json").read_text())
    assert report["accepted_rows"] == 0
    assert report["rejected_rows"] == 2
    assert report["rejection_counts"] == {"unaligned_aspect_ratio": 2}


def test_canonical_builder_rejects_unaligned_magicbrush_geometry(tmp_path):
    root = tmp_path / "train"
    images = root / "images/0017_1"
    images.mkdir(parents=True)
    Image.new("RGB", (1024, 1023), (1, 2, 3)).save(images / "source.png")
    Image.new("RGB", (1024, 1024), (4, 5, 6)).save(images / "target.png")
    Image.new("L", (1024, 1024), 255).save(images / "mask_edit.png")
    manifest_row = {
        "sample_key": "0017_1",
        "img_id": "0017",
        "turn_index": 1,
        "instruction": "Change the sign",
        "source": "images/0017_1/source.png",
        "target": "images/0017_1/target.png",
        "mask_edit": "images/0017_1/mask_edit.png",
    }
    manifest = root / "manifest.jsonl"
    manifest.write_text(json.dumps(manifest_row) + "\n")
    canonical = tmp_path / "canonical.jsonl"
    script = Path(export.__file__).with_name("build_magicbrush_canonical.py")
    subprocess.run([sys.executable, str(script), "--manifest", str(manifest),
                    "--dataset-root", str(root), "--revision", export.REVISION, "--output", str(canonical)], check=True)
    assert canonical.read_text() == ""
    rejections = [json.loads(line) for line in canonical.with_suffix(".rejections.jsonl").read_text().splitlines()]
    assert len(rejections) == 1
    assert rejections[0]["reason"] == "unaligned_aspect_ratio"
    assert rejections[0]["sample_key"] == "0017_1"
    report = json.loads(canonical.with_suffix(".report.json").read_text())
    assert report["accepted_rows"] == 0
    assert report["rejected_rows"] == 1
    assert report["rejection_counts"] == {"unaligned_aspect_ratio": 1}
