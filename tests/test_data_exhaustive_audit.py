import json

import pytest
from PIL import Image

from scripts.data.audit_fixed200k import audit_dataset, montage
from src.explicit_region.canonical import sample_uid_for, sha256_bytes
from src.explicit_region.fixed_corpus import canonical_freeze_order
from src.explicit_region.fixed_dataset import FixedCorpusDataset
from tests.fixed200k_helpers import fixture_record


def dataset(tmp_path, count=3):
    assets = tmp_path / "assets"
    rows = canonical_freeze_order([fixture_record(assets, "magicbrush", index) for index in range(count)])
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return FixedCorpusDataset(manifest, {"magicbrush": assets}, resolution=32), {"magicbrush": assets}


def test_exhaustive_integrity_checks_last_record_and_reports_partial_coverage(tmp_path):
    ds, roots = dataset(tmp_path)
    # Corrupt the final record, outside an imagined small prefix/sample.
    last = ds.rows[-1]
    (roots["magicbrush"] / last["source_locator"]["relative_path"]).unlink()
    report = audit_dataset(ds, roots)
    assert report["status"] == "FAIL"
    assert report["attempted_rows"] == 3
    assert report["completed_checks"]["english"] == 3
    assert report["completed_checks"]["hash"] == 2
    assert report["failure_counts"] == {"missing_assets": 1}
    assert report["failure_examples"][0]["sample_uid"] == last["sample_uid"]


def test_exhaustive_integrity_checks_native_mask_metadata(tmp_path):
    ds, roots = dataset(tmp_path)
    ds.rows[-1]["region_fraction"] = 0.99
    report = audit_dataset(ds, roots)
    assert report["status"] == "FAIL"
    assert report["completed_checks"]["geometry"] == 3
    assert report["completed_checks"]["mask"] == 2
    assert report["failure_counts"] == {"mask_failures": 1}


def test_exhaustive_integrity_success_and_montage_single_channel_mask(tmp_path):
    ds, roots = dataset(tmp_path)
    report = audit_dataset(ds, roots)
    assert report["status"] == "PASS"
    assert all(count == 3 for count in report["completed_checks"].values())
    assert report["datasets"]["magicbrush"]["source_width"]["mean"] == 32
    output = tmp_path / "montage.jpg"
    montage((ds[index] for index in range(3)), output, count=3)
    with Image.open(output) as image:
        assert image.mode == "RGB"
        assert image.size == (768, 758)


@pytest.mark.parametrize("kind", ["hash", "decode", "geometry", "english"])
def test_exhaustive_failure_categories(tmp_path, kind):
    ds, roots = dataset(tmp_path, count=1)
    row = ds.rows[0]
    source = roots["magicbrush"] / row["source_locator"]["relative_path"]
    if kind in ("hash", "decode"):
        source.write_bytes(b"not an image")
        if kind == "decode":
            row["source_sha256"] = sha256_bytes(source.read_bytes())
    elif kind == "geometry":
        Image.new("RGB", (12, 40)).save(source)
        row["source_sha256"] = sha256_bytes(source.read_bytes())
    else:
        row["instruction_en"] = "   "
    row["sample_uid"] = sample_uid_for(row)
    report = audit_dataset(ds, roots)
    assert report["status"] == "FAIL"
    assert report["attempted_rows"] == 1
    expected = {"hash": "hash_failures", "decode": "decode_failures",
                "geometry": "geometry_failures", "english": "english_failures"}[kind]
    assert expected in report["failure_counts"]
