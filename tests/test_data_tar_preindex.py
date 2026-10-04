import io
import json
import sqlite3
import tarfile

import pytest

from scripts.data.preindex_canonical_tars import preindex
from src.explicit_region import canonical


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    canonical.clear_locator_caches()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    yield
    canonical.clear_locator_caches()


def asset(path, value=b"exact payload"):
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("image.png"); info.size = len(value)
        archive.addfile(info, io.BytesIO(value))


def test_preindex_build_once_then_reuse_shared_without_replacing(tmp_path, monkeypatch):
    root = tmp_path / "raw"; root.mkdir(); path = root / "a.tar"; asset(path)
    report = preindex(root, report_path=tmp_path / "report.json")
    assert report["status"] == "PASS" and report["completed"] == 1
    shared = canonical._tar_index_path(canonical._asset_identity(path))
    assert shared.is_file()
    before = shared.stat()
    monkeypatch.setattr(canonical.tarfile, "open", lambda *a, **k: pytest.fail("completed shared index must not rescan"))
    repeated = preindex(root, report_path=tmp_path / "report2.json")
    assert repeated["results"][0]["action"] == "existing"
    assert shared.stat().st_mtime_ns == before.st_mtime_ns
    assert canonical.read_locator_bytes({"backend": "tar", "archive": "a.tar", "member": "image.png"}, root) == b"exact payload"


def test_preindex_promotes_private_index_after_validation(tmp_path, monkeypatch):
    root = tmp_path / "raw"; root.mkdir(); path = root / "a.tar"; asset(path)
    identity = canonical._asset_identity(path); shared = canonical._tar_index_path(identity)
    private = canonical._build_tar_index(path, identity, shared.with_name(shared.stem + ".private.sqlite3"))
    assert private.is_file() and not shared.exists()
    monkeypatch.setattr(canonical.tarfile, "open", lambda *a, **k: pytest.fail("promotion must not scan"))
    report = preindex(root, report_path=tmp_path / "report.json")
    assert report["results"][0]["action"] == "promoted"
    assert shared.is_file() and not private.exists()


def test_preindex_refuses_stale_owner_and_does_not_remove_it(tmp_path):
    root = tmp_path / "raw"; root.mkdir(); path = root / "a.tar"; asset(path)
    cache = canonical._tar_index_path(canonical._asset_identity(path)).parent
    owner = cache / "PREINDEX_OWNER"; owner.mkdir()
    with pytest.raises(RuntimeError, match="owner lock exists"):
        preindex(root, report_path=tmp_path / "report.json")
    assert owner.is_dir()


def test_preindex_refuses_bad_offset_or_schema_without_publication(tmp_path):
    root = tmp_path / "raw"; root.mkdir(); path = root / "a.tar"; asset(path)
    identity = canonical._asset_identity(path); shared = canonical._tar_index_path(identity)
    private = canonical._build_tar_index(path, identity, shared.with_name(shared.stem + ".private.sqlite3"))
    connection = sqlite3.connect(f"{private.as_uri()}?mode=rw&nolock=1", uri=True)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("UPDATE members SET offset=?", (identity[3] + 1,)); connection.commit(); connection.close()
    with pytest.raises(canonical.FrozenCorpusIntegrityError, match="invalid tar index offsets"):
        preindex(root, report_path=tmp_path / "report.json")
    assert not shared.exists()
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "FAIL"
    assert not (shared.parent / "PREINDEX_OWNER").exists()


def test_preindex_parallel_distinct_archives(tmp_path):
    root = tmp_path / "raw"; root.mkdir()
    for index in range(4):
        asset(root / f"{index}.tar", str(index).encode())
    report = preindex(root, workers=2, report_path=tmp_path / "report.json")
    assert report["status"] == "PASS" and report["completed"] == 4
    assert len({row["index"] for row in report["results"]}) == 4
