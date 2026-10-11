import copy
import hashlib
import io
from pathlib import Path
import tarfile

import pytest

from src.explicit_region.config import _load_unexpanded
from src.explicit_region.dense_runtime import validate_dense_config
from src.explicit_region.mirror_download import MirrorDownloader
from scripts.data.prepare_mixed30k_corpus import exclude_connected_groups, materialize
from tests.fixed200k_helpers import fixture_record, synthetic_record


def config():
    return _load_unexpanded(Path("configs/train/dense_mixed20k_4a800_5epoch.yaml"))


def test_registered_mixed_recipe_is_independent_of_formal_and_magicbrush():
    c = config();validate_dense_config(c)
    assert c["initial_dense_checkpoint"].endswith("checkpoint-3843")
    assert c["data"]["expected_rows"] == 20000
    assert c["max_optimizer_steps"] == 6250
    assert c["checkpoint_steps"] == [1250, 2500, 3750, 5000, 6250]
    with pytest.raises(RuntimeError):
        validate_dense_config(c, formal=True)
    formal = _load_unexpanded(Path("configs/train/dense_d200k_5epoch.yaml"))
    validate_dense_config(formal, formal=True)
    magic = _load_unexpanded(Path("configs/train/dense_magicbrush_4a800_10epoch.yaml"))
    validate_dense_config(magic)


def test_registered_dataset_quotas_match_selection_config():
    from src.explicit_region.mixed_experiment import COUNTS, ROWS
    c = _load_unexpanded(Path("configs/data/mixed20k_total.yaml"))
    assert ROWS == c["train_total"] == 20000
    assert COUNTS == {"magicbrush": c["dataset_policy"]["magicbrush"]["fixed_count"],
                     "crispedit": c["dataset_policy"]["crispedit"]["cap"],
                     "scaleedit": c["dataset_policy"]["scaleedit"]["cap"],
                     "interedit": 1616}


def test_explicit_recovery_archives_preparation_failure_and_forbids_replaying_training(tmp_path):
    import json
    from scripts.train.run_mixed30k_continuation import recover_pretraining
    marker = tmp_path / "PIPELINE_STARTED.json"; marker.write_text('{}')
    state = tmp_path / "STATUS.json"
    state.write_text(json.dumps({"status": "STOPPED_HARD_ERROR", "phase": "screen_translate_freeze", "pid": 999999999}))
    (tmp_path / "sft/train/checkpoint-1").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="PRETRAINING_ONLY"): recover_pretraining(tmp_path)
    (tmp_path / "sft/train/checkpoint-1").rmdir()
    recover_pretraining(tmp_path)
    assert not marker.exists()
    assert len(list((tmp_path / "recovery").glob('*/PIPELINE_STARTED.json'))) == 1
    assert len(list((tmp_path / "recovery").glob('*/RECOVERY.json'))) == 1


@pytest.mark.parametrize("field,value", [("world_size", 8), ("epochs", 10),
    ("max_optimizer_steps", 5490), ("scheduler_horizon_steps", 5490), ("initial_dense_checkpoint", None)])
def test_mixed_budget_and_warmstart_cannot_silently_change(field, value):
    c = config();c[field] = value
    with pytest.raises(RuntimeError):
        validate_dense_config(c)


def test_protected_target_source_edges_remove_entire_connected_groups():
    protected = synthetic_record("magicbrush", 0, group="official-dev")
    a = synthetic_record("crispedit", 1, group="a")
    b = synthetic_record("crispedit", 2, group="a")
    c = synthetic_record("scaleedit", 3, group="c")
    kept = synthetic_record("interedit", 4)
    a["target_sha256"] = protected["source_sha256"]
    c["source_sha256"] = b["target_sha256"]
    assert exclude_connected_groups([c, kept, b, a], [protected]) == [kept]


def test_materialization_preserves_bytes_identity_and_original_manifest(tmp_path):
    row = fixture_record(tmp_path / "original", "crispedit", 1)
    before = copy.deepcopy(row)
    result = materialize([row], {"crispedit": tmp_path / "original"}, tmp_path / "materialized")[0]
    assert row == before and result["sample_uid"] == row["sample_uid"]
    for role in ("source", "target", "region"):
        path = tmp_path / "materialized/crispedit" / result[f"{role}_locator"]["relative_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row[f"{role}_sha256"]


def test_interedit_reader_reuses_one_open_handle_per_archive(tmp_path, monkeypatch):
    from src.explicit_region.interedit import TarMemberReader
    path = tmp_path / "test.tar"
    with tarfile.open(path, "w") as handle:
        info = tarfile.TarInfo("image");info.size = 3
        handle.addfile(info, io.BytesIO(b"abc"))
    original = tarfile.open;opened = []
    def counted(*args, **kwargs):
        opened.append(args[0]);return original(*args, **kwargs)
    monkeypatch.setattr(tarfile, "open", counted)
    reader = TarMemberReader(tmp_path)
    try:
        assert reader.read("test.tar", "image") == reader.read("test.tar", "image") == b"abc"
        assert len(opened) == 1
    finally:
        reader.close()


def test_mirror_ignores_proxy_environment_and_requires_safe_pinned_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://must-not-use.invalid:9999")
    downloader = MirrorDownloader()
    with downloader.session() as session:
        assert session.trust_env is False
    with pytest.raises(ValueError):
        downloader.metadata("owner/repo", "main")
    with pytest.raises(ValueError):
        downloader.download("owner/repo", "a" * 40, {"rfilename": "../escape"}, tmp_path)


def test_cached_mirror_payload_is_checked_against_lfs_sha(tmp_path):
    path = tmp_path / "model.bin";path.write_bytes(b"correct")
    entry = {"rfilename": path.name, "size": 7,
             "lfs": {"sha256": hashlib.sha256(b"correct").hexdigest()}}
    downloader = MirrorDownloader()
    assert downloader.download("owner/repo", "a" * 40, entry, tmp_path) == path
    path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="content hash mismatch"):
        downloader.download("owner/repo", "a" * 40, entry, tmp_path)


def test_quality_selection_requires_improvement_ci_and_preservation():
    from scripts.train.run_mixed30k_continuation import choose
    good = {"step": 2424, "inside_mean": .6,
            "paired_inside": {"mean_delta": -.01, "bootstrap_95_ci": [-.02, -.001]},
            "paired_outside": {"mean_delta": .009}}
    assert choose([good]) == good
    for field, value in (("mean_delta", -.004), ("bootstrap_95_ci", [-.02, .001])):
        row = copy.deepcopy(good);row["paired_inside"][field] = value
        assert choose([row]) is None
    row = copy.deepcopy(good);row["paired_outside"]["mean_delta"] = .011
    assert choose([row]) is None
    row = copy.deepcopy(good);row["step"] = 4848
    assert choose([row, good])["step"] == 2424


@pytest.mark.parametrize("status,body", [(206, b"cdefg"), (200, b"abcdefg")])
def test_mirror_resumes_range_or_rewrites_when_range_is_ignored(tmp_path, monkeypatch, status, body):
    (tmp_path / "file.partial").write_bytes(b"ab")
    class Response:
        status_code = status
        headers = {"Content-Range": "bytes 2-6/7"}
        def raise_for_status(self): pass
        def iter_content(self, size): yield body
    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            assert url.startswith("https://hf-mirror.com/datasets/")
            assert kwargs["headers"] == {"Range": "bytes=2-"}
            return Response()
    downloader = MirrorDownloader();monkeypatch.setattr(downloader, "session", Session)
    entry = {"rfilename": "file", "size": 7,
             "lfs": {"sha256": hashlib.sha256(b"abcdefg").hexdigest()}}
    path = downloader.download("owner/repo", "a" * 40, entry, tmp_path)
    assert path.read_bytes() == b"abcdefg" and not (tmp_path / "file.partial").exists()
