import io
import multiprocessing
import tarfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.explicit_region import canonical


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    canonical.clear_locator_caches()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    yield
    canonical.clear_locator_caches()


def tar_asset(path, members):
    with tarfile.open(path, "w") as archive:
        for name, value in members:
            info = tarfile.TarInfo(name); info.size = len(value)
            archive.addfile(info, io.BytesIO(value))


def locator(archive, member):
    return {"backend": "tar", "archive": archive, "member": member}


def test_tar_index_once_exact_bytes_duplicate_semantics_and_reopen(tmp_path, monkeypatch):
    path = tmp_path / "assets.tar"
    tar_asset(path, [("a", b"first"), ("b", b"second"), ("a", b"last")])
    original = canonical.tarfile.open
    opened = []

    def instrument(*args, **kwargs):
        opened.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(canonical.tarfile, "open", instrument)
    for _ in range(3):
        assert canonical.read_locator_bytes(locator(path.name, "a"), tmp_path) == b"last"
        assert canonical.read_locator_bytes(locator(path.name, "b"), tmp_path) == b"second"
    assert len(opened) == 1
    canonical.clear_locator_caches()
    assert canonical.read_locator_bytes(locator(path.name, "b"), tmp_path) == b"second"
    assert len(opened) == 1  # Disk index survives process-local eviction.
    assert canonical.expected_locator_hash(locator(path.name, "a"), tmp_path) == canonical.sha256_bytes(b"last")


def test_tar_members_safe_missing_and_nonregular(tmp_path):
    path = tmp_path / "links.tar"
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("link"); info.type = tarfile.SYMTYPE; info.linkname = "../outside"
        archive.addfile(info)
    with pytest.raises(canonical.FrozenCorpusIntegrityError, match="regular"):
        canonical.read_locator_bytes(locator(path.name, "link"), tmp_path)
    with pytest.raises(canonical.FrozenCorpusIntegrityError, match="missing"):
        canonical.read_locator_bytes(locator(path.name, "absent"), tmp_path)
    with pytest.raises(ValueError, match="safe relative"):
        canonical.read_locator_bytes(locator(path.name, "../outside"), tmp_path)


def test_compressed_tar_is_explicitly_rejected_not_misindexed(tmp_path):
    path = tmp_path / "compressed.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("a"); info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(canonical.FrozenCorpusIntegrityError, match="uncompressed archive"):
        canonical.read_locator_bytes(locator(path.name, "a"), tmp_path)


def test_tar_stat_identity_invalidates_index_and_handle_lru(tmp_path, monkeypatch):
    monkeypatch.setattr(canonical, "_MAX_OPEN_ASSETS", 2)
    path = tmp_path / "assets.tar"
    tar_asset(path, [("a", b"before")])
    assert canonical.read_locator_bytes(locator(path.name, "a"), tmp_path) == b"before"
    tar_asset(path, [("a", b"after mutation")])
    assert canonical.read_locator_bytes(locator(path.name, "a"), tmp_path) == b"after mutation"
    for index in range(4):
        path = tmp_path / f"{index}.tar"; tar_asset(path, [("a", str(index).encode())])
        assert canonical.read_locator_bytes(locator(path.name, "a"), tmp_path) == str(index).encode()
        assert len(canonical._TAR_READERS) <= 2


def child_read(root, queue):
    try:
        before = len(canonical._TAR_READERS)
        value = canonical.read_locator_bytes(locator("fork.tar", "a"), root)
        queue.put((before, value))
    except Exception as exc:
        queue.put(str(exc))


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="fork-specific worker guard")
def test_tar_worker_fork_discards_parent_connections(tmp_path):
    tar_asset(tmp_path / "fork.tar", [("a", b"fork-safe")])
    assert canonical.read_locator_bytes(locator("fork.tar", "a"), tmp_path) == b"fork-safe"
    context = multiprocessing.get_context("fork"); queue = context.Queue()
    child = context.Process(target=child_read, args=(str(tmp_path), queue)); child.start()
    result = queue.get(timeout=20); child.join(20)
    assert child.exitcode == 0 and result == (0, b"fork-safe")
    assert canonical.read_locator_bytes(locator("fork.tar", "a"), tmp_path) == b"fork-safe"


def spawn_cold_reader(root, event, queue):
    import traceback
    try:
        event.wait(30)
        queue.put(canonical.read_locator_bytes(locator("parallel.tar", "a"), root))
    except Exception:
        queue.put(traceback.format_exc())


def test_tar_concurrent_cold_spawn_index_on_external_mount(tmp_path):
    tar_asset(tmp_path / "parallel.tar", [("a", b"parallel-safe")] +
              [(f"member-{i}", b"x") for i in range(100)])
    context = multiprocessing.get_context("spawn")
    event, queue = context.Event(), context.Queue()
    children = [context.Process(target=spawn_cold_reader, args=(str(tmp_path), event, queue)) for _ in range(4)]
    for child in children:
        child.start()
    event.set()
    results = [queue.get(timeout=90) for _ in children]
    for child in children:
        child.join(30)
    assert all(child.exitcode == 0 for child in children)
    assert results == [b"parallel-safe"] * len(children), results


def parquet_locator(path, row, column="image"):
    return {"backend": "parquet", "file": path.name, "row_index": row, "column": column}


def test_parquet_selects_row_group_column_and_caches(tmp_path, monkeypatch):
    path = tmp_path / "images.parquet"
    values = [{"bytes": f"value-{i}".encode(), "path": None} for i in range(12)]
    pq.write_table(pa.table({"image": values, "unrelated": [b"x" * 1000] * 12}), path, row_group_size=3)
    original = pq.ParquetFile
    calls = []

    class Instrumented:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def read_row_group(self, index, **kwargs):
            calls.append((index, kwargs["columns"]))
            return self.inner.read_row_group(index, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", Instrumented)
    monkeypatch.setattr(pq, "read_table", lambda *a, **k: pytest.fail("whole-column read forbidden"))
    assert canonical.read_locator_bytes(parquet_locator(path, 7), tmp_path) == b"value-7"
    assert canonical.read_locator_bytes(parquet_locator(path, 8), tmp_path) == b"value-8"
    assert calls == [(2, ["image"])]
    assert canonical.read_locator_bytes(parquet_locator(path, 1), tmp_path) == b"value-1"
    assert calls[-1] == (0, ["image"])
    assert canonical._PARQUET_BLOCK_BYTES <= canonical._PARQUET_CACHE_BYTES


def test_parquet_large_group_batch_cache_and_bounds(tmp_path, monkeypatch):
    path = tmp_path / "large.parquet"
    pq.write_table(pa.table({"image": [str(i).encode() * 100 for i in range(80)]}), path, row_group_size=80)
    monkeypatch.setattr(canonical, "_PARQUET_GROUP_BYTES", 1)
    monkeypatch.setattr(canonical, "_PARQUET_CACHE_BYTES", 4000)
    original = pq.ParquetFile

    class BatchedOnly:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def read_row_group(self, *args, **kwargs):
            pytest.fail("large row group must use bounded batches")

    monkeypatch.setattr(pq, "ParquetFile", BatchedOnly)
    for index in (3, 79, 20, 33):
        assert canonical.read_locator_bytes(parquet_locator(path, index), tmp_path) == str(index).encode() * 100
        assert canonical._PARQUET_BLOCK_BYTES <= 4000
    with pytest.raises(canonical.FrozenCorpusIntegrityError, match="out of bounds"):
        canonical.read_locator_bytes(parquet_locator(path, 80), tmp_path)
    with pytest.raises(ValueError, match="non-negative"):
        canonical.read_locator_bytes(parquet_locator(path, -1), tmp_path)
    with pytest.raises(canonical.FrozenCorpusIntegrityError, match="column missing"):
        canonical.read_locator_bytes(parquet_locator(path, 0, "missing"), tmp_path)


def test_parquet_cache_accounts_for_retained_slice_buffers(monkeypatch):
    monkeypatch.setattr(canonical, "_PARQUET_CACHE_BYTES", 256)
    array = pa.array([b"x" * 100] * 100)
    sliced = array.slice(0, 1)
    assert sliced.nbytes < 256 < sliced.get_total_buffer_size()
    canonical._cache_parquet_block(("slice",), sliced)
    assert not canonical._PARQUET_BLOCKS
    assert canonical._PARQUET_BLOCK_BYTES == 0


def test_parquet_mutation_invalidates_cached_bytes(tmp_path):
    path = tmp_path / "mutable.parquet"
    pq.write_table(pa.table({"image": [b"original"]}), path)
    assert canonical.read_locator_bytes(parquet_locator(path, 0), tmp_path) == b"original"
    pq.write_table(pa.table({"image": [b"changed-longer"]}), path)
    assert canonical.read_locator_bytes(parquet_locator(path, 0), tmp_path) == b"changed-longer"
