"""Portable, content-addressed records for the fixed four-dataset corpus."""

from __future__ import annotations

import atexit
import errno
from bisect import bisect_right
from collections import OrderedDict
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import tarfile
import tempfile
import threading
import time
from typing import Any
import uuid

from PIL import Image

from .deterministic import canonical_json


SCHEMA_VERSION = "fixed200k-v2"
DATASET_NAMES = {"magicbrush", "crispedit", "scaleedit", "interedit"}
CANONICAL_EDIT_TYPES = {
    "add", "remove", "replace", "local_attribute", "texture",
    "background", "style", "motion", "other",
}
REQUIRED_FIELDS = {
    "schema_version", "sample_uid", "manifest_index", "dataset_name",
    "dataset_revision", "group_id", "source_locator", "target_locator",
    "region_locator", "source_sha256", "target_sha256", "region_sha256",
    "instruction_original", "instruction_en", "language_original",
    "edit_type_original", "edit_type_canonical", "mask_semantics",
    "region_fraction", "translation_status", "translation_cache_key",
}


class FrozenCorpusIntegrityError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalized_instruction(text: str) -> str:
    return " ".join(str(text).strip().split())


def sample_uid_for(record: dict[str, Any]) -> str:
    payload = "\0".join([
        str(record["dataset_name"]), str(record["dataset_revision"]),
        str(record["source_sha256"]), str(record["target_sha256"]),
        str(record["region_sha256"]),
        normalized_instruction(record["instruction_original"]),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_locator(locator: dict[str, Any]) -> None:
    backend = locator.get("backend")
    path_fields = {
        "file": ("relative_path",), "tar": ("archive", "member"),
        "parquet": ("file",), "derived_bbox": (),
    }
    if backend not in path_fields:
        raise ValueError(f"unsupported locator backend: {backend}")
    for field in path_fields[backend]:
        value = str(locator.get(field, ""))
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"locator {field} must be a safe relative path: {value!r}")
    if backend == "parquet":
        if (type(locator.get("row_index")) is not int or locator["row_index"] < 0 or
                not isinstance(locator.get("column"), str) or not locator["column"]):
            raise ValueError("parquet locator requires non-negative integer row_index and column")
    if backend == "derived_bbox":
        bbox = locator.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("derived_bbox locator requires bbox=[x0,y0,x1,y1]")
        if min(int(locator.get("reference_width", 0)), int(locator.get("reference_height", 0))) <= 0:
            raise ValueError("derived_bbox locator requires positive reference dimensions")


def locator_identity_bytes(locator: dict[str, Any]) -> bytes:
    validate_locator(locator)
    return canonical_json(locator).encode("utf-8")


def _resolve(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise FrozenCorpusIntegrityError(f"asset escapes dataset root: {relative}") from exc
    return candidate


# Process-local LRUs: no asset bytes are copied to disk; only tar header indexes
# live in the external cache. Arrow may decode one page larger than the cache
# budget, but no whole image column or unbounded row group is retained.
_MAX_OPEN_ASSETS = 8
_PARQUET_CACHE_BYTES = 128 * 1024 * 1024
_PARQUET_GROUP_BYTES = 32 * 1024 * 1024
_PARQUET_BATCH_ROWS = 16
_PARQUET_MAX_METADATA_BYTES = 32 * 1024 * 1024
_CACHE_PID = os.getpid()
_CACHE_PROCESS_TOKEN = uuid.uuid4().hex
_CACHE_LOCK = threading.RLock()
_TAR_READERS = OrderedDict()
_PARQUET_READERS = OrderedDict()
_PARQUET_BLOCKS = OrderedDict()
_PARQUET_BLOCK_BYTES = 0


def _asset_identity(path: Path) -> tuple:
    stat = path.stat()
    return (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def clear_locator_caches() -> None:
    """Release process-local handles/tables; safe to call at worker shutdown."""
    global _PARQUET_BLOCK_BYTES
    for connection, handle in _TAR_READERS.values():
        connection.close(); handle.close()
    _TAR_READERS.clear()
    for reader, _ in _PARQUET_READERS.values():
        reader.close()
    _PARQUET_READERS.clear()
    _PARQUET_BLOCKS.clear()
    _PARQUET_BLOCK_BYTES = 0


def _after_fork() -> None:
    global _CACHE_PID, _CACHE_LOCK, _CACHE_PROCESS_TOKEN
    # Child must not use SQLite connections or inherited file-position state.
    clear_locator_caches()
    _CACHE_PID = os.getpid()
    _CACHE_PROCESS_TOKEN = uuid.uuid4().hex
    _CACHE_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
atexit.register(clear_locator_caches)


def _check_pid() -> None:
    if _CACHE_PID != os.getpid():
        _after_fork()


TAR_INDEX_SCHEMA = "canonical-tar-offset-v2"


def _tar_index_path(identity: tuple) -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    cache = Path(base) if base else Path.home() / ".cache"
    cache = cache / "editmgt" / TAR_INDEX_SCHEMA
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    return cache / f"{key}.sqlite3"


def _build_tar_index(path: Path, identity: tuple, index: Path) -> Path:
    # Streaming TarInfo iteration plus disk SQLite keeps million-member indexes
    # out of Python RAM. Atomic publication allows independent DDP workers to
    # converge on the same immutable index. Object-backed mounts without flock
    # use process-private files: their rename/open is not safely concurrent.
    import fcntl
    if index.exists():
        return index
    with index.with_suffix(".lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
        except OSError as exc:
            if exc.errno not in (errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP):
                raise
            index = index.with_name(f"{index.stem}.{_CACHE_PROCESS_TOKEN}.sqlite3")
        if index.exists():
            return index
        fd, temporary_name = tempfile.mkstemp(prefix=index.stem + ".", suffix=".tmp", dir=index.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        connection = None
        try:
            # The temporary DB has a unique filename and one writer. nolock
            # avoids unsupported POSIX locks on object-backed RUN_ROOT mounts.
            connection = sqlite3.connect(f"{temporary.as_uri()}?mode=rw&nolock=1", uri=True)
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA cache_size=-2048")
            connection.execute("CREATE TABLE members(name TEXT PRIMARY KEY, offset INTEGER, size INTEGER, regular INTEGER) WITHOUT ROWID")
            connection.execute("CREATE TABLE identity(schema TEXT NOT NULL, value TEXT NOT NULL)")
            connection.execute("INSERT INTO identity VALUES (?,?)", (TAR_INDEX_SCHEMA, canonical_json(identity)))
            with tarfile.open(path, "r:") as archive:
                for member in archive:
                    # Match tarfile's last-entry-wins lookup, but never follow
                    # symlink/hardlink members outside their declared identity.
                    connection.execute("INSERT OR REPLACE INTO members VALUES (?,?,?,?)", (
                        member.name, member.offset_data, member.size,
                        int(member.isfile() and not member.issparse()),
                    ))
                    archive.members.clear()  # TarFile otherwise retains every header.
            _require_asset_unchanged(path, identity)
            connection.commit(); connection.close(); connection = None
            temporary.replace(index)
            return index
        except tarfile.ReadError as exc:
            raise FrozenCorpusIntegrityError(
                f"indexed canonical tar requires an uncompressed archive: {path}"
            ) from exc
        finally:
            if connection is not None:
                connection.close()
            temporary.unlink(missing_ok=True)


def _require_asset_unchanged(path: Path, identity: tuple) -> None:
    if _asset_identity(path) != identity:
        raise FrozenCorpusIntegrityError(f"asset changed during locator read: {path}")


def validate_tar_index(index: Path, identity: tuple) -> dict:
    """Validate a completed index before single-writer shared publication."""
    connection = _open_tar_index(index)
    try:
        recorded = connection.execute("SELECT schema,value FROM identity").fetchone()
        if recorded != (TAR_INDEX_SCHEMA, canonical_json(identity)):
            raise FrozenCorpusIntegrityError(f"tar index identity/schema mismatch: {index}")
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise FrozenCorpusIntegrityError(f"corrupt tar index: {index}")
        invalid = connection.execute(
            "SELECT COUNT(*) FROM members WHERE regular NOT IN (0,1) OR offset<0 OR size<0 OR offset+size>?",
            (identity[3],),
        ).fetchone()[0]
        if invalid:
            raise FrozenCorpusIntegrityError(f"invalid tar index offsets: {index}")
        counts = dict(connection.execute("SELECT regular,COUNT(*) FROM members GROUP BY regular"))
        return {"members": sum(counts.values()), "regular_members": counts.get(1, 0),
                "nonregular_members": counts.get(0, 0)}
    finally:
        connection.close()


def _open_tar_index(index: Path):
    # Object-store mounts can expose a just-renamed immutable file slightly
    # late. Retry only open/I/O errors, never malformed schema or byte reads.
    for attempt in range(6):
        try:
            return sqlite3.connect(f"{index.as_uri()}?mode=ro&immutable=1", uri=True, check_same_thread=False)
        except sqlite3.OperationalError as exc:
            if ("unable to open" not in str(exc) and "disk I/O" not in str(exc)) or attempt == 5:
                raise
            time.sleep(0.1 * 2 ** attempt)


def _read_tar(path: Path, member: str) -> bytes:
    identity = _asset_identity(path)
    readers = _TAR_READERS.get(identity)
    if readers is None:
        index = _build_tar_index(path, identity, _tar_index_path(identity))
        connection = _open_tar_index(index)
        connection.execute("PRAGMA cache_size=-2048")
        recorded = connection.execute("SELECT schema,value FROM identity").fetchone()
        if recorded != (TAR_INDEX_SCHEMA, canonical_json(identity)):
            connection.close()
            raise FrozenCorpusIntegrityError(f"tar index identity mismatch: {path}")
        readers = (connection, path.open("rb"))
        _TAR_READERS[identity] = readers
        while len(_TAR_READERS) > _MAX_OPEN_ASSETS:
            _, (old_connection, old_handle) = _TAR_READERS.popitem(last=False)
            old_connection.close(); old_handle.close()
    _TAR_READERS.move_to_end(identity)
    connection, handle = readers
    entry = connection.execute("SELECT offset,size,regular FROM members WHERE name=?", (member,)).fetchone()
    if entry is None:
        raise FrozenCorpusIntegrityError(f"tar member missing: {member}")
    offset, size, regular = entry
    if not regular or offset < 0 or size < 0 or offset + size > identity[3]:
        raise FrozenCorpusIntegrityError(f"tar member must be a regular in-bounds file: {member}")
    # pread is independent of file position and safe even when users call from
    # multiple threads. Reads return exactly the original member payload bytes.
    chunks, position = [], offset
    while position < offset + size:
        value = os.pread(handle.fileno(), min(offset + size - position, 8 * 1024 * 1024), position)
        if not value:
            raise FrozenCorpusIntegrityError(f"truncated tar member: {member}")
        chunks.append(value); position += len(value)
    _require_asset_unchanged(path, identity)
    return b"".join(chunks)


def _parquet_reader(path: Path, identity: tuple):
    import pyarrow.parquet as pq
    cached = _PARQUET_READERS.get(identity)
    if cached is None:
        # Cap retained footer metadata too, rather than keeping an arbitrarily
        # large count of row groups in every worker.
        with path.open("rb") as handle:
            handle.seek(-8, os.SEEK_END)
            footer = handle.read(8)
        if footer[4:] != b"PAR1" or int.from_bytes(footer[:4], "little") > _PARQUET_MAX_METADATA_BYTES:
            raise FrozenCorpusIntegrityError("invalid or oversized Parquet footer")
        reader = pq.ParquetFile(path, memory_map=False, pre_buffer=False)
        ends, total = [], 0
        for index in range(reader.num_row_groups):
            total += reader.metadata.row_group(index).num_rows
            ends.append(total)
        cached = (reader, ends)
        _PARQUET_READERS[identity] = cached
        while len(_PARQUET_READERS) > _MAX_OPEN_ASSETS:
            _, (old_reader, _) = _PARQUET_READERS.popitem(last=False)
            old_reader.close()
    _PARQUET_READERS.move_to_end(identity)
    return cached


def _cache_parquet_block(key, value) -> None:
    global _PARQUET_BLOCK_BYTES
    size = value.get_total_buffer_size()
    if size > _PARQUET_CACHE_BYTES:
        return
    while _PARQUET_BLOCKS and _PARQUET_BLOCK_BYTES + size > _PARQUET_CACHE_BYTES:
        _, old = _PARQUET_BLOCKS.popitem(last=False)
        _PARQUET_BLOCK_BYTES -= old.get_total_buffer_size()
    _PARQUET_BLOCKS[key] = value
    _PARQUET_BLOCK_BYTES += size


def _read_parquet(path: Path, column: str, row_index: int) -> bytes:
    identity = _asset_identity(path)
    reader, ends = _parquet_reader(path, identity)
    if not ends or row_index >= ends[-1]:
        raise FrozenCorpusIntegrityError(f"parquet row_index out of bounds: {row_index}")
    if column not in reader.schema_arrow.names:
        raise FrozenCorpusIntegrityError(f"parquet column missing: {column}")
    group_index = bisect_right(ends, row_index)
    local_index = row_index - (ends[group_index - 1] if group_index else 0)
    metadata = reader.metadata.row_group(group_index)
    # Struct image columns have paths such as image.bytes/image.path.
    projected_bytes = sum(metadata.column(i).total_uncompressed_size for i in range(metadata.num_columns)
                          if metadata.column(i).path_in_schema.split(".", 1)[0] == column)
    whole_group = projected_bytes <= _PARQUET_GROUP_BYTES
    block_index = -1 if whole_group else local_index // _PARQUET_BATCH_ROWS
    key = (identity, column, group_index, block_index)
    block = _PARQUET_BLOCKS.get(key)
    if block is None:
        if whole_group:
            block = reader.read_row_group(group_index, columns=[column], use_threads=False).column(column)
        else:
            # A giant image row group must not become a giant retained table.
            # Arrow still decodes pages; iter_batches bounds returned data. For
            # random access in giant groups an offline small-row-group export
            # is preferable; no second image reader or asset rewrite happens here.
            for batch_index, batch in enumerate(reader.iter_batches(
                    batch_size=_PARQUET_BATCH_ROWS, row_groups=[group_index],
                    columns=[column], use_threads=False)):
                if batch_index == block_index:
                    block = batch.column(0)
                    break
            if block is None:
                raise FrozenCorpusIntegrityError("parquet batch missing")
        _cache_parquet_block(key, block)
    else:
        _PARQUET_BLOCKS.move_to_end(key)
    value = block[local_index if whole_group else local_index % _PARQUET_BATCH_ROWS].as_py()
    if isinstance(value, dict) and "bytes" in value:
        value = value["bytes"]
    if isinstance(value, memoryview):
        value = value.tobytes()
    if not isinstance(value, bytes):
        raise FrozenCorpusIntegrityError("parquet image cell is not bytes")
    _require_asset_unchanged(path, identity)
    return value


def read_locator_bytes(locator: dict[str, Any], root: str | Path) -> bytes:
    """Read an asset without allowing a manifest to escape its dataset root."""
    validate_locator(locator)
    root = Path(root)
    backend = locator["backend"]
    if backend == "file":
        return _resolve(root, locator["relative_path"]).read_bytes()
    if backend in ("tar", "parquet"):
        _check_pid()
        with _CACHE_LOCK:
            if backend == "tar":
                return _read_tar(_resolve(root, locator["archive"]), locator["member"])
            return _read_parquet(_resolve(root, locator["file"]), locator["column"], locator["row_index"])
    if backend == "derived_bbox":
        return locator_identity_bytes(locator)
    raise AssertionError(backend)


def image_from_locator(locator: dict[str, Any], root: str | Path, *,
                       payload: bytes | None = None) -> Image.Image:
    """Decode via the canonical rules; optional already-read bytes avoid rereads."""
    if locator["backend"] == "derived_bbox":
        width, height = int(locator["reference_width"]), int(locator["reference_height"])
        x0, y0, x1, y1 = map(int, locator["bbox"])
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise FrozenCorpusIntegrityError(f"invalid bbox coordinates: {locator['bbox']}")
        image = Image.new("L", (width, height), 0)
        image.paste(255, (x0, y0, x1, y1))
        return image
    with Image.open(io.BytesIO(read_locator_bytes(locator, root) if payload is None else payload)) as image:
        return image.copy()


def expected_locator_hash(locator: dict[str, Any], root: str | Path) -> str:
    return sha256_bytes(
        locator_identity_bytes(locator)
        if locator["backend"] == "derived_bbox"
        else read_locator_bytes(locator, root)
    )


def validate_record(record: dict[str, Any], *, require_frozen_index: bool = False) -> None:
    missing = sorted(REQUIRED_FIELDS - record.keys())
    if missing:
        raise ValueError(f"canonical record missing fields: {missing}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported canonical schema: {record['schema_version']}")
    if record["dataset_name"] not in DATASET_NAMES:
        raise ValueError(f"invalid dataset_name: {record['dataset_name']}")
    if record["edit_type_canonical"] not in CANONICAL_EDIT_TYPES:
        raise ValueError(f"invalid canonical edit type: {record['edit_type_canonical']}")
    for field in ("source_locator", "target_locator", "region_locator"):
        validate_locator(record[field])
    for field in ("source_sha256", "target_sha256", "region_sha256", "sample_uid"):
        value = str(record[field])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{field} must be lowercase SHA256")
    if record["sample_uid"] != sample_uid_for(record):
        raise ValueError("sample_uid does not match stable content identity")
    if require_frozen_index and not isinstance(record["manifest_index"], int):
        raise ValueError("frozen record requires integer manifest_index")
    if not 0 <= float(record["region_fraction"]) <= 1:
        raise ValueError("region_fraction must be in [0,1]")


def verify_record_assets(record: dict[str, Any], roots: dict[str, str | Path]) -> None:
    validate_record(record)
    root = roots.get(record["dataset_name"])
    if root is None:
        raise FrozenCorpusIntegrityError(f"missing root for {record['dataset_name']}")
    for prefix in ("source", "target", "region"):
        actual = expected_locator_hash(record[f"{prefix}_locator"], root)
        if actual != record[f"{prefix}_sha256"]:
            raise FrozenCorpusIntegrityError(
                f"{prefix} hash mismatch for {record['sample_uid']}: "
                f"expected={record[f'{prefix}_sha256']} actual={actual}"
            )
