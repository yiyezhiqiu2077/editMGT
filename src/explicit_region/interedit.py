"""Inter-Edit metadata parsing without requiring local full-dataset assets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
import tarfile
from typing import Iterator


@dataclass(frozen=True)
class InterEditRecord:
    sample_id: int
    source_id: int
    edit_type: str
    instruction_original: str
    better_data: bool
    source_archive: str
    source_file: str
    asset_archive: str
    target_file: str
    mask_file: str
    mask_semantics: str = "user_guidance_region"

    def to_dict(self) -> dict:
        return asdict(self)


REQUIRED = {
    "sample_id",
    "source_id",
    "edit_type",
    "instruction",
    "better_data",
    "source_archive",
    "source_file",
    "asset_archive",
    "target_file",
    "mask_file",
}


def parse_record(row: dict) -> InterEditRecord:
    missing = sorted(REQUIRED - row.keys())
    if missing:
        raise ValueError(f"Inter-Edit row missing keys: {missing}")
    if row["edit_type"] not in {"Add", "Remove", "Local", "Texture"}:
        raise ValueError(f"unsupported Inter-Edit edit_type: {row['edit_type']}")
    if type(row["better_data"]) is not bool:
        raise ValueError("Inter-Edit better_data must be a JSON boolean")
    return InterEditRecord(
        sample_id=int(row["sample_id"]),
        source_id=int(row["source_id"]),
        edit_type=row["edit_type"],
        instruction_original=str(row["instruction"]),
        better_data=row["better_data"],
        source_archive=str(row["source_archive"]),
        source_file=str(row["source_file"]),
        asset_archive=str(row["asset_archive"]),
        target_file=str(row["target_file"]),
        mask_file=str(row["mask_file"]),
    )


def iter_metadata(path: str | Path, *, only_better_data: bool = True) -> Iterator[InterEditRecord]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            rows = json.load(handle)
        else:
            rows = (json.loads(line) for line in handle if line.strip())
        for row in rows:
            record = parse_record(row)
            if not only_better_data or record.better_data:
                yield record


class TarMemberReader:
    """Deterministic row -> archive/member reader; no streaming cursor state."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._handles: dict[Path, tarfile.TarFile] = {}

    def read(self, archive: str, member: str) -> bytes:
        path = (self.root / archive).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError(f"archive escapes Inter-Edit root: {archive}") from exc
        handle = self._handles.setdefault(path, tarfile.open(path, "r:*"))
        extracted = handle.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"{member} not found in {path}")
        return extracted.read()

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
