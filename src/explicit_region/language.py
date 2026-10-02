"""Deterministic language detection and revision-aware translation caching."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .deterministic import canonical_json


HAN_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF]")


def contains_han(text: str) -> bool:
    return bool(HAN_RE.search(text))


def translation_cache_key(
    source_text: str,
    backend_name: str,
    immutable_model_revision: str,
    source_language: str,
    target_language: str,
    decoding_config: dict,
) -> str:
    payload = "\0".join(
        [
            source_text,
            backend_name,
            immutable_model_revision,
            source_language,
            target_language,
            canonical_json(decoding_config),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JsonlTranslationCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        if line_number == sum(1 for _ in self.path.open(encoding="utf-8")):
                            break
                        raise
                    key = row["cache_key"]
                    if key in self.records and self.records[key] != row:
                        raise ValueError(f"translation cache inconsistency for {key}")
                    self.records[key] = row

    def get(self, key: str) -> dict | None:
        return self.records.get(key)

    def append(self, row: dict) -> None:
        key = row["cache_key"]
        if key in self.records:
            if self.records[key] != row:
                raise ValueError(f"translation cache collision for {key}")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        self.records[key] = row


def translation_qa_flags(source: str, translated: str) -> list[str]:
    flags = []
    if not translated.strip():
        flags.append("empty_output")
    if translated.strip() == source.strip() and contains_han(source):
        flags.append("copy_output")
    if contains_han(translated):
        flags.append("han_remaining")
    if source and translated and not 0.2 <= len(translated) / len(source) <= 5.0:
        flags.append("length_ratio_outlier")
    source_numbers = re.findall(r"\d+(?:\.\d+)?", source)
    translated_numbers = re.findall(r"\d+(?:\.\d+)?", translated)
    if source_numbers != translated_numbers:
        flags.append("digit_mismatch")
    return flags
