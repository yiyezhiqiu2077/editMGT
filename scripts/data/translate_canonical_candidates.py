#!/usr/bin/env python3
"""Translate only candidate/reserve prompts requested by iterative corpus build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.translate_interedit import iter_nllb_translation_batches, mock_translate
from src.explicit_region.language import (
    JsonlTranslationCache,
    contains_han,
    translation_cache_key,
    translation_qa_flags,
)


def repair_cache_tail(cache: JsonlTranslationCache) -> None:
    """Make the cache's ignored partial final row safe for the next append.

    The constructor already rejects corrupt nonfinal rows. Retain any ignored
    tail as a content-addressed sidecar before truncating it; committed rows are
    never rewritten. Also separate a valid final JSON row lacking a newline.
    """
    if not cache.path.exists():
        return
    end = 0
    tail_start = 0
    tail = b""
    with cache.path.open("rb") as handle:
        for line in handle:
            if line.strip():
                tail_start, tail = end, line
            end += len(line)
    if not tail:
        return
    try:
        json.loads(tail)
    except json.JSONDecodeError:
        # JsonlTranslationCache only permits a malformed final physical line.
        if tail_start + len(tail) != end:
            raise ValueError("translation cache has a corrupt nonfinal row")
        digest = hashlib.sha256(tail).hexdigest()
        backup = cache.path.with_name(f"{cache.path.name}.partial-{digest}")
        if not backup.exists():
            with backup.open("xb") as handle:
                handle.write(tail)
                handle.flush()
                os.fsync(handle.fileno())
        with cache.path.open("r+b") as handle:
            handle.truncate(tail_start)
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({"event": "translation_cache_tail_recovered", "bytes": len(tail),
                          "preserved_tail": str(backup)}), flush=True)
    else:
        with cache.path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            needs_newline = handle.read(1) != b"\n"
        if needs_newline:
            with cache.path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())


def sync_cache(cache: JsonlTranslationCache) -> None:
    # append() closes/flushed each JSON row; fsync once per bounded batch before
    # acknowledging progress so an interrupted later batch loses no prior batch.
    if cache.path.exists():
        with cache.path.open("ab") as handle:
            handle.flush()
            os.fsync(handle.fileno())


def write_qa(output: Path, rows: list[dict]) -> None:
    # Replace rather than truncate the prior QA report before the new one exists.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--required", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--qa-output", required=True)
    parser.add_argument("--backend", choices=("nllb", "mock"), default="nllb")
    parser.add_argument("--model-path")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--src-lang", default="zho_Hans")
    parser.add_argument("--tgt-lang", default="eng_Latn")
    args = parser.parse_args()
    if args.formal and args.backend == "mock":
        raise SystemExit("MOCK_TRANSLATION_FORBIDDEN_IN_FORMAL")
    if args.backend == "nllb" and not args.model_path:
        raise SystemExit("TRANSLATOR_MODEL_NOT_FOUND")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    decoding = {"do_sample": False, "num_beams": 1, "max_new_tokens": 128}
    requested = []
    seen = set()
    with Path(args.required).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                text = json.loads(line)["instruction_original"]
                if contains_han(text) and text not in seen:
                    requested.append(text)
                    seen.add(text)
    cache = JsonlTranslationCache(args.cache)
    repair_cache_tail(cache)
    pending, keys = [], []
    for text in requested:
        key = translation_cache_key(text, args.backend, args.revision, args.src_lang, args.tgt_lang, decoding)
        if cache.get(key) is None:
            pending.append(text)
            keys.append(key)
    # Include flagged cases from every earlier request/backfill, not just this
    # invocation. A previously QA-flagged cached translation is not rerun.
    qa = [row for row in cache.records.values()
          if translation_qa_flags(row["source_text"], row["translated_text"])]
    output = Path(args.qa_output)
    write_qa(output, qa)
    started = time.monotonic()
    committed = 0
    completed_batches = 0

    def progress(event):
        print(json.dumps({"event": event, "requested_unique": len(requested),
                          "cached_requested": len(requested) - len(pending),
                          "pending_initial": len(pending), "translated_now": committed,
                          "remaining": len(pending) - committed, "completed_batches": completed_batches,
                          "qa_failed": len(qa), "elapsed_seconds": round(time.monotonic() - started, 3)}),
              flush=True)

    progress("translation_started")
    try:
        batches = ()
        if pending:
            batches = (
                (mock_translate(pending[start:start + args.batch_size])
                 for start in range(0, len(pending), args.batch_size))
                if args.backend == "mock" else iter_nllb_translation_batches(
                    pending, args.model_path, args.revision, decoding, args.batch_size, args.src_lang, args.tgt_lang
                )
            )
        for translated in batches:
            expected = min(args.batch_size, len(pending) - committed)
            if not expected or len(translated) != expected:
                raise RuntimeError("translator returned an incomplete batch")
            # Validate and construct the entire returned batch before committing
            # any row of it. Generation failures keep all earlier cache batches.
            rows = []
            for text, value, key in zip(pending[committed:committed + expected], translated,
                                        keys[committed:committed + expected]):
                flags = translation_qa_flags(text, value)
                rows.append({"cache_key": key, "source_text": text, "translated_text": value,
                             "backend": args.backend, "revision": args.revision,
                             "source_language": args.src_lang, "target_language": args.tgt_lang,
                             "decoding_config": decoding, "status": "ok" if not flags else "qa_flagged",
                             "qa_flags": flags})
            for row in rows:
                cache.append(row)
                committed += 1
                if row["qa_flags"]:
                    qa.append(row)
            sync_cache(cache)
            completed_batches += 1
            write_qa(output, qa)
            progress("translation_batch_committed")
        if committed != len(pending):
            raise RuntimeError("translator returned an incomplete batch")
    except BaseException:
        # A signal can arrive just after append() closes its row but before the
        # in-memory counters update. Re-read durable rows for accurate QA/resume.
        sync_cache(cache)
        saved = JsonlTranslationCache(cache.path)
        committed = sum(saved.get(key) is not None for key in keys)
        qa = [row for row in saved.records.values()
              if translation_qa_flags(row["source_text"], row["translated_text"])]
        progress("translation_interrupted")
        raise
    finally:
        sync_cache(cache)
        write_qa(output, qa)
    progress("translation_complete")


if __name__ == "__main__":
    main()
