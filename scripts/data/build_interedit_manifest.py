#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explicit_region.interedit import iter_metadata
from src.explicit_region.language import JsonlTranslationCache, contains_han, translation_cache_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", nargs="+")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    decoding = {"do_sample": False, "num_beams": 1, "max_new_tokens": 128}
    cache = JsonlTranslationCache(args.cache)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for metadata in args.metadata:
            for record in iter_metadata(metadata, only_better_data=True):
                language = "zho_Hans" if contains_han(record.instruction_original) else "eng_Latn"
                key = translation_cache_key(
                    record.instruction_original, args.backend, args.revision, language, "eng_Latn", decoding
                )
                cached = cache.get(key)
                if cached is None or cached["status"] != "ok":
                    raise SystemExit(f"TRANSLATION_NOT_READY sample_id={record.sample_id}")
                instruction_en = cached["translated_text"]
                if contains_han(instruction_en):
                    raise SystemExit(f"HAN_IN_FORMAL_MANIFEST sample_id={record.sample_id}")
                row = record.to_dict() | {
                    "instruction_en": instruction_en,
                    "language_original": language,
                    "translation_status": cached["status"],
                    "translation_backend": args.backend,
                    "translation_revision": args.revision,
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
    print(json.dumps({"records": count, "han_count": 0, "output": str(output)}))


if __name__ == "__main__":
    main()
