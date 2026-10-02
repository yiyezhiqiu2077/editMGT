#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.explicit_region.interedit import iter_metadata
from src.explicit_region.language import (
    JsonlTranslationCache,
    contains_han,
    translation_cache_key,
    translation_qa_flags,
)


def mock_translate(texts: list[str]) -> list[str]:
    table = {
        "添加一只狗": "Add a dog",
        "移除汽车": "Remove the car",
        "把杯子变成蓝色": "Make the cup blue",
        "添加纹理": "Add texture",
    }
    return [table.get(text, "Translated instruction") for text in texts]


def nllb_translate(texts, model_path, revision, decoding, batch_size, src_lang, tgt_lang):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, revision=revision, local_files_only=True, src_lang=src_lang
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, revision=revision, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    tokenizer.src_lang = src_lang
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    outputs = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start : start + batch_size], return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=forced_bos,
                do_sample=False,
                num_beams=int(decoding["num_beams"]),
                max_new_tokens=int(decoding["max_new_tokens"]),
            )
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", nargs="+")
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
    decoding = {"do_sample": False, "num_beams": 1, "max_new_tokens": 128}
    unique = {}
    for metadata in args.metadata:
        for row in iter_metadata(metadata, only_better_data=True):
            unique.setdefault(row.instruction_original, row.edit_type)
    cache = JsonlTranslationCache(args.cache)
    pending = []
    keys = []
    for text in sorted(unique):
        source_language = "zho_Hans" if contains_han(text) else "eng_Latn"
        key = translation_cache_key(text, args.backend, args.revision, source_language, args.tgt_lang, decoding)
        if cache.get(key) is None:
            pending.append(text)
            keys.append((key, source_language))
    chinese = [text for text in pending if contains_han(text)]
    translated_chinese = (
        mock_translate(chinese)
        if args.backend == "mock"
        else nllb_translate(
            chinese, args.model_path, args.revision, decoding, args.batch_size,
            args.src_lang, args.tgt_lang,
        )
    )
    translated_iter = iter(translated_chinese)
    for text, (key, language) in zip(pending, keys):
        translated = next(translated_iter) if language == "zho_Hans" else text
        cache.append(
            {
                "cache_key": key,
                "source_text": text,
                "source_sha256": __import__("hashlib").sha256(text.encode()).hexdigest(),
                "translated_text": translated,
                "backend": args.backend,
                "revision": args.revision,
                "source_language": language,
                "target_language": args.tgt_lang,
                "decoding_config": decoding,
                "status": "ok" if not translation_qa_flags(text, translated) else "qa_flagged",
                "qa_flags": translation_qa_flags(text, translated),
            }
        )
    qa_rows = [row for row in cache.records.values() if row.get("qa_flags")]
    output = Path(args.qa_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in qa_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"unique": len(unique), "translated_now": len(pending), "qa_flagged": len(qa_rows)}))


if __name__ == "__main__":
    main()
