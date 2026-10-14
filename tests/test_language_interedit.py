import json
import gzip

import pytest

from src.explicit_region.config import load_config
from src.explicit_region.interedit import iter_metadata
from src.explicit_region.language import (
    JsonlTranslationCache,
    contains_han,
    translation_cache_key,
    translation_qa_flags,
)


def test_interedit_fixture_and_better_filter():
    records = list(iter_metadata("tests/fixtures/interedit/metadata.jsonl"))
    assert len(records) == 3
    assert {row.edit_type for row in records} == {"Add", "Remove", "Local"}
    assert all(row.mask_semantics == "user_guidance_region" for row in records)


def test_interedit_jsonl_gzip_and_strict_better_data(tmp_path):
    row = {
        "sample_id": 1, "source_id": 2, "edit_type": "Add",
        "instruction": "添加一只狗", "better_data": True,
        "source_archive": "source.tar", "source_file": "source.png",
        "asset_archive": "asset.tar", "target_file": "target.png",
        "mask_file": "mask.png",
    }
    path = tmp_path / "metadata.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    assert [record.sample_id for record in iter_metadata(path)] == [1]
    row["better_data"] = "false"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="JSON boolean"):
        list(iter_metadata(path))


def test_language_detection_and_english_passthrough():
    assert contains_han("添加一只狗")
    assert not contains_han("Add a dog")
    assert translation_qa_flags("Add a dog", "Add a dog") == []
    assert "han_remaining" in translation_qa_flags("添加一只狗", "添加一只狗")


def test_translation_qa_flag_names():
    assert "empty_output" in translation_qa_flags("添加一只狗", "")
    assert "copy_output" in translation_qa_flags("添加一只狗", "添加一只狗")
    assert "length_ratio_outlier" in translation_qa_flags("短", "a" * 20)
    assert "digit_mismatch" in translation_qa_flags("添加 2 只狗", "Add three dogs")


def test_formal_nllb_config_has_explicit_languages(monkeypatch):
    monkeypatch.setenv("TRANSLATOR_REVISION", "immutable-test-revision")
    config = load_config("configs/translation/nllb.yaml")
    assert config["backend"] == "nllb"
    assert config["src_lang"] == "zho_Hans"
    assert config["tgt_lang"] == "eng_Latn"
    assert config["immutable_revision"] == "immutable-test-revision"
    assert config["decoding"] == {"do_sample": False, "num_beams": 1, "max_new_tokens": 128}


def test_revision_aware_cache_resume(tmp_path):
    decoding = {"num_beams": 1}
    key = translation_cache_key("添加一只狗", "mock", "rev1", "zho_Hans", "eng_Latn", decoding)
    other = translation_cache_key("添加一只狗", "mock", "rev2", "zho_Hans", "eng_Latn", decoding)
    assert key != other
    assert key != translation_cache_key(
        "添加一只狗", "mock", "rev1", "zho_Hans", "eng_Latn", {"num_beams": 2}
    )
    path = tmp_path / "cache.jsonl"
    row = {"cache_key": key, "translated_text": "Add a dog"}
    cache = JsonlTranslationCache(path)
    cache.append(row)
    resumed = JsonlTranslationCache(path)
    assert resumed.get(key) == row
    resumed.append(row)
    assert len(path.read_text().splitlines()) == 1
