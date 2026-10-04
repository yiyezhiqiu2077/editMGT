import json
import sys

import pytest

from scripts.data import translate_canonical_candidates as translator
from scripts.data import translate_interedit as interedit_translator
from src.explicit_region.language import JsonlTranslationCache, translation_cache_key


DECODING = {"do_sample": False, "num_beams": 1, "max_new_tokens": 128}
REVISION = "fixture-rev"


def cache_row(text, translated=None):
    return {
        "cache_key": translation_cache_key(text, "nllb", REVISION, "zho_Hans", "eng_Latn", DECODING),
        "source_text": text,
        "translated_text": translated if translated is not None else text,
        "qa_flags": ["copy_output", "han_remaining"] if translated is None else [],
    }


def run_args(tmp_path, monkeypatch, texts, batch_size=2):
    required = tmp_path / "required.jsonl"
    required.write_text("".join(json.dumps({"instruction_original": text}) + "\n" for text in texts))
    cache = tmp_path / "cache.jsonl"
    qa = tmp_path / "qa.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "translate", "--required", str(required), "--cache", str(cache),
        "--qa-output", str(qa), "--model-path", "unused", "--revision", REVISION,
        "--batch-size", str(batch_size),
    ])
    return cache, qa


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_complete_cache_does_not_load_model_and_preserves_flagged_cases(tmp_path, monkeypatch):
    text = "把物体改成红色"
    cache, qa = run_args(tmp_path, monkeypatch, [text])
    cache.write_text(json.dumps(cache_row(text)) + "\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("complete cache must not load NLLB or access a GPU")

    monkeypatch.setattr(translator, "iter_nllb_translation_batches", forbidden)
    translator.main()
    assert json.loads(qa.read_text())["source_text"] == text


@pytest.mark.parametrize("failure", [RuntimeError("mock translator interrupted"), KeyboardInterrupt()])
def test_streamed_batches_persist_qa_and_resume_only_pending(tmp_path, monkeypatch, capsys, failure):
    prior = "把物体改成红色"
    requested = ["添加一只狗", "移除汽车", "把杯子变成蓝色", "添加纹理", "画一个三角形"]
    cache, qa = run_args(tmp_path, monkeypatch, [prior, *requested[:2], "English passthrough", requested[0], *requested[2:]])
    cache.write_text(json.dumps(cache_row(prior)) + "\n")
    calls = []

    def interrupting(texts, model_path, revision, decoding, batch_size, src_lang, tgt_lang):
        calls.append(list(texts))
        assert (model_path, revision, decoding, batch_size, src_lang, tgt_lang) == (
            "unused", REVISION, DECODING, 2, "zho_Hans", "eng_Latn"
        )
        yield ["Add a dog", requested[1]]  # Second output is a durable QA failure.
        assert [row["source_text"] for row in read_rows(cache)] == [prior, *requested[:2]]
        assert [row["source_text"] for row in read_rows(qa)] == [prior, requested[1]]
        raise failure

    monkeypatch.setattr(translator, "iter_nllb_translation_batches", interrupting)
    with pytest.raises(type(failure)):
        translator.main()
    assert calls == [requested]
    before = cache.read_bytes()
    assert len(JsonlTranslationCache(cache).records) == 3
    assert [row["source_text"] for row in read_rows(qa)] == [prior, requested[1]]
    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["event"] for row in progress] == ["translation_started", "translation_batch_committed", "translation_interrupted"]
    assert progress[-1]["translated_now"] == 2
    assert progress[-1]["remaining"] == 3
    assert progress[-1]["elapsed_seconds"] >= 0

    def resumed(texts, *args):
        calls.append(list(texts))
        assert texts == requested[2:]
        yield ["Make the cup blue", "Add texture"]
        yield ["Draw a triangle"]

    monkeypatch.setattr(translator, "iter_nllb_translation_batches", resumed)
    translator.main()
    assert calls == [requested, requested[2:]]
    assert cache.read_bytes().startswith(before)
    assert [row["source_text"] for row in read_rows(cache)] == [prior, *requested]
    assert [row["source_text"] for row in read_rows(qa)] == [prior, requested[1]]
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["translated_now"] == 3


@pytest.mark.parametrize("bad_batch", [["Only one result"], ["one", "two", "three"], ["valid", None]])
def test_bad_batch_is_rejected_before_any_of_its_rows_commit(tmp_path, monkeypatch, bad_batch):
    texts = ["添加一只狗", "移除汽车", "把杯子变成蓝色", "添加纹理"]
    cache, qa = run_args(tmp_path, monkeypatch, texts)

    def batches(*args):
        yield ["Add a dog", "Remove the car"]
        yield bad_batch

    monkeypatch.setattr(translator, "iter_nllb_translation_batches", batches)
    with pytest.raises((RuntimeError, AttributeError)):
        translator.main()
    assert [row["source_text"] for row in read_rows(cache)] == texts[:2]
    assert read_rows(qa) == []


def test_premature_stream_end_keeps_completed_batch(tmp_path, monkeypatch):
    texts = ["添加一只狗", "移除汽车", "添加纹理"]
    cache, _ = run_args(tmp_path, monkeypatch, texts)
    monkeypatch.setattr(translator, "iter_nllb_translation_batches", lambda *args: iter([["Add a dog", "Remove the car"]]))
    with pytest.raises(RuntimeError, match="incomplete batch"):
        translator.main()
    assert [row["source_text"] for row in read_rows(cache)] == texts[:2]


def test_interrupt_after_cache_write_rebuilds_qa_from_disk(tmp_path, monkeypatch, capsys):
    texts = ["添加一只狗", "移除汽车"]
    cache, qa = run_args(tmp_path, monkeypatch, texts)
    monkeypatch.setattr(translator, "iter_nllb_translation_batches", lambda *args: iter([texts]))
    original = JsonlTranslationCache.append

    def append_then_interrupt(self, row):
        original(self, row)
        raise KeyboardInterrupt()

    monkeypatch.setattr(JsonlTranslationCache, "append", append_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        translator.main()
    assert [row["source_text"] for row in read_rows(cache)] == texts[:1]
    assert read_rows(qa) == read_rows(cache)
    progress = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert progress["translated_now"] == 1
    assert progress["remaining"] == 1


def test_load_failure_preserves_prior_cache_and_refreshes_all_qa(tmp_path, monkeypatch):
    prior = "把物体改成红色"
    cache, qa = run_args(tmp_path, monkeypatch, ["添加纹理"])
    cache.write_text(json.dumps(cache_row(prior)) + "\n")
    before = cache.read_bytes()
    qa.write_text("stale\n")

    def failing(*args):
        raise RuntimeError("model load failed")

    monkeypatch.setattr(translator, "iter_nllb_translation_batches", failing)
    with pytest.raises(RuntimeError, match="model load failed"):
        translator.main()
    assert cache.read_bytes() == before
    assert read_rows(qa) == [cache_row(prior)]


@pytest.mark.parametrize("tail", [b'{"cache_key": "interrupted', b'{"cache_key": "interrupted\n'])
def test_partial_cache_tail_recovered_before_resume(tmp_path, monkeypatch, tail):
    prior = "把物体改成红色"
    pending = "添加纹理"
    cache, qa = run_args(tmp_path, monkeypatch, [prior, pending])
    committed = (json.dumps(cache_row(prior)) + "\n").encode()
    cache.write_bytes(committed + tail)

    def batches(texts, *args):
        assert texts == [pending]
        yield ["Add texture"]

    monkeypatch.setattr(translator, "iter_nllb_translation_batches", batches)
    translator.main()
    assert cache.read_bytes().startswith(committed)
    assert len(JsonlTranslationCache(cache).records) == 2
    assert [row["source_text"] for row in read_rows(cache)] == [prior, pending]
    assert read_rows(qa) == [cache_row(prior)]
    assert next(tmp_path.glob("cache.jsonl.partial-*")).read_bytes() == tail


def test_valid_cache_last_row_without_newline_is_preserved(tmp_path, monkeypatch):
    prior = "把物体改成红色"
    cache, _ = run_args(tmp_path, monkeypatch, [prior, "添加纹理"])
    cache.write_text(json.dumps(cache_row(prior)))
    monkeypatch.setattr(translator, "iter_nllb_translation_batches", lambda *args: iter([["Add texture"]]))
    translator.main()
    assert len(JsonlTranslationCache(cache).records) == 2
    assert read_rows(cache)[0] == cache_row(prior)


def test_corrupt_nonfinal_cache_row_is_not_repaired(tmp_path, monkeypatch):
    cache, _ = run_args(tmp_path, monkeypatch, ["添加纹理"])
    cache.write_text('{"broken"\n' + json.dumps(cache_row("把物体改成红色")) + "\n")
    before = cache.read_bytes()
    with pytest.raises(json.JSONDecodeError):
        translator.main()
    assert cache.read_bytes() == before
    assert not list(tmp_path.glob("cache.jsonl.partial-*"))


def test_mock_backend_remains_bounded_and_does_not_load_nllb(tmp_path, monkeypatch):
    cache, _ = run_args(tmp_path, monkeypatch, ["添加一只狗", "移除汽车", "添加纹理"])
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--backend", "mock"])
    original = translator.mock_translate
    sizes = []

    def mock(texts):
        sizes.append(len(texts))
        return original(texts)

    def forbidden(*args):
        raise AssertionError("mock must not load model")

    monkeypatch.setattr(translator, "mock_translate", mock)
    monkeypatch.setattr(translator, "iter_nllb_translation_batches", forbidden)
    translator.main()
    assert sizes == [2, 1]
    assert [row["translated_text"] for row in read_rows(cache)] == ["Add a dog", "Remove the car", "Add texture"]


def test_streaming_nllb_loads_once_and_retains_greedy_recipe(monkeypatch):
    import transformers

    calls = []

    class Encoded(dict):
        def to(self, device):
            assert device == "cuda"
            return self

    class Tokenizer:
        def convert_tokens_to_ids(self, language):
            assert language == "eng_Latn"
            return 256047

        def __call__(self, texts, **kwargs):
            assert self.src_lang == "zho_Hans"
            assert kwargs == {"return_tensors": "pt", "padding": True}
            calls.append(("encode", list(texts)))
            return Encoded(input_ids=list(texts))

        def batch_decode(self, generated, **kwargs):
            assert kwargs == {"skip_special_tokens": True}
            return generated

    class Model:
        def to(self, device):
            assert device == "cuda"
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            texts = kwargs.pop("input_ids")
            assert kwargs == {"forced_bos_token_id": 256047, **DECODING}
            calls.append(("generate", list(texts)))
            return [f"translated {text}" for text in texts]

    def load_tokenizer(*args, **kwargs):
        calls.append(("load_tokenizer", args, kwargs))
        return Tokenizer()

    def load_model(*args, **kwargs):
        calls.append(("load_model", args, kwargs))
        return Model()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForSeq2SeqLM, "from_pretrained", load_model)
    iterator = interedit_translator.iter_nllb_translation_batches(
        ["甲", "乙", "丙"], "pinned-model", REVISION, DECODING, 2, "zho_Hans", "eng_Latn"
    )
    assert calls == []  # Lazy until the first requested batch.
    assert next(iterator) == ["translated 甲", "translated 乙"]
    assert [call[0] for call in calls] == ["load_tokenizer", "load_model", "encode", "generate"]
    assert list(iterator) == [["translated 丙"]]
    assert sum(call[0] == "load_model" for call in calls) == 1
    assert calls[0] == ("load_tokenizer", ("pinned-model",), {
        "revision": REVISION, "local_files_only": True, "src_lang": "zho_Hans"
    })
    assert calls[1] == ("load_model", ("pinned-model",), {
        "revision": REVISION, "local_files_only": True, "torch_dtype": interedit_translator.torch.bfloat16
    })
    assert [call[1] for call in calls if call[0] == "encode"] == [["甲", "乙"], ["丙"]]


def test_legacy_nllb_list_api_flattens_stream_in_order(monkeypatch):
    calls = []

    def batches(*args):
        calls.append(args)
        yield ["one", "two"]
        yield ["three"]

    monkeypatch.setattr(interedit_translator, "iter_nllb_translation_batches", batches)
    args = (["甲", "乙", "丙"], "pinned-model", REVISION, DECODING, 2, "zho_Hans", "eng_Latn")
    assert interedit_translator.nllb_translate(*args) == ["one", "two", "three"]
    assert calls == [args]


def test_empty_stream_and_invalid_batch_do_not_load_model():
    assert list(interedit_translator.iter_nllb_translation_batches(
        [], "unused", REVISION, DECODING, 32, "zho_Hans", "eng_Latn"
    )) == []
    with pytest.raises(ValueError, match="positive"):
        list(interedit_translator.iter_nllb_translation_batches(
            ["甲"], "unused", REVISION, DECODING, 0, "zho_Hans", "eng_Latn"
        ))
