from pathlib import Path
import pytest

from src.explicit_region.fixed_corpus import write_corpus_ready,verify_corpus_ready


def test_corpus_ready_recomputes_hashes(tmp_path):
    train=tmp_path/"train.jsonl";train.write_text("frozen\n")
    marker=tmp_path/"CORPUS_READY.json"
    write_corpus_ready(marker,files={"train_200k":train},metadata={"total_rows":200000})
    assert verify_corpus_ready(marker)["status"]=="READY"
    train.write_text("mutated\n")
    with pytest.raises(RuntimeError,match="CORPUS_NOT_READY"):verify_corpus_ready(marker)
