import pytest

from src.explicit_region.deterministic import global_index_for_rank, stable_seed
from src.explicit_region.config import load_config
from src.v2_utils import resolve_lora_scope


def test_stateless_seed_namespace_and_resume():
    assert stable_seed(42, 10, "a", "geometry") == stable_seed(42, 10, "a", "geometry")
    assert stable_seed(42, 10, "a", "geometry") != stable_seed(42, 10, "a", "mask")
    before = [global_index_for_rank(0, m, 0, 2, 1) for m in range(3)]
    after = [global_index_for_rank(6, m, 0, 2, 1) for m in range(3)]
    assert before == [0, 2, 4]
    assert after == [6, 8, 10]


def test_lora_scope_legacy_mapping():
    assert resolve_lora_scope(None, True) == "reference_only"
    assert resolve_lora_scope(None, False) == "both"
    assert resolve_lora_scope("target_only", True) == "target_only"
    with pytest.raises(ValueError):
        resolve_lora_scope("ambiguous", False)


def test_unresolved_environment_fails(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("root: ${THIS_MUST_NOT_EXIST}\n")
    monkeypatch.delenv("THIS_MUST_NOT_EXIST", raising=False)
    with pytest.raises(KeyError):
        load_config(path)


def test_overridden_parent_environment_is_not_required(tmp_path, monkeypatch):
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text("data:\n  manifest: ${OVERRIDDEN_PARENT_ENV}\n")
    child.write_text("extends: parent.yaml\ndata:\n  manifest: local.jsonl\n")
    monkeypatch.delenv("OVERRIDDEN_PARENT_ENV", raising=False)
    assert load_config(child)["data"]["manifest"] == "local.jsonl"
