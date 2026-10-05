import json

import pytest

from src.dimo.contracts import DIMO_EDIT_FORMAL_READY, enforce_run_guard, load_teacher_contract


def test_formal_status_is_immutable_false_and_smoke_is_two_step_limited():
    assert DIMO_EDIT_FORMAL_READY is False
    metadata = enforce_run_guard(None, prep_smoke=True, max_optimizer_steps=2, formal_ready=False)
    assert metadata["PREP_ONLY_TEACHER"] is True
    with pytest.raises(RuntimeError, match="PREP_SMOKE_MAX_STEPS_EXCEEDED"):
        enforce_run_guard(None, prep_smoke=True, max_optimizer_steps=3, formal_ready=False)
    with pytest.raises(RuntimeError, match="DIMO_TEACHER_NOT_SELECTED"):
        enforce_run_guard(None, prep_smoke=False, max_optimizer_steps=1, formal_ready=False)


def test_teacher_contract_requires_all_identity_fields(tmp_path):
    (tmp_path / "dimo_teacher_manifest.json").write_text(json.dumps({"formal_teacher": False}))
    with pytest.raises(RuntimeError, match="incomplete"):
        load_teacher_contract(tmp_path)
