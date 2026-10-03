import copy

import pytest

from src.dimo.checkpoint import SCHEMA_VERSION, validate_inference_checkpoint


def _contract():
    teacher = {"bundle_sha256": "teacher-bundle", "adapter_sha256": "adapter"}
    inference = {"sha256": "inference", "teacher_bundle_sha256": "teacher-bundle"}
    state = {
        "schema": SCHEMA_VERSION,
        "step_phase": "complete",
        "teacher_bundle_fingerprint": teacher,
        "teacher_bundle_sha256": "teacher-bundle",
        "inference_fingerprint": inference,
        "dimo_upstream_reference_commit": "upstream",
        "config_hash": "config",
    }
    return state, teacher, inference


def test_inference_checkpoint_contract_accepts_exact_identity():
    state, teacher, inference = _contract()
    validate_inference_checkpoint(
        state, expected_teacher_bundle_fingerprint=teacher,
        expected_inference_fingerprint=inference, expected_upstream_commit="upstream",
    )


@pytest.mark.parametrize("field", [
    "schema", "teacher_bundle_sha256", "inference_fingerprint",
    "dimo_upstream_reference_commit",
])
def test_inference_checkpoint_contract_fails_closed(field):
    state, teacher, inference = _contract()
    broken = copy.deepcopy(state)
    broken[field] = "wrong" if field != "inference_fingerprint" else {"sha256": "wrong"}
    with pytest.raises(RuntimeError, match="DIMO_INFERENCE_CHECKPOINT_MISMATCH"):
        validate_inference_checkpoint(
            broken, expected_teacher_bundle_fingerprint=teacher,
            expected_inference_fingerprint=inference, expected_upstream_commit="upstream",
        )


def test_inference_checkpoint_contract_requires_a_recorded_config_hash():
    state, teacher, inference = _contract()
    state["config_hash"] = ""
    with pytest.raises(RuntimeError, match="DIMO_INFERENCE_CHECKPOINT_MISMATCH"):
        validate_inference_checkpoint(
            state, expected_teacher_bundle_fingerprint=teacher,
            expected_inference_fingerprint=inference, expected_upstream_commit="upstream",
        )
