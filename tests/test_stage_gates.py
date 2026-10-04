import json
from pathlib import Path

import pytest

from src.explicit_region import gates


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    ready = tmp_path / "READY.json"
    ready.write_text('{"status":"READY"}')
    approval = tmp_path / "external-human-review.json"
    approval.write_text(json.dumps({"schema": "fixed200k-human-approval-v1", "actor_type": "human",
        "decision": "go", "approver": "CPU test fixture (not real corpus approval)",
        "approved_at": "2026-10-03T00:00:00Z", "evidence": "synthetic unit fixture only",
        "corpus_ready_sha256": gates.sha256_file(ready)}))
    identity = {"schema": "stage-gate-identity-v1", "corpus_ready_sha256": gates.sha256_file(ready), "git_sha": "abc"}
    monkeypatch.setattr(gates, "verify_corpus_ready", lambda path: json.loads(Path(path).read_text()))
    monkeypatch.setattr(gates, "build_gate_identity", lambda *args, **kwargs: identity)
    monkeypatch.setattr(gates, "verify_preregistration", lambda *args, **kwargs: {"identity_sha256": "frozen"})
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({"schema": "fixed200k-smoke-verification-v2", "status": "PASS", "gate_identity": identity,
        "samples": 640, "unique_samples": 640, "checkpoint_evidence": {"10": {}, "20": {}},
        **{key: True for key in ("sample_sequence_match", "geometry_seed_sequence_match", "corruption_seed_sequence_match",
            "learning_rate_sequence_match", "finite_state", "exact_weights_match", "optimizer_match", "scheduler_match",
            "quality_state_match", "sampler_cursor_match", "rank_rng_match", "exact_metrics_match")}}))
    return {"corpus_ready": ready, "corpus_approval": approval, "smoke_report": smoke,
            "preregistration": tmp_path / "prereg.json"}


def test_smoke_needs_human_approval_but_not_smoke_or_prereg(evidence):
    evidence.pop("smoke_report")
    evidence.pop("preregistration")
    result = gates.enforce_stage_gate("smoke", **evidence)
    assert result["status"] == "PASS"
    evidence["corpus_approval"] = None
    with pytest.raises(ValueError, match="real external human"):
        gates.enforce_stage_gate("smoke", **evidence)


@pytest.mark.parametrize("stage", ["probe", "formal", "eval"])
def test_later_stages_need_smoke_and_committed_prereg(evidence, stage):
    assert gates.enforce_stage_gate(stage, **evidence)["status"] == "PASS"
    no_smoke = evidence | {"smoke_report": None}
    with pytest.raises(ValueError, match="smoke PASS"):
        gates.enforce_stage_gate(stage, **no_smoke)
    with pytest.raises(ValueError, match="preregistration"):
        gates.enforce_stage_gate(stage, **(evidence | {"preregistration": None}))
    Path(evidence["smoke_report"]).write_text(json.dumps({"status": "PASS", "gate_identity": {"git_sha": "stale"}}))
    with pytest.raises(ValueError, match="stale"):
        gates.enforce_stage_gate(stage, **evidence)


@pytest.mark.parametrize("field,value", [("actor_type", "assistant"), ("decision", "no-go"),
                                         ("corpus_ready_sha256", "different"), ("evidence", ""),
                                         ("schema", "approved-plan")])
def test_plan_auto_approval_no_go_and_stale_corpus_fail(evidence, field, value):
    approval = Path(evidence["corpus_approval"])
    payload = json.loads(approval.read_text()); payload[field] = value
    approval.write_text(json.dumps(payload))
    with pytest.raises(ValueError): gates.enforce_stage_gate("smoke", **evidence)


def test_bare_pass_and_missing_finite_state_do_not_pass(evidence):
    smoke = Path(evidence["smoke_report"])
    report = json.loads(smoke.read_text())
    report.pop("finite_state")
    smoke.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="state evidence"):
        gates.enforce_stage_gate("formal", **evidence)


def test_ready_verifier_failure_is_not_bypassed(evidence, monkeypatch):
    def reject(path): raise RuntimeError("corpus invalid")
    monkeypatch.setattr(gates, "verify_corpus_ready", reject)
    with pytest.raises(RuntimeError, match="corpus invalid"):
        gates.enforce_stage_gate("smoke", **evidence)


def test_short_arbitrary_recipe_is_not_smoke(monkeypatch):
    exact = {"data": {"name": "fixed_200k"}, "max_optimizer_steps": 20, "output_dir": "smoke-exact"}
    monkeypatch.setattr(gates, "load_config", lambda path: exact)
    assert gates.training_stage(exact) == "smoke"
    assert gates.training_stage(exact | {"output_dir": "arbitrary"}) == "formal"
    assert gates.training_stage(exact | {"output_dir": "lr-probe-1e-5"}) == "probe"
