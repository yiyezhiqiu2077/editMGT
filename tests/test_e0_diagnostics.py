import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.explicit_region import gates, selection
from src.explicit_region.config import load_config
from src.explicit_region.contracts import COMPONENT_FILES

ROOT = Path(__file__).resolve().parents[1]
REVISION = "0a00f14ea159728fb76ac99ed741f9deca5fee28"


@pytest.fixture
def environment(tmp_path, monkeypatch):
    for key in ("DERIVED_ROOT", "EDITMGT_OUTPUT_ROOT", "EDITMGT_MODEL_ROOT", "MAGICBRUSH_ROOT", "MAGICBRUSH_DEV_ROOT",
                "CRISPEDIT_ROOT", "SCALEEDIT_ROOT", "INTEREDIT_ROOT"):
        monkeypatch.setenv(key, str(tmp_path / key.lower()))
    monkeypatch.setenv("TRANSLATOR_REVISION", "fixture-revision")


@pytest.fixture
def released(tmp_path, environment):
    model = tmp_path / "snapshots" / REVISION
    entries = []
    for folder, marker in COMPONENT_FILES.values():
        for filename, raw in ((marker, b'{}'), ("fixture-weights.bin", b"pinned model bytes")):
            path = model / folder / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            entries.append({"rfilename": f"{folder}/{filename}", "size": len(raw),
                            **({"lfs": {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}} if filename.endswith("bin")
                               else {"blobId": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()})})
    metadata = tmp_path / "release.json"
    metadata.write_text(json.dumps({"id": "WeiChow/EditMGT", "sha": REVISION, "siblings": entries}))
    manifest = tmp_path / "magicbrush_official_dev.jsonl"
    manifest.write_text(json.dumps({"sample_uid": "fixture-sample", "dataset_name": "magicbrush"}) + "\n")
    frozen = {"identity_sha256": "fixture-frozen", "eval_config": selection.config_identity(ROOT / "configs/eval/formal.yaml"),
              "primary_manifest": selection.manifest_identity(manifest, primary=True)}
    return model, metadata, manifest, frozen


def identity(released, experiment="E0-official", mode="official_upstream_timestep"):
    model, metadata, manifest, frozen = released
    return selection.baseline_evaluation_identity(experiment=experiment, model_root=model, release_metadata=metadata,
        config_path=ROOT / "configs/eval/formal.yaml", manifest=manifest, dataset="magicbrush", timestep_mode=mode,
        preregistration=frozen)


def test_baseline_has_pinned_identity_protocol_and_no_checkpoint(released):
    official = identity(released)
    region = identity(released, "E0-region", "roi_relative")
    assert official["identity_sha256"] != region["identity_sha256"]
    assert official["released_model"] == region["released_model"]
    assert "checkpoint" not in official
    assert official["role"] == "released_baseline" and not official["ranking_eligible"]
    assert official["inference_settings"] == {"lora_scope": "both", "persistent_conditioning": False}
    with pytest.raises(ValueError, match="protocol"):
        identity(released, "E0-official", "roi_relative")


@pytest.mark.parametrize("change", ["weight", "blob", "extra", "revision"])
def test_baseline_rejects_changed_released_assets(released, change):
    model, metadata, _, _ = released
    if change == "weight": (model / "editmgt/fixture-weights.bin").write_bytes(b"wrong model bytes")
    elif change == "blob": (model / "editmgt/config.json").write_bytes(b'[]')
    elif change == "extra": (model / "editmgt/other.bin").write_bytes(b'extra')
    else:
        value = json.loads(metadata.read_text()); value["sha"] = "0" * 40; metadata.write_text(json.dumps(value))
    with pytest.raises(ValueError): identity(released)


def test_baseline_rejects_partial_or_probe_manifest_and_ranking(released):
    model, metadata, manifest, frozen = released
    expected = identity(released)
    assessment = selection.assess_candidate({"schema": "formal-evaluator-v2", "identity": expected},
        expected_identity=expected, sample_keys=["fixture-sample"], selection={})
    assert not assessment["eligible"] and "baseline" in assessment["reason"]
    probe = manifest.with_name("magicbrush_probe128.jsonl"); probe.write_bytes(manifest.read_bytes())
    with pytest.raises(ValueError, match="full official"):
        selection.baseline_evaluation_identity(experiment="E0-official", model_root=model, release_metadata=metadata,
            config_path=ROOT / "configs/eval/formal.yaml", manifest=probe, dataset="magicbrush",
            timestep_mode="official_upstream_timestep", preregistration=frozen)


@pytest.fixture
def diagnostic(environment, monkeypatch, tmp_path):
    # Only hash/READY I/O is mocked; exact real diagnostic recipe matching executes.
    ready = Path(load_config(ROOT / gates.DIAGNOSTIC_CONFIGS[0])["data"]["corpus_ready"])
    ready.parent.mkdir(parents=True, exist_ok=True); ready.write_text('{"status":"READY"}')
    approval = tmp_path / "fixture-human.json"
    approval.write_text(json.dumps({"schema": "fixed200k-human-approval-v1", "actor_type": "human", "decision": "go",
        "approver": "synthetic unit fixture only", "approved_at": "fixture-date", "evidence": "not real approval",
        "corpus_ready_sha256": gates.sha256_file(ready)}))
    monkeypatch.setattr(gates, "verify_corpus_ready", lambda path: {"status": "READY"})
    monkeypatch.setattr(gates, "build_gate_identity", lambda *a, **k: {"fixture": "code-identity"})
    monkeypatch.setattr(gates, "committed_config_files", lambda *a, **k: {"fixture": "committed"})
    monkeypatch.setenv("CORPUS_APPROVAL", str(approval))
    monkeypatch.delenv("FIXED200K_SMOKE_REPORT", raising=False)
    monkeypatch.delenv("EDITMGT_PREREGISTRATION", raising=False)
    return ready, approval


@pytest.mark.parametrize("relative", gates.DIAGNOSTIC_CONFIGS)
def test_exact_branch_diagnostics_need_ready_human_but_no_smoke(diagnostic, relative):
    config = load_config(ROOT / relative)
    assert gates.training_stage(config) == "diagnostic"
    result = gates.enforce_training_gate(config)
    assert result["stage"] == "diagnostic"
    assert result["diagnostic_recipe"]["successful_optimizer_updates"] == 2
    assert not result["diagnostic_recipe"]["ranking_eligible"]
    ready, approval = diagnostic
    with pytest.raises(ValueError, match="real external human"):
        gates.enforce_stage_gate("diagnostic", corpus_ready=str(ready), corpus_approval=None, diagnostic_config=config)
    with pytest.raises(ValueError, match="whitelisted"):
        gates.enforce_stage_gate("diagnostic", corpus_ready=str(ready), corpus_approval=approval)


@pytest.mark.parametrize("mutation", ["steps", "output", "scope", "conditioning", "resolution"])
def test_changed_diagnostic_recipe_is_not_a_gate_bypass(diagnostic, mutation):
    config = copy.deepcopy(load_config(ROOT / gates.DIAGNOSTIC_CONFIGS[0]))
    if mutation == "steps": config["max_optimizer_steps"] = 3
    elif mutation == "output": config["output_dir"] = str(Path(config["output_dir"]).parent / "e1-full-target-both")
    elif mutation == "scope": config["lora"]["scope"] = "reference_only"
    elif mutation == "conditioning": config["corruption"]["persistent_conditioning"] = True
    else: config["resolution"] = 512
    assert gates.training_stage(config) == "formal"
    ready, approval = diagnostic
    with pytest.raises(ValueError, match="whitelisted"):
        gates.enforce_stage_gate("diagnostic", corpus_ready=str(ready), corpus_approval=approval, diagnostic_config=config)


def test_diagnostic_requires_committed_config(diagnostic, monkeypatch):
    def not_committed(*a, **k): raise ValueError("not committed")
    monkeypatch.setattr(gates, "committed_config_files", not_committed)
    with pytest.raises(ValueError, match="not committed"):
        gates.enforce_training_gate(load_config(ROOT / gates.DIAGNOSTIC_CONFIGS[0]))
