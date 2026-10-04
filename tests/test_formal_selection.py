import copy
import json
from pathlib import Path

import pytest

from src.explicit_region import selection

ROOT = Path(__file__).resolve().parents[1]


def metric_row(key, seed, identity, *, inside=.02, outside=.04, progress=.2):
    return {"sample_key": key, "seed": seed, "dataset_name": "magicbrush", "identity": identity,
            "metrics": {"inside_masked_lpips": inside, "outside_masked_lpips": outside,
                        "d_ST": .1, "d_SO": .1 * progress, "d_OT": inside,
                        "progress": (.1 * progress) / .1, "no_op": progress <= .1}}


def report_for(identity, keys=("a", "b"), **kwargs):
    return {"schema": "formal-evaluator-v2", "identity": identity,
            "per_generation": [metric_row(key, seed, identity, **kwargs) for key in keys for seed in range(4)]}


@pytest.fixture
def rules(monkeypatch, tmp_path):
    monkeypatch.setenv("DERIVED_ROOT", str(tmp_path))
    return selection.validate_selection_config(selection.load_config(ROOT / "configs/eval/formal.yaml"))


def assess(report, identity, rules):
    return selection.assess_candidate(report, expected_identity=identity, sample_keys=["a", "b"], selection=rules)


def test_preservation_boundary_and_seed_then_sample_mean(rules):
    identity = {"dataset": "magicbrush", "split": "official_dev"}
    report = report_for(identity, outside=.05)
    assert assess(report, identity, rules)["eligible"]
    for row in report["per_generation"]:
        value = .01 if row["sample_key"] == "a" else .03
        row["metrics"]["inside_masked_lpips"] = row["metrics"]["d_OT"] = value
    assert assess(report, identity, rules)["metrics"]["inside_masked_lpips"] == .02
    report["per_generation"][0]["metrics"]["outside_masked_lpips"] = .051
    assert not assess(report, identity, rules)["eligible"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_seed", "wrong_sample", "wrong_identity",
                                     "wrong_dataset", "nan", "inf", "missing_metric", "bad_progress", "bad_noop"])
def test_incomplete_or_inconsistent_candidate_is_ineligible(rules, mutation):
    identity = {"dataset": "magicbrush", "split": "official_dev"}
    report = report_for(identity)
    row = report["per_generation"][0]
    if mutation == "missing": report["per_generation"].pop()
    elif mutation == "duplicate": report["per_generation"].append(copy.deepcopy(row))
    elif mutation == "wrong_seed": row["seed"] = 9
    elif mutation == "wrong_sample": row["sample_key"] = "extra"
    elif mutation == "wrong_identity": row["identity"] = {}
    elif mutation == "wrong_dataset": row["dataset_name"] = "crispedit"
    elif mutation == "nan": row["metrics"]["d_ST"] = float("nan")
    elif mutation == "inf": row["metrics"]["inside_masked_lpips"] = float("inf")
    elif mutation == "missing_metric": del row["metrics"]["outside_masked_lpips"]
    elif mutation == "bad_progress": row["metrics"]["progress"] = 1
    elif mutation == "bad_noop": row["metrics"]["no_op"] = True
    assert not assess(report, identity, rules)["eligible"]


def test_auxiliary_and_probe_never_rank(rules):
    for identity in ({"dataset": "crispedit", "split": "diagnostic"},
                     {"dataset": "magicbrush", "split": "diagnostic"}):
        assert not assess(report_for(identity), identity, rules)["eligible"]


def test_progress_zero_denominator_and_noop_thresholds(rules):
    identity = {"dataset": "magicbrush", "split": "official_dev"}
    report = report_for(identity)
    first = report["per_generation"][0]["metrics"]
    first.update(d_ST=0., d_SO=0., progress=0., no_op=False)
    assert assess(report, identity, rules)["eligible"]
    first.update(d_ST=.05, d_SO=.005, progress=.005 / .05, no_op=True)
    assert assess(report, identity, rules)["metrics"]["no_op"] == .125


@pytest.fixture
def frozen_repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    env = {"DERIVED_ROOT": str(tmp_path / "derived"), "EDITMGT_OUTPUT_ROOT": str(tmp_path / "runs"),
           "EDITMGT_MODEL_ROOT": str(tmp_path / "model"), "TRANSLATOR_REVISION": "fixed-revision"}
    for name in ("MAGICBRUSH_ROOT", "MAGICBRUSH_DEV_ROOT", "CRISPEDIT_ROOT", "SCALEEDIT_ROOT", "INTEREDIT_ROOT"):
        env[name] = str(tmp_path / name)
    for name, value in env.items(): monkeypatch.setenv(name, value)
    paths = [ROOT / "configs/eval/formal.yaml", ROOT / "configs/train/_cluster_8g_base.yaml"]
    paths += [ROOT / relative for relative in selection.IMPLEMENTATION_FILES]
    paths += [ROOT / f"configs/train/cluster_8g_{name}.yaml" for name in
              ("e1", "e2", "e3", "e4", "lr_1e-5", "lr_3e-5", "lr_5e-5", "smoke", "smoke_fresh10", "smoke_resume20")]
    committed = {}
    for source in paths:
        target = root / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        committed[str(source.relative_to(ROOT))] = source.read_bytes()
    def git(_root, *args):
        if args[0] == "show": return committed[args[1].split(":", 1)[1]]
        if args[0] == "rev-parse": return b"a" * 40
        if args[0] == "merge-base": return b""
        raise AssertionError(args)
    monkeypatch.setattr(selection, "_git", git)
    manifest = Path(env["DERIVED_ROOT"]) / "fixed200k/validation/magicbrush_official_dev.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(''.join(json.dumps({"sample_uid": key, "dataset_name": "magicbrush"}) + "\n" for key in ("a", "b")))
    ready = manifest.parents[1] / "CORPUS_READY.json"
    ready_payload = {"status": "READY", "files": {"validation_magicbrush_official_dev": selection.file_identity(manifest)}}
    ready.write_text(json.dumps(ready_payload))
    import src.explicit_region.fixed_corpus as fixed
    monkeypatch.setattr(fixed, "verify_corpus_ready", lambda path: json.loads(Path(path).read_text()))
    prereg = tmp_path / "preregistration.json"
    payload = selection.freeze_preregistration(root / "configs/eval/formal.yaml", prereg, corpus_ready=ready, repo_root=root)
    return root, prereg, payload


def make_candidate(candidate, frozen, tmp_path, **kwargs):
    checkpoint = Path(candidate["checkpoint_path"])
    checkpoint.mkdir(parents=True)
    config = selection.load_config(candidate["train_config"]["path"])
    trainable = {field: config.get(field) for field in ("data", "validation", "resolution", "seed", "corruption", "token_mask", "lora", "optimizer",
                     "scheduler_horizon_steps", "warmup_steps", "precision", "component_dtypes", "torch_compile")}
    trainable.update(world_size=8, per_device_batch=config["batch_per_gpu"], gradient_accumulation=config["gradient_accumulation"],
                     data_content_hashes={"data.corpus_ready": {"resolved_path": frozen["corpus_ready"]["path"],
                                                               "sha256": frozen["corpus_ready"]["sha256"]}})
    trainable["training_config"] = {k: v for k, v in config.items()
                                   if k not in {"output_dir", "max_optimizer_steps", "checkpoint_steps", "run_type"}}
    fingerprint = selection.object_sha256(trainable)
    for name in selection.CHECKPOINT_FILES: (checkpoint / name).write_bytes(b"fixture")
    (checkpoint / "trainable_config.json").write_text(json.dumps(trainable))
    (checkpoint / "fingerprint.json").write_text(json.dumps({"sha256": fingerprint, "payload": trainable}))
    (checkpoint / "checkpoint_metadata.json").write_text(json.dumps({
        "schema_version": 2, "global_optimizer_step": candidate["step"], "fingerprint": fingerprint,
        "committed_global_sample_count": candidate["step"] * 32,
        "resolved_config_sha256": candidate["train_config"]["resolved_sha256"]}))
    identity = selection.evaluation_identity(experiment=candidate["experiment"], checkpoint=checkpoint,
                  config_path=frozen["eval_config"]["path"], manifest=frozen["primary_manifest"]["path"],
                  dataset="magicbrush", preregistration=frozen)
    path = tmp_path / f"{candidate['experiment']}-{candidate['step']}.json"
    report = report_for(identity, **kwargs)
    predictions = tmp_path / f"{candidate['experiment']}-{candidate['step']}-predictions.jsonl"
    predictions.write_text(''.join(json.dumps({k: v for k, v in row.items() if k != "metrics"}) + '\n' for row in report["per_generation"]))
    report["predictions_manifest"] = selection.file_identity(predictions)
    path.write_text(json.dumps(report))
    return path


def test_frozen_lr_universe_and_missing_candidates(frozen_repo, tmp_path):
    root, prereg, frozen = frozen_repo
    assert len(frozen["candidate_sets"]["lr"]) == 3
    assert {c["step"] for c in frozen["candidate_sets"]["lr"]} == {1500}
    assert len(frozen["candidate_sets"]["formal"]) == 40
    candidate = frozen["candidate_sets"]["lr"][0]
    report = make_candidate(candidate, frozen, tmp_path)
    result = selection.select_candidates(prereg, [report], "lr", repo_root=root)
    assert result["selected"]["experiment"] == "LR-1e-5"
    assert len([c for c in result["candidates"] if not c["eligible"]]) == 2
    assert not result["test_authorized"]


def test_none_eligible_no_test_and_diagnostic_excluded(frozen_repo, tmp_path):
    root, prereg, frozen = frozen_repo
    candidate = frozen["candidate_sets"]["lr"][0]
    report = make_candidate(candidate, frozen, tmp_path, outside=.051)
    result = selection.select_candidates(prereg, [report], "lr", repo_root=root)
    assert result["selected"] is None and not result["test_authorized"]
    assert result["status"] == "NO_ELIGIBLE_CANDIDATE"
    value = json.loads(report.read_text())
    value["identity"]["checkpoint"]["step"] = 500
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="diagnostic"):
        selection.select_candidates(prereg, [report], "lr", repo_root=root)


def test_exact_ties_noop_then_step_then_fixed_name(frozen_repo, tmp_path):
    root, prereg, frozen = frozen_repo
    candidates = [c for c in frozen["candidate_sets"]["formal"] if
                  (c["experiment"], c["step"]) in (("E1", 625), ("E2", 625), ("E3", 1250), ("E4", 625))]
    reports = [make_candidate(c, frozen, tmp_path, progress=.05 if c["experiment"] == "E1" else .2) for c in candidates]
    result = selection.select_candidates(prereg, reports, "formal", repo_root=root)
    assert result["selected"]["experiment"] == "E2"
    assert len({selection.output_identity_name(c["experiment"], selection.checkpoint_identity(c["checkpoint_path"], c["experiment"]))
                for c in candidates}) == 4


@pytest.mark.parametrize("change", ["weights", "step", "train_config", "manifest", "eval_config", "base_config"])
def test_identity_changes_fail_closed(frozen_repo, tmp_path, change):
    root, prereg, frozen = frozen_repo
    candidate = frozen["candidate_sets"]["lr"][0]
    report = make_candidate(candidate, frozen, tmp_path)
    checkpoint = Path(candidate["checkpoint_path"])
    if change == "weights": (checkpoint / "adapter_model.safetensors").write_bytes(b"changed")
    elif change in ("step", "train_config"):
        path = checkpoint / "checkpoint_metadata.json"
        metadata = json.loads(path.read_text())
        metadata["global_optimizer_step" if change == "step" else "resolved_config_sha256"] = 500 if change == "step" else "0" * 64
        path.write_text(json.dumps(metadata))
    elif change == "manifest":
        path = Path(frozen["primary_manifest"]["path"])
        path.write_text(path.read_text() + '\n')
    elif change == "eval_config":
        path = root / "configs/eval/formal.yaml"; path.write_text(path.read_text() + '\n')
    else:
        path = root / "configs/train/_cluster_8g_base.yaml"; path.write_text(path.read_text() + '\n')
    if change in ("manifest", "eval_config", "base_config"):
        with pytest.raises(ValueError): selection.select_candidates(prereg, [report], "lr", repo_root=root)
    else:
        assert selection.select_candidates(prereg, [report], "lr", repo_root=root)["selected"] is None


def test_prepare_generation_honors_e4_and_lr_diagnostic_not_ranking(frozen_repo, tmp_path, monkeypatch):
    from scripts.eval import generate_evaluation as generation
    root, prereg, frozen = frozen_repo
    monkeypatch.setattr(generation, "verify_preregistration", lambda path: selection.verify_preregistration(path, repo_root=root))
    monkeypatch.setattr(generation, "enforce_stage_gate", lambda *args, **kwargs: {"status": "PASS"})
    candidate = next(c for c in frozen["candidate_sets"]["formal"] if c["experiment"] == "E4" and c["step"] == 625)
    make_candidate(candidate, frozen, tmp_path)
    _, identity = generation.prepare_evaluation("E4", candidate["checkpoint_path"], frozen["eval_config"]["path"],
                        frozen["primary_manifest"]["path"], "magicbrush", prereg)
    assert identity["inference_settings"] == {"lora_scope": "reference_only", "persistent_conditioning": True}
    parent = frozen["candidate_sets"]["lr"][0]
    diagnostic = parent | {"step": 500, "checkpoint_path": str(Path(parent["checkpoint_path"]).parent / "checkpoint-500")}
    make_candidate(diagnostic, frozen, tmp_path)
    _, identity = generation.prepare_evaluation(parent["experiment"], diagnostic["checkpoint_path"], frozen["eval_config"]["path"],
                        frozen["primary_manifest"]["path"], "magicbrush", prereg)
    assert identity["checkpoint"]["step"] == 500
    assert 500 not in {c["step"] for c in frozen["candidate_sets"]["lr"]}


def test_committed_parent_config_cannot_hide_inherited_changes(frozen_repo):
    root, _, frozen = frozen_repo
    path = root / "configs/train/cluster_8g_e4.yaml"
    committed = selection.committed_config_files(path, root)
    assert "configs/train/_cluster_8g_base.yaml" in committed
    base = root / "configs/train/_cluster_8g_base.yaml"
    base.write_text(base.read_text() + "\n# changed fixture\n")
    with pytest.raises(ValueError, match="not committed"):
        selection.committed_config_files(path, root)


def test_freeze_is_exclusive_and_precedes_runs(frozen_repo, tmp_path):
    root, prereg, frozen = frozen_repo
    kwargs = dict(corpus_ready=frozen["corpus_ready"]["path"], repo_root=root)
    with pytest.raises(FileExistsError):
        selection.freeze_preregistration(frozen["eval_config"]["path"], prereg, **kwargs)
    Path(frozen["candidate_sets"]["lr"][0]["checkpoint_path"]).parent.mkdir(parents=True)
    with pytest.raises(ValueError, match="before|precede"):
        selection.freeze_preregistration(frozen["eval_config"]["path"], tmp_path / "late.json", **kwargs)
