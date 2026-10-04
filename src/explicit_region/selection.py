"""Frozen DEV-only evaluation identities and fail-closed checkpoint selection.

Aggregates in evaluator output are descriptive only. Selection recomputes every
sample-after-seed mean from the exact preregistered manifest and four seeds.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from .config import load_config
from .contracts import audit_component_identity, sha256_file
from .deterministic import canonical_json

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [0, 1, 2, 3]
FORMAL_STEPS = list(range(625, 6251, 625))
IMPLEMENTATION_FILES = (
    "src/explicit_region/selection.py", "src/explicit_region/gates.py", "src/explicit_region/metrics.py",
    "src/pipeline.py", "src/editmgt.py", "scripts/eval/select_checkpoint.py",
    "scripts/eval/generate_evaluation.py", "scripts/eval/formal_eval.py",
)
CHECKPOINT_FILES = (
    "adapter_model.safetensors", "mask_conditioning.safetensors",
    "mask_conditioning_config.json", "trainable_config.json", "fingerprint.json",
    "checkpoint_metadata.json",
)
REQUIRED_METRICS = (
    "inside_masked_lpips", "outside_masked_lpips", "d_ST", "d_SO", "d_OT", "progress",
)


def object_sha256(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_identity(path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def config_identity(path) -> dict:
    return file_identity(path) | {"resolved_sha256": object_sha256(load_config(path))}


def _git(repo_root, *args) -> bytes:
    return subprocess.run(["git", *args], cwd=repo_root, check=True, capture_output=True).stdout


def committed_config_files(path, repo_root=ROOT) -> dict:
    """Bind inherited YAML too; an unchanged child cannot conceal a dirty base."""
    import yaml
    root = Path(repo_root).resolve()
    result = {}
    path = Path(path).resolve()
    while True:
        relative = str(path.relative_to(root))
        raw = path.read_bytes()
        if _git(root, "show", f"HEAD:{relative}") != raw:
            raise ValueError(f"preregistration config is not committed: {relative}")
        result[relative] = hashlib.sha256(raw).hexdigest()
        parent = yaml.safe_load(raw).get("extends")
        if parent is None:
            break
        parent = Path(parent)
        path = parent.resolve() if parent.is_absolute() else (path.parent / parent).resolve()
        if str(path.relative_to(root)) in result:
            raise ValueError("cyclic config inheritance")
    return result


def validate_selection_config(config) -> dict:
    selection = config.get("selection", {})
    expected = {
        "status": "preregistered", "schema": "fixed200k-selection-v1",
        "primary_dataset": "magicbrush", "primary_split": "official_dev",
        "required_seeds": SEEDS, "averaging": "sample_after_seed_mean",
        "preservation_metric": "outside_masked_lpips", "threshold_preserve": 0.05,
        "objective": "inside_masked_lpips", "direction": "minimize",
        "tau_edit": 0.05, "tau_noop": 0.10, "progress": "d_SO/max(d_ST,1e-12)",
        "tie_break": ["no_op", "earlier_step", "fixed_experiment_name"],
        "invalid_candidate": "ineligible", "no_eligible_candidate": "no_selection_no_test",
        "auxiliary_ranking": False, "probe128_ranking": False,
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise ValueError(f"unapproved selection rule: {key}")
    if config.get("generation_seeds") != SEEDS:
        raise ValueError("formal evaluation requires seeds 0,1,2,3 exactly")
    if Path(selection.get("primary_manifest", "")).name != "magicbrush_official_dev.jsonl":
        raise ValueError("primary must be the full frozen official DEV manifest, never probe128/TEST")
    for stage, names, steps in (
        ("lr", ["LR-1e-5", "LR-3e-5", "LR-5e-5"], [1500]),
        ("formal", ["E1", "E2", "E3", "E4"], FORMAL_STEPS),
    ):
        group = selection.get("candidate_sets", {}).get(stage, {})
        if sorted(group.get("experiments", {})) != names or group.get("checkpoint_steps") != steps:
            raise ValueError(f"unapproved {stage} candidate universe")
    if selection["candidate_sets"]["lr"].get("diagnostic_steps") != [500, 1000]:
        raise ValueError("LR steps 500/1000 are diagnostics only")
    return selection


def manifest_identity(path, *, primary=False) -> dict:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [row.get("sample_uid", row.get("sample_key")) for row in rows]
    if not rows or any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("manifest must contain nonempty, unique sample identities")
    if any("test" in str(row.get("split", row.get("source_split", ""))).lower() for row in rows):
        raise ValueError("TEST manifest is not authorized for selection/evaluation")
    if primary:
        if Path(path).name != "magicbrush_official_dev.jsonl":
            raise ValueError("not the frozen official DEV manifest")
        if any(row.get("dataset_name") != "magicbrush" for row in rows):
            raise ValueError("primary manifest contains another dataset")
    return file_identity(path) | {"samples": len(keys), "sample_keys_sha256": object_sha256(keys)}


def freeze_preregistration(config_path, output, *, corpus_ready, repo_root=ROOT) -> dict:
    """Create once, before any candidate run exists; never manufactures approval."""
    from .fixed_corpus import verify_corpus_ready
    config = load_config(config_path)
    selection = validate_selection_config(config)
    ready = verify_corpus_ready(corpus_ready)
    manifest = manifest_identity(selection["primary_manifest"], primary=True)
    attested = ready["files"]["validation_magicbrush_official_dev"]
    if file_identity(attested["path"]) != {k: manifest[k] for k in ("path", "sha256")}:
        raise ValueError("primary DEV manifest is not the READY-attested manifest")
    committed = committed_config_files(config_path, repo_root)
    candidates = {}
    for stage, group in selection["candidate_sets"].items():
        candidates[stage] = []
        for experiment, relative in sorted(group["experiments"].items()):
            path = Path(repo_root) / relative
            train_config = load_config(path)
            output_dir = Path(train_config["output_dir"]).resolve()
            if output_dir.exists():
                raise ValueError(f"freeze must precede candidate outcomes/run directory: {output_dir}")
            committed.update(committed_config_files(path, repo_root))
            identity = config_identity(path)
            for step in group["checkpoint_steps"]:
                candidates[stage].append({
                    "experiment": experiment, "step": step, "train_config": identity,
                    "checkpoint_path": str(output_dir / f"checkpoint-{step}"),
                })
    payload = {
        "schema": "fixed200k-preregistration-v1",
        "git_sha": _git(repo_root, "rev-parse", "HEAD").decode().strip(),
        "committed_config_files": committed, "eval_config": config_identity(config_path),
        "implementation_files": {
            relative: hashlib.sha256(_git(repo_root, "show", f"HEAD:{relative}")).hexdigest()
            for relative in IMPLEMENTATION_FILES
        },
        "corpus_ready": file_identity(corpus_ready), "primary_manifest": manifest,
        "candidate_sets": candidates, "selection": selection,
    }
    for relative, digest in payload["implementation_files"].items():
        if sha256_file(Path(repo_root) / relative) != digest:
            raise ValueError(f"preregistration implementation is not committed: {relative}")
    payload["identity_sha256"] = object_sha256(payload)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return payload


def verify_preregistration(path, *, repo_root=ROOT, corpus_ready=None) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = {k: v for k, v in payload.items() if k != "identity_sha256"}
    if payload.get("schema") != "fixed200k-preregistration-v1" or payload.get("identity_sha256") != object_sha256(unsigned):
        raise ValueError("preregistration identity mismatch")
    _git(repo_root, "merge-base", "--is-ancestor", payload["git_sha"], "HEAD")
    if set(payload["implementation_files"]) != set(IMPLEMENTATION_FILES):
        raise ValueError("preregistration implementation identities are incomplete")
    expected_config_files = committed_config_files(payload["eval_config"]["path"], repo_root)
    train_paths = {candidate["train_config"]["path"] for group in payload["candidate_sets"].values() for candidate in group}
    for train_path in sorted(train_paths):
        expected_config_files.update(committed_config_files(train_path, repo_root))
    if expected_config_files != payload["committed_config_files"]:
        raise ValueError("preregistration config inheritance identities are incomplete")
    for relative, digest in (payload["committed_config_files"] | payload["implementation_files"]).items():
        raw = (Path(repo_root) / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest or _git(repo_root, "show", f"{payload['git_sha']}:{relative}") != raw:
            raise ValueError(f"changed/uncommitted preregistered config: {relative}")
    if config_identity(payload["eval_config"]["path"]) != payload["eval_config"]:
        raise ValueError("evaluation config identity mismatch")
    config = load_config(payload["eval_config"]["path"])
    selection = validate_selection_config(config)
    if selection != payload["selection"]:
        raise ValueError("preregistration rules mismatch")
    if manifest_identity(selection["primary_manifest"], primary=True) != payload["primary_manifest"]:
        raise ValueError("full DEV manifest identity mismatch")
    expected_ready = payload["corpus_ready"]
    if file_identity(corpus_ready or expected_ready["path"]) != expected_ready:
        raise ValueError("preregistration READY identity mismatch")
    for stage, group in selection["candidate_sets"].items():
        expected = []
        for experiment, relative in sorted(group["experiments"].items()):
            cfg_path = Path(repo_root) / relative
            cfg = load_config(cfg_path)
            for step in group["checkpoint_steps"]:
                expected.append({"experiment": experiment, "step": step, "train_config": config_identity(cfg_path),
                                 "checkpoint_path": str(Path(cfg["output_dir"]).resolve() / f"checkpoint-{step}")})
        if payload["candidate_sets"].get(stage) != expected:
            raise ValueError(f"{stage} candidate set/config identity mismatch")
    return payload


def checkpoint_identity(checkpoint, experiment, *, expected=None) -> dict:
    root = Path(checkpoint).resolve()
    files = {name: sha256_file(root / name) for name in CHECKPOINT_FILES}
    trainable = json.loads((root / "trainable_config.json").read_text())
    fingerprint = json.loads((root / "fingerprint.json").read_text())
    metadata = json.loads((root / "checkpoint_metadata.json").read_text())
    digest = object_sha256(trainable)
    if fingerprint.get("sha256") != digest or fingerprint.get("payload") != trainable or metadata.get("fingerprint") != digest:
        raise ValueError("checkpoint fingerprint/trainable identity mismatch")
    step = metadata.get("global_optimizer_step")
    if type(step) is not int or step <= 0:
        raise ValueError("checkpoint lacks a verified optimizer step")
    if metadata.get("schema_version") != 2 or metadata.get("committed_global_sample_count") != step * 32:
        raise ValueError("checkpoint metadata schema/committed cursor mismatch")
    resolved_digest = metadata.get("resolved_config_sha256")
    if not isinstance(resolved_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", resolved_digest):
        raise ValueError("checkpoint lacks resolved training config identity")
    if expected is not None:
        if (experiment != expected["experiment"] or step != expected["step"]
                or str(root) != expected["checkpoint_path"]
                or resolved_digest != expected["train_config"]["resolved_sha256"]):
            raise ValueError("checkpoint experiment/step/training config mismatch")
        training_config = load_config(expected["train_config"]["path"])
        semantic_config = {k: v for k, v in training_config.items()
                           if k not in {"output_dir", "max_optimizer_steps", "checkpoint_steps", "run_type"}}
        if trainable.get("training_config") != semantic_config:
            raise ValueError("checkpoint semantic training configuration mismatch")
        # The full config digest lives in metadata (resume changes output/limits),
        # while semantic recipe fields are independently bound by the fingerprint.
        for field in ("data", "validation", "resolution", "seed", "corruption", "token_mask", "lora", "optimizer",
                      "scheduler_horizon_steps", "warmup_steps", "precision", "component_dtypes", "torch_compile"):
            if trainable.get(field) != training_config.get(field):
                raise ValueError(f"checkpoint saved recipe mismatch: {field}")
        if (trainable.get("world_size") != 8 or trainable.get("per_device_batch") != training_config["batch_per_gpu"]
                or trainable.get("gradient_accumulation") != training_config["gradient_accumulation"]):
            raise ValueError("checkpoint batch/world-size recipe mismatch")
    result = {"path": str(root), "experiment": experiment, "step": step,
              "resolved_config_sha256": resolved_digest, "files": files}
    return result | {"identity_sha256": object_sha256(result)}


def inference_settings(checkpoint) -> dict:
    payload = json.loads((Path(checkpoint) / "trainable_config.json").read_text())
    scope = payload["lora"]["scope"]
    persistent = payload["corruption"]["persistent_conditioning"]
    if scope not in ("both", "reference_only") or type(persistent) is not bool:
        raise ValueError("invalid saved LoRA scope/persistent conditioning")
    return {"lora_scope": scope, "persistent_conditioning": persistent}


def evaluation_identity(*, experiment, checkpoint, config_path, manifest, dataset, preregistration=None) -> dict:
    primary = dataset == "magicbrush" and Path(manifest).name == "magicbrush_official_dev.jsonl"
    result = {"schema": "evaluation-identity-v1", "dataset": dataset,
              "split": "official_dev" if primary else "diagnostic",
              "manifest": manifest_identity(manifest, primary=primary), "eval_config": config_identity(config_path),
              "checkpoint": checkpoint_identity(checkpoint, experiment),
              "inference_settings": inference_settings(checkpoint),
              "preregistration_sha256": preregistration["identity_sha256"] if preregistration else None}
    return result | {"identity_sha256": object_sha256(result)}


def released_model_identity(model_root, release_metadata, *, repo_id, revision) -> dict:
    """Rehash pinned official component bytes; do not invent a finetuned checkpoint."""
    root = Path(model_root).resolve()
    if root.parent.name != "snapshots" or root.name != revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("E0 requires the pinned released model snapshot path")
    metadata = json.loads(Path(release_metadata).read_text())
    if metadata.get("id") != repo_id or metadata.get("sha") != revision:
        raise ValueError("released model metadata repo/revision mismatch")
    component_identity = audit_component_identity(root)
    folders = {Path(component["resolved_path"]).name for component in component_identity["components"].values()}
    expected = {row["rfilename"]: row for row in metadata["siblings"]
                if Path(row["rfilename"]).parts[0] in folders}
    if not expected or len(expected) != len([row for row in metadata["siblings"] if Path(row["rfilename"]).parts[0] in folders]):
        raise ValueError("missing/duplicate released component metadata")
    actual = {str(path.relative_to(root)) for folder in folders for path in (root / folder).rglob("*") if path.is_file()}
    if actual != set(expected):
        raise ValueError("released component file set differs from pinned metadata")
    verified = {}
    for relative, row in sorted(expected.items()):
        path = root / relative
        if not path.resolve().is_relative_to(root) or path.stat().st_size != row["size"]:
            raise ValueError(f"released component size/path mismatch: {relative}")
        if row.get("lfs"):
            digest = sha256_file(path)
            if digest != row["lfs"]["sha256"] or row["lfs"]["size"] != row["size"]:
                raise ValueError(f"released component LFS mismatch: {relative}")
        else:
            raw = path.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            if blob != row["blobId"]:
                raise ValueError(f"released component blob mismatch: {relative}")
            digest = hashlib.sha256(raw).hexdigest()
        verified[relative] = digest
    value = {"repo_id": repo_id, "revision": revision, "components": component_identity,
             "release_metadata": file_identity(release_metadata), "files_sha256": verified}
    return value | {"identity_sha256": object_sha256(value)}


def baseline_evaluation_identity(*, experiment, model_root, release_metadata, config_path,
                                 manifest, dataset, timestep_mode, preregistration) -> dict:
    config = load_config(config_path)
    if experiment not in ("E0-official", "E0-region"):
        raise ValueError("baseline experiment must explicitly name E0-official or E0-region")
    mode = config["baselines"][experiment]["inference_timestep_mode"]
    if timestep_mode != mode:
        raise ValueError("baseline protocol differs from its fixed experiment name")
    baseline = config["released_baseline"]
    if (baseline.get("repo_id") != "WeiChow/EditMGT" or baseline.get("revision") != "0a00f14ea159728fb76ac99ed741f9deca5fee28"
            or baseline.get("persistent_conditioning") is not False or baseline.get("lora_scope") != "both"
            or baseline.get("trainable_adapter") is not False or baseline.get("ranking_eligible") is not False):
        raise ValueError("released baseline recipe changed")
    primary = dataset == "magicbrush" and Path(manifest).name == "magicbrush_official_dev.jsonl"
    if not primary:
        raise ValueError("canonical E0 entry point permits full official MagicBrush DEV only")
    value = {"schema": "evaluation-identity-v1", "role": "released_baseline", "ranking_eligible": False,
             "experiment": experiment, "dataset": dataset, "split": "official_dev",
             "manifest": manifest_identity(manifest, primary=True), "eval_config": config_identity(config_path),
             "baseline_protocol": mode, "inference_settings": {"lora_scope": "both", "persistent_conditioning": False},
             "released_model": released_model_identity(model_root, release_metadata,
                                                        repo_id=baseline["repo_id"], revision=baseline["revision"]),
             "preregistration_sha256": preregistration["identity_sha256"]}
    if value["manifest"] != preregistration["primary_manifest"] or value["eval_config"] != preregistration["eval_config"]:
        raise ValueError("baseline evaluation DEV/config differs from frozen preregistration")
    return value | {"identity_sha256": object_sha256(value)}


def output_identity_name(experiment, checkpoint_identity_value) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", experiment):
        raise ValueError("experiment name must be a safe fixed identifier")
    return f"{experiment}-step{checkpoint_identity_value['step']}-{checkpoint_identity_value['identity_sha256']}"


def _finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def assess_candidate(report, *, expected_identity, sample_keys, selection) -> dict:
    """Malformed data never silently disappear through finite_mean."""
    try:
        if report.get("schema") != "formal-evaluator-v2" or report.get("identity") != expected_identity:
            raise ValueError("evaluation identity mismatch")
        if expected_identity.get("role") == "released_baseline" or expected_identity.get("ranking_eligible") is False:
            raise ValueError("released baseline metrics cannot rank candidates")
        if expected_identity["dataset"] != "magicbrush" or expected_identity["split"] != "official_dev":
            raise ValueError("auxiliary/probe128 metrics cannot rank candidates")
        rows = report["per_generation"]
        expected_pairs = {(key, seed) for key in sample_keys for seed in SEEDS}
        pairs = []
        by_sample = {key: [] for key in sample_keys}
        for row in rows:
            key, seed = row.get("sample_key"), row.get("seed")
            if not isinstance(key, str) or type(seed) is not int or (key, seed) not in expected_pairs:
                raise ValueError("unexpected DEV sample or generation seed")
            if row.get("identity") != expected_identity or row.get("dataset_name") != "magicbrush":
                raise ValueError("generation identity mismatch")
            pairs.append((key, seed))
            metrics = row["metrics"]
            if any(not _finite_number(metrics.get(name)) for name in REQUIRED_METRICS):
                raise ValueError("missing/nonfinite required metric")
            if any(metrics[name] < 0 for name in REQUIRED_METRICS):
                raise ValueError("negative LPIPS/progress metric")
            progress = metrics["d_SO"] / max(metrics["d_ST"], 1e-12)
            noop = metrics["d_ST"] >= selection["tau_edit"] and progress <= selection["tau_noop"]
            if metrics["progress"] != progress or metrics["d_OT"] != metrics["inside_masked_lpips"]:
                raise ValueError("inconsistent distance/progress metrics")
            if type(metrics.get("no_op")) is not bool or metrics["no_op"] != noop:
                raise ValueError("inconsistent no-op metric")
            by_sample[key].append(metrics)
        if Counter(pairs) != Counter({pair: 1 for pair in expected_pairs}):
            raise ValueError("missing/duplicate DEV samples or seeds")
        means = {}
        for name in ("inside_masked_lpips", "outside_masked_lpips", "no_op"):
            sample_means = [math.fsum(float(m[name]) for m in by_sample[key]) / len(SEEDS) for key in sample_keys]
            means[name] = math.fsum(sample_means) / len(sample_keys)
        if not all(math.isfinite(value) for value in means.values()):
            raise ValueError("nonfinite sample-after-seed aggregate")
        eligible = means["outside_masked_lpips"] <= selection["threshold_preserve"]
        return {"eligible": eligible, "reason": None if eligible else "preservation gate failed", "metrics": means}
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        return {"eligible": False, "reason": str(exc), "metrics": None}


def select_candidates(preregistration, reports, stage, *, repo_root=ROOT) -> dict:
    """reports is an explicit list; missing frozen candidates are ineligible."""
    frozen = verify_preregistration(preregistration, repo_root=repo_root)
    if stage not in ("lr", "formal"):
        raise ValueError("only lr/formal selection is permitted; no TEST entry point")
    manifest = frozen["primary_manifest"]["path"]
    sample_keys = [row.get("sample_uid", row.get("sample_key")) for row in
                   (json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip())]
    indexed = {}
    unknown = []
    known = {(c["experiment"], c["step"]) for c in frozen["candidate_sets"][stage]}
    for path in reports:
        try:
            report = json.loads(Path(path).read_text())
            identity = report["identity"]["checkpoint"]
            key = (identity["experiment"], identity["step"])
            if key not in known:
                unknown.append(str(path))
            else:
                indexed.setdefault(key, []).append((str(Path(path).resolve()), report))
        except (AttributeError, OSError, ValueError, KeyError, TypeError):
            unknown.append(str(path))
    if unknown:
        raise ValueError(f"unrecognized/diagnostic/invalid reports must not enter selection: {unknown}")
    results = []
    for candidate in frozen["candidate_sets"][stage]:
        key = (candidate["experiment"], candidate["step"])
        matches = indexed.get(key, [])
        result = {"experiment": key[0], "step": key[1], "checkpoint_path": candidate["checkpoint_path"]}
        try:
            if len(matches) != 1:
                raise ValueError("missing or duplicate candidate report")
            checkpoint_identity(candidate["checkpoint_path"], key[0], expected=candidate)
            saved = json.loads((Path(candidate["checkpoint_path"]) / "trainable_config.json").read_text())
            if saved.get("data_content_hashes", {}).get("data.corpus_ready") != {
                "resolved_path": frozen["corpus_ready"]["path"], "sha256": frozen["corpus_ready"]["sha256"]
            }:
                raise ValueError("checkpoint training corpus identity mismatch")
            report = matches[0][1]
            predictions = report["predictions_manifest"]
            if file_identity(predictions["path"]) != predictions:
                raise ValueError("prediction manifest changed after evaluation")
            generated = [json.loads(line) for line in Path(predictions["path"]).read_text().splitlines() if line.strip()]
            if generated != [{k: v for k, v in row.items() if k != "metrics"} for row in report["per_generation"]]:
                raise ValueError("reported generations differ from frozen prediction manifest")
            identity = evaluation_identity(experiment=key[0], checkpoint=candidate["checkpoint_path"],
                                           config_path=frozen["eval_config"]["path"], manifest=manifest,
                                           dataset="magicbrush", preregistration=frozen)
            result.update(assess_candidate(matches[0][1], expected_identity=identity,
                                           sample_keys=sample_keys, selection=frozen["selection"]))
            result["report"] = matches[0][0]
        except (AttributeError, OSError, ValueError, KeyError, TypeError) as exc:
            result.update(eligible=False, reason=str(exc), metrics=None)
        results.append(result)
    eligible = [result for result in results if result["eligible"]]
    eligible.sort(key=lambda row: (row["metrics"]["inside_masked_lpips"], row["metrics"]["no_op"], row["step"], row["experiment"]))
    return {"schema": "fixed200k-selection-result-v1", "stage": stage,
            "preregistration_sha256": frozen["identity_sha256"], "candidates": results,
            "selected": eligible[0] if eligible else None,
            "status": "SELECTED" if eligible else "NO_ELIGIBLE_CANDIDATE",
            "test_authorized": False}
