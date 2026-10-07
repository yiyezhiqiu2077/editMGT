"""Contracts and provenance primitives for the empty-server formal pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable

import yaml


STAGES = (
    "00_environment", "01_model", "02_datasets", "03_eval_models",
    "04_schema_contract", "05_canonical", "06_validation_freeze",
    "07_selection", "08_translation", "09_dedup_backfill",
    "10_exact200k", "11_corpus_audit", "12_ready",
)
HF_COMMIT = re.compile(r"^[0-9a-f]{40}$")
HF_ASSETS = {
    "editmgt", "nllb", "magicbrush", "magicbrush_test", "crispedit",
    "scaleedit", "interedit", "dino", "clip",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git_sha(root: str | Path = ".") -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def load_formal_assets(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "editmgt-formal-assets-v3":
        raise ValueError("FORMAL_ASSET_MANIFEST_INVALID: schema_version")
    assets = config.get("assets", {})
    missing = sorted(HF_ASSETS - set(assets))
    if missing:
        raise ValueError(f"FORMAL_ASSET_MANIFEST_INVALID: missing {missing}")
    for name, spec in assets.items():
        if spec.get("provider") == "huggingface":
            if not spec.get("repo_id") or not HF_COMMIT.fullmatch(str(spec.get("revision", ""))):
                raise ValueError(f"ASSET_NOT_PINNED: {name}")
            if not str(spec.get("identity_policy", "")).startswith("exact_commit"):
                raise ValueError(f"ASSET_NOT_PINNED: {name} identity policy")
            if not spec.get("local_subdir") or not spec.get("expected_structure"):
                raise ValueError(f"FORMAL_ASSET_MANIFEST_INVALID: {name}")
        elif spec.get("identity_policy") != "locked_package_and_backbone":
            raise ValueError(f"ASSET_NOT_PINNED: {name}")
    return config


def asset_path(asset_root: str | Path, spec: dict) -> Path:
    root = Path(asset_root).expanduser().resolve()
    target = (root / spec["local_subdir"]).resolve()
    if root != target and root not in target.parents:
        raise ValueError("asset local_subdir escapes ASSET_ROOT")
    return target


def validate_asset_structure(target: str | Path, spec: dict) -> None:
    target = Path(target)
    missing = [entry for entry in spec.get("expected_structure", []) if not (target / entry).exists()]
    if missing:
        code = "EDITMGT_SNAPSHOT_INVALID" if spec.get("repo_id") == "WeiChow/EditMGT" else "ASSET_SNAPSHOT_INVALID"
        raise RuntimeError(f"{code}: missing {missing} in {target}")
    if "expected_rows" in spec:
        candidates = (target / "dataset_manifest.json", target / "manifest.json")
        manifest = next((path for path in candidates if path.is_file()), None)
        if manifest is None:
            raise RuntimeError(f"ASSET_SNAPSHOT_INVALID: row manifest missing in {target}")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        actual = payload.get("rows", payload.get("num_samples"))
        if actual != spec["expected_rows"]:
            raise RuntimeError(f"ASSET_SNAPSHOT_INVALID: expected {spec['expected_rows']} rows, got {actual}")


def validate_resolved_revision(requested: str, resolved: str) -> None:
    if not HF_COMMIT.fullmatch(str(resolved)) or requested != resolved:
        raise RuntimeError(f"ASSET_REVISION_MISMATCH: requested={requested} resolved={resolved}")


def identity_path(target: str | Path) -> Path:
    return Path(target) / "asset_identity.json"


def validated_identity(target: str | Path, spec: dict) -> dict | None:
    marker = identity_path(target)
    success = Path(target) / "_SUCCESS"
    if not marker.is_file() or not success.is_file():
        return None
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
        validate_resolved_revision(spec["revision"], value["resolved_revision"])
        if value["repo_id"] != spec["repo_id"] or value["requested_revision"] != spec["revision"]:
            return None
        if success.read_text(encoding="utf-8").strip() != spec["revision"]:
            return None
        validate_asset_structure(target, spec)
        return value
    except (KeyError, ValueError, RuntimeError, json.JSONDecodeError):
        return None


def write_json_atomic(path: str | Path, value: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def schema_columns(path: str | Path) -> dict[str, str]:
    import pyarrow.parquet as pq
    schema = pq.ParquetFile(path).schema_arrow
    result = {}
    for field in schema:
        actual = str(field.type)
        if actual.startswith("struct<"):
            actual = "struct"
        result[field.name] = actual
    return result


def match_schema_contract(contract: dict, actual_columns: dict[str, str]) -> None:
    aliases = {"integer": {"int32", "int64"}, "image": {"struct", "binary"}}
    mismatches = []
    for name, expected in contract.get("required_columns", {}).items():
        actual = actual_columns.get(name)
        allowed = aliases.get(str(expected), {str(expected)})
        if actual not in allowed:
            mismatches.append({"column": name, "expected": expected, "actual": actual})
    if mismatches:
        raise RuntimeError("SCHEMA_CONTRACT_MISMATCH: " + json.dumps(mismatches, sort_keys=True))


def load_schema_contract(path: str | Path) -> dict:
    contract = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if contract.get("schema_version") != "formal-schema-contract-v1":
        raise ValueError("SCHEMA_CONTRACT_MISMATCH: unsupported contract version")
    return contract


def verify_contract_file(contract_path: str | Path, data_root: str | Path) -> dict:
    contract = load_schema_contract(contract_path)
    if contract["format"] == "parquet":
        files = sorted(Path(data_root).glob(contract["files_glob"]))
        if not files:
            raise RuntimeError("SCHEMA_CONTRACT_MISMATCH: no parquet files")
        for file in files:
            match_schema_contract(contract, schema_columns(file))
        return {"status": "PASS", "files": len(files), "dataset": contract["dataset_name"]}
    return {"status": "PASS", "dataset": contract["dataset_name"], "deferred_row_validation": True}


def stage_fingerprint(*, stage: str, git: str, config_files: Iterable[str | Path],
                      inputs: Iterable[str | Path] = ()) -> dict:
    if stage not in STAGES:
        raise ValueError(f"unknown formal stage: {stage}")
    configs = {str(Path(p).resolve()): sha256_file(p) for p in config_files}
    input_hashes = {str(Path(p).resolve()): sha256_file(p) for p in inputs if Path(p).is_file()}
    value = {"stage": stage, "git_sha": git, "config_hashes": configs, "input_hashes": input_hashes}
    value["fingerprint"] = stable_json_hash(value)
    return value


def valid_stage_marker(marker: str | Path, expected: dict, outputs: Iterable[str | Path] = ()) -> bool:
    outputs = list(outputs)
    try:
        payload = json.loads(Path(marker).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if payload.get("status") != "SUCCESS" or payload.get("fingerprint") != expected["fingerprint"]:
        return False
    current = {str(Path(path).resolve()): sha256_file(path) for path in outputs if Path(path).is_file()}
    return len(current) == len(outputs) and payload.get("output_hashes", {}) == current


def write_stage_marker(marker: str | Path, expected: dict, resolved_revisions: dict,
                       outputs: Iterable[str | Path] = ()) -> dict:
    outputs = list(outputs)
    missing = [str(path) for path in outputs if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"stage outputs missing: {missing}")
    payload = expected | {"status": "SUCCESS", "resolved_revisions": resolved_revisions,
                          "output_hashes": {str(Path(path).resolve()): sha256_file(path) for path in outputs}}
    write_json_atomic(marker, payload)
    return payload


def invalidate_from(stage_root: str | Path, stage: str) -> list[str]:
    start = STAGES.index(stage); removed = []
    for name in STAGES[start:]:
        marker = Path(stage_root) / name / "_SUCCESS.json"
        if marker.exists():
            marker.unlink(); removed.append(name)
    return removed


def require_selection_ready(path: str | Path) -> dict:
    selection = yaml.safe_load(Path(path).read_text(encoding="utf-8")).get("selection", {})
    required = ("threshold_preserve", "tau_edit", "tau_noop")
    if selection.get("status") != "READY" or any(not isinstance(selection.get(key), (int, float)) for key in required):
        raise RuntimeError("SELECTION_RULE_NOT_PREREGISTERED")
    return selection


def verify_magicbrush_test_counts(metadata: dict, rows: list[dict]) -> None:
    sessions = {str(row.get("session_id")) for row in rows}
    if metadata.get("sessions") != 535 or metadata.get("turns") != 1053 or len(rows) != 1053 or len(sessions) != 535:
        raise RuntimeError("MAGICBRUSH_TEST_IDENTITY_MISMATCH: expected 535 sessions/1053 turns")


def verify_formal_asset_corpus(formal_assets: str | Path, corpus_ready: str | Path,
                               current_git: str) -> tuple[dict, dict]:
    from .fixed_corpus import verify_corpus_ready
    assets_path, corpus_path = map(Path, (formal_assets, corpus_ready))
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    corpus = verify_corpus_ready(corpus_path)
    if assets.get("git_sha") != current_git or corpus.get("git_sha") != current_git:
        raise RuntimeError("FORMAL_PIPELINE_NOT_READY: provenance Git SHA mismatch")
    if corpus.get("formal_assets_sha256") != assets.get("formal_identity_sha256"):
        raise RuntimeError("FORMAL_PIPELINE_NOT_READY: corpus/formal asset identity mismatch")
    if assets.get("corpus_ready", {}).get("sha256") != sha256_file(corpus_path):
        raise RuntimeError("FORMAL_PIPELINE_NOT_READY: formal asset ledger/corpus mismatch")
    return assets, corpus


def verify_formal_ready(formal_assets: str | Path, corpus_ready: str | Path,
                        formal_ready: str | Path, current_git: str) -> dict:
    assets_path, corpus_path, ready_path = map(Path, (formal_assets, corpus_ready, formal_ready))
    assets, corpus = verify_formal_asset_corpus(assets_path, corpus_path, current_git)
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if ready.get("status") != "READY" or ready.get("world_size") != 8:
        raise RuntimeError("FORMAL_PIPELINE_NOT_READY: smoke status/world size")
    smoke_record = ready.get("smoke_verification", {})
    smoke_path = Path(smoke_record.get("path", ""))
    if not smoke_path.is_file() or smoke_record.get("sha256") != sha256_file(smoke_path):
        raise RuntimeError("FORMAL_PIPELINE_NOT_READY: smoke verification hash")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS" or smoke.get("nan_inf_count") != 0 or smoke.get("no_oom") is not True or smoke.get("no_nccl_error") is not True:
        raise RuntimeError("FORMAL_PIPELINE_NOT_READY: smoke verification failed")
    checks = {
        "git_sha": current_git,
        "formal_assets_sha256": sha256_file(assets_path),
        "corpus_ready_sha256": sha256_file(corpus_path),
    }
    for key, value in checks.items():
        if ready.get(key) != value:
            raise RuntimeError(f"FORMAL_PIPELINE_NOT_READY: {key} mismatch")
    return ready


def require_bound_smoke(smoke, corpus_ready, current_git):
    """Reject stale or incomplete smoke evidence, even when its label says PASS."""
    corpus = json.loads(Path(corpus_ready).read_text())
    expected = {'git_sha': current_git, 'world_size': 8, 'data_name': 'fixed_200k',
                'train_manifest_sha256': corpus['train_manifest_sha256'],
                'corpus_ready_sha256': sha256_file(corpus_ready)}
    proofs = ('lora_grad_finite_nonzero', 'region_grad_finite_nonzero',
              'frozen_base_gradients_zero', 'optimizer_scheduler_updates_match',
              'checkpoint_exact_replay', 'optimizer_exact_replay', 'scheduler_exact_replay',
              'sampler_state_exact_replay', 'rank_rng_exact_replay', 'committed_cursor_match',
              'fingerprint_match', 'dataset_identity_bound')
    if (smoke.get('status') != 'PASS' or any(smoke.get(k) != v for k, v in expected.items())
            or any(smoke.get(k) is not True for k in proofs)):
        raise RuntimeError('FORMAL_PIPELINE_NOT_READY: current-code exact smoke evidence required')


def verify_pipeline_mode(*, mode, formal_assets, corpus_ready, current_git,
                         formal_ready=None, pre_smoke=False, selection_config=None,
                         selected_checkpoint=None, require_reattestation=False):
    """One provenance implementation; training never depends on selection."""
    if mode not in ('train', 'select'):
        raise ValueError('unknown verification mode')
    if pre_smoke:
        verify_formal_asset_corpus(formal_assets, corpus_ready, current_git)
        ready = {'world_size': None}
    else:
        if formal_ready is None:
            raise RuntimeError('FORMAL_PIPELINE_NOT_READY: --formal-ready is required')
        ready = verify_formal_ready(formal_assets, corpus_ready, formal_ready, current_git)
    if require_reattestation:
        corpus = json.loads(Path(corpus_ready).read_text())
        attestation = corpus.get('reattestation', {})
        if (attestation.get('status') != 'READY' or attestation.get('git_sha') != current_git
                or attestation.get('manifest_unchanged') is not True
                or attestation.get('validation_unchanged') is not True
                or attestation.get('loader_rows') != 200000):
            raise RuntimeError('FORMAL_PIPELINE_NOT_READY: current-code re-attestation required')
        files = corpus.get('files', {})
        try:
            alpha = json.loads(Path(files['magicbrush_alpha_probe']['path']).read_text())
            loader = json.loads(Path(files['current_loader_compatibility']['path']).read_text())
            before = json.loads(Path(files['reattest_identity_before']['path']).read_text())
            after = json.loads(Path(files['reattest_identity_after']['path']).read_text())
        except (KeyError, FileNotFoundError) as exc:
            raise RuntimeError('FORMAL_PIPELINE_NOT_READY: re-attestation evidence missing') from exc
        if (alpha.get('status') != 'PASS' or alpha.get('mode') not in ('RAW_COMPARE', 'PROVENANCE_REUSE')
                or len(alpha.get('records', [])) != 32
                or len({r['sample_uid'] for r in alpha['records']}) != 32
                or any(r.get('status') != 'PASS' or r.get('binary_disagreement') != 0
                       or r.get('exact_disagreement') != 0 for r in alpha['records'])
                or loader.get('status') != 'PASS' or loader.get('rows') != 200000
                or before != after or before.get('train_sha256') != corpus['train_manifest_sha256']):
            raise RuntimeError('FORMAL_PIPELINE_NOT_READY: re-attestation evidence failed')
        if not pre_smoke:
            smoke = json.loads(Path(ready['smoke_verification']['path']).read_text())
            require_bound_smoke(smoke, corpus_ready, current_git)
    if mode == 'select':
        if selection_config is None:
            raise RuntimeError('SELECTION_RULE_NOT_PREREGISTERED: --selection-config required')
        require_selection_ready(selection_config)
        if selected_checkpoint is None:
            raise RuntimeError('SELECTED_CHECKPOINT_IDENTITY_REQUIRED')
        selected = json.loads(Path(selected_checkpoint).read_text())
        if (selected.get('status') != 'READY' or selected.get('git_sha') != current_git
                or selected.get('formal_assets_sha256') != sha256_file(formal_assets)):
            raise RuntimeError('SELECTED_CHECKPOINT_IDENTITY_MISMATCH')
        files = selected.get('files', {})
        if not {'adapter_model.safetensors', 'mask_conditioning.safetensors'} <= files.keys():
            raise RuntimeError('SELECTED_CHECKPOINT_CANDIDATE_INCOMPLETE')
        if selected.get('checkpoint_identity_sha256') != stable_json_hash(files):
            raise RuntimeError('SELECTED_CHECKPOINT_IDENTITY_MISMATCH')
        for name, expected in files.items():
            if Path(name).name != name or sha256_file(Path(selected['checkpoint_path']) / name) != expected:
                raise RuntimeError('SELECTED_CHECKPOINT_CANDIDATE_HASH_MISMATCH')
    return ready
