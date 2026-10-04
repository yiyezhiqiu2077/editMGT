"""Deterministic selection and attestation primitives for D200K-v2."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

import yaml

from .canonical import sample_uid_for, validate_record
from .contracts import sha256_file
from .language import contains_han, translation_cache_key, translation_qa_flags


class MissingTranslation(RuntimeError):
    """A selected slot is pending, not rejected or available for backfill."""


def cached_translator(cache_path: str | Path, config: dict) -> Callable:
    """Strict, revision-aware cache lookup shared by train and validation."""
    cache = {}
    path = Path(cache_path)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row["cache_key"]
                if key in cache and cache[key] != row:
                    raise ValueError(f"translation cache inconsistency for {key}")
                cache[key] = row

    def translate(text: str, _row: dict) -> tuple[str, str]:
        key = translation_cache_key(
            text, config["backend"], config["immutable_revision"],
            config["src_lang"], config["tgt_lang"], config["decoding"],
        )
        entry = cache.get(key)
        if entry is None:
            raise MissingTranslation(text)
        if entry["source_text"] != text:
            raise ValueError(f"translation cache source mismatch for {key}")
        return entry["translated_text"], key

    return translate


def validate_frozen_language(row: dict) -> None:
    """Enforce the frozen English/translation contract (not language guessing)."""
    original, english = row["instruction_original"], row["instruction_en"]
    if not isinstance(english, str) or not english.strip() or contains_han(english):
        raise ValueError("invalid frozen instruction_en")
    if contains_han(original):
        if (row["translation_status"] != "translated" or
                row["language_original"] != "zho_Hans" or
                not row["translation_cache_key"] or translation_qa_flags(original, english)):
            raise ValueError("invalid frozen translation or QA failure")
    elif (english != original or row["translation_status"] != "passthrough_en" or
          row["language_original"] != "en" or row["translation_cache_key"] is not None):
        raise ValueError("invalid frozen English passthrough")


DATASET_PRIORITY = ("magicbrush", "crispedit", "scaleedit", "interedit")
INTEREDIT_TYPES = ("add", "remove", "local", "texture")
BUCKETS = ("small", "medium", "large")
VALIDATION_NAMES = (
    "magicbrush_official_dev", "magicbrush_probe128", "crispedit_aux128",
    "scaleedit_aux128", "interedit_aux128",
)
READY_REQUIRED_FILES = frozenset({
    "train_200k", "train_meta", "candidate_selection", "reserve_order", "final_selection",
    "translation_rejections", "backfill_history", "selection_report", "duplicate_report",
    "translation_report", "translation_cache", "audit_report", "audit_failures",
    "translation_manual_audit", "translation_manual_audit_csv", "selection_config",
    "validation_meta", "validation_translation_rejections",
    *(f"validation_{name}" for name in VALIDATION_NAMES),
    *(f"montage_{name}_100" for name in DATASET_PRIORITY), "montage_final_200k_montage",
})
INTEGRITY_CHECK_NAMES = ("assets", "decode", "hash", "geometry", "mask", "english")


def _digest(*values) -> str:
    return hashlib.sha256("\0".join(map(str, values)).encode("utf-8")).hexdigest()


def selection_hash(row: dict, seed: int = 42) -> str:
    return _digest(seed, row["dataset_name"], row["dataset_revision"], row["sample_uid"])


def deterministic_order(rows: Iterable[dict], seed: int = 42) -> list[dict]:
    return sorted(rows, key=lambda row: (selection_hash(row, seed), row["sample_uid"]))


def collapse_exact_sample_duplicates(rows: Iterable[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["source_sha256"], row["target_sha256"], row["region_sha256"],
            " ".join(row["instruction_original"].strip().split()),
        )
        groups[key].append(row)
    kept, removed = [], []
    for members in groups.values():
        ordered = deterministic_order(members, seed)
        kept.append(ordered[0]); removed.extend(ordered[1:])
    return deterministic_order(kept, seed), removed


def freeze_magicbrush_probe(rows: Iterable[dict], count: int = 128, seed: int = 42) -> list[dict]:
    keyed = sorted(
        rows,
        key=lambda row: (_digest("magicbrush-periodic-probe", f"seed={seed}", row["sample_uid"]), row["sample_uid"]),
    )
    if len(keyed) < count:
        raise ValueError(f"MagicBrush official dev has only {len(keyed)} rows; probe requires {count}")
    return keyed[:count]


def freeze_aux_groups(rows: Iterable[dict], dataset_name: str, count: int = 128, seed: int = 42) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    ordered_groups = sorted(
        groups,
        key=lambda group: (_digest(dataset_name, "aux-validation", f"seed={seed}", group), group),
    )
    chosen = []
    for group in ordered_groups:
        chosen.append(min(groups[group], key=lambda row: (selection_hash(row, seed), row["sample_uid"])))
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise ValueError(f"{dataset_name} has only {len(chosen)} eligible groups; aux requires {count}")
    return chosen


def _interedit_type(row: dict) -> str:
    original = str(row.get("edit_type_original", "")).strip().lower()
    canonical = str(row.get("edit_type_canonical", "")).strip().lower()
    if original in INTEREDIT_TYPES:
        return original
    if canonical == "local_attribute":
        return "local"
    if canonical in INTEREDIT_TYPES:
        return canonical
    raise ValueError(f"unsupported Inter-Edit type for stratification: {original}/{canonical}")


def _source_round_robin(rows: list[dict], seed: int) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    sources = sorted(groups, key=lambda group: (_digest(seed, "source-order", group), group))
    queues = {
        group: deque(sorted(groups[group], key=lambda row: (selection_hash(row, seed), row["sample_uid"])))
        for group in sources
    }
    result = []
    while any(queues.values()):
        for group in sources:
            if queues[group]:
                result.append(queues[group].popleft())
    return result


def interedit_strata(rows: Iterable[dict], seed: int = 42) -> dict[tuple[str, str], list[dict]]:
    by_type = defaultdict(list)
    for row in rows:
        by_type[_interedit_type(row)].append(row)
    result = {}
    for edit_type in INTEREDIT_TYPES:
        ordered = sorted(
            by_type[edit_type], key=lambda row: (float(row["region_fraction"]), selection_hash(row, seed), row["sample_uid"])
        )
        third = len(ordered) // 3
        buckets = {
            "small": ordered[:third], "medium": ordered[third : 2 * third],
            "large": ordered[2 * third :],
        }
        for bucket, members in buckets.items():
            result[(edit_type, bucket)] = _source_round_robin(members, seed)
    return result


def _split_quota(total: int, names: tuple[str, ...]) -> dict[str, int]:
    base, remainder = divmod(total, len(names))
    return {name: base + (index < remainder) for index, name in enumerate(names)}


def select_interedit(rows: Iterable[dict], quota: int, seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    """Return selected, deterministic reserve, and redistribution events."""
    strata = interedit_strata(rows, seed)
    type_quotas = _split_quota(quota, INTEREDIT_TYPES)
    selected, events = [], []
    remaining = {key: deque(value) for key, value in strata.items()}
    selected_uids = set()
    fallback = {
        "small": ("medium", "large"), "medium": ("small", "large"),
        "large": ("medium", "small"),
    }
    type_deficits = {}
    for edit_type in INTEREDIT_TYPES:
        bucket_quotas = _split_quota(type_quotas[edit_type], BUCKETS)
        type_selected = 0
        for bucket in BUCKETS:
            need = bucket_quotas[bucket]
            direct = min(need, len(remaining[(edit_type, bucket)]))
            for _ in range(direct):
                row = remaining[(edit_type, bucket)].popleft() | {"selection_stratum": f"{edit_type}/{bucket}"}; selected.append(row)
                selected_uids.add(row["sample_uid"]); type_selected += 1
            deficit = need - direct
            for alternate in fallback[bucket]:
                while deficit and remaining[(edit_type, alternate)]:
                    row = remaining[(edit_type, alternate)].popleft() | {"selection_stratum": f"{edit_type}/{alternate}"}; selected.append(row)
                    selected_uids.add(row["sample_uid"]); type_selected += 1; deficit -= 1
                    events.append({"reason": "region_bucket_shortage", "from": [edit_type, bucket],
                                   "to": [edit_type, alternate], "sample_uid": row["sample_uid"]})
        type_deficits[edit_type] = type_quotas[edit_type] - type_selected
    # Redistribute remaining edit-type deficits in fixed cyclic type order.
    for deficient_type in INTEREDIT_TYPES:
        deficit = type_deficits[deficient_type]
        start = INTEREDIT_TYPES.index(deficient_type)
        candidates = INTEREDIT_TYPES[start + 1 :] + INTEREDIT_TYPES[: start + 1]
        while deficit:
            progressed = False
            for donor_type in candidates:
                for bucket in BUCKETS:
                    if remaining[(donor_type, bucket)]:
                        row = remaining[(donor_type, bucket)].popleft() | {"selection_stratum": f"{donor_type}/{bucket}"}; selected.append(row)
                        selected_uids.add(row["sample_uid"]); deficit -= 1; progressed = True
                        events.append({"reason": "edit_type_shortage", "from": deficient_type,
                                       "to": donor_type, "sample_uid": row["sample_uid"]})
                        break
                if deficit == 0:
                    break
            if not progressed:
                raise RuntimeError("INSUFFICIENT_INTEREDIT_ELIGIBLE_DATA")
    reserve = []
    for edit_type in INTEREDIT_TYPES:
        for bucket in BUCKETS:
            for row in remaining[(edit_type, bucket)]:
                reserve.append(row | {"selection_stratum": f"{edit_type}/{bucket}"})
    return selected, reserve, events


def apply_translation(
    row: dict, translate: Callable[[str, dict], tuple[str, str | None]],
) -> tuple[dict | None, list[str]]:
    result = dict(row)
    original = result["instruction_original"]
    if contains_han(original):
        translated, cache_key = translate(original, result)
        flags = translation_qa_flags(original, translated)
        if flags:
            return None, flags
        result.update(
            instruction_en=translated, language_original="zho_Hans",
            translation_status="translated", translation_cache_key=cache_key,
        )
    else:
        flags = translation_qa_flags(original, original)
        if "empty_output" in flags:
            return None, flags
        result.update(
            instruction_en=original, language_original="en",
            translation_status="passthrough_en", translation_cache_key=None,
        )
    # sample_uid intentionally binds original instruction, so translation does not alter identity.
    if result["sample_uid"] != sample_uid_for(result):
        raise ValueError("translation unexpectedly changed stable sample identity")
    return result, []


def canonical_freeze_order(rows: Iterable[dict], seed: int = 42) -> list[dict]:
    ordered = deterministic_order(rows, seed)
    return [dict(row, manifest_index=index, train_row_id=row["sample_uid"]) for index, row in enumerate(ordered)]


def assert_frozen_corpus(rows: list[dict], total: int = 200000) -> None:
    if len(rows) != total:
        raise ValueError(f"final corpus must contain exactly {total} rows, got {len(rows)}")
    if len({row["sample_uid"] for row in rows}) != total:
        raise ValueError("final corpus contains duplicate sample_uid")
    if [row["manifest_index"] for row in rows] != list(range(total)):
        raise ValueError("manifest_index is not contiguous 0..N-1")
    for row in rows:
        validate_record(row, require_frozen_index=True)
        validate_frozen_language(row)
        if row.get("train_row_id") != row["sample_uid"]:
            raise ValueError("train_row_id does not match sample_uid")


def _jsonl(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_ready_payload(payload: dict) -> None:
    _require(payload.get("status") == "READY", "marker status is not READY")
    _require(payload.get("schema_version") == "fixed200k-ready-v2", "unsupported READY schema")
    _require(type(payload.get("total_rows")) is int and payload["total_rows"] == 200000,
             "READY total_rows must be exactly 200000")
    files = payload.get("files")
    _require(isinstance(files, dict) and READY_REQUIRED_FILES <= files.keys(),
             "required readiness artifacts missing")
    paths = {}
    for name, record in files.items():
        target = Path(record["path"])
        _require(target.is_absolute(), f"artifact path must be absolute: {name}")
        digest = record["sha256"]
        _require(isinstance(digest, str) and len(digest) == 64 and
                 all(c in "0123456789abcdef" for c in digest), f"invalid SHA256 for {name}")
        _require(target.is_file() and sha256_file(target) == digest, f"hash mismatch for {name}")
        paths[name] = target

    def load(name):
        return json.loads(paths[name].read_text(encoding="utf-8"))

    # Presence alone is insufficient for structured reports.
    for name in ("duplicate_report", "translation_report"):
        _require(isinstance(load(name), dict), f"malformed report: {name}")
    val_meta = load("validation_meta")
    _require(val_meta.get("schema") == "fixed200k-validation-v2" and val_meta.get("status") == "FROZEN",
             "validation is not frozen")
    validation, manifest_counts = {}, {"train_200k": 200000}
    val_sources, val_groups = set(), set()
    for name in VALIDATION_NAMES:
        key = f"validation_{name}"
        rows = list(_jsonl(paths[key]))
        _require(len(rows) > 0 and (name == "magicbrush_official_dev" or len(rows) == 128),
                 f"invalid validation count: {name}")
        _require(len({row["sample_uid"] for row in rows}) == len(rows), f"duplicate validation UID: {name}")
        expected_dataset = name.split("_", 1)[0]
        for row in rows:
            validate_record(row); validate_frozen_language(row)
            _require(row["dataset_name"] == expected_dataset, f"wrong validation dataset: {name}")
        _require(val_meta["counts"].get(f"{name}.jsonl") == len(rows) and
                 val_meta["sha256"].get(f"{name}.jsonl") == files[key]["sha256"],
                 f"validation metadata mismatch: {name}")
        if "aux128" in name:
            _require(len({row["group_id"] for row in rows}) == 128, f"duplicate aux group: {name}")
            _require(not val_sources & {row["source_sha256"] for row in rows}, f"cross-validation source leak: {name}")
        if name != "magicbrush_probe128":
            val_sources.update(row["source_sha256"] for row in rows)
            val_groups.update((row["dataset_name"], row["group_id"]) for row in rows)
        validation[name] = rows
        manifest_counts[key] = len(rows)
    dev_by_uid = {row["sample_uid"]: row for row in validation["magicbrush_official_dev"]}
    _require(all(dev_by_uid.get(row["sample_uid"]) == row for row in validation["magicbrush_probe128"]),
             "probe is not an exact DEV subset")

    train_meta, selection = load("train_meta"), load("selection_report")
    _require(train_meta.get("schema_version") == "fixed200k-meta-v2", "unsupported train metadata schema")
    _require(selection.get("schema_version") == "fixed200k-selection-v2", "unsupported selection schema")
    config = yaml.safe_load(paths["selection_config"].read_text(encoding="utf-8"))
    _require(config.get("schema_version") == "fixed200k-v2" and config.get("train_total") == 200000,
             "selection config is not production fixed200k")
    _require(train_meta.get("train_sha256") == files["train_200k"]["sha256"] and
             train_meta.get("selection_config_sha256") == files["selection_config"]["sha256"] and
             train_meta.get("translation_cache_sha256") == files["translation_cache"]["sha256"],
             "train metadata artifact binding mismatch")
    _require(val_meta.get("selection_config_sha256") == files["selection_config"]["sha256"],
             "validation selection config binding mismatch")
    _require(val_meta.get("translation_rejections_sha256") == files["validation_translation_rejections"]["sha256"],
             "validation translation rejection binding mismatch")
    frozen_validation_hashes = train_meta.get("validation_hashes", {})
    for name in VALIDATION_NAMES:
        if name == "magicbrush_probe128":
            continue
        key = f"validation_{name}"
        _require(frozen_validation_hashes.get(str(paths[key])) == files[key]["sha256"],
                 f"train was not selected against current validation: {name}")
    contract = train_meta["translation_contract"]
    _require(contract == val_meta.get("translation_contract"), "train/validation translation contract mismatch")
    _require(contract.get("backend") == "nllb" and contract.get("src_lang") == "zho_Hans" and
             contract.get("tgt_lang") == "eng_Latn" and
             contract.get("decoding") == {"do_sample": False, "num_beams": 1, "max_new_tokens": 128} and
             bool(contract.get("immutable_revision")), "invalid frozen translation contract")
    translate = cached_translator(paths["translation_cache"], contract)

    def check_translation(row):
        if contains_han(row["instruction_original"]):
            english, key = translate(row["instruction_original"], row)
            _require(english == row["instruction_en"] and key == row["translation_cache_key"],
                     "frozen instruction disagrees with translation cache")

    for rows in validation.values():
        for row in rows:
            check_translation(row)
    uids, sources = set(), defaultdict(set)
    counts, revisions = Counter(), defaultdict(set)
    translated_count = 0
    manual = list(_jsonl(paths["translation_manual_audit"]))
    manual_by_uid = {row["sample_uid"]: row for row in manual}
    _require(len(manual_by_uid) == len(manual), "duplicate manual translation sample")
    manual_seen = set()
    total = 0
    for index, row in enumerate(_jsonl(paths["train_200k"])):
        validate_record(row, require_frozen_index=True); validate_frozen_language(row); check_translation(row)
        _require(type(row["manifest_index"]) is int and row["manifest_index"] == index,
                 "train manifest_index is not contiguous")
        _require(row.get("train_row_id") == row["sample_uid"], "train_row_id mismatch")
        _require(row["sample_uid"] not in uids, "duplicate train sample_uid")
        _require(row["source_sha256"] not in val_sources and
                 (row["dataset_name"], row["group_id"]) not in val_groups, "train/validation leakage")
        uids.add(row["sample_uid"]); counts[row["dataset_name"]] += 1
        sources[row["dataset_name"]].add(row["source_sha256"])
        revisions[row["dataset_name"]].add(row["dataset_revision"])
        if contains_han(row["instruction_original"]):
            translated_count += 1
            if row["sample_uid"] in manual_by_uid:
                audited = manual_by_uid[row["sample_uid"]]
                _require(audited.get("dataset") == row["dataset_name"] and
                         audited.get("instruction_original") == row["instruction_original"] and
                         audited.get("instruction_en") == row["instruction_en"] and audited.get("qa_flags") == [],
                         "manual translation sample differs from frozen record")
                manual_seen.add(row["sample_uid"])
        total += 1
    _require(total == 200000, f"train must contain exactly 200000 unique rows, got {total}")
    _require(len(manual) == min(500, translated_count) and manual_seen == set(manual_by_uid),
             "manual translation sample has incomplete or foreign rows")
    with paths["translation_manual_audit_csv"].open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    _require(len(csv_rows) == len(manual) and {row["sample_uid"] for row in csv_rows} == set(manual_by_uid),
             "manual translation CSV differs from JSONL")
    for left_index, left in enumerate(DATASET_PRIORITY):
        for right in DATASET_PRIORITY[left_index + 1:]:
            _require(not sources[left] & sources[right], "cross-dataset training source duplicate")
    for report in (payload, train_meta, selection):
        _require(report.get("total_rows") == total and report.get("dataset_counts") == dict(counts),
                 "reported corpus counts disagree with train")
    _require(payload.get("dataset_revisions") == train_meta.get("dataset_revisions"), "dataset revisions mismatch")
    for name, selected_revisions in revisions.items():
        _require(selected_revisions <= set(train_meta["dataset_revisions"].get(name, [])),
                 f"missing selected revision: {name}")
    _require(payload.get("unique_source_counts") == {name: len(sources[name]) for name in counts},
             "unique source counts mismatch")
    _require(files["final_selection"]["sha256"] == files["train_200k"]["sha256"],
             "final selection differs from frozen train")

    audit = load("audit_report")
    integrity = audit.get("integrity", {})
    _require(audit.get("schema") == "fixed200k-audit-v2" and audit.get("status") == "PASS" and
             integrity.get("status") == "PASS" and integrity.get("mode") == "exhaustive" and
             integrity.get("resolution") == 1024, "successful exhaustive integrity audit required")
    _require(audit.get("total") == 200000 and integrity.get("total_rows") == sum(manifest_counts.values()),
             "audit total coverage mismatch")
    _require(set(integrity.get("manifests", {})) == set(manifest_counts), "audit manifest coverage incomplete")
    for name, count in manifest_counts.items():
        result = integrity["manifests"][name]
        _require(result.get("status") == "PASS" and result.get("mode") == "exhaustive" and
                 result.get("total_rows") == count and result.get("attempted_rows") == count and
                 result.get("sha256") == files[name]["sha256"], f"audit identity/coverage mismatch: {name}")
        _require(result.get("completed_checks") == {check: count for check in INTEGRITY_CHECK_NAMES} and
                 not any(result.get("failure_counts", {"missing": 1}).values()),
                 f"audit checks incomplete or failed: {name}")
    _require(paths["audit_failures"].stat().st_size == 0, "audit failures are nonempty")
    vq = audit.get("vq_containment", {})
    _require(vq.get("status") == "PASS" and vq.get("mode") == "sampled" and vq.get("dtype") == "float32",
             "successful FP32 sampled VQ audit required")
    for name, count in counts.items():
        result = vq.get("datasets", {}).get(name, {})
        expected = min(500, count)
        sampled_uids = result.get("sample_uids", [])
        _require(result.get("status") == "PASS" and result.get("mode") == "sampled" and
                 result.get("samples", 0) >= expected and len(set(sampled_uids)) == result.get("samples") and
                 set(sampled_uids) <= uids, f"incomplete sampled VQ audit: {name}")
    for name in READY_REQUIRED_FILES:
        if name.startswith("montage_"):
            _require(audit.get("montages", {}).get(paths[name].name) == files[name]["sha256"],
                     f"montage audit binding mismatch: {name}")
            from PIL import Image
            with Image.open(paths[name]) as image:
                image.verify()
    _require(bool(payload.get("created_from_git_sha")), "missing source code identity")


def write_corpus_ready(
    path: str | Path, *, files: dict[str, str | Path], metadata: dict,
) -> dict:
    if {"status", "schema_version", "files"} & metadata.keys():
        raise ValueError("metadata cannot override reserved READY keys")
    payload = {
        **metadata, "status": "READY", "schema_version": "fixed200k-ready-v2",
        "files": {name: {"path": str(Path(value).resolve()), "sha256": sha256_file(value)}
                  for name, value in sorted(files.items())},
    }
    _validate_ready_payload(payload)
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return payload


def verify_corpus_ready(path: str | Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        _validate_ready_payload(payload)
    except Exception as exc:
        raise RuntimeError(f"CORPUS_NOT_READY: {exc}") from exc
    return payload


def corpus_counts(rows: Iterable[dict]) -> dict:
    return dict(Counter(row["dataset_name"] for row in rows))
