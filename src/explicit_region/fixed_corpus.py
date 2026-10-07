"""Deterministic selection and attestation primitives for D200K-v2."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

from .canonical import sample_uid_for, validate_record
from .contracts import sha256_file
from .language import contains_han, translation_qa_flags


DATASET_PRIORITY = ("magicbrush", "crispedit", "scaleedit", "interedit")
INTEREDIT_TYPES = ("add", "remove", "local", "texture")
BUCKETS = ("small", "medium", "large")


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


def write_corpus_ready(
    path: str | Path, *, files: dict[str, str | Path], metadata: dict,
) -> dict:
    hashes = {name: sha256_file(value) for name, value in sorted(files.items())}
    payload = {
        "status": "READY", "schema_version": metadata.get("schema_version", "fixed200k-ready-v2"),
        "files": {name: {"path": str(Path(value).resolve()), "sha256": hashes[name]}
                  for name, value in sorted(files.items())},
        **metadata,
    }
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_corpus_ready(path: str | Path) -> dict:
    marker = Path(path)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("status") != "READY":
        raise RuntimeError("CORPUS_NOT_READY: marker status is not READY")
    if payload.get("schema_version") == "fixed200k-ready-v3":
        required = ("git_sha", "formal_assets_sha256", "selection_config_sha256",
                    "train_manifest_sha256", "total_rows", "dataset_counts",
                    "unique_source_counts", "schema_contract", "asset_integrity",
                    "translation", "dedup", "geometry", "train_dev_leakage",
                    "exact_200k", "vq_audit")
        missing = [key for key in required if key not in payload]
        failed = [key for key in required[7:] if payload.get(key) != "PASS"]
        if missing or failed or payload.get("total_rows") != 200000:
            raise RuntimeError(f"CORPUS_NOT_READY: machine contract missing={missing} failed={failed}")
    for name, record in payload.get("files", {}).items():
        target = Path(record["path"])
        if not target.is_file() or sha256_file(target) != record["sha256"]:
            raise RuntimeError(f"CORPUS_NOT_READY: hash mismatch for {name}")
    if payload.get("schema_version") == "fixed200k-ready-v3":
        train = payload.get("files", {}).get("train_200k", {})
        if payload.get("train_manifest_sha256") != train.get("sha256"):
            raise RuntimeError("CORPUS_NOT_READY: train manifest identity mismatch")
    return payload


def verify_mini_corpus_ready(path: str | Path, *, expected_rows: int) -> dict:
    marker = Path(path)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "READY"
        or payload.get("schema_version") != "mini-corpus-ready-v1"
        or payload.get("dev_only") is not True
        or payload.get("total_rows") != expected_rows
    ):
        raise RuntimeError("MINI_CORPUS_NOT_READY: invalid dev corpus attestation")
    for name, record in payload.get("files", {}).items():
        target = Path(record["path"])
        if not target.is_file() or sha256_file(target) != record["sha256"]:
            raise RuntimeError(f"MINI_CORPUS_NOT_READY: hash mismatch for {name}")
    return payload


def corpus_counts(rows: Iterable[dict]) -> dict:
    return dict(Counter(row["dataset_name"] for row in rows))
