"""Audited lineage clusters and sample-weighted paired cluster bootstrap."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .contracts import sha256_file


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def audited_clusters(rows, contract):
    """A group field name alone is not evidence of original-source identity.

    Require a reviewed semantic contract with an immutable evidence document.
    Join groups sharing source hashes or a target->source multi-turn edge.
    Neither canonical records nor corpus-ready markers are changed.
    """
    if contract.get("schema") != "source-group-identity-v1":
        raise RuntimeError("GROUP_IDENTITY_CONTRACT_MISSING")
    if not rows:
        raise RuntimeError("GROUP_IDENTITY_EMPTY_MANIFEST")
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(i, j):
        parent[find(j)] = find(i)
    group_owner, image_owner, uids = {}, {}, set()
    evidence_checked = set()
    for i, row in enumerate(rows):
        dataset = row.get("dataset_name")
        specification = contract.get("datasets", {}).get(dataset, {})
        if (specification.get("group_semantics") not in ("original_source", "editing_sequence")
                or specification.get("reviewed") is not True):
            raise RuntimeError(f"GROUP_SEMANTICS_NOT_CONFIRMED: {dataset}")
        if dataset not in evidence_checked:
            if sha256_file(specification["evidence_path"]) != specification.get("evidence_sha256"):
                raise RuntimeError("GROUP_SEMANTIC_EVIDENCE_CHANGED")
            evidence_checked.add(dataset)
        group = row.get("group_id")
        uid = row.get("sample_uid")
        if not group or not uid or uid in uids or not row.get("source_sha256") or not row.get("target_sha256"):
            raise RuntimeError("GROUP_IDENTITY_MISSING_OR_CONFLICTING")
        uids.add(uid)
        key = (dataset, str(group))
        if key in group_owner:
            join(i, group_owner[key])
        else:
            group_owner[key] = i
        # Shared image content crosses namespaces; this also catches reused
        # originals and multi-turn chains with inconsistent local group labels.
        for digest in (row["source_sha256"], row["target_sha256"]):
            if digest in image_owner:
                join(i, image_owner[digest])
            else:
                image_owner[digest] = i
    members = {}
    for i, row in enumerate(rows):
        members.setdefault(find(i), []).append(row["sample_uid"])
    cluster_names = {root: min(values) for root, values in members.items()}
    return {row["sample_uid"]: cluster_names[find(i)] for i, row in enumerate(rows)}


def audit_split_isolation(train_path, dev_paths, test_paths, contract):
    splits = {"train": read_rows(train_path), "dev": [], "test": []}
    for name, paths in (("dev", dev_paths), ("test", test_paths)):
        for path in paths:
            splits[name].extend(read_rows(path))
    all_rows = [row for rows in splits.values() for row in rows]
    # A duplicate UID is itself a split-isolation failure.
    clusters = audited_clusters(all_rows, contract)
    cluster_splits = {}
    for name, rows in splits.items():
        for row in rows:
            cluster_splits.setdefault(clusters[row["sample_uid"]], set()).add(name)
    leaks = [key for key, names in cluster_splits.items() if len(names) > 1]
    if leaks:
        raise RuntimeError(f"SOURCE_GROUP_SPLIT_LEAKAGE: {len(leaks)} clusters")
    return {"status": "PASS", "split_samples": {k: len(v) for k, v in splits.items()},
        "independent_groups": len(cluster_splits), "checks": ["group", "source_hash", "target_source_lineage"]}


def cluster_bootstrap(values, groups, *, seed=42, resamples=10000, confidence_level=.95):
    values = np.asarray(values, dtype=np.float64)
    if len(values) != len(groups) or not len(values) or not np.isfinite(values).all():
        raise ValueError("invalid paired cluster bootstrap inputs")
    if len(set(groups)) < 2:
        raise RuntimeError("CLUSTER_CI_REQUIRES_AT_LEAST_TWO_INDEPENDENT_GROUPS")
    if resamples <= 0 or not 0 < confidence_level < 1:
        raise ValueError("invalid bootstrap configuration")
    names = sorted(set(groups))
    counts = np.asarray([groups.count(k) for k in names], dtype=np.int64)
    sums = np.asarray([values[np.asarray(groups) == k].sum() for k in names])
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples)
    for start in range(0, resamples, 256):
        n = min(256, resamples - start)
        picks = rng.integers(len(names), size=(n, len(names)))
        # Ratio of total sample sums/counts, not mean of group means.
        boot[start:start+n] = sums[picks].sum(1) / counts[picks].sum(1)
    alpha = (1 - confidence_level) / 2
    return {"n": len(values), "independent_groups": len(names),
        "group_sizes": dict(zip(names, counts.tolist())), "mean_delta": float(values.mean()),
        "bootstrap_95_ci": np.quantile(boot, [alpha, 1-alpha]).tolist(),
        "confidence_level": confidence_level, "resamples": resamples, "seed": seed,
        "bootstrap_unit": "source_group_cluster", "estimand": "sample_weighted_mean_delta"}
