import json
import numpy as np
import pytest

from src.explicit_region.group_statistics import cluster_bootstrap, audited_clusters, audit_split_isolation
from src.explicit_region.dense_evaluation import paired_comparison, select_checkpoint
from src.explicit_region.contracts import sha256_file


def test_bootstrap_is_sample_weighted_and_reproducible():
    values = np.array([0., 0., 0., 10.])
    result = cluster_bootstrap(values, ["A", "A", "A", "B"], resamples=100)
    picks = np.random.default_rng(42).integers(2, size=(100, 2))
    expected = np.array([0., 10.])[picks].sum(1) / np.array([3, 1])[picks].sum(1)
    assert result["mean_delta"] == 2.5
    assert result["group_sizes"] == {"A": 3, "B": 1}
    assert result["bootstrap_95_ci"] == np.quantile(expected, [.025, .975]).tolist()
    assert result == cluster_bootstrap(values, ["A", "A", "A", "B"], resamples=100)


def prediction(offset=0):
    return {"per_sample_after_seed_mean": [{"dataset_name": "magicbrush", "sample_key": str(i),
        "sample_uid": str(i), "group_id": f"group{i//2}", "cluster_id": f"cluster{i//2}",
        "generation_seeds": [0, 1, 2, 3], "source_sha256": "source", "target_sha256": "target",
        "region_sha256": "mask", "manifest_sha256": "manifest",
        "metrics": {"inside_masked_lpips": .5+offset, "outside_masked_lpips": .1}}
        for i in range(4)]}


def test_paired_cluster_identity_and_mean():
    result = paired_comparison(prediction(-.01), prediction(), strict_groups=True, resamples=100)
    metric = result["magicbrush"]["inside_masked_lpips"]
    assert metric["mean_delta"] == pytest.approx(-.01)
    assert metric["n"] == 4 and metric["independent_groups"] == 2


@pytest.mark.parametrize("field", ["sample_uid", "cluster_id", "group_id", "generation_seeds", "source_sha256"])
def test_paired_identity_conflicts_fail_closed(field):
    candidate = prediction(-.01)
    candidate["per_sample_after_seed_mean"][0][field] = None
    with pytest.raises(RuntimeError):
        paired_comparison(candidate, prediction(), strict_groups=True)


def test_group_semantic_evidence_and_multiturn_leakage(tmp_path):
    evidence = tmp_path / "evidence.md"; evidence.write_text("Reviewed original-source group mapping")
    contract = {"schema": "source-group-identity-v1", "datasets": {"magicbrush": {
        "reviewed": True, "group_semantics": "editing_sequence", "evidence_path": str(evidence),
        "evidence_sha256": sha256_file(evidence)}}}
    train = {"dataset_name": "magicbrush", "sample_uid": "train", "group_id": "a",
             "source_sha256": "original", "target_sha256": "intermediate"}
    dev = {"dataset_name": "magicbrush", "sample_uid": "dev", "group_id": "b",
           "source_sha256": "intermediate", "target_sha256": "final"}
    clusters = audited_clusters([train, dev], contract)
    assert clusters["train"] == clusters["dev"]
    train_path = tmp_path / "train.jsonl"; train_path.write_text(json.dumps(train)+"\n")
    dev_path = tmp_path / "dev.jsonl"; dev_path.write_text(json.dumps(dev)+"\n")
    with pytest.raises(RuntimeError, match="LEAKAGE"):
        audit_split_isolation(train_path, [dev_path], [], contract)
    contract["datasets"]["magicbrush"]["reviewed"] = False
    with pytest.raises(RuntimeError, match="NOT_CONFIRMED"):
        audited_clusters([train], contract)
