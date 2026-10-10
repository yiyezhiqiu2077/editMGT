import json
import pytest

from src.explicit_region.dense_evaluation import select_checkpoint
from tests.test_dense_pipeline import make_checkpoint


def candidate(root, step, *, improvement=.006, upper=-.001, degradation=.01):
    common = {"bootstrap_unit": "source_group_cluster", "estimand": "sample_weighted_mean_delta"}
    return {"backend": "dense", "checkpoint": str(root), "step": step,
        "metrics": {"magicbrush": {"inside_masked_lpips": .3}},
        "paired_to_E0-region": {"magicbrush": {
            "inside_masked_lpips": common | {"mean_delta": -improvement, "bootstrap_95_ci": [-.011, upper]},
            "outside_masked_lpips": common | {"mean_delta": degradation, "bootstrap_95_ci": [-.01, .02]}}}}


def registration():
    return {"protocol": "full-dense-selection-v2", "plan": {"selection": {
        "minimum_inside_lpips_improvement": .005, "maximum_outside_lpips_degradation": .01,
        "require_inside_ci_improvement": True}}}


def test_point_threshold_and_positive_ci_are_distinct_and_tie_earlier(tmp_path):
    _, first = make_checkpoint(tmp_path / "first", step=3125)
    _, later = make_checkpoint(tmp_path / "later", step=6250)
    items = [candidate(later, 6250), candidate(first, 3125, improvement=.005)]
    result = select_checkpoint({"candidate_steps": [3125, 6250]}, registration(), items, tmp_path / "selected.json")
    # CI supports >0 improvement but does not establish >.005. The frozen
    # protocol imposes .005 on the mean, not on the entire confidence interval.
    assert result["status"] == "READY" and result["selected_step"] == 3125


@pytest.mark.parametrize("changes", [{"improvement": .0049}, {"upper": 0.}, {"degradation": .0101}])
def test_no_eligible_means_no_teacher_export(tmp_path, changes):
    result = select_checkpoint({"candidate_steps": [3125]}, registration(),
        [candidate(tmp_path / "must-not-load", 3125, **changes)], tmp_path / "selected.json")
    assert result == {"status": "PENDING", "reason": "NO_ELIGIBLE_CHECKPOINT", "action": "STOP_TEACHER_EXPORT"}


def test_incomplete_formal_candidates_or_noncluster_stats_stop(tmp_path):
    with pytest.raises(RuntimeError, match="INCOMPLETE_CANDIDATES"):
        select_checkpoint({"candidate_steps": [3125, 6250]}, registration(),
                          [candidate(tmp_path, 3125)], tmp_path / "selected.json")
    item = candidate(tmp_path, 3125)
    item["paired_to_E0-region"]["magicbrush"]["inside_masked_lpips"]["bootstrap_unit"] = "sample"
    with pytest.raises(RuntimeError, match="UNAUDITED_STATISTICS"):
        select_checkpoint({"candidate_steps": [3125]}, registration(), [item], tmp_path / "selected.json")
