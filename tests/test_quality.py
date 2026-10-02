from src.explicit_region.quality import QualityGate


CONFIG = dict(baseline_start=1, baseline_end=4, window_size=2, max_loss_ratio=1.5,
              consecutive_bad_windows=2, max_abs_logit=100, max_clip_fraction=1.0,
              max_update_ratio=1.0)


def test_quality_gate_requires_baseline_then_detects_regression():
    gate=QualityGate(CONFIG)
    for step in range(1,5): assert gate.update(step,loss=1,max_abs_logit=2,clip_applied=False,update_ratio=.01,finite_parameters=True)
    assert gate.update(5,loss=3,max_abs_logit=2,clip_applied=False,update_ratio=.01,finite_parameters=True)
    assert not gate.update(6,loss=3,max_abs_logit=2,clip_applied=False,update_ratio=.01,finite_parameters=True)
    assert gate.status == "QUALITY_FAILED"


def test_quality_state_roundtrip_and_finite_gate():
    gate=QualityGate(CONFIG);gate.update(1,loss=1,max_abs_logit=2,clip_applied=False,update_ratio=.01,finite_parameters=True)
    resumed=QualityGate.from_state(CONFIG,gate.state_dict())
    assert resumed.state_dict()==gate.state_dict()
    assert not resumed.update(2,loss=float("nan"),max_abs_logit=2,clip_applied=False,update_ratio=.01,finite_parameters=True)
