import torch

from src.dimo.ema import TrainableEMA, apply_ema_to_student_role
from tests.dimo.toy import ToyDiMOModel
from src.dimo.roles import DiMOModelRoles


def test_ema_inference_strictly_applies_distinct_student_weights():
    roles = DiMOModelRoles(ToyDiMOModel())
    raw = roles.role_state_dict("student")
    ema = TrainableEMA(roles.role_named_parameters("student"), decay=0.9).state_dict()
    for value in ema["shadow"].values():
        value.add_(0.125)
    apply_ema_to_student_role(roles, ema)
    applied = roles.role_state_dict("student")
    assert any(not torch.equal(raw[name], applied[name]) for name in raw)
    for name, value in applied.items():
        assert torch.equal(value.float(), ema["shadow"][name])


def test_ema_inference_rejects_shape_and_dtype_mismatch():
    roles = DiMOModelRoles(ToyDiMOModel())
    ema = TrainableEMA(roles.role_named_parameters("student")).state_dict()
    name = next(iter(ema["shadow"]))
    ema["shadow"][name] = ema["shadow"][name].double()
    try:
        apply_ema_to_student_role(roles, ema)
    except RuntimeError as exc:
        assert "dtype" in str(exc)
    else:
        raise AssertionError("dtype mismatch was accepted")
