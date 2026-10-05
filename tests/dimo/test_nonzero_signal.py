import json

from src.dimo.diagnostics import run_nonzero_signal_diagnostic
from src.dimo.roles import DiMOModelRoles
from tests.dimo.toy import ToyDiMOModel, batch


def test_auxiliary_epsilon_perturbation_proves_nonzero_student_signal(tmp_path):
    roles = DiMOModelRoles(ToyDiMOModel())
    inputs = batch()
    output = tmp_path / "dimo_nonzero_signal_test.json"
    before = roles.role_state_dict("auxiliary")
    report = run_nonzero_signal_diagnostic(
        roles,
        target_tokens=inputs["source_tokens"],
        reference_tokens=inputs["source_tokens"],
        edit_region_mask=inputs["edit_region_mask"],
        prompt_condition=inputs["prompt_condition"],
        model_kwargs=inputs["model_kwargs"],
        output_path=output,
    )
    assert report["passed"] is True
    assert report["probability_difference_norm"] > 0
    assert report["dimo_gradient_norm"] > 0
    assert report["student_grad_norm"] > 0
    assert report["teacher_grad_norm"] == report["auxiliary_grad_norm"] == 0
    assert json.loads(output.read_text())["epsilon"] == 1e-3
    after = roles.role_state_dict("auxiliary")
    assert all(before[name].equal(after[name]) for name in before)
