import pytest


def test_teacher_inherits_dropout_while_student_and_auxiliary_are_zero():
    pytest.importorskip("peft")
    from src.dimo.initialization import build_role_lora_configs

    configs = build_role_lora_configs({
        "rank": 2, "alpha": 4, "dropout": 0.17, "target_modules": ["to_q"],
    }, {
        "lora_dropout_policy": "zero_for_dimo",
        "teacher_lora_dropout": "inherit",
        "student_lora_dropout": 0.0,
        "auxiliary_lora_dropout": 0.0,
    })
    assert configs["teacher"].lora_dropout == 0.17
    assert configs["student"].lora_dropout == 0.0
    assert configs["auxiliary"].lora_dropout == 0.0


def test_nonzero_student_dropout_is_rejected():
    pytest.importorskip("peft")
    from src.dimo.initialization import build_role_lora_configs

    with pytest.raises(ValueError, match="must be zero"):
        build_role_lora_configs({
            "rank": 2, "alpha": 4, "dropout": 0.1, "target_modules": ["to_q"],
        }, {"student_lora_dropout": 0.1})

