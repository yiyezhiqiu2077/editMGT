"""Versioned joint DiMO recipe; legacy configuration remains fail-closed."""
from .rng import RNG_NAMESPACES

RECIPE = "dimo-v2"


def is_v2(config):
    recipe = config.get("recipe", "dimo-v1")
    if recipe not in ("dimo-v1", RECIPE):
        raise RuntimeError("DIMO_UNKNOWN_RECIPE")
    return recipe == RECIPE


def batches_per_step(config):
    return 2 if is_v2(config) else 1


def consumption_budget(config):
    world = config["distributed"]["world_size"]
    batch = config["batch_per_gpu"] * config["gradient_accumulation"]
    per_step = batches_per_step(config) * world * batch
    rows = config["data"]["expected_rows"]
    if rows % per_step:
        raise RuntimeError("DIMO_EPOCH_CANNOT_PAD_DROP_OR_SPLIT_BATCH_PAIR")
    steps = config["epochs"] * rows // per_step
    return {"budget_basis": "total_data_consumption", "rows_per_epoch": rows,
            "data_epochs": config["epochs"], "samples_per_optimizer_step": per_step,
            "steps_per_epoch": rows // per_step, "optimizer_steps": steps,
            "total_sample_consumptions": config["epochs"] * rows,
            "student_updates": steps, "auxiliary_updates": steps,
            "student_update_sample_consumptions": steps * world * batch,
            "auxiliary_update_sample_consumptions": steps * world * batch}


def validate_joint_recipe(config):
    if not is_v2(config):
        if (config["model_roles"].get("freeze_embed", False)
                or config["model_roles"].get("freeze_mlm_head", False)
                or config["auxiliary"].get("label_smoothing", 0.) != 0.
                or config["optimizer"].get("student_max_grad_norm") is not None):
            raise RuntimeError("DIMO_V1_RECIPE_DRIFT")
        return
    if (config.get("budget_basis") != "total_data_consumption"
            or config["model_roles"].get("freeze_embed") is not True
            or config["model_roles"].get("freeze_mlm_head") is not True
            or config["optimizer"].get("name") != "AdamW"
            or config["optimizer"].get("student_max_grad_norm") != 10.
            or config["auxiliary"].get("label_smoothing") != .1
            or config["auxiliary"]["batch_policy"] != "fresh_batch"
            or config["auxiliary"]["soft_target_weight"] != 0.
            or config["auxiliary"]["updates_per_student"] != 1
            or config["distillation"].get("student_cfg") != 1.
            or config["distillation"].get("auxiliary_train_cfg") != 1.
            or config["distillation"]["teacher_cfg"] != 4.
            or config["distillation"]["auxiliary_cfg"] != 1.):
        raise RuntimeError("DIMO_V2_JOINT_RECIPE_MISMATCH")
    budget = consumption_budget(config)
    if (config["max_optimizer_steps"] != budget["optimizer_steps"]
            or config["optimizer"]["horizon_steps"] != budget["optimizer_steps"]):
        raise RuntimeError("DIMO_V2_DATA_CONSUMPTION_BUDGET_MISMATCH")
    diagnostics = config.get("full_update_diagnostic_steps", [])
    if (not isinstance(diagnostics, list) or len(set(diagnostics)) != len(diagnostics)
            or any(type(s) is not int or not 1 <= s <= budget["optimizer_steps"] for s in diagnostics)):
        raise RuntimeError("DIMO_V2_INVALID_FULL_UPDATE_DIAGNOSTICS")


def role_parameter_contract(roles):
    return {role: [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype)}
                   for n, p in roles.role_named_parameters(role) if p.requires_grad]
            for role in ("student", "auxiliary")}


def recipe_identity(config, roles):
    return {"recipe": RECIPE, "budget": consumption_budget(config),
            "freeze_policy": {"embed": True, "mlm_layer": True},
            "optimizer_parameters": role_parameter_contract(roles),
            "ema_parameter_names": [n for n, p in roles.role_named_parameters("student") if p.requires_grad],
            "rng": {"schema": "dimo-stateless-per-sample-v2", "namespaces": list(RNG_NAMESPACES)},
            "inference": {"r_init": 1., "sigma": .3, "student_cfg": 1., "student_steps": 1},
            "teacher_query_cfg": 4., "auxiliary_label_smoothing": .1,
            "student_max_grad_norm": 10., "batch_policy": "fresh_batch"}
