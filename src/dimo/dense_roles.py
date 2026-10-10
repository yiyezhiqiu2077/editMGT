"""Frozen Dense backbone + zero-delta student/aux adapters; three region vectors."""
from contextlib import contextmanager
import json
from pathlib import Path
import torch
from .roles import DiMOModelRoles
from .dense_teacher import load_dense_teacher_contract
from src.explicit_region.dense_checkpoint import load_dense_weights


class DenseDiMOModelRoles(DiMOModelRoles):
    teacher_backend = "dense"

    def __init__(self, model):
        super().__init__(model, adapter_names={"teacher": "__dense_teacher__", "student": "student", "auxiliary": "auxiliary"},
                         initialize_from_teacher=False)

    def _forward(self, role, *, training=None, **kwargs):
        from contextlib import nullcontext
        device = next(self.base_model.parameters()).device
        context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with context:
            return super()._forward(role, training=training, **kwargs)

    @contextmanager
    def _activate(self, role, training=None):
        if role not in self.adapter_names:
            raise ValueError(role)
        previous = getattr(self.base_model, "active_adapter", "student")
        if isinstance(previous, (list, tuple)):
            previous = previous[0]
        was_training = self.base_model.training
        self.activate_adapter(role)
        self.base_model.train(False if role == "teacher" else (True if training is None else training))
        try:
            yield
        finally:
            self.activate_adapter(previous if previous in ("student", "auxiliary") else "student")
            self.base_model.train(was_training)

    def activate_adapter(self, role):
        if role == "teacher":
            self.base_model.disable_adapters()
            self._set_active_trainability("teacher")
            self.base_model.eval()
        elif role in ("student", "auxiliary"):
            self.base_model.enable_adapters()
            self.base_model.set_adapter(role)
            self._set_active_trainability(role)
            self.base_model.train()
        else:
            raise ValueError(role)


def initialize_dense_model_roles(base_model, teacher_checkpoint, *, lora, expected_base_identity=None):
    from peft import LoraConfig
    contract = load_dense_teacher_contract(teacher_checkpoint, expected_base_identity=expected_base_identity)
    if any("lora_" in n for n, _ in base_model.named_parameters()):
        raise RuntimeError("DENSE_TEACHER_BACKBONE_ALREADY_HAS_ADAPTERS")
    if any(p.dtype != torch.float32 for p in base_model.parameters()):
        raise RuntimeError("DENSE_TEACHER_BACKBONE_MUST_BE_FP32")
    load_dense_weights(base_model, teacher_checkpoint)
    for role in ("student", "auxiliary"):
        base_model.add_adapter(LoraConfig(r=int(lora["rank"]), lora_alpha=int(lora["alpha"]),
            lora_dropout=0.0, target_modules=lora["target_modules"], init_lora_weights=True), adapter_name=role)
    roles = DenseDiMOModelRoles(base_model)
    for name, parameter in base_model.named_parameters():
        if "lora_B" in name and bool(parameter.detach().ne(0).any()):
            raise RuntimeError("DENSE_DIMO_NONZERO_INITIAL_DELTA")
    roles.teacher_contract = contract
    return roles


def initial_logits_parity(roles, kwargs, *, atol=0.0, rtol=0.0):
    was_training = roles.base_model.training
    try:
        with torch.no_grad():
            teacher = roles.forward_teacher(**kwargs)
            student = roles.forward_student(training=False, **kwargs)
            auxiliary = roles.forward_aux(training=False, **kwargs)
        errors = {"student": float((student.float() - teacher.float()).abs().max()),
                  "auxiliary": float((auxiliary.float() - teacher.float()).abs().max())}
        if not torch.allclose(teacher, student, atol=atol, rtol=rtol) or not torch.allclose(teacher, auxiliary, atol=atol, rtol=rtol):
            raise RuntimeError(f"DENSE_DIMO_INITIAL_PARITY_FAILURE: {errors}")
        return {"status": "PASS", "atol": atol, "rtol": rtol, "max_abs_error": errors}
    finally:
        roles.base_model.train(was_training)


def initialize_teacher_roles(base_model, checkpoint, *, model_roles=None, teacher_backend=None):
    from .initialization import initialize_shared_model_roles
    root = Path(checkpoint)
    marker = root / "dimo_teacher_manifest.json"
    detected = json.loads(marker.read_text()).get("teacher_backend", "lora") if marker.exists() else "lora"
    if teacher_backend is not None and teacher_backend != detected:
        raise RuntimeError("DIMO_TEACHER_BACKEND_MISMATCH")
    if detected == "dense":
        return initialize_dense_model_roles(base_model, checkpoint, lora={"rank": 64, "alpha": 64,
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0", "ff.net.2", "proj_mlp", "proj_out"]})
    return initialize_shared_model_roles(base_model, checkpoint, model_roles=model_roles)
