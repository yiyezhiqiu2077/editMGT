"""One frozen base with role-specific adapters and region embeddings."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import json
from pathlib import Path

import torch
from torch import nn


ROLES = ("teacher", "student", "auxiliary")


def _is_adapter_parameter(name: str, adapter_name: str) -> bool:
    wrapped = f".{name}."
    marker = f".{adapter_name}."
    return marker in wrapped and ("lora_" in name or "adapter" in name)


class DiMOModelRoles(nn.Module):
    """Manage three logical roles while allocating the transformer base once.

    The base model must already contain one named adapter per role and expose
    ``set_adapter(name)``. Student/auxiliary adapter tensors are copied from the
    teacher adapter during construction.
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        adapter_names: Mapping[str, str] | None = None,
        initialize_from_teacher: bool = True,
    ) -> None:
        super().__init__()
        if not hasattr(base_model, "edit_region_embedding"):
            raise TypeError("base model must expose edit_region_embedding")
        if not callable(getattr(base_model, "set_adapter", None)):
            raise TypeError("shared-base role backend requires named adapter switching")
        self.base_model = base_model
        self.adapter_names = dict(adapter_names or {role: role for role in ROLES})
        if set(self.adapter_names) != set(ROLES):
            raise ValueError("adapter_names must define teacher, student, and auxiliary")
        teacher_region = base_model.edit_region_embedding.detach().clone()
        self.region_embeddings = nn.ParameterDict(
            {role: nn.Parameter(teacher_region.clone()) for role in ROLES}
        )
        self.base_model.requires_grad_(False)
        if initialize_from_teacher:
            self._copy_teacher_adapters()
        for role in ("student", "auxiliary"):
            parameters = list(self._adapter_parameters(role))
            if not parameters:
                raise RuntimeError(f"no parameters found for {role} adapter")
            for parameter in parameters:
                parameter.requires_grad_(True)
        self.region_embeddings["teacher"].requires_grad_(False)
        self.region_embeddings["student"].requires_grad_(True)
        self.region_embeddings["auxiliary"].requires_grad_(True)

    def _adapter_parameters(self, role: str) -> Iterator[nn.Parameter]:
        adapter = self.adapter_names[role]
        for name, parameter in self.base_model.named_parameters():
            if _is_adapter_parameter(name, adapter):
                yield parameter

    def _copy_teacher_adapters(self) -> None:
        named = dict(self.base_model.named_parameters())
        teacher_name = self.adapter_names["teacher"]
        teacher_items = {
            name: parameter for name, parameter in named.items()
            if _is_adapter_parameter(name, teacher_name)
        }
        if not teacher_items:
            raise RuntimeError("no teacher adapter parameters found")
        with torch.no_grad():
            for role in ("student", "auxiliary"):
                target_name = self.adapter_names[role]
                copied = 0
                for source_name, source in teacher_items.items():
                    target_key = source_name.replace(f".{teacher_name}.", f".{target_name}.")
                    target = named.get(target_key)
                    if target is not None:
                        if target.shape != source.shape:
                            raise RuntimeError("role adapter shape mismatch")
                        target.copy_(source)
                        copied += 1
                if copied != len(teacher_items):
                    raise RuntimeError(f"incomplete teacher -> {role} adapter initialization")

    def _set_active_trainability(self, role: str) -> None:
        for candidate in ROLES:
            enabled = candidate == role and candidate != "teacher"
            for parameter in self._adapter_parameters(candidate):
                parameter.requires_grad_(enabled)

    @contextmanager
    def _activate(self, role: str, training: bool | None = None):
        if role not in ROLES:
            raise ValueError(f"unknown model role: {role}")
        previous = getattr(self.base_model, "active_adapter", None)
        was_training = self.base_model.training
        self.base_model.set_adapter(self.adapter_names[role])
        self._set_active_trainability(role)
        effective_training = role != "teacher" if training is None else bool(training)
        if role == "teacher":
            effective_training = False
        self.base_model.train(effective_training)
        try:
            yield
        finally:
            self.base_model.train(was_training)
            if isinstance(previous, str):
                self.base_model.set_adapter(previous)
                previous_role = next((key for key, value in self.adapter_names.items() if value == previous), None)
                if previous_role is not None:
                    self._set_active_trainability(previous_role)
            elif isinstance(previous, (list, tuple)) and len(previous) == 1:
                self.base_model.set_adapter(previous[0])
                previous_role = next((key for key, value in self.adapter_names.items() if value == previous[0]), None)
                if previous_role is not None:
                    self._set_active_trainability(previous_role)

    def _forward(self, role: str, *, training: bool | None = None, **kwargs):
        if "edit_region_embedding_override" in kwargs:
            raise ValueError("role manager owns edit_region_embedding_override")
        with self._activate(role, training=training):
            return self.base_model(
                **kwargs,
                edit_region_embedding_override=self.region_embeddings[role],
            )

    def activate_adapter(self, role: str) -> None:
        """Select the adapter used by gradient-checkpoint recomputation."""
        if role not in ROLES:
            raise ValueError(f"unknown model role: {role}")
        self.base_model.set_adapter(self.adapter_names[role])
        self._set_active_trainability(role)
        self.base_model.train(role != "teacher")

    @torch.no_grad()
    def forward_teacher(self, *, training: bool | None = None, **kwargs):
        return self._forward("teacher", training=False, **kwargs)

    def forward_student(self, *, training: bool | None = None, **kwargs):
        return self._forward("student", training=training, **kwargs)

    def forward_aux(self, *, training: bool | None = None, **kwargs):
        return self._forward("auxiliary", training=training, **kwargs)

    def forward_role(self, role: str, *, training: bool | None = None, **kwargs):
        if role == "teacher":
            return self.forward_teacher(training=False, **kwargs)
        if role == "student":
            return self.forward_student(training=training, **kwargs)
        if role == "auxiliary":
            return self.forward_aux(training=training, **kwargs)
        raise ValueError(f"unknown model role: {role}")

    def role_named_parameters(self, role: str) -> list[tuple[str, nn.Parameter]]:
        adapter_name = self.adapter_names[role]
        adapter = [
            (f"adapter.{name.replace(f'.{adapter_name}.', '.role.')}", parameter)
            for name, parameter in self.base_model.named_parameters()
            if _is_adapter_parameter(name, adapter_name)
        ]
        return adapter + [("edit_region_embedding", self.region_embeddings[role])]

    def role_state_dict(self, role: str) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.role_named_parameters(role)
        }

    def load_role_state_dict(self, role: str, state: Mapping[str, torch.Tensor]) -> None:
        current = dict(self.role_named_parameters(role))
        if set(current) != set(state):
            raise RuntimeError(f"{role} state identity mismatch")
        with torch.no_grad():
            for name, parameter in current.items():
                parameter.copy_(state[name].to(parameter.device, parameter.dtype))


def audit_role_optimizers(
    roles: DiMOModelRoles,
    student_optimizer: torch.optim.Optimizer,
    auxiliary_optimizer: torch.optim.Optimizer,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    student_expected = {id(p) for _, p in roles.role_named_parameters("student")}
    auxiliary_expected = {id(p) for _, p in roles.role_named_parameters("auxiliary")}
    student_actual = {id(p) for group in student_optimizer.param_groups for p in group["params"]}
    auxiliary_actual = {id(p) for group in auxiliary_optimizer.param_groups for p in group["params"]}
    if student_actual != student_expected or auxiliary_actual != auxiliary_expected:
        raise RuntimeError("optimizer membership violates DiMO role isolation")
    if (sum(len(group["params"]) for group in student_optimizer.param_groups) != len(student_expected)
            or sum(len(group["params"]) for group in auxiliary_optimizer.param_groups) != len(auxiliary_expected)):
        raise RuntimeError("duplicate optimizer parameter membership")
    if student_actual & auxiliary_actual:
        raise RuntimeError("student and auxiliary optimizers overlap")
    teacher = roles.role_named_parameters("teacher")
    if any(parameter.requires_grad for _, parameter in teacher):
        raise RuntimeError("teacher role must be frozen")
    if getattr(roles, "backend", None) == "full_dense":
        roles.audit_storage()
        if any(not p.requires_grad for role in ("student", "auxiliary") for _, p in roles.role_named_parameters(role)):
            raise RuntimeError("FULL_DENSE_TRAINABLE_COVERAGE_INCOMPLETE")
        report = {"backend": "full_dense", "teacher_trainable": 0,
            "student_trainable": sum(p.numel() for _, p in roles.role_named_parameters("student")),
            "auxiliary_trainable": sum(p.numel() for _, p in roles.role_named_parameters("auxiliary")),
            "optimizer_overlap": 0, "base_trainable": 0}
        if output_path is not None:
            Path(output_path).write_text(json.dumps(report, indent=2) + "\n")
        return report
    base_trainable = [
        name for name, parameter in roles.base_model.named_parameters()
        if parameter.requires_grad
        and not _is_adapter_parameter(name, roles.adapter_names["student"])
        and not _is_adapter_parameter(name, roles.adapter_names["auxiliary"])
    ]
    if base_trainable:
        raise RuntimeError(f"base/teacher parameters are unexpectedly trainable: {base_trainable[:5]}")
    report = {
        "backend": "shared_base_adapters",
        "teacher_trainable": 0,
        "student_trainable": sum(p.numel() for _, p in roles.role_named_parameters("student")),
        "auxiliary_trainable": sum(p.numel() for _, p in roles.role_named_parameters("auxiliary")),
        "optimizer_overlap": 0,
        "base_trainable": 0,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
