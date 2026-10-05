import copy

import torch
from torch import nn

from src.dimo.ema import TrainableEMA
from src.dimo.roles import DiMOModelRoles


class ToyDiMOModel(nn.Module):
    def __init__(self, codebook_size=5):
        super().__init__()
        self.codebook_size = codebook_size
        self.inner_dim = codebook_size
        self.edit_region_embedding = nn.Parameter(torch.linspace(0.0, 0.2, codebook_size))
        self.adapters = nn.ModuleDict({
            role: nn.Linear(codebook_size + 1, codebook_size, bias=False)
            for role in ("teacher", "student", "auxiliary")
        })
        with torch.no_grad():
            teacher = torch.zeros(codebook_size, codebook_size + 1)
            teacher[:, :codebook_size] = torch.eye(codebook_size)
            teacher[:, -1] = torch.linspace(-0.2, 0.2, codebook_size)
            self.adapters["teacher"].weight.copy_(teacher)
        self.active_adapter = "teacher"

    def set_adapter(self, name):
        self.active_adapter = name

    def forward(self, hidden_states, edit_region_mask, edit_region_embedding_override, **kwargs):
        encoded = torch.nn.functional.one_hot(hidden_states, self.codebook_size + 1).float()
        logits = self.adapters[self.active_adapter](encoded)
        return logits + edit_region_mask[..., None] * edit_region_embedding_override


def step_config():
    return {
        "initialization": {"mask_ratio": 0.5},
        "embedding_perturbation": {"enabled": False, "sigma": 0.0},
        "distillation": {
            "mode": "FKL", "jeffreys_beta": 0.5,
            "teacher_temperature": 1.0, "auxiliary_temperature": 1.0,
            "teacher_cfg": 1.0, "auxiliary_cfg": 1.0,
        },
        "pseudo_forward": {
            "teacher_ratio_mode": "cosine", "auxiliary_ratio_mode": "cosine",
            "ratio_min": 0.02, "ratio_max": 0.98,
        },
        "auxiliary": {"updates_per_student": 1, "batch_policy": "same_batch", "soft_target_weight": 0.0},
        "sampling": {"temperature": 1.0, "top_k": 0, "top_p": 0.0},
    }


def make_bundle():
    roles = DiMOModelRoles(ToyDiMOModel())
    with torch.no_grad():
        for _, parameter in roles.role_named_parameters("auxiliary"):
            parameter.add_(torch.linspace(-0.07, 0.07, parameter.numel()).reshape_as(parameter))
    student_parameters = [p for _, p in roles.role_named_parameters("student")]
    auxiliary_parameters = [p for _, p in roles.role_named_parameters("auxiliary")]
    student_optimizer = torch.optim.Adam(student_parameters, lr=1e-2)
    auxiliary_optimizer = torch.optim.Adam(auxiliary_parameters, lr=1e-2)
    student_scheduler = torch.optim.lr_scheduler.LambdaLR(student_optimizer, lambda _: 1.0)
    auxiliary_scheduler = torch.optim.lr_scheduler.LambdaLR(auxiliary_optimizer, lambda _: 1.0)
    ema = TrainableEMA(roles.role_named_parameters("student"), decay=0.9)
    return roles, student_optimizer, auxiliary_optimizer, student_scheduler, auxiliary_scheduler, ema


def batch(uid="sample-a"):
    source = (torch.arange(16).reshape(1, 4, 4) % 5).long()
    region = torch.zeros_like(source, dtype=torch.bool)
    region[:, 1:4, 1:4] = True
    return {
        "source_tokens": source,
        "edit_region_mask": region,
        "sample_uids": [uid],
        "prompt_condition": {"conditional": {}, "unconditional": {}},
        "model_kwargs": {},
    }
