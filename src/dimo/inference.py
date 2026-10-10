"""Full Dense inference owns exactly one Transformer, with no optimizers."""
from contextlib import nullcontext
import json
from pathlib import Path

import torch

from .full_checkpoint import verify, load_weights


class DenseInferenceRole(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = model.eval().requires_grad_(False)

    def model_for(self, role):
        if role != "student":
            raise ValueError("inference has only the selected student/EMA role")
        return self.base_model

    def forward_role(self, role, *, training=None, **kwargs):
        if role != "student" or training:
            raise RuntimeError("INFERENCE_ROLE_CANNOT_TRAIN_OR_QUERY_OTHER_ROLES")
        device = next(self.base_model.parameters()).device
        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with context, torch.no_grad():
            return self.base_model(**kwargs)


def load_dense_inference_role(model, checkpoint, *, weights, teacher_bundle):
    identity, _ = verify(checkpoint, weights=weights)
    if identity.get("teacher_bundle_fingerprint") != teacher_bundle or identity.get("backend") != "ddp":
        raise RuntimeError("FULL_DIMO_INFERENCE_TEACHER_OR_BACKEND_MISMATCH")
    if identity.get("surrogate") != {"implementation": "linear", "version": "roi-vocabulary-sum-v1"}:
        raise RuntimeError("FULL_DIMO_INFERENCE_OBJECTIVE_IDENTITY_MISMATCH")
    load_weights(checkpoint, model, weights=weights, expected_identity=identity)
    return DenseInferenceRole(model), identity
