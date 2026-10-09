import json

import pytest
import torch
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

from src.dimo.contracts import DIMO_MODEL_ROLES_V11
from src.dimo.initialization import initialize_shared_model_roles
from src.transformer import Transformer2DModel


def tiny():
    return Transformer2DModel(in_channels=6, num_layers=0, num_single_layers=0,
        attention_head_dim=6, num_attention_heads=1, joint_attention_dim=6,
        pooled_projection_dim=6, axes_dims_rope=(2, 2, 2), vocab_size=6,
        codebook_size=5, text_encoder_architecture='CLIP', connector_type='none')


def test_teacher_peft_and_region_exact_round_trip(tmp_path):
    teacher = tiny()
    target = next(name for name, module in teacher.named_modules()
                  if isinstance(module, torch.nn.Linear))
    lora = dict(rank=2, alpha=2, dropout=.05, target_modules=[target])
    teacher.add_adapter(LoraConfig(r=2, lora_alpha=2, lora_dropout=.05, target_modules=[target]))
    with torch.no_grad():
        for name, value in teacher.named_parameters():
            if 'lora_' in name:
                value.copy_(torch.arange(value.numel()).reshape_as(value) * .01)
        teacher.edit_region_embedding.fill_(.137)
    adapter = get_peft_model_state_dict(teacher)
    save_file({k: v.clone() for k, v in adapter.items()}, tmp_path / 'adapter_model.safetensors')
    save_file({'edit_region_embedding': teacher.edit_region_embedding.clone()},
              tmp_path / 'mask_conditioning.safetensors')
    (tmp_path / 'trainable_config.json').write_text(json.dumps({'lora': lora}))
    roles = initialize_shared_model_roles(tiny(), str(tmp_path), model_roles=DIMO_MODEL_ROLES_V11)
    for role in ('teacher', 'student', 'auxiliary'):
        loaded = get_peft_model_state_dict(roles.base_model, adapter_name=role)
        assert loaded.keys() == adapter.keys()
        assert all(torch.equal(loaded[k], adapter[k]) for k in adapter)
        assert torch.equal(roles.region_embeddings[role], teacher.edit_region_embedding)
    save_file(dict(list(adapter.items())[1:]), tmp_path / 'adapter_model.safetensors')
    with pytest.raises(RuntimeError, match='not loaded exactly'):
        initialize_shared_model_roles(tiny(), str(tmp_path), model_roles=DIMO_MODEL_ROLES_V11)
