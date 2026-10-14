"""Load every released EditMGT component from one offline snapshot."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import VQModel
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTextModelWithProjection, CLIPTokenizer

from src.scheduler import Scheduler
from src.transformer import Transformer2DModel

from .contracts import audit_component_identity


@dataclass
class ReleasedComponents:
    transformer: Transformer2DModel
    text_encoder: CLIPTextModelWithProjection
    tokenizer: CLIPTokenizer
    llm_encoder: AutoModelForCausalLM
    llm_tokenizer: AutoTokenizer
    vqvae: VQModel
    scheduler: Scheduler
    identity: dict


def load_released_components(
    model_root: str, *, identity_output=None, torch_dtype: torch.dtype | None = None,
    vq_dtype: torch.dtype | None = None,
    transformer_dtype: torch.dtype | None = None,
) -> ReleasedComponents:
    identity = audit_component_identity(model_root, identity_output)
    common = {"local_files_only": True}
    model_common = common | ({"torch_dtype": torch_dtype} if torch_dtype is not None else {})
    return ReleasedComponents(
        # The released snapshot predates the zero-initialized region vector.
        # Disable low-memory meta loading so that this one new parameter keeps
        # its constructor initialization while every released key is loaded.
        transformer=Transformer2DModel.from_pretrained(
            model_root, subfolder="editmgt", low_cpu_mem_usage=False,
            **(model_common | ({"torch_dtype": transformer_dtype} if transformer_dtype is not None else {}))
        ),
        text_encoder=CLIPTextModelWithProjection.from_pretrained(
            model_root, subfolder="text_encoder", **model_common
        ),
        tokenizer=CLIPTokenizer.from_pretrained(model_root, subfolder="tokenizer", **common),
        llm_encoder=AutoModelForCausalLM.from_pretrained(
            model_root, subfolder="llm_encoder", output_hidden_states=True, **model_common
        ),
        llm_tokenizer=AutoTokenizer.from_pretrained(model_root, subfolder="llm_encoder", **common),
        vqvae=VQModel.from_pretrained(
            model_root, subfolder="vqvae",
            **(common | ({"torch_dtype": vq_dtype} if vq_dtype is not None else {})),
        ),
        scheduler=Scheduler.from_pretrained(model_root, subfolder="scheduler", **common),
        identity=identity,
    )
