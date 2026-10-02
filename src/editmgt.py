# Copyright 2025 EditMGT Team. All rights reserved.
import torch
import json
from pathlib import Path
from peft import LoraConfig
from diffusers import VQModel
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, AutoModelForCausalLM, AutoTokenizer
from src.scheduler import Scheduler
from src.pipeline import Pipeline
from src.transformer import Transformer2DModel

def init_edit_mgt(
    device,
    enable_bf16=False,
    base_model_path='WeiChow/EditMGT',
    local_files_only=False,
    trainable_state_path=None,
):
    text_encoder_clip = CLIPTextModelWithProjection.from_pretrained(
        base_model_path, subfolder="text_encoder", local_files_only=local_files_only
    )
    tokenizer_clip = CLIPTokenizer.from_pretrained(
        base_model_path, subfolder="tokenizer", local_files_only=local_files_only
    )
    model_gemma = AutoModelForCausalLM.from_pretrained(base_model_path, subfolder="llm_encoder", output_hidden_states=True, local_files_only=local_files_only)
    tokenizer_gemma = AutoTokenizer.from_pretrained(base_model_path, subfolder="llm_encoder", local_files_only=local_files_only)
    text_encoder = [text_encoder_clip,model_gemma]
    tokenizer = [tokenizer_clip,tokenizer_gemma]
    vq_model = VQModel.from_pretrained(
        base_model_path, subfolder="vqvae", local_files_only=local_files_only,
    ) 
    scheduler = Scheduler.from_pretrained(
        base_model_path, subfolder="scheduler", local_files_only=local_files_only,
    )
    if enable_bf16:
        text_encoder = [t.to(torch.bfloat16) for t in text_encoder]
        # VQ remains FP32: the explicit-region audit found only ~89.6% token
        # agreement between BF16 and FP32 assignments on 32 fixed examples.
        vq_model = vq_model.to(torch.float32)
        # model = Transformer2DModel.from_pretrained(base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16)
        model = Transformer2DModel.from_pretrained(
            base_model_path, subfolder="editmgt", torch_dtype=torch.bfloat16,
            local_files_only=local_files_only, low_cpu_mem_usage=False,
        )
    else:
        # model = Transformer2DModel.from_pretrained(base_model_path, subfolder="transformer")
        model = Transformer2DModel.from_pretrained(
            base_model_path, subfolder="editmgt", local_files_only=local_files_only,
            low_cpu_mem_usage=False,
        )
    
    if trainable_state_path is not None:
        from src.explicit_region.checkpoint import load_trainable_state

        state_root = Path(trainable_state_path)
        payload = json.loads((state_root / "trainable_config.json").read_text(encoding="utf-8"))
        lora = payload["lora"]
        model.add_adapter(
            LoraConfig(
                r=lora["rank"], lora_alpha=lora["alpha"],
                lora_dropout=lora["dropout"], target_modules=lora["target_modules"],
            )
        )
        load_trainable_state(model, state_root)

    pipe = Pipeline(
        transformer=model,
        tokenizer=tokenizer[0],
        text_encoder=text_encoder[0],
        vqvae=vq_model,
        scheduler=scheduler,
        text_encoder_t5=text_encoder[1],
        tokenizer_t5=tokenizer[1]
    )
    pipe = pipe.to(device)
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    pipe.text_encoder_t5.eval()
    pipe.vqvae.eval()

    return pipe
