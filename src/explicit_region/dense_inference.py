"""FP32 Dense Transformer + BF16-autocast inference, with token-lock checks."""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch

from .checkpoint import load_trainable_state
from .contracts import freeze_modules, sha256_file
from .dense_checkpoint import load_dense_weights, write_json
from .modeling import load_released_components
from .fixed_dataset import CanonicalAlignedDataset
from .config import load_config


def check_base_identity(checkpoint, identity):
    payload = json.loads((Path(checkpoint) / "fingerprint.json").read_text())["payload"]
    expected = payload["model_identity"]
    actual = {"repo_id": identity["repo_id"], "resolved_revision": identity["resolved_revision"],
              "components": {n: {"config_sha256": r["config_sha256"]} for n, r in sorted(identity["components"].items())}}
    if actual != expected:
        raise RuntimeError("DENSE_INFERENCE_RELEASED_IDENTITY_MISMATCH")


def pipeline_from_components(components, transformer, device):
    from src.pipeline import Pipeline
    freeze_modules([components.text_encoder, components.llm_encoder, components.vqvae])
    components.text_encoder.to(device, dtype=torch.bfloat16)
    components.llm_encoder.to(device, dtype=torch.bfloat16)
    components.vqvae.to(device, dtype=torch.float32)
    transformer.to(device).eval()
    if any(p.dtype != torch.float32 for p in transformer.parameters()):
        raise RuntimeError("DENSE_INFERENCE_TRANSFORMER_NOT_FP32")
    pipe = Pipeline(transformer=transformer, tokenizer=components.tokenizer,
        text_encoder=components.text_encoder, vqvae=components.vqvae,
        scheduler=copy.deepcopy(components.scheduler), tokenizer_t5=components.llm_tokenizer,
        text_encoder_t5=components.llm_encoder)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_evaluation_pipeline(model_root, *, backend, checkpoint=None, device="cuda"):
    components = load_released_components(model_root, torch_dtype=torch.bfloat16,
                                          transformer_dtype=torch.float32, vq_dtype=torch.float32)
    model = components.transformer
    if backend == "dense":
        if not checkpoint:
            raise RuntimeError("DENSE_CHECKPOINT_REQUIRED")
        check_base_identity(checkpoint, components.identity)
        load_dense_weights(model, checkpoint)
    elif backend == "lora":
        from peft import LoraConfig
        root = Path(checkpoint)
        recipe = json.loads((root / "trainable_config.json").read_text())
        check_base_identity(root, components.identity)
        lora = recipe["lora"]
        model.add_adapter(LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"], target_modules=lora["target_modules"]))
        load_trainable_state(model, root)
    elif backend != "released" or checkpoint is not None:
        raise RuntimeError("INVALID_EVALUATION_BACKEND")
    return pipeline_from_components(components, model, torch.device(device))


@contextmanager
def inference_safety(pipe):
    """Observe, never repair, pipeline logits and scheduler exterior tokens."""
    captured = {}
    def inputs(module, args, kwargs):
        region = kwargs.get("edit_region_mask")
        reference = kwargs.get("reference_image_hidden_states")
        if region is None or reference is None:
            raise RuntimeError("DENSE_INFERENCE_REGION_OR_REFERENCE_MISSING")
        captured["region"], captured["reference"] = region.bool(), reference
    def outputs(module, args, output):
        if not isinstance(output, torch.Tensor) or not bool(torch.isfinite(output).all()):
            raise RuntimeError("DENSE_INFERENCE_NONFINITE_LOGITS")
    pre = pipe.transformer.register_forward_pre_hook(inputs, with_kwargs=True)
    post = pipe.transformer.register_forward_hook(outputs)
    original = pipe.scheduler.step
    decode = pipe.vqvae.decode
    def checked_step(*args, **kwargs):
        result = original(*args, **kwargs)
        tokens = result.prev_sample
        region, reference = captured["region"][:tokens.shape[0]], captured["reference"][:tokens.shape[0]]
        if not torch.equal(tokens[~region], reference[~region]):
            raise RuntimeError("DENSE_INFERENCE_OUTSIDE_TOKEN_LOCK_VIOLATION")
        return result
    def checked_decode(*args, **kwargs):
        result = decode(*args, **kwargs)
        if not bool(torch.isfinite(result.sample).all()):
            raise RuntimeError("DENSE_INFERENCE_NONFINITE_DECODE")
        return result
    pipe.scheduler.step, pipe.vqvae.decode = checked_step, checked_decode
    try:
        yield
    finally:
        pre.remove(); post.remove()
        pipe.scheduler.step, pipe.vqvae.decode = original, decode


def pil(tensor):
    return Image.fromarray((tensor.numpy().transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8))


@torch.inference_mode()
def generate_fixed_evaluation(pipe, *, manifest, canonical_root, dataset_name, config,
                              timestep_mode, output_dir, device="cuda", count=None):
    if "test" in dataset_name.lower():
        raise RuntimeError("DENSE_DEV_PIPELINE_FORBIDS_TEST")
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    if any("test" in str(r.get("split", "")).lower() for r in rows):
        raise RuntimeError("DENSE_DEV_MANIFEST_CONTAINS_TEST")
    if any(r["dataset_name"] != dataset_name for r in rows):
        raise RuntimeError("DENSE_EVAL_DATASET_IDENTITY_MISMATCH")
    ds = CanonicalAlignedDataset(rows, canonical_root, resolution=config["resolution"],
        base_seed=42, minimum_mask_retention=.75, max_random_attempts=8, verify_hashes=True)
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("DENSE_EVAL_OUTPUT_EXISTS")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    device = torch.device(device)
    autocast = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    with inference_safety(pipe):
        for i in range(min(len(ds), count if count is not None else len(ds))):
            item = ds[i]
            source, target = pil(item["source_image"]), pil(item["target_image"])
            mask = Image.fromarray(item["edit_region_mask"].numpy().astype(np.uint8) * 255)
            for seed in config["generation_seeds"]:
                generator = torch.Generator(device=device).manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                with autocast():
                    result = pipe(prompt=item["instruction_en"], reference_image=source, mask_image=mask,
                        height=config["resolution"], width=config["resolution"],
                        num_inference_steps=config["steps"], guidance_scale=config["guidance_scale"],
                        reference_strength=config["reference_strength"], generator=generator,
                        lora_scope="both", inference_timestep_mode=timestep_mode).images[0]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                stem = f"{dataset_name}_{i:06d}_seed{seed}"
                paths = {}
                for name, image in (("source", source), ("target", target), ("mask", mask), ("output", result)):
                    path = out / f"{stem}_{name}.png"
                    image.save(path)
                    paths[name] = str(path.resolve())
                records.append(paths | {"sample_key": item["sample_key"], "sample_uid": rows[i]["sample_uid"],
                    "dataset_name": dataset_name, "edit_type": item["edit_type"], "seed": seed,
                    "runtime_seconds": elapsed, "timestep_mode": timestep_mode,
                    "token_outside_lock": "PASS", "manifest_sha256": sha256_file(manifest)})
    predictions = out / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    return predictions


def periodic_dense_validation(components, model, train_config, output, step, device):
    from scripts.eval.formal_eval import evaluate
    validation = train_config["validation"]
    infer = load_config(validation["inference_config"])
    # Preserve all stochastic state; evaluation never changes training mode,
    # sampler, optimizer or shared training scheduler.
    state = (torch.get_rng_state(), torch.cuda.get_rng_state(device), random.getstate(), np.random.get_state())
    training = model.training
    try:
        pipe = pipeline_from_components(components, model, device)
        manifest = generate_fixed_evaluation(pipe, manifest=validation["manifest"],
            canonical_root=validation["dataset_root"], dataset_name="magicbrush", config=infer,
            timestep_mode=infer["inference_timestep_mode"], output_dir=Path(output) / f"validation-{step}", device=device)
        result = evaluate(manifest, infer, manifest.parent / "metrics.json", str(device))
        return result
    finally:
        model.train(training)
        torch.set_rng_state(state[0]); torch.cuda.set_rng_state(state[1], device)
        random.setstate(state[2]); np.random.set_state(state[3])
