"""Evaluate v2 with the frozen native DEV geometry and verified old results."""
import gc
import json
from pathlib import Path

from PIL import Image
import torch

from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.contracts import freeze_modules, sha256_file
from src.explicit_region.dense_checkpoint import write_json
from src.explicit_region.dense_evaluation import comparison_visual, paired_comparison
from src.explicit_region.dense_inference import pil
from src.explicit_region.latency import repeated_latency
from src.explicit_region.metrics import finite_mean
from src.explicit_region.group_statistics import cluster_bootstrap
from .contracts import released_base_model_identity, teacher_bundle_fingerprint
from .dense_teacher import load_dense_teacher_contract
from .dev_protocol import validate_plan, validate_predictions, STUDENT_PROTOCOL, TEACHER_PROTOCOL
from .evaluation import checkpoint_r_init, teacher_formal_requirement
from .inference import load_dense_inference_role
from .one_step import one_step_edit_tokens
from .rng import stable_seed, normal_noise_per_sample


def roi_crop_box(mask):
    """Match the previous DEV128 contextual ROI embedding metric exactly."""
    box = mask.convert("L").getbbox()
    if box is None:
        raise RuntimeError("DIMO_DEV_EMPTY_ROI_CROP")
    x0, y0, x1, y1 = box
    margin = max(8, round(.1 * max(x1 - x0, y1 - y0)))
    return (max(0, x0-margin), max(0, y0-margin), min(mask.width, x1+margin), min(mask.height, y1+margin))


@torch.inference_mode()
def generate_native_student(plan, rows, checkpoint, output, run_identity, weights, device):
    from src.explicit_region.modeling import load_released_components
    from scripts.eval.formal_eval import tensor
    from scripts.train.train_dimo_editing import prepare_batch
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("DIMO_DEV_GENERATION_OUTPUT_EXISTS")
    components = load_released_components(plan["model_root"], torch_dtype=torch.bfloat16,
        transformer_dtype=torch.float32, vq_dtype=torch.float32)
    bundle = teacher_bundle_fingerprint(plan["teacher_checkpoint"],
        base_model_identity=released_base_model_identity(components.identity), formal=teacher_formal_requirement(run_identity))
    roles, identity = load_dense_inference_role(components.transformer, checkpoint, weights=weights, teacher_bundle=bundle)
    if identity != run_identity or identity.get("recipe") != "dimo-v2":
        raise RuntimeError("DIMO_V2_DEV_CHECKPOINT_RUN_IDENTITY_MISMATCH")
    ratio = checkpoint_r_init(identity)
    if ratio != plan["student_protocol"]["r_init"]:
        raise RuntimeError("DIMO_V2_DEV_INITIALIZATION_MISMATCH")
    roles.to(device).eval()
    freeze_modules([components.text_encoder, components.llm_encoder, components.vqvae])
    for model in (components.text_encoder, components.llm_encoder, components.vqvae):
        model.to(device)
    output.mkdir(parents=True, exist_ok=True)
    predictions = []
    modules = {"transformer": (roles.base_model, "forward"), "vq_encode": (components.vqvae, "encode"),
        "vq_decode": (components.vqvae, "decode"), "clip": (components.text_encoder, "forward"),
        "gemma": (components.llm_encoder, "forward")}
    for i, row in enumerate(rows):
        batch = {"source_image": tensor(row["source"]), "edit_region_mask": tensor(row["mask"], mask=True)[:, 0],
                 "instruction_en": [row["instruction_en"]], "sample_uid": [row["sample_uid"]]}
        for seed in STUDENT_PROTOCOL["generation_seeds"]:
            args = (seed, 0, row["sample_uid"], 0)
            def operation():
                prepared = prepare_batch(batch, components, {"resolution": 1024, "token_mask": STUDENT_PROTOCOL["token_mask"]}, device)
                source, region = prepared["source_tokens"], prepared["edit_region_mask"]
                noise = normal_noise_per_sample(torch.empty(source.shape[0], roles.base_model.inner_dim,
                    *source.shape[1:], device=device, dtype=torch.float32), [stable_seed(*args, "dimo-embedding-noise")])
                result = one_step_edit_tokens(roles, source_tokens=source, edit_region_mask=region,
                    prompt_condition=prepared["prompt_condition"], timestep_model_kwargs=prepared["model_kwargs"],
                    mask_token_id=roles.base_model.config.vocab_size-1, codebook_size=roles.base_model.config.codebook_size,
                    r_init=ratio, init_mask_seeds=stable_seed(*args, "dimo-init-mask"),
                    init_token_seeds=stable_seed(*args, "dimo-init-token"), sample_seeds=stable_seed(*args, "dimo-student-sample"),
                    cfg_scale=1., target_embedding_noise=noise, embedding_noise_sigma=.3)
                decoded = components.vqvae.decode(result.tokens, force_not_quantize=True,
                    shape=(*source.shape, components.vqvae.config.latent_channels)).sample.clip(0, 1)
                if not torch.isfinite(decoded).all():
                    raise RuntimeError("DIMO_DEV_NONFINITE_DECODE")
                return pil(decoded[0].float().cpu())
            image, timing = repeated_latency(operation, modules, device, warmup=1 if i == 0 and seed == 0 else 0, repeats=1)
            if any(p["transformer"]["calls"] != 1 for p in timing["phases"]):
                raise RuntimeError("DIMO_DEV_ONE_STEP_FORWARD_COUNT_MISMATCH")
            path = output / f"{i:04d}_seed{seed}.png"
            image.save(path)
            predictions.append(row | {"output": str(path.resolve()), "seed": seed,
                "manifest_sha256": plan["manifest_sha256"], "runtime_seconds": timing["mean_seconds"],
                "latency_protocol": timing, "transformer_forward_count": 1, "token_outside_lock": "PASS",
                "pixel_outside_lock": "NOT_GUARANTEED", "inference_protocol": STUDENT_PROTOCOL | {"role": weights}})
    path = output / "predictions.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in predictions))
    validate_predictions(path, rows, plan["manifest_sha256"], STUDENT_PROTOCOL)
    del roles, components
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return path


@torch.inference_mode()
def evaluate_roi_metrics(predictions, plan, output, device):
    """Recompute all models' metrics with one implementation, including old PNGs."""
    from scripts.eval.formal_eval import evaluate, embedding_backend
    from src.explicit_region.group_statistics import cluster_bootstrap
    configuration = {"generation_seeds": [0, 1], "bootstrap": plan["bootstrap"], "embedding_models": plan["embedding_models"]}
    result = evaluate(predictions, configuration, str(output) + ".base.partial", device)
    backends = embedding_backend(configuration, device)
    if set(backends) != {"clip", "dino"}:
        raise RuntimeError("DIMO_DEV_ROI_EMBEDDING_BACKENDS_REQUIRED")
    def feature(image, name):
        processor, model = backends[name]
        inputs = {k: v.to(device) for k, v in processor(images=image, return_tensors="pt").items()}
        values = model.get_image_features(**inputs) if name == "clip" else model(**inputs).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(values.float(), dim=-1)
    refs, boxes = {}, {}
    for row in result["per_generation"]:
        uid = row["sample_uid"]
        if uid not in boxes:
            with Image.open(row["mask"]) as mask:
                boxes[uid] = roi_crop_box(mask)
            with Image.open(row["target"]) as image:
                refs[uid] = {name: feature(image.convert("RGB").crop(boxes[uid]), name) for name in backends}
        with Image.open(row["output"]) as image:
            for name in backends:
                row["metrics"][f"roi_{name}_to_target"] = float((feature(image.convert("RGB").crop(boxes[uid]), name) * refs[uid][name]).sum())
    for sample in result["per_sample_after_seed_mean"]:
        repeated = [r for r in result["per_generation"] if r["sample_uid"] == sample["sample_uid"]]
        for name in backends:
            key = f"roi_{name}_to_target"
            sample["metrics"][key] = finite_mean(r["metrics"][key] for r in repeated)
    result["roi_crop_rule"] = plan["roi_crop_rule"]
    result["roi_crop_xyxy_by_sample_uid"] = boxes
    for name in backends:
        key = f"roi_{name}_to_target"
        samples = result["per_sample_after_seed_mean"]
        result["aggregate"]["dataset:magicbrush_dev"][key] = cluster_bootstrap(
            [r["metrics"][key] for r in samples], [r["cluster_id"] for r in samples], **{k: v for k, v in plan["bootstrap"].items() if k in ("seed", "resamples", "confidence_level")})
    write_json(output, result)
    return result


def evaluate_native_plan(plan, root, *, steps=None, device="cuda", diagnostic_count=None):
    from scripts.train.train_dense_region import configure_determinism
    configure_determinism()
    device, root = torch.device(device), Path(root)
    rows = validate_plan(plan)
    if diagnostic_count is not None:
        raise RuntimeError("DIMO_DEV128_FIXED_PROTOCOL_FORBIDS_PARTIAL_COMPARISON")
    identity = json.loads((root / "experiment_manifest.json").read_text())
    teacher_formal_requirement(identity)
    if (identity.get("recipe") != "dimo-v2" or identity["teacher_bundle_fingerprint"]["bundle_sha256"] != plan["teacher_bundle_sha256"]
            or load_dense_teacher_contract(plan["teacher_checkpoint"]).checkpoint_hash != plan["teacher_bundle_sha256"]):
        raise RuntimeError("DIMO_DEV128_RUN_TEACHER_IDENTITY_MISMATCH")
    selected = steps or plan["steps"]
    if any(step not in plan["steps"] for step in selected):
        raise RuntimeError("DIMO_DEV128_UNREGISTERED_STEP")
    output = root / "evaluation/dev128"
    output.mkdir(parents=True, exist_ok=True)
    ledger = {"evaluation_source_sha256": {name: sha256_file(Path(__file__).parent / name)
                 for name in ("native_evaluation.py", "dev_protocol.py", "evaluation.py")}, "run_identity": identity, "plan_sha256": recipe_fingerprint(plan), "manifest_sha256": plan["manifest_sha256"]}
    ledger_path = output / "evaluation_identity.json"
    if ledger_path.exists() and json.loads(ledger_path.read_text()) != ledger:
        raise RuntimeError("DIMO_DEV128_EVALUATION_IDENTITY_CHANGED")
    write_json(ledger_path, ledger)
    paths, metrics = {}, {}
    for name, entry in plan["historical"].items():
        paths[name] = Path(entry["predictions"])
        metric_path = output / (name.replace(" ", "_") + "_metrics.json")
        metrics[name] = (json.loads(metric_path.read_text()) if metric_path.exists() else
                         evaluate_roi_metrics(paths[name], plan, metric_path, device))
    for step in selected:
        selected_paths = dict(paths)
        for weights, label in (("student", "v2 Raw"), ("ema", "v2 EMA")):
            folder = output / f"step-{step}" / weights
            predictions = folder / "predictions.jsonl"
            if not predictions.exists():
                generate_native_student(plan, rows, root / f"inference/checkpoint-{step}", folder, identity, weights, device)
            validate_predictions(predictions, rows, plan["manifest_sha256"], STUDENT_PROTOCOL)
            metric_path = folder / "metrics.json"
            result = (json.loads(metric_path.read_text()) if metric_path.exists() else
                      evaluate_roi_metrics(predictions, plan, metric_path, device))
            comparisons = {name: paired_comparison(result, previous, strict_groups=True,
                **{k: v for k, v in plan["bootstrap"].items() if k in ("seed", "resamples", "confidence_level")})
                for name, previous in metrics.items()}
            for name, previous in metrics.items():
                a = {r["sample_uid"]: r for r in result["per_sample_after_seed_mean"]}
                b = {r["sample_uid"]: r for r in previous["per_sample_after_seed_mean"]}
                for feature_name in ("roi_clip_to_target", "roi_dino_to_target"):
                    comparisons[name]["magicbrush_dev"][feature_name] = cluster_bootstrap(
                        [a[k]["metrics"][feature_name] - b[k]["metrics"][feature_name] for k in sorted(a)],
                        [a[k]["cluster_id"] for k in sorted(a)], **{k: v for k, v in plan["bootstrap"].items() if k in ("seed", "resamples", "confidence_level")})
            write_json(folder / "paired_to_previous.json", comparisons)
            selected_paths[label] = predictions
        comparison_visual(selected_paths, output / f"step-{step}/comparison.jpg", count=8)
