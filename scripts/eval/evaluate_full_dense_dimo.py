#!/usr/bin/env python3
"""Local-only paired Teacher/Raw/EMA DEV evaluation at fixed optimizer steps."""
import argparse
import gc
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.explicit_region.config import load_config
from src.explicit_region.contracts import freeze_modules, sha256_file
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.dense_checkpoint import write_json
from src.explicit_region.dense_inference import load_evaluation_pipeline, generate_fixed_evaluation
from src.explicit_region.dense_evaluation import assert_matching_predictions, comparison_visual, paired_comparison
from src.explicit_region.modeling import load_released_components
from src.dimo.contracts import released_base_model_identity, teacher_bundle_fingerprint
from src.dimo.inference import load_dense_inference_role
from src.dimo.evaluation import generate_one_step_evaluation
from scripts.eval.formal_eval import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", default="configs/eval/full_dense_dimo_plan.yaml")
    p.add_argument("--run-root", required=True)
    p.add_argument("--steps", nargs="+", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--diagnostic-count", type=int)
    a = p.parse_args()
    plan = load_config(a.plan)
    config = load_config(plan["inference_config"])
    if a.diagnostic_count:
        config["generation_seeds"] = [0]
        config["latency"] = {"warmup": 0, "repeats": 1}
    contract = json.loads(Path(plan["group_identity_contract"]).read_text())
    root = Path(a.run_root)
    output = root / "evaluation" / ("diagnostic" if a.diagnostic_count else "full_dev")
    output.mkdir(parents=True, exist_ok=True)
    run_identity = json.loads((root / "experiment_manifest.json").read_text())
    evaluation_identity = {"run_identity": run_identity, "plan_sha256": recipe_fingerprint(plan),
        "inference_sha256": recipe_fingerprint(config), "diagnostic_count": a.diagnostic_count,
        "manifest_sha256": {d["name"]: sha256_file(d["manifest"]) for d in plan["datasets"]},
        "group_contract_sha256": sha256_file(plan["group_identity_contract"])}
    ledger = output / "evaluation_identity.json"
    if ledger.exists() and json.loads(ledger.read_text()) != evaluation_identity:
        raise RuntimeError("DIMO_EVALUATION_IDENTITY_CHANGED")
    write_json(ledger, evaluation_identity)
    from scripts.train.train_dense_region import configure_determinism
    configure_determinism()
    for dataset in plan["datasets"]:
        name = dataset["name"]
        folder = output / "teacher" / name
        predictions = folder / "predictions.jsonl"
        if not predictions.exists():
            pipeline = load_evaluation_pipeline(plan["model_root"], backend="dense",
                checkpoint=plan["teacher_checkpoint"], device=a.device)
            predictions = generate_fixed_evaluation(pipeline, manifest=dataset["manifest"], canonical_root=dataset["root"],
                dataset_name=name, config=config, timestep_mode="roi_relative", output_dir=folder, device=a.device,
                count=a.diagnostic_count, group_contract=contract)
            del pipeline; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        teacher_metrics = folder / "metrics.json"
        if not teacher_metrics.exists(): evaluate(predictions, config, teacher_metrics, a.device)
    steps = a.steps or plan["steps"]
    if any(step not in plan["steps"] for step in steps):
        raise RuntimeError("DIMO_EVALUATION_UNREGISTERED_STEP")
    for step in steps:
        checkpoint = root / "inference" / f"checkpoint-{step}"
        if not checkpoint.is_dir():
            raise RuntimeError(f"DIMO_EVALUATION_CHECKPOINT_NOT_AVAILABLE: {step}")
        for weights in ("student", "ema"):
            components = load_released_components(plan["model_root"], torch_dtype=torch.bfloat16,
                transformer_dtype=torch.float32, vq_dtype=torch.float32)
            bundle = teacher_bundle_fingerprint(plan["teacher_checkpoint"],
                base_model_identity=released_base_model_identity(components.identity), formal=True)
            roles, identity = load_dense_inference_role(components.transformer, checkpoint, weights=weights, teacher_bundle=bundle)
            if identity != run_identity:
                raise RuntimeError("DIMO_EVALUATION_RUN_CHECKPOINT_MISMATCH")
            roles.to(a.device).eval()
            freeze_modules([components.text_encoder, components.llm_encoder, components.vqvae])
            for model in (components.text_encoder, components.llm_encoder, components.vqvae): model.to(a.device)
            for dataset in plan["datasets"]:
                name = dataset["name"]
                folder = output / f"step-{step}" / weights / name
                path = folder / "predictions.jsonl"
                if not path.exists():
                    path = generate_one_step_evaluation(components, roles, manifest=dataset["manifest"],
                        canonical_root=dataset["root"], dataset_name=name, config=config,
                        output_dir=folder, group_contract=contract, device=a.device, count=a.diagnostic_count)
                metric_path = folder / "metrics.json"
                if not metric_path.exists(): evaluate(path, config, metric_path, a.device)
                teacher_path = output / "teacher" / name / "predictions.jsonl"
                assert_matching_predictions([teacher_path, path], expected_seeds=config["generation_seeds"])
                teacher_result = json.loads((teacher_path.parent / "metrics.json").read_text())
                result = json.loads(metric_path.read_text())
                # A four-case diagnostic cannot establish formal cluster CIs.
                if not a.diagnostic_count:
                    paired = paired_comparison(result, teacher_result, strict_groups=True)
                    write_json(folder / "paired_to_teacher.json", paired)
            del roles, components; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        for dataset in plan["datasets"]:
            name = dataset["name"]
            paths = {"Teacher": output / "teacher" / name / "predictions.jsonl",
                     "Student": output / f"step-{step}" / "student" / name / "predictions.jsonl",
                     "EMA": output / f"step-{step}" / "ema" / name / "predictions.jsonl"}
            comparison_visual(paths, output / f"step-{step}" / f"{name}_montage.png")


if __name__ == "__main__":
    main()
