import ast
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.eval import formal_eval
from src.explicit_region.selection import inference_settings, sha256_file

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("scope,persistent", [("both", False), ("both", True), ("reference_only", True)])
def test_saved_scope_and_conditioning_are_restored(tmp_path, scope, persistent):
    (tmp_path / "trainable_config.json").write_text(json.dumps({
        "lora": {"scope": scope}, "corruption": {"persistent_conditioning": persistent}}))
    assert inference_settings(tmp_path) == {"lora_scope": scope, "persistent_conditioning": persistent}


def test_pipeline_conditioning_flag_controls_both_transformer_branches():
    # Exercise the actual call expressions without importing heavyweight model backends.
    tree = ast.parse((ROOT / "src/pipeline.py").read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "__call__")
    defaults = dict(zip([arg.arg for arg in method.args.args][-len(method.args.defaults):], method.args.defaults))
    assert ast.literal_eval(defaults["persistent_conditioning"]) is True
    expressions = [keyword.value for node in ast.walk(method) if isinstance(node, ast.Call)
                   for keyword in node.keywords if keyword.arg == "edit_region_conditioning_active"]
    assert len(expressions) == 2
    for expression in expressions:
        compiled = compile(ast.Expression(expression), "<pipeline-conditioning>", "eval")
        for persistent in (True, False):
            for mask in (None, object()):
                assert eval(compiled, {"persistent_conditioning": persistent, "model_edit_region_mask": mask}) == (persistent and mask is not None)


@pytest.mark.parametrize("persistent", [True, False])
def test_periodic_validation_propagates_saved_conditioning(persistent):
    tree = ast.parse((ROOT / "src/explicit_region/validation.py").read_text())
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "pipe")
    keyword = next(item for item in call.keywords if item.arg == "persistent_conditioning")
    compiled = compile(ast.Expression(keyword.value), "<periodic-validation-conditioning>", "eval")
    assert eval(compiled, {"train_config": {"corruption": {"persistent_conditioning": persistent}}}) is persistent


class FakeLPIPS:
    def __init__(self, device): pass
    def __call__(self, a, b, mask):
        valid = mask.flatten(1).any(1)
        numerator = ((a - b).abs() * mask).flatten(1).sum(1)
        return numerator / (mask.flatten(1).sum(1).clamp_min(1) * 3), valid
    def full(self, a, b): return (a - b).abs().flatten(1).mean(1)


def test_evaluator_binds_images_and_identity_before_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(formal_eval, "MaskedLPIPS", FakeLPIPS)
    paths = {}
    for name, value in (("source", 0), ("target", 100), ("output", 20)):
        path = tmp_path / f"{name}.png"
        Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)
        paths[name] = str(path)
    mask = np.zeros((16, 16), dtype=np.uint8); mask[4:12, 4:12] = 255
    Image.fromarray(mask).save(tmp_path / "mask.png")
    paths["mask"] = str(tmp_path / "mask.png")
    dataset_manifest = tmp_path / "magicbrush_official_dev.jsonl"
    dataset_manifest.write_text(json.dumps({"sample_uid": "a", "dataset_name": "magicbrush"}) + '\n')
    identity = {"experiment": "E4", "checkpoint": "full-content-digest", "manifest": {"path": str(dataset_manifest)}}
    records = [paths | {"identity": identity, "sample_key": "a", "dataset_name": "magicbrush", "seed": seed,
                       "image_sha256": {key: sha256_file(path) for key, path in paths.items()}} for seed in range(4)]
    manifest = tmp_path / "predictions.jsonl"
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in records))
    config = {"selection": {"tau_edit": .05, "tau_noop": .1}, "bootstrap": {"seed": 42, "resamples": 10}}
    report = formal_eval.evaluate(manifest, config, tmp_path / "metrics.json", "cpu", expected_identity=identity)
    assert report["schema"] == "formal-evaluator-v2"
    assert report["identity"] == identity
    assert report["per_generation"][0]["metrics"]["progress"] == pytest.approx(.2)
    assert not report["per_generation"][0]["metrics"]["no_op"]
    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text(''.join(json.dumps(row) + '\n' for row in records[:-1]))
    with pytest.raises(ValueError, match="missing/duplicate"):
        formal_eval.evaluate(incomplete, config, tmp_path / "incomplete-metrics.json", "cpu", expected_identity=identity)
    with pytest.raises(ValueError, match="identities"):
        formal_eval.evaluate(manifest, config, tmp_path / "bad.json", "cpu", expected_identity={})
    Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8)).save(paths["output"])
    with pytest.raises(ValueError, match="image identity"):
        formal_eval.evaluate(manifest, config, tmp_path / "bad-image.json", "cpu", expected_identity=identity)
