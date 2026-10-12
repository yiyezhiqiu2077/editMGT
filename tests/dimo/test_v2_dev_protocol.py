import copy
import json
from PIL import Image
import pytest

from src.dimo.dev_protocol import validate_native_manifest, validate_predictions, STUDENT_PROTOCOL
from src.dimo.evaluation import checkpoint_r_init, teacher_formal_requirement
from src.dimo.native_evaluation import roi_crop_box
from src.explicit_region.contracts import sha256_file


def native_fixture(tmp_path):
    for name in ("source", "target", "output"):
        Image.new("RGB", (1024, 1024), "gray").save(tmp_path / f"{name}.png")
    Image.new("L", (1024, 1024), 255).save(tmp_path / "mask.png")
    row = {"sample_uid": "uid", "sample_key": "key", "dataset_name": "magicbrush_dev",
           "group_id": "group", "cluster_id": "group", "instruction_en": "change color"}
    for name in ("source", "target", "mask"):
        row[name] = str(tmp_path / f"{name}.png")
        row["region_sha256" if name == "mask" else name + "_sha256"] = sha256_file(row[name])
    path = tmp_path / "dev.jsonl"; path.write_text(json.dumps(row)+"\n")
    predictions = [row | {"seed": seed, "manifest_sha256": sha256_file(path),
        "output": str(tmp_path / "output.png"), "inference_protocol": STUDENT_PROTOCOL} for seed in [0, 1]]
    output = tmp_path / "predictions.jsonl"
    output.write_text("".join(json.dumps(r)+"\n" for r in predictions))
    return row, path, output, predictions


def test_native_pixels_manifest_and_prediction_protocol_are_content_bound(tmp_path):
    row, manifest, predictions, values = native_fixture(tmp_path)
    rows = validate_native_manifest(manifest, sha256_file(manifest), count=1)
    hashes = validate_predictions(predictions, rows, sha256_file(manifest), STUDENT_PROTOCOL)
    validate_predictions(predictions, rows, sha256_file(manifest), STUDENT_PROTOCOL, expected_output_hashes=hashes)
    Image.new("RGB", (1024, 1024), "white").save(tmp_path / "output.png")
    with pytest.raises(RuntimeError, match="OUTPUT_CHANGED"):
        validate_predictions(predictions, rows, sha256_file(manifest), STUDENT_PROTOCOL, expected_output_hashes=hashes)
    Image.new("RGB", (1024, 1024), "white").save(tmp_path / "source.png")
    with pytest.raises(RuntimeError, match="HASH_MISMATCH"):
        validate_native_manifest(manifest, sha256_file(manifest), count=1)


@pytest.mark.parametrize("field,value", [("r_init", .5), ("cfg", 4.), ("sigma", 0.),
                                       ("generation_seeds", [0]), ("steps", 2)])
def test_old_results_cannot_be_compared_under_different_protocol(tmp_path, field, value):
    row, manifest, predictions, values = native_fixture(tmp_path)
    values = copy.deepcopy(values)
    values[0]["inference_protocol"][field] = value
    predictions.write_text("".join(json.dumps(r)+"\n" for r in values))
    with pytest.raises(RuntimeError, match="PROTOCOL"):
        validate_predictions(predictions, [row], sha256_file(manifest), STUDENT_PROTOCOL)


def test_checkpoint_controls_rinit_with_no_silent_reference_default():
    assert checkpoint_r_init({"initialization": {"mask_ratio": 1.}}) == 1.
    assert checkpoint_r_init({"initialization": {"mask_ratio": .5}}) == .5
    with pytest.raises(RuntimeError, match="UNRECORDED"):
        checkpoint_r_init({})
    with pytest.raises(RuntimeError, match="OVERRIDE"):
        checkpoint_r_init({"initialization": {"mask_ratio": 1.}}, .5)


def test_nonformal_teacher_requires_exact_registered_experiment_identity():
    identity = {"formal": False, "experiment_mode": "mixed20k_last_sft_dimo",
                "selection_scope": "mixed20k_last_sft_dimo", "teacher_bundle_fingerprint": {"formal_teacher": False}}
    assert teacher_formal_requirement(identity) is False
    identity["selection_scope"] = "unknown"
    with pytest.raises(RuntimeError, match="IDENTITY"):
        teacher_formal_requirement(identity)


def test_roi_context_crop_matches_historical_rounding_rule():
    mask = Image.new("L", (1024, 1024))
    from PIL import ImageDraw
    ImageDraw.Draw(mask).rectangle((100, 200, 184, 209), fill=255)
    assert roi_crop_box(mask) == (92, 192, 193, 218)


def test_roi_metric_backend_order_seed_mean_and_cluster_bootstrap(tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace
    from scripts.eval import formal_eval
    from src.dimo.native_evaluation import evaluate_roi_metrics
    row, _, predictions, _ = native_fixture(tmp_path)
    records = [row | {"sample_uid": str(i), "cluster_id": str(i), "seed": seed,
                      "output": row["source"], "metrics": {}} for i in range(2) for seed in [0, 1]]
    result = {"per_generation": records,
              "per_sample_after_seed_mean": [{"sample_uid": str(i), "cluster_id": str(i), "metrics": {}}
                                             for i in range(2)],
              "aggregate": {"dataset:magicbrush_dev": {}}}
    monkeypatch.setattr(formal_eval, "evaluate", lambda *a: result)
    def processor(*, images, return_tensors):
        assert images.size == (1024, 1024) and return_tensors == "pt"
        return {"pixel_values": torch.ones(1, 2)}
    clip = SimpleNamespace(get_image_features=lambda pixel_values: pixel_values)
    dino = lambda pixel_values: SimpleNamespace(last_hidden_state=pixel_values[:, None])
    monkeypatch.setattr(formal_eval, "embedding_backend", lambda *a: {"clip": (processor, clip), "dino": (processor, dino)})
    plan = {"bootstrap": {"unit": "source_group_cluster", "seed": 7, "resamples": 10, "confidence_level": .9},
            "embedding_models": {}, "roi_crop_rule": "fixture"}
    output = tmp_path / "metrics.json"
    measured = evaluate_roi_metrics(predictions, plan, output, torch.device("cpu"))
    assert output.is_file()
    for name in ("clip", "dino"):
        key = f"roi_{name}_to_target"
        assert measured["per_sample_after_seed_mean"][0]["metrics"][key] == pytest.approx(1.)
        aggregate = measured["aggregate"]["dataset:magicbrush_dev"][key]
        assert aggregate["n"] == aggregate["independent_groups"] == 2
        assert aggregate["seed"] == 7 and aggregate["confidence_level"] == .9
