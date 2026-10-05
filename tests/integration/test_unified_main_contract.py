import json
from pathlib import Path

import pytest
import torch

from scripts.train import train_dimo_editing
from src.dimo.contracts import (
    DIMO_EDIT_FORMAL_READY,
    enforce_run_guard,
    load_teacher_contract,
    released_base_model_identity,
)
from src.dimo.forward_process import apply_target_embedding_perturbation
from src.dimo.teacher_registration import register_teacher
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.fixed_dataset import FixedCorpusDataset
from src.explicit_region.formal_pipeline import load_formal_assets, sha256_file, stable_json_hash
from src.transformer import Transformer2DModel
from tests.fixed200k_helpers import fixture_record


ROOT = Path(__file__).resolve().parents[2]
MODEL_REVISION = "a" * 40
FORMAL_GIT = "b" * 40


def _write_registration_inputs(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "mask_conditioning.safetensors").write_bytes(b"region")
    (checkpoint / "trainable_config.json").write_text(
        json.dumps({"lora": {"rank": 8}}), encoding="utf-8"
    )
    fingerprint_payload = {
            "git_sha": FORMAL_GIT,
            "model_identity": {
                "repo_id": "WeiChow/EditMGT",
                "resolved_revision": MODEL_REVISION,
                "components": {"transformer": {"config_sha256": "c" * 64}},
            },
    }
    fingerprint = {
        "sha256": recipe_fingerprint(fingerprint_payload),
        "payload": fingerprint_payload,
    }
    (checkpoint / "fingerprint.json").write_text(
        json.dumps(fingerprint), encoding="utf-8"
    )
    formal = tmp_path / "formal_assets.json"
    formal.write_text(
        json.dumps(
            {
                "git_sha": FORMAL_GIT,
                "assets": {
                    "editmgt": {
                        "repo_id": "WeiChow/EditMGT",
                        "resolved_revision": MODEL_REVISION,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    record = tmp_path / "SELECTED_CHECKPOINT.json"
    _refresh_selected_record(checkpoint, record, formal)
    return checkpoint, record, formal, tmp_path / "dimo_teacher"


def _refresh_selected_record(checkpoint: Path, record: Path, formal: Path) -> None:
    files = {
        name: sha256_file(checkpoint / name)
        for name in (
            "adapter_model.safetensors",
            "mask_conditioning.safetensors",
            "trainable_config.json",
            "fingerprint.json",
        )
        if (checkpoint / name).is_file()
    }
    value = {
        "schema_version": "selected-checkpoint-v1",
        "status": "READY",
        "git_sha": FORMAL_GIT,
        "formal_assets_sha256": sha256_file(formal),
        "checkpoint_path": str(checkpoint.resolve()),
        "files": files,
        "checkpoint_identity_sha256": stable_json_hash(files),
    }
    record.write_text(json.dumps(value), encoding="utf-8")


def _rewrite_fingerprint(checkpoint: Path, record: Path, formal: Path, update) -> None:
    path = checkpoint / "fingerprint.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    update(value)
    value["sha256"] = recipe_fingerprint(value["payload"])
    path.write_text(json.dumps(value), encoding="utf-8")
    _refresh_selected_record(checkpoint, record, formal)


def _tiny_transformer() -> Transformer2DModel:
    return Transformer2DModel(
        in_channels=6,
        num_layers=0,
        num_single_layers=0,
        attention_head_dim=6,
        num_attention_heads=1,
        joint_attention_dim=6,
        pooled_projection_dim=6,
        axes_dims_rope=(2, 2, 2),
        vocab_size=6,
        codebook_size=5,
        text_encoder_architecture="CLIP",
        connector_type="none",
    ).eval()


def _forward(model: Transformer2DModel, **extra):
    common = {
        "hidden_states": torch.tensor([[[0, 1], [2, 3]]]),
        "encoder_hidden_states": torch.randn(1, 1, 6),
        "pooled_projections": torch.randn(1, 6),
        "timestep": torch.tensor([0.5]),
        "img_ids": torch.zeros(4, 3),
        "txt_ids": torch.zeros(1, 3),
        "micro_conds": torch.zeros(1, 5),
        "edit_region_mask": torch.ones(1, 2, 2, dtype=torch.bool),
    }
    return model(**common, **extra)


def test_formal_manifest_and_shared_fixed_dataset_contract(tmp_path):
    assert load_formal_assets(ROOT / "configs/formal_assets.yaml")["schema_version"] == "editmgt-formal-assets-v3"
    assert train_dimo_editing.FixedCorpusDataset is FixedCorpusDataset
    data_root = tmp_path / "magicbrush"
    row = fixture_record(data_root, "magicbrush", 0)
    row["manifest_index"] = 0
    manifest = tmp_path / "train_200k.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    sample = FixedCorpusDataset(manifest, {"magicbrush": data_root}, resolution=32)[0]
    assert sample["sample_uid"] == row["sample_uid"]
    assert sample["instruction_en"] == row["instruction_en"]
    assert sample["edit_region_mask"].any()


def test_sft_transformer_defaults_are_finite_and_use_original_region_embedding():
    model = _tiny_transformer()
    with torch.no_grad():
        torch.manual_seed(9)
        default = _forward(model)
        torch.manual_seed(9)
        explicit = _forward(
            model,
            edit_region_embedding_override=model.edit_region_embedding,
            target_embedding_noise=None,
            target_embedding_noise_sigma=0,
        )
    assert torch.isfinite(default).all()
    assert torch.equal(default, explicit)


def test_embedding_noise_none_and_sigma_zero_are_strict_noops():
    hidden = torch.randn(2, 3, 4, 4)
    assert apply_target_embedding_perturbation(hidden, None, None, 0.5) is hidden
    assert apply_target_embedding_perturbation(hidden, torch.randn_like(hidden), None, 0) is hidden


def test_teacher_registration_happy_path_and_idempotence(tmp_path):
    checkpoint, record, formal, output = _write_registration_inputs(tmp_path)
    result = register_teacher(checkpoint, record, formal, output)
    assert result["status"] == "TEACHER_REGISTRATION_PASS"
    runtime_identity = released_base_model_identity(
        {
            "repo_id": "WeiChow/EditMGT",
            "resolved_revision": MODEL_REVISION,
            "components": {"transformer": {"config_sha256": "c" * 64}},
        }
    )
    assert runtime_identity == result["base_model_identity"]
    contract = load_teacher_contract(output, expected_base_identity=result["base_model_identity"])
    assert contract.formal_teacher
    again = register_teacher(checkpoint, record, formal, output)
    assert again["status"] == "ALREADY_REGISTERED_IDENTICAL"


@pytest.mark.parametrize("name", ["trainable_config.json", "fingerprint.json"])
def test_teacher_registration_rejects_missing_mandatory_file(tmp_path, name):
    checkpoint, record, formal, output = _write_registration_inputs(tmp_path)
    (checkpoint / name).unlink()
    with pytest.raises(RuntimeError, match="DIMO_TEACHER_BUNDLE_INCOMPLETE"):
        register_teacher(checkpoint, record, formal, output)


@pytest.mark.parametrize(
    "name",
    [
        "adapter_model.safetensors",
        "mask_conditioning.safetensors",
        "trainable_config.json",
        "fingerprint.json",
    ],
)
def test_teacher_registration_rejects_tampered_checkpoint_file(tmp_path, name):
    checkpoint, record, formal, output = _write_registration_inputs(tmp_path)
    with (checkpoint / name).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(RuntimeError, match="DIMO_TEACHER_REGISTRATION_MISMATCH"):
        register_teacher(checkpoint, record, formal, output)


def test_teacher_registration_rejects_tampered_selected_record(tmp_path):
    checkpoint, record, formal, output = _write_registration_inputs(tmp_path)
    value = json.loads(record.read_text(encoding="utf-8"))
    value["checkpoint_identity_sha256"] = "0" * 64
    record.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="DIMO_TEACHER_REGISTRATION_MISMATCH"):
        register_teacher(checkpoint, record, formal, output)


def test_teacher_registration_rejects_tampered_formal_assets(tmp_path):
    checkpoint, record, formal, output = _write_registration_inputs(tmp_path)
    value = json.loads(formal.read_text(encoding="utf-8"))
    value["tampered"] = True
    formal.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="DIMO_TEACHER_REGISTRATION_MISMATCH"):
        register_teacher(checkpoint, record, formal, output)


@pytest.mark.parametrize(
    ("field", "value"),
    [("repo_id", "other/EditMGT"), ("resolved_revision", "d" * 40)],
)
def test_teacher_registration_rejects_base_identity_mismatch(tmp_path, field, value):
    checkpoint, record, formal, output = _write_registration_inputs(tmp_path)
    _rewrite_fingerprint(
        checkpoint,
        record,
        formal,
        lambda data: data["payload"]["model_identity"].__setitem__(field, value),
    )
    with pytest.raises(RuntimeError, match="DIMO_TEACHER_REGISTRATION_MISMATCH"):
        register_teacher(checkpoint, record, formal, output)


def test_teacher_registration_rejects_missing_base_identity(tmp_path):
    checkpoint, record, formal, output = _write_registration_inputs(tmp_path)
    _rewrite_fingerprint(
        checkpoint,
        record,
        formal,
        lambda data: data["payload"].pop("model_identity"),
    )
    with pytest.raises(RuntimeError, match="DIMO_BASE_MODEL_IDENTITY_MISSING"):
        register_teacher(checkpoint, record, formal, output)


def test_teacher_registration_refuses_existing_different_bundle(tmp_path):
    checkpoint, record, formal, output = _write_registration_inputs(tmp_path)
    register_teacher(checkpoint, record, formal, output)
    with (output / "dimo_teacher_manifest.json").open("ab") as handle:
        handle.write(b"different")
    with pytest.raises(RuntimeError, match="DIMO_TEACHER_OUTPUT_CONFLICT"):
        register_teacher(checkpoint, record, formal, output)


def test_teacher_contract_rejects_post_registration_weight_tamper(tmp_path):
    checkpoint, record, formal, output = _write_registration_inputs(tmp_path)
    register_teacher(checkpoint, record, formal, output)
    with (output / "adapter_model.safetensors").open("ab") as handle:
        handle.write(b"tampered-after-registration")
    with pytest.raises(RuntimeError, match="contract identity mismatch"):
        load_teacher_contract(output)


def test_formal_dimo_remains_fail_closed_with_registered_teacher(tmp_path):
    checkpoint, record, formal, output = _write_registration_inputs(tmp_path)
    register_teacher(checkpoint, record, formal, output)
    contract = load_teacher_contract(output)
    assert DIMO_EDIT_FORMAL_READY is False
    with pytest.raises(RuntimeError, match="DIMO_EDIT_FORMAL_READY=false"):
        enforce_run_guard(contract, prep_smoke=False, max_optimizer_steps=1, formal_ready=False)
