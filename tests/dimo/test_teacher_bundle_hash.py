import json

import pytest

from src.dimo.contracts import teacher_bundle_fingerprint


def _write_core(root):
    (root / "adapter_model.safetensors").write_bytes(b"adapter")
    (root / "mask_conditioning.safetensors").write_bytes(b"mask")
    (root / "trainable_config.json").write_text(json.dumps({"lora": {}}))


def test_bundle_hash_covers_every_behavior_defining_input(tmp_path):
    _write_core(tmp_path)
    first = teacher_bundle_fingerprint(tmp_path, base_model_identity={"snapshot": "a"})
    assert first["missing_optional_files"] == ["dimo_teacher_manifest.json", "fingerprint.json"]
    (tmp_path / "mask_conditioning.safetensors").write_bytes(b"changed-mask")
    second = teacher_bundle_fingerprint(tmp_path, base_model_identity={"snapshot": "a"})
    assert first["bundle_sha256"] != second["bundle_sha256"]
    third = teacher_bundle_fingerprint(tmp_path, base_model_identity={"snapshot": "b"})
    assert second["bundle_sha256"] != third["bundle_sha256"]


def test_formal_bundle_requires_optional_prep_artifacts(tmp_path):
    _write_core(tmp_path)
    teacher_bundle_fingerprint(tmp_path, base_model_identity="base", formal=False)
    with pytest.raises(RuntimeError, match="formal teacher bundle is incomplete"):
        teacher_bundle_fingerprint(tmp_path, base_model_identity="base", formal=True)

