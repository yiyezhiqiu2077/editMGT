"""Content-bound exploratory teacher export, never formal registration."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import torch
from safetensors.torch import load_file

from .contracts import TeacherCheckpointContract, _canonical_hash, _sha256, released_base_model_identity
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.contracts import audit_component_identity

MANIFEST = 'dimo_provisional_teacher_manifest.json'
SCHEMA = 'dimo-provisional-teacher-v1'
REQUIRED = ('adapter_model.safetensors', 'mask_conditioning.safetensors',
            'trainable_config.json', 'fingerprint.json')


def load_provisional_teacher(root, *, expected_base_identity=None):
    root = Path(root).resolve()
    manifest = json.loads((root / MANIFEST).read_text())
    if (manifest.get('schema') != SCHEMA or manifest.get('formal_teacher') is not False
            or manifest.get('dev_only') is not True or manifest.get('source_step') != 3125
            or manifest.get('source_epoch') != .5
            or manifest.get('source_samples') != 100000
            or not re.fullmatch(r'[0-9a-f]{40}', str(manifest.get('source_git_sha')))
            or not re.fullmatch(r'[0-9a-f]{64}', str(manifest.get('source_d200k_sha256')))
            or manifest.get('source_experiment') != 'E3-D200K-5Epoch'):
        raise RuntimeError('INVALID_PROVISIONAL_TEACHER_CONTRACT')
    identity = manifest.get('base_model_identity')
    if not identity or (expected_base_identity is not None and identity != expected_base_identity):
        raise RuntimeError('PROVISIONAL_TEACHER_BASE_IDENTITY_MISMATCH')
    files = manifest.get('artifact_sha256', {})
    if (not set(REQUIRED) <= files.keys()
            or not set(files) <= set(REQUIRED) | {'mask_conditioning_config.json'}
            or set(p.name for p in root.iterdir()) != set(files) | {MANIFEST}):
        raise RuntimeError('PROVISIONAL_TEACHER_BUNDLE_INCOMPLETE')
    for name, digest in files.items():
        if Path(name).name != name or _sha256(root / name) != digest:
            raise RuntimeError('PROVISIONAL_TEACHER_ARTIFACT_HASH_MISMATCH')
    core = {k: v for k, v in manifest.items() if k != 'bundle_sha256'}
    if manifest.get('bundle_sha256') != _canonical_hash(core):
        raise RuntimeError('PROVISIONAL_TEACHER_MANIFEST_HASH_MISMATCH')
    return TeacherCheckpointContract(root, manifest, _canonical_hash(
        {'manifest_sha256': _sha256(root / MANIFEST), 'artifacts': files}))


def export_provisional_teacher(checkpoint, model_root, output):
    source, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise RuntimeError('TEACHER_EXPORT_MUST_BE_INDEPENDENT')
    required = (*REQUIRED, 'training_state.pt')
    if any(not (source / name).is_file() for name in required):
        raise RuntimeError('E3_CHECKPOINT_INCOMPLETE')
    state_hash = _sha256(source / 'training_state.pt')
    state = torch.load(source / 'training_state.pt', map_location='cpu', weights_only=False)
    if (state.get('global_optimizer_step') != 3125
            or state.get('committed_global_sample_count') != 100000):
        raise RuntimeError('E3_STEP_OR_SAMPLE_CURSOR_MISMATCH')
    fingerprint = json.loads((source / 'fingerprint.json').read_text())
    payload = fingerprint.get('payload', {})
    if (fingerprint.get('sha256') != recipe_fingerprint(payload)
            or state.get('fingerprint') != fingerprint['sha256']
            or json.loads((source / 'trainable_config.json').read_text()) != payload):
        raise RuntimeError('E3_CHECKPOINT_FINGERPRINT_MISMATCH')
    corruption = payload.get('corruption', {})
    if (corruption.get('mode') != 'roi_hardlock'
            or corruption.get('persistent_conditioning') is not True
            or payload.get('lora', {}).get('scope') != 'both'
            or payload.get('epochs') != 5 or payload.get('scheduler_horizon_steps') != 31250
            or payload.get('world_size') != 8 or payload.get('per_device_batch') != 1
            or payload.get('gradient_accumulation') != 4
            or payload.get('resolution') != 1024 or payload.get('data', {}).get('name') != 'fixed_200k'
            or not re.fullmatch(r'[0-9a-f]{64}', str(payload.get('data_content_hashes', {})
                                                     .get('data.manifest', {}).get('sha256')))
            or not re.fullmatch(r'[0-9a-f]{40}', str(payload.get('git_sha')))):
        raise RuntimeError('SOURCE_IS_NOT_E3_D200K_FIVE_EPOCH')
    actual = released_base_model_identity(audit_component_identity(model_root))
    source_identity = released_base_model_identity(payload.get('model_identity', {}))
    if (source_identity != actual or actual.get('repo_id') != 'WeiChow/EditMGT'
            or not re.fullmatch(r'[0-9a-f]{40}', str(actual.get('resolved_revision')))
            or payload.get('model_snapshot') != actual['resolved_revision']):
        raise RuntimeError('E3_RELEASED_BASE_MODEL_IDENTITY_MISMATCH')
    adapter = load_file(source / 'adapter_model.safetensors')
    region = load_file(source / 'mask_conditioning.safetensors')
    if (not adapter or set(region) != {'edit_region_embedding'}
            or any(not torch.isfinite(t).all() for t in (*adapter.values(), *region.values()))):
        raise RuntimeError('E3_TEACHER_WEIGHTS_INVALID')
    names = list(REQUIRED)
    if (source / 'mask_conditioning_config.json').is_file():
        names.append('mask_conditioning_config.json')
    hashes = {name: _sha256(source / name) for name in names}
    manifest = {
        'schema': SCHEMA, 'source_experiment': 'E3-D200K-5Epoch',
        'source_step': 3125, 'source_epoch': .5, 'source_samples': 100000,
        'formal_teacher': False, 'dev_only': True,
        'base_model_identity': actual, 'source_git_sha': payload['git_sha'],
        'source_d200k_sha256': payload['data_content_hashes']['data.manifest']['sha256'],
        'artifact_sha256': hashes, 'source_checkpoint': str(source),
        'source_training_state_sha256': state_hash,
        'source_recipe_fingerprint': fingerprint['sha256'],
    }
    manifest['bundle_sha256'] = _canonical_hash(manifest)
    if output.exists():
        contract = load_provisional_teacher(output, expected_base_identity=actual)
        if contract.manifest != manifest:
            raise RuntimeError('IMMUTABLE_TEACHER_DESTINATION_EXISTS')
        return contract.manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{output.name}.partial-', dir=output.parent))
    try:
        for name in names:
            shutil.copyfile(source / name, temporary / name)
        (temporary / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        load_provisional_teacher(temporary, expected_base_identity=actual)
        if (_sha256(source / 'training_state.pt') != state_hash
                or any(_sha256(source / name) != digest for name, digest in hashes.items())):
            raise RuntimeError('E3_CHECKPOINT_CHANGED_DURING_EXPORT')
        for path in temporary.iterdir():
            path.chmod(0o444)
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return load_provisional_teacher(output, expected_base_identity=actual).manifest
