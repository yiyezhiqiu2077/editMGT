import ast
import copy
import json
from pathlib import Path
import subprocess

import pytest
from safetensors.torch import save_file
import torch

from src.dimo.contracts import DIMO_EDIT_FORMAL_READY, enforce_run_guard, released_base_model_identity
from src.dimo.pilot_contract import (SMOKE_PROOFS, gpu_allocation_report, require_smoke_certificate,
    smoke_compatibility, training_identity, validate_pilot_config)
from src.dimo.provisional_teacher import MANIFEST, export_provisional_teacher, load_provisional_teacher
from src.dimo.checkpoint import config_hash
from src.dimo.smoke import compare_states, verify_real_smoke
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.config import _load_unexpanded
from src.explicit_region.contracts import COMPONENT_FILES, audit_component_identity

ROOT = Path(__file__).resolve().parents[2]
BASE = '36dff5ff9ad10cee3d3910eb980932ec0a251b62'


def config(name='pilot'):
    name = {'pilot': 'pilot', 'smoke': 'smoke', 'full': 'd200k'}[name]
    return _load_unexpanded(ROOT / f'configs/dimo/{name}_e3_step3125_8g.yaml')


def e3_fixture(tmp_path):
    # Deliberately synthetic export contract fixture, never real teacher evidence.
    model = tmp_path / 'model'
    for subfolder, filename in COMPONENT_FILES.values():
        folder = model / subfolder; folder.mkdir(parents=True, exist_ok=True)
        (folder / filename).write_text('{}')
    (model / 'asset_identity.json').write_text(json.dumps({'repo_id': 'WeiChow/EditMGT',
        'resolved_revision': 'b' * 40}))
    identity = audit_component_identity(model)
    payload = {'git_sha': 'a' * 40, 'model_identity': identity, 'model_snapshot': 'b' * 40,
        'corruption': {'mode': 'roi_hardlock', 'persistent_conditioning': True},
        'lora': {'scope': 'both'}, 'epochs': 5, 'scheduler_horizon_steps': 31250,
        'world_size': 8, 'per_device_batch': 1, 'gradient_accumulation': 4,
        'resolution': 1024, 'data': {'name': 'fixed_200k'},
        'data_content_hashes': {'data.manifest': {'sha256': 'c' * 64}},
        'unit_test_fixture': True}
    source = tmp_path / 'e3/checkpoint-3125'; source.mkdir(parents=True)
    save_file({'unit_test_adapter': torch.ones(2)}, source / 'adapter_model.safetensors')
    save_file({'edit_region_embedding': torch.ones(2)}, source / 'mask_conditioning.safetensors')
    (source / 'trainable_config.json').write_text(json.dumps(payload))
    fp = recipe_fingerprint(payload)
    (source / 'fingerprint.json').write_text(json.dumps({'sha256': fp, 'payload': payload}))
    (source / 'mask_conditioning_config.json').write_text('{}')
    torch.save({'global_optimizer_step': 3125, 'committed_global_sample_count': 100000,
                'fingerprint': fp}, source / 'training_state.pt')
    return source, model


def test_provisional_export_is_atomic_complete_immutable_and_not_formal(tmp_path):
    source, model = e3_fixture(tmp_path)
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / 'separate-teacher'
    result = export_provisional_teacher(source, model, output)
    assert result['formal_teacher'] is False and result['dev_only'] is True
    assert result['source_step'] == 3125 and result['source_epoch'] == .5
    assert not (output / 'training_state.pt').exists()
    assert not (output / 'SELECTED_CHECKPOINT.json').exists()
    assert not (output / 'dimo_teacher_manifest.json').exists()
    assert result == export_provisional_teacher(source, model, output)
    contract = load_provisional_teacher(output)
    assert contract.formal_teacher is False
    assert original == {p.name: p.read_bytes() for p in source.iterdir()}
    for name in result['artifact_sha256']:
        assert (output / name).read_bytes() == original[name]
    assert not list(tmp_path.glob('.separate-teacher.partial-*'))
    assert enforce_run_guard(contract, prep_smoke=False, max_optimizer_steps=500,
        formal_ready=False, experiment_mode='pilot_8gpu', world_size=8)['dev_only'] is True
    with pytest.raises(RuntimeError, match='DIMO_TEACHER_NOT_SELECTED'):
        enforce_run_guard(contract, prep_smoke=False, max_optimizer_steps=500, formal_ready=False)


@pytest.mark.parametrize('field,value', [('global_optimizer_step', 300), ('committed_global_sample_count', 99999)])
def test_export_rejects_wrong_step_or_cursor(tmp_path, field, value):
    source, model = e3_fixture(tmp_path)
    state = torch.load(source / 'training_state.pt', weights_only=False)
    state[field] = value; torch.save(state, source / 'training_state.pt')
    with pytest.raises(RuntimeError, match='STEP_OR_SAMPLE_CURSOR'):
        export_provisional_teacher(source, model, tmp_path / 'teacher')


def test_export_and_contract_reject_mutated_artifacts_and_base(tmp_path):
    source, model = e3_fixture(tmp_path)
    (model / 'editmgt/config.json').write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match='BASE_MODEL_IDENTITY'):
        export_provisional_teacher(source, model, tmp_path / 'teacher')
    (model / 'editmgt/config.json').write_text('{}')
    output = tmp_path / 'teacher'
    export_provisional_teacher(source, model, output)
    artifact = output / 'mask_conditioning.safetensors'
    artifact.chmod(0o644); artifact.write_bytes(b'corrupted')
    with pytest.raises(RuntimeError, match='ARTIFACT_HASH'):
        load_provisional_teacher(output)


def test_missing_checkpoint_and_overlapping_output_fail(tmp_path):
    source, model = e3_fixture(tmp_path)
    with pytest.raises(RuntimeError, match='INDEPENDENT'):
        export_provisional_teacher(source, model, source / 'teacher')
    (source / 'fingerprint.json').unlink()
    with pytest.raises(RuntimeError, match='INCOMPLETE'):
        export_provisional_teacher(source, model, tmp_path / 'teacher')


def test_reference_configs_and_budget_are_explicit():
    for name, steps in [('pilot', 500), ('smoke', 20), ('full', 25000)]:
        c = config(name); result = validate_pilot_config(c)
        assert result['global_batch'] == 8 and result['target_steps'] == steps
        assert c['optimizer']['horizon_steps'] == steps
        assert c['optimizer']['student_learning_rate'] == c['optimizer']['auxiliary_learning_rate'] == 1e-6
        assert c['distillation']['mode'] == 'FKL'
        assert c['initialization']['mask_ratio'] == .5
        assert c['embedding_perturbation']['sigma'] == .3
        assert c['ema']['decay'] == .9999 and c['gt_anchor']['weight'] == 0
    assert config()['checkpoint_steps'] == [0, 50, 100, 250, 500]
    assert config('full')['checkpoint_steps'] == list(range(2500, 25001, 2500))
    assert DIMO_EDIT_FORMAL_READY is False


@pytest.mark.parametrize('field,value', [('formal_ready', True), ('gradient_accumulation', 4),
    ('batch_per_gpu', 2), ('dev_only', False), ('resolution', 512)])
def test_pilot_cannot_bypass_reference_guard(field, value):
    c = config(); c[field] = value
    with pytest.raises(RuntimeError, match='PILOT'):
        validate_pilot_config(c)


def test_config_fingerprint_excludes_only_output_and_binds_horizon():
    c = config(); d = copy.deepcopy(c); d['output_dir'] = 'other-run'
    args = ({'bundle_sha256': 'teacher'}, {'model': 'released'}, {'data': 'frozen'})
    assert config_hash(training_identity(c, *args)) == config_hash(training_identity(d, *args))
    d['optimizer']['horizon_steps'] += 1
    assert config_hash(training_identity(c, *args)) != config_hash(training_identity(d, *args))


def test_pilot_requires_real_complete_smoke_certificate(tmp_path):
    c = config(); data = {'train_manifest_sha256': 'data', 'corpus_ready_sha256': 'ready'}
    assert smoke_compatibility(c, 'teacher', data) == smoke_compatibility(config('smoke'), 'teacher', data)
    with pytest.raises(RuntimeError, match='REAL_8GPU_SMOKE_REQUIRED'):
        require_smoke_certificate(None, c, 'teacher', data, 'git')
    report = {'schema': 'dimo-real-8gpu-smoke-v1', 'status': 'PASS', 'world_size': 8,
        'data_name': 'fixed_200k', 'git_sha': 'git', 'teacher_hash': 'teacher', 'data_identity': data,
        'compatibility_sha256': smoke_compatibility(c, 'teacher', data),
        'resume_equivalence': 'bitwise_exact', 'checks': dict.fromkeys(SMOKE_PROOFS, True)}
    path = tmp_path / 'smoke.json'; path.write_text(json.dumps(report))
    require_smoke_certificate(path, c, 'teacher', data, 'git')
    report['checks'].pop('gradient_sync_and_role_isolation')
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match='CERTIFICATE_MISMATCH'):
        require_smoke_certificate(path, c, 'teacher', data, 'git')


def test_cpu_evidence_cannot_be_called_real_gpu_smoke(tmp_path):
    roots = [tmp_path / name for name in ('fresh20', 'first10', 'resume20')]
    for root in roots:
        root.mkdir()
        (root / 'run_manifest.json').write_text(json.dumps({'world_size': 8, 'dev_only': True,
            'formal_teacher': False, 'resolved_config': {'data': {'name': 'fixed_200k'}},
            'actual_execution': {'backend': 'gloo', 'device_type': 'cpu', 'visible_gpu_count': 0}}))
    with pytest.raises(RuntimeError, match='NOT_REAL_EIGHT_GPU'):
        verify_real_smoke(*roots)


def test_equivalence_levels_are_not_conflated():
    a = {'floating': torch.ones(3), 'rng': torch.zeros(3, dtype=torch.uint8)}
    b = copy.deepcopy(a); b['floating'][0] += 1e-6
    result = compare_states(a, b)
    assert result['within_tolerance'] and not result['bitwise_exact']
    b['rng'][0] = 1
    assert not compare_states(a, b)['within_tolerance']


def test_busy_allocation_is_rejected_without_stopping_processes(monkeypatch):
    devices = '\n'.join(f'{i}, GPU-{i}, 30000, 100' for i in range(8))
    def query(command, **kwargs):
        return devices if '--query-gpu=index,uuid,memory.free,utilization.gpu' in command else 'GPU-0, 123\n'
    monkeypatch.setattr(subprocess, 'check_output', query)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    with pytest.raises(RuntimeError, match='NO_INDEPENDENT'):
        gpu_allocation_report(config())


def test_training_math_and_e3_files_are_unchanged():
    unchanged = ['src/dimo/divergence.py', 'src/dimo/surrogate.py', 'src/dimo/auxiliary.py',
        'src/dimo/forward_process.py', 'src/dimo/conditioning.py', 'src/dimo/one_step.py',
        'src/dimo/rng.py', 'src/dimo/ema.py', 'src/dimo/roles.py',
        'src/dimo/teacher_registration.py', 'scripts/dimo/register_teacher.py',
        'src/explicit_region/corruption.py', 'src/explicit_region/epoch_sampler.py',
        'scripts/train/train_explicit_region.py', 'configs/train/cluster_8g_e3.yaml']
    for name in unchanged:
        assert (ROOT / name).read_bytes() == subprocess.check_output(['git', 'show', f'{BASE}:{name}'], cwd=ROOT)
    name = 'src/dimo/initialization.py'
    before = ast.parse(subprocess.check_output(['git', 'show', f'{BASE}:{name}'], cwd=ROOT, text=True))
    after = ast.parse((ROOT / name).read_text())
    for function in ('build_editing_initial_state', 'sample_student_tokens', 'build_role_lora_configs'):
        get = lambda tree: next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        assert ast.dump(get(before)) == ast.dump(get(after))
