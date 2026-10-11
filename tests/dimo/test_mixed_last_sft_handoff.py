"""Content-bound final SFT handoff and safe post-training ownership transfer."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.explicit_region.config import _load_unexpanded
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.dense_checkpoint import save_dense_checkpoint
from src.dimo import mixed_experiment as mixed
from src.dimo.dense_teacher import load_dense_teacher_contract, dense_teacher_fingerprint
from src.dimo.runtime import validate_full_config, disk_budget
from src.dimo import mixed_storage
from scripts.train import auto_mixed_last_sft_dimo as handoff
from tests.test_dense_pipeline import ToyDense, payload


def config():
    return _load_unexpanded(Path('configs/dimo/full_dense_mixed20k_4a800_5epoch.yaml'))


def request_fixture(tmp_path):
    c = config()
    c['user_transition_record'] = str(tmp_path / 'request.json')
    c['sft_final_checkpoint'] = str(tmp_path / 'sft/train/checkpoint-6250')
    request = {'schema': 'mixed20k-auto-last-sft-dimo-v1',
               'requested_action': 'automatic_dimo_after_five_sft_epochs_using_final_checkpoint',
               'user_instruction': '续训5个epoch后请自动用最后的权重进行DiMO训练',
               'sft_final_step': 6250, 'sft_additional_epochs': 5, 'dimo_epochs': 5,
               'dimo_training_dataset': 'same_mixed20k', 'dev_quality_gate': False,
               'sft_final_checkpoint': c['sft_final_checkpoint']}
    Path(c['user_transition_record']).write_text(json.dumps(request))
    return c, request


def test_mixed_dimo_preserves_math_and_reuses_sft_data_template():
    c = config(); validate_full_config(c)
    sft = _load_unexpanded(Path('configs/train/dense_mixed20k_4a800_5epoch.yaml'))
    assert c['data'] == {**sft['data'], 'verify_hashes_at_read': True}
    assert c['max_optimizer_steps'] * c['distributed']['world_size'] == 5 * 20000
    assert 500 not in c['evaluation_steps']
    with pytest.raises(RuntimeError, match='REQUIRES_REAL_DATA'):
        validate_full_config(c, synthetic=True)


@pytest.mark.parametrize('mutate', [
    lambda c: c.update(epochs=10),
    lambda c: c['distributed'].update(world_size=8),
    lambda c: c['optimizer'].update(horizon_steps=50000),
    lambda c: c['data'].update(expected_rows=8784),
    lambda c: c.update(formal_ready=True),
    lambda c: c['initialization'].update(mask_ratio=.5),
    lambda c: c['surrogate'].update(implementation='squared'),
    lambda c: c.update(evaluation_steps=[0, 500, 25000]),
])
def test_mixed_dimo_rejects_unrequested_budget_or_math_changes(mutate):
    c = config(); mutate(c)
    with pytest.raises(RuntimeError): validate_full_config(c)


@pytest.mark.parametrize('field,value', [('sft_final_step', 5000), ('dimo_epochs', 10),
    ('dimo_training_dataset', 'magicbrush_only'), ('user_instruction', ''), ('dev_quality_gate', True)])
def test_request_cannot_substitute_another_teacher_or_budget(tmp_path, field, value):
    c, request = request_fixture(tmp_path)
    assert mixed.verify_request(c) == request
    request[field] = value
    Path(c['user_transition_record']).write_text(json.dumps(request))
    with pytest.raises(RuntimeError, match='REQUEST_MISMATCH'): mixed.verify_request(c)


def make_final_checkpoint(tmp_path, monkeypatch, *, step=6250, cursor=100000, manifest_hash='d'*64):
    c, _ = request_fixture(tmp_path)
    c['data']['corpus_ready'] = str(tmp_path / 'READY.json')
    Path(c['data']['corpus_ready']).write_text('{}')
    sft = {'output_dir': str(tmp_path / 'sft/train'), 'continuation': {'new_optimizer': True}}
    ready = {'train_manifest_sha256': 'd'*64, 'warm_start_checkpoint_sha256': 'e'*64}
    p = payload() | {'experiment_mode': 'mixed20k_epoch7_continuation', 'max_optimizer_steps': 6250,
                     'continuation': sft['continuation'], 'warm_start_dense_identity': 'e'*64,
                     'data_content_hashes': {'data.manifest': {'sha256': manifest_hash}}}
    save_dense_checkpoint(c['sft_final_checkpoint'], model=ToyDense(), fingerprint_payload=p,
                          global_optimizer_step=step, committed_global_sample_count=cursor)
    monkeypatch.setattr(mixed, 'verify_ready', lambda _: ready)
    monkeypatch.setattr(mixed, 'load_config', lambda _: sft)
    return c


def test_final_teacher_export_binds_real_weights_epoch_and_authorization(tmp_path, monkeypatch):
    c = make_final_checkpoint(tmp_path, monkeypatch)
    record = mixed.select_last_teacher(c)
    assert record['selected_step'] == 6250 and record['cumulative_sft_epoch'] == 12
    assert record['dev_quality_gate'] is False
    mixed.export_last_teacher(record, tmp_path / 'teacher')
    contract = load_dense_teacher_contract(tmp_path / 'teacher')
    assert contract.manifest['status'] == 'mixed20k_user_last'
    assert contract.manifest['dense_checkpoint_sha256'] == record['selected_checkpoint_hash']
    assert contract.formal_teacher is False
    with pytest.raises(RuntimeError, match='NOT_SELECTED'):
        dense_teacher_fingerprint(tmp_path / 'teacher', expected_base_identity=contract.manifest['base_model_identity'], formal=True)
    for name in ('model-00001.safetensors', 'model.safetensors.index.json', 'fingerprint.json', 'metadata.json'):
        assert (tmp_path / 'teacher' / name).read_bytes() == (Path(c['sft_final_checkpoint']) / name).read_bytes()


@pytest.mark.parametrize('changes', [{'step': 5000}, {'cursor': 99996}, {'manifest_hash': 'f'*64}])
def test_final_handoff_rejects_complete_but_wrong_checkpoint(tmp_path, monkeypatch, changes):
    c = make_final_checkpoint(tmp_path, monkeypatch, **changes)
    with pytest.raises(RuntimeError, match='FINAL_SFT_IDENTITY'): mixed.select_last_teacher(c)


def test_rehashed_record_cannot_claim_wrong_epoch_or_formal_teacher(tmp_path, monkeypatch):
    record = mixed.select_last_teacher(make_final_checkpoint(tmp_path, monkeypatch))
    for key, value in [('selected_step', 5000), ('cumulative_sft_epoch', 11), ('formal_teacher', True)]:
        forged = copy.deepcopy(record); forged[key] = value
        forged['record_sha256'] = recipe_fingerprint({k:v for k,v in forged.items() if k != 'record_sha256'})
        with pytest.raises(RuntimeError, match='RECORD_INVALID'):
            mixed.verify_teacher_record(forged, record['selected_checkpoint_hash'])


def test_smoke_budget_does_not_reserve_unexecuted_future_epoch_saves(tmp_path, monkeypatch):
    monkeypatch.setattr(mixed_storage.shutil, 'disk_usage', lambda _: SimpleNamespace(free=10**13))
    report = disk_budget(1000, config(), tmp_path, stop_at=20)
    assert report['stop_at_step'] == 20
    assert sum(r['retained_bytes'] for r in report['volumes'].values()) == (28 + 8) * 1000
    existing = tmp_path / 'checkpoints/checkpoint-20'; existing.mkdir(parents=True)
    (existing / 'COMPLETED.json').write_text('{}')
    report = disk_budget(1000, config(), tmp_path, stop_at=20)
    assert sum(r['retained_bytes'] for r in report['volumes'].values()) == 8 * 1000
    monkeypatch.setattr(mixed_storage.shutil, 'disk_usage', lambda _: SimpleNamespace(free=1))
    with pytest.raises(RuntimeError, match='DISK_BUDGET_FAILED'):
        disk_budget(1000, config(), tmp_path, stop_at=20)


def test_split_inference_exports_are_budgeted_on_actual_filesystem(tmp_path, monkeypatch):
    train = tmp_path / 'train'; train.mkdir()
    secondary = tmp_path / 'secondary'; secondary.mkdir()
    (train / 'inference').symlink_to(secondary, target_is_directory=True)
    original = mixed_storage.os.stat
    def devices(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if not kwargs.get('follow_symlinks', True): return result
        # Use the resolved parent without calling resolve inside the mocked stat.
        text = str(path)
        return SimpleNamespace(st_dev=2 if '/secondary' in text else 1, st_mode=result.st_mode)
    monkeypatch.setattr(mixed_storage.os, 'stat', devices)
    monkeypatch.setattr(mixed_storage.shutil, 'disk_usage', lambda _: SimpleNamespace(free=10**13))
    report = mixed_storage.segment_budget(1000, config(), train, 25000)
    assert report['volumes'][1]['retained_bytes'] == 3 * 28 * 1000
    assert report['volumes'][2]['retained_bytes'] == 4 * 8 * 1000


def test_handoff_stops_only_owned_post_training_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff, 'live_process', lambda _: True)
    killed = []; monkeypatch.setattr(handoff.os, 'kill', lambda *args: killed.append(args))
    state = {'phase': 'sft:train:stepfull', 'pid': 123}
    assert not handoff.sft_success_state(state)
    assert handoff.safe_stop_sft_evaluation(tmp_path, state) is False and not killed
    state['phase'] = 'four_gpu_full_dev_evaluation'
    monkeypatch.setattr(Path, 'read_bytes', lambda _: b'python\0unrelated.py\0' + str(tmp_path).encode() + b'\0')
    with pytest.raises(RuntimeError, match='PROCESS_IDENTITY_CHANGED'):
        handoff.safe_stop_sft_evaluation(tmp_path, state)
    assert not killed
    monkeypatch.setattr(Path, 'read_bytes', lambda _: b'python\0scripts/train/run_mixed30k_continuation.py\0--experiment-root\0' + str(tmp_path).encode() + b'\0')
    assert handoff.safe_stop_sft_evaluation(tmp_path, state) is True
    assert killed == [(123, handoff.signal.SIGTERM)]
    assert json.loads((tmp_path / 'SFT_EVALUATION_USER_HANDOFF.json').read_text())['sft_training_command_completed_successfully'] is True


def test_waiter_requires_successful_sft_phase_even_with_completed_checkpoint(tmp_path, monkeypatch):
    final = tmp_path / 'checkpoint-6250'; final.mkdir()
    (final / 'COMPLETED.json').write_text('{}')
    (tmp_path / 'STATUS.json').write_text(json.dumps({'phase': 'sft:train:stepfull', 'status': 'RUNNING', 'pid': 123}))
    run = object.__new__(handoff.AutoRun); run.root = tmp_path
    run.assert_code_request = lambda: None
    events = []; run.state = lambda status, **values: events.append(status)
    monkeypatch.setattr(handoff, 'live_process', lambda _: True)
    def finish(_):
        assert events == ['WAITING_FOR_SFT']
        (tmp_path / 'STATUS.json').write_text(json.dumps({'phase': 'four_gpu_full_dev_evaluation', 'status': 'RUNNING', 'pid': 123}))
    monkeypatch.setattr(handoff.time, 'sleep', finish)
    assert run.wait_sft({'sft_final_checkpoint': str(final)})['phase'] == 'four_gpu_full_dev_evaluation'


def test_waiter_does_not_start_from_paused_or_failed_training(tmp_path, monkeypatch):
    final = tmp_path / 'checkpoint-6250'; final.mkdir()
    (final / 'COMPLETED.json').write_text('{}')
    (tmp_path / 'STATUS.json').write_text(json.dumps({'phase': 'sft:train:stepfull', 'status': 'USER_PAUSED', 'pid': 123}))
    run = object.__new__(handoff.AutoRun); run.root = tmp_path
    run.assert_code_request = lambda: None
    events = []; run.state = lambda status, **values: events.append(status)
    def failed(_):
        assert events == ['WAITING_USER_PAUSED_SFT']
        (tmp_path / 'STATUS.json').write_text(json.dumps({'phase': 'sft:train:stepfull', 'status': 'STOPPED_HARD_ERROR', 'pid': 123}))
    monkeypatch.setattr(handoff.time, 'sleep', failed)
    with pytest.raises(RuntimeError, match='SFT_FAILED'):
        run.wait_sft({'sft_final_checkpoint': str(final)})


def test_frozen_manifest_and_roots_must_match_sft_exactly(tmp_path, monkeypatch):
    c, _ = request_fixture(tmp_path)
    sft = {'data': copy.deepcopy(c['data'])}
    monkeypatch.setattr(mixed, 'load_config', lambda _: sft)
    monkeypatch.setattr(mixed, 'verify_mixed_ready', lambda _: {'verified': True})
    assert mixed.verify_ready(c) == {'verified': True}
    c['data']['roots']['interedit'] = '/another/corpus'
    with pytest.raises(RuntimeError, match='REUSE_EXACT_FROZEN_SFT_DATA'):
        mixed.verify_ready(c)
