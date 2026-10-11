"""The fixed-data audit must render a boolean ROI as an alpha mask."""
from PIL import Image
import torch
from scripts.data.audit_fixed200k import montage


def test_montage_renders_boolean_mask_overlay_and_preserves_tensors(tmp_path):
    mask = torch.zeros(32, 32, dtype=torch.bool); mask[:, :16] = True
    item = {'source_image': torch.zeros(3, 32, 32), 'target_image': torch.ones(3, 32, 32),
            'edit_region_mask': mask, 'dataset_name': 'crispedit', 'sample_uid': 'synthetic',
            'edit_type': 'color', 'mask_semantics': 'precise', 'instruction_en': 'Change the color.'}
    output = tmp_path / 'montage.png'; montage([item], output)
    with Image.open(output) as im:
        assert im.size == (768, 246)
        inside, outside = im.getpixel((400, 80)), im.getpixel((560, 80))
        assert inside[0] > 0 and inside[1:] == (0, 0)
        assert outside == (0, 0, 0)
    assert mask.dtype == torch.bool and mask.sum() == 512
    assert item['source_image'].eq(0).all()


def test_mixed_sft_waits_for_other_gpu_jobs_without_preempting(tmp_path, monkeypatch):
    from scripts.train import run_mixed30k_continuation as coordinator
    run = object.__new__(coordinator.MixedRun)
    run.assert_code = lambda: None
    states = []; run.state = lambda status, **values: states.append(status)
    attempts = iter([RuntimeError('FULL_DIMO_GPU_HAS_OTHER_JOB_DO_NOT_PREEMPT'), None])
    def admission(**kwargs):
        outcome = next(attempts)
        if outcome is not None: raise outcome
    called = []
    monkeypatch.setattr(coordinator, 'gpu_admission', admission)
    monkeypatch.setattr(coordinator.time, 'sleep', lambda _: None)
    monkeypatch.setattr(coordinator.Run, 'train', lambda *args: called.append(args))
    run.train('sft', tmp_path, 2)
    assert states == ['WAITING_FOR_IDLE_GPUS']
    assert len(called) == 1 and called[0][1:] == ('sft', tmp_path, 2, None)


def test_mixed_sft_does_not_wait_away_hardware_safety_failure(tmp_path, monkeypatch):
    import pytest
    from scripts.train import run_mixed30k_continuation as coordinator
    run = object.__new__(coordinator.MixedRun); run.assert_code = lambda: None
    def invalid(**kwargs): raise RuntimeError('MAGICBRUSH_REQUIRES_FOUR_A800_80GB_GPUS')
    monkeypatch.setattr(coordinator, 'gpu_admission', invalid)
    with pytest.raises(RuntimeError, match='REQUIRES_FOUR_A800'):
        run.train('sft', tmp_path, 2)


def test_gpu_sharing_requires_explicit_flag_and_keeps_formal_gate(monkeypatch):
    import subprocess, psutil
    import pytest
    from types import SimpleNamespace
    from src.dimo.runtime import gpu_admission
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    inventory = '\n'.join(f'{i}, gpu{i}, NVIDIA A800, 81920, 12000' for i in range(4))
    monkeypatch.setattr(subprocess, 'check_output', lambda argv, **kwargs:
                        inventory if '--query-gpu=index,uuid,name,memory.total,memory.used' in argv else 'gpu0, 987654\n')
    monkeypatch.setattr(psutil, 'Process', lambda _: SimpleNamespace(ppid=lambda: -1))
    with pytest.raises(RuntimeError, match='GPU_HAS_OTHER_JOB'): gpu_admission(magicbrush=True)
    result = gpu_admission(magicbrush=True, allow_other_jobs=True)
    assert result['gpu_sharing_user_authorized'] is True
    assert result['existing_other_jobs'] == [{'gpu_uuid': 'gpu0', 'pid': 987654}]
    with pytest.raises(RuntimeError, match='FORMAL_GPU_SHARING_NOT_AUTHORIZED'):
        gpu_admission(allow_other_jobs=True)


def test_parallel_loader_audit_verifies_every_row_and_propagates_failures():
    from types import SimpleNamespace
    import pytest
    from scripts.data.prepare_mixed30k_corpus import verify_loader_rows
    class Data:
        rows = [{'sample_uid': str(i)} for i in range(12)]
        backends = {'toy': (SimpleNamespace(tar_reader=None), {})}
        def __init__(self): self.seen = []
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            self.seen.append(i)
            if getattr(self, 'broken', False) and i == 7: raise ValueError('bad asset')
            return self.rows[i]
    ds = Data(); verify_loader_rows(ds, workers=4)
    assert sorted(ds.seen) == list(range(12))
    ds.broken = True
    with pytest.raises(ValueError, match='bad asset'): verify_loader_rows(ds, workers=4)


def test_separate_dev_storage_reserves_images_without_reducing_checkpoint_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.dimo import mixed_storage as storage
    import pytest
    paths = [tmp_path / name for name in ('primary', 'secondary', 'evaluation')]
    for path in paths: path.mkdir()
    original = storage.os.stat
    def devices(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        text = str(path)
        device = 3 if '/evaluation' in text else 2 if '/secondary' in text else 1
        return SimpleNamespace(st_dev=device, st_mode=result.st_mode)
    monkeypatch.setattr(storage.os, 'stat', devices)
    monkeypatch.setattr(storage.shutil, 'disk_usage', lambda _: SimpleNamespace(free=10**13))
    ordinary = storage.whole_handoff_budget(*paths[:2], 1000)
    separate = storage.whole_handoff_budget(*paths[:2], 1000, evaluation_root=paths[2])
    assert ordinary['filesystems'][0]['required_bytes'] - separate['filesystems'][0]['required_bytes'] == 24 * 1024**3
    assert separate['filesystems'][2]['required_bytes'] == 32 * 1024**3
    assert separate['filesystems'][1] == ordinary['filesystems'][1]
    assert separate['status'] == 'PASS'
    with pytest.raises(RuntimeError, match='THIRD_FILESYSTEM'):
        storage.whole_handoff_budget(*paths[:2], 1000, evaluation_root=paths[0])


def skip_request(root):
    import json
    record = {'schema': 'mixed20k-user-skip-sft-gpu-acceptance-v1',
              'user_instruction': 'Skip SFT smoke and resume comparison; start long training',
              'skip_sft_gpu_acceptance': True, 'sft_epochs': 5, 'sft_steps': 6250, 'data_rows': 20000}
    (root / 'USER_SKIP_SFT_GPU_ACCEPTANCE.json').write_text(json.dumps(record))
    return record


def test_direct_ready_recovery_requires_user_request_and_cannot_replay_long_training(tmp_path, monkeypatch):
    import json, pytest
    from scripts.train import run_mixed30k_continuation as coordinator
    (tmp_path / 'data').mkdir(); (tmp_path / 'data/MIXED_READY.json').write_text('{}')
    (tmp_path / 'configs').mkdir(); (tmp_path / 'configs/sft.yaml').write_text('fixture: true')
    (tmp_path / 'PIPELINE_STARTED.json').write_text('{}')
    (tmp_path / 'STATUS.json').write_text(json.dumps({'status': 'STOPPED_HARD_ERROR', 'phase': 'sft:smoke2:step2', 'pid': 999999999}))
    monkeypatch.setattr(coordinator, 'verify_mixed_ready', lambda _: {'verified': True})
    with pytest.raises(FileNotFoundError): coordinator.recover_ready_training(tmp_path)
    skip_request(tmp_path)
    train = tmp_path / 'sft/train'; train.mkdir(parents=True)
    (train / 'train_metrics.jsonl').write_text('{"global_step":1}\n')
    with pytest.raises(RuntimeError, match='CANNOT_REPLAY_EXISTING_LONG_TRAINING'):
        coordinator.recover_ready_training(tmp_path)
    assert (tmp_path / 'PIPELINE_STARTED.json').exists()
    (train / 'train_metrics.jsonl').unlink()
    smoke = tmp_path / 'sft/smoke2'; smoke.mkdir()
    (smoke / 'partial.log').write_text('retained')
    coordinator.recover_ready_training(tmp_path)
    assert not (tmp_path / 'PIPELINE_STARTED.json').exists()
    assert (smoke / 'partial.log').read_text() == 'retained'
    receipt = next((tmp_path / 'recovery').glob('*/RECOVERY.json'))
    assert json.loads(receipt.read_text())['long_training_replayed'] is False


def test_skip_request_does_not_accept_missing_instruction_or_changed_budget(tmp_path):
    import json, pytest
    from scripts.train.run_mixed30k_continuation import verify_skip_sft_acceptance
    record = skip_request(tmp_path)
    assert len(verify_skip_sft_acceptance(tmp_path)) == 64
    for key, value in [('user_instruction', ''), ('sft_steps', 25000), ('skip_sft_gpu_acceptance', False)]:
        forged = dict(record); forged[key] = value
        (tmp_path / 'USER_SKIP_SFT_GPU_ACCEPTANCE.json').write_text(json.dumps(forged))
        with pytest.raises(RuntimeError, match='EXPLICIT_USER_REQUEST'):
            verify_skip_sft_acceptance(tmp_path)
