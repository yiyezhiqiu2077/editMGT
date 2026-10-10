import copy
from pathlib import Path
import pytest
import torch
from src.explicit_region.config import _load_unexpanded
from src.explicit_region.dense_runtime import validate_dense_config
from src.dimo.runtime import validate_full_config
from src.dimo.magicbrush_selection import choose
from scripts.data.audit_magicbrush_training_input import session_subset
from scripts.dimo.merge_e3_dense import compare_logits


def configs():
    return (_load_unexpanded(Path('configs/train/dense_magicbrush_4a800_10epoch.yaml')),
            _load_unexpanded(Path('configs/dimo/full_dense_magicbrush_4a800_10epoch.yaml')))


def test_modes_admit_registered_recipes_and_legal_session_budget():
    s,d=configs();validate_dense_config(s);validate_full_config(d)
    s['data']['expected_rows']=8784;s['max_optimizer_steps']=s['scheduler_horizon_steps']=5490
    s['checkpoint_steps']=[e*549 for e in (1,2,3,5,7,10)]
    d['data']['expected_rows']=8784;d['max_optimizer_steps']=d['optimizer']['horizon_steps']=21960
    validate_dense_config(s);validate_full_config(d)


@pytest.mark.parametrize('field,value',[('world_size',8),('epochs',1),('max_optimizer_steps',25000)])
def test_magicbrush_cannot_inherit_d200k_budget(field,value):
    _,d=configs()
    if field=='world_size':d['distributed'][field]=value
    else:d[field]=value
    with pytest.raises(RuntimeError):validate_full_config(d)


def test_magicbrush_cannot_inherit_half_random_initialization_or_formal_flag():
    _,d=configs();d['initialization']={'mask_ratio':.5,'random_tokens_inside_roi':True}
    with pytest.raises(RuntimeError):validate_full_config(d)
    _,d=configs();d['formal_ready']=True
    with pytest.raises(RuntimeError):validate_full_config(d)
    _,d=configs()
    with pytest.raises(RuntimeError):validate_full_config(d,synthetic=True)


def test_original_full_d200k_gate_unchanged():
    d=_load_unexpanded(Path('configs/dimo/full_dense.yaml'))
    d['data']={'expected_rows':200000,'name':'fixed_200k'}
    validate_full_config(d)
    d['distributed']['world_size']=4
    with pytest.raises(RuntimeError):validate_full_config(d)
    s=_load_unexpanded(Path('configs/train/dense_d200k_5epoch.yaml'))
    validate_dense_config(s,formal=True)
    s['scheduler_horizon_steps']=5500
    with pytest.raises(RuntimeError):validate_dense_config(s,formal=True)


def candidate(step,inside=-.005,ci=-.001,outside=.01):
    return {'step':step,'inside_mean':.5,'identity_protocol_valid':True,'numerical_roi_valid':True,
            'paired_inside':{'mean_delta':inside,'bootstrap_95_ci':[-.02,ci]},
            'paired_outside':{'mean_delta':outside}}


def test_exploratory_selection_exact_boundaries_and_fallback():
    assert choose([candidate(550)])['step']==550
    assert choose([candidate(550,ci=0.)]) is None
    assert choose([candidate(550,inside=-.004999)]) is None
    assert choose([candidate(550,outside=.010001)]) is None
    assert choose([candidate(1100),candidate(550)])['step']==550
    c=candidate(550);c['numerical_roi_valid']=False
    assert choose([c]) is None


def test_invalid_turn_excludes_entire_session_then_only_whole_singles():
    rows=[{'split':'train','group_id':str(i),'sample_key':str(i),'valid':True} for i in range(8807)]
    rows[0]['valid']=False;rows[1]['group_id']='0'
    result=session_subset(rows)
    assert result['kept_rows']==8800 and result['excluded_rows']==7
    assert '0' in result['excluded_sample_keys'] and '1' in result['excluded_sample_keys']
    assert result['padding'] is False and result['contains_invalid_rows'] is False


def test_greedy_parity_compares_vocabulary_axis():
    a=torch.tensor([[[[1.,2.,3.],[1.,2.,3.]]]])
    b=a.clone();b[...,0]=4
    assert compare_logits(a,a,'fp32')['greedy_token_agreement']==1.
    assert compare_logits(a,b,'fp32')['greedy_token_agreement']==0.


@pytest.mark.parametrize('cursor',[0,4,8,12])
def test_epoch_boundary_and_step0_loader_resume_preserves_sequence_and_rng(cursor):
    from src.explicit_region.dense_runtime import DenseEpochLoader
    from src.explicit_region.epoch_sampler import DeterministicEpochSampler
    class Data(torch.utils.data.Dataset):
        def __len__(self):return 8
        def set_epoch(self,e):self.epoch=e
        def __getitem__(self,i):return torch.tensor([self.epoch,i])
    def create(state=None):
        ds=Data();epoch=state['epoch'] if state else 0
        sampler=DeterministicEpochSampler(8,base_seed=43,epoch=epoch,train_manifest_sha256='a'*64,
            world_size=2,rank=0,samples_consumed_in_epoch=state['cursor'] if state else 0)
        generator=torch.Generator().manual_seed(10043)
        if state:generator.set_state(state['rng'])
        loader=DenseEpochLoader(ds,sampler,epochs=3,batch_size=1,num_workers=0,generator=generator,
            resume_epoch_start_rng=state['start'] if state else None)
        return sampler,generator,loader
    sf,gf,lf=create();fresh=[]
    for batch in lf:
        fresh.append(batch[0].tolist());sf.samples_consumed_in_epoch+=2
    sp,gp,lp=create();prefix=[]
    it=iter(lp)
    if cursor==0:
        next(it) # bootstrap step0 materializes the loader without committing.
    else:
        for _ in range(cursor//2):
            prefix.append(next(it)[0].tolist());sp.samples_consumed_in_epoch+=2
    state={'epoch':sp.epoch,'cursor':sp.samples_consumed_in_epoch,'rng':gp.get_state().clone(),'start':lp.epoch_start_rng.clone()}
    sr,gr,lr=create(state);suffix=[]
    for batch in lr:
        suffix.append(batch[0].tolist());sr.samples_consumed_in_epoch+=2
    assert prefix+suffix==fresh
    assert torch.equal(gr.get_state(),gf.get_state())


def test_coordinator_hard_error_stops_peer_process_group(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import threading
    from argparse import Namespace
    from scripts.train.run_magicbrush_two_stage import Run
    run=Run(Namespace(experiment_root=tmp_path,upload_curves=False,monitor_token_file=None))
    peer=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
    with run.children_lock:run.children.add(peer)
    try:
        with pytest.raises(subprocess.CalledProcessError):
            run.command([sys.executable,'-c','raise SystemExit(17)'],'expected_failure')
        assert run.failed.is_set()
        assert peer.wait(timeout=5)==-signal.SIGTERM
        with pytest.raises(RuntimeError,match='PEER_PROCESS_FAILED'):
            run.command([sys.executable,'-c','pass'],'must_not_start')
    finally:
        if peer.poll() is None:os.killpg(peer.pid,signal.SIGKILL)
        peer.wait()


def test_pipeline_disk_budget_rejects_before_gpu_and_retains_evidence(tmp_path,monkeypatch):
    from collections import namedtuple
    from scripts.train import run_magicbrush_two_stage as coordinator
    (tmp_path/'admission').mkdir()
    usage=namedtuple('Usage','total used free')
    monkeypatch.setattr(coordinator.shutil,'disk_usage',lambda _:usage(100,99,1))
    with pytest.raises(RuntimeError,match='DISK_BUDGET_FAILED'):
        coordinator.pipeline_disk_budget(tmp_path,1009307648)
    assert '"status": "FAIL"' in (tmp_path/'admission/DISK_BUDGET.json').read_text()


def test_lightweight_report_includes_dependencies_but_excludes_weights(tmp_path):
    import tarfile
    from scripts.report.build_experiment_report import build_report
    stage=tmp_path/'stage';(stage/'code').mkdir(parents=True)
    (stage/'code/pyproject.toml').write_text('[project]\nname="audit"\n')
    (stage/'code/uv.lock').write_text('version = 1\n')
    (stage/'code/model.safetensors').write_bytes(b'weight')
    (stage/'REPORT.md').write_text('NOT_TESTED\n')
    archive=tmp_path/'report.tar.gz';build_report({'e3':stage},archive,stage='e3')
    with tarfile.open(archive) as handle:
        names=handle.getnames()
    assert 'e3/code/pyproject.toml' in names and 'e3/code/uv.lock' in names
    assert all('safetensors' not in n for n in names)
