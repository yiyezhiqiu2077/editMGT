"""Explicit exploratory selection binds real checkpoint contents without DEV claims."""
import copy
import json
import pytest
from tests.test_dense_pipeline import ToyDense, payload
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dense_checkpoint import save_dense_checkpoint
from src.dimo.magicbrush_selection import user_select, verify_selection_record, export_teacher
from src.dimo.dense_teacher import load_dense_teacher_contract, dense_teacher_fingerprint


def selection(tmp_path):
    train=tmp_path/'train.jsonl';train.write_text('{}\n'*16)
    dev=tmp_path/'dev.jsonl';dev.write_text('{}\n'*2)
    p=payload()
    p.update(experiment_mode='magicbrush_exploratory',
             data_content_hashes={'data.manifest':{'sha256':sha256_file(train)}})
    for step in (0,7):
        save_dense_checkpoint(tmp_path/f'checkpoint-{step}',model=ToyDense(),fingerprint_payload=p,
                              global_optimizer_step=step)
    source=tmp_path/'checkpoint-7'
    plan={'train_manifest':str(train),'dev_manifest':str(dev),'dev32_manifest':str(dev),
          'baseline_checkpoint':str(tmp_path/'checkpoint-0'),'candidate_checkpoints':{'7':str(source)}}
    record=user_select(plan,source,epoch=7,instruction='Use epoch7 for DiMO',authorization_sha256='a'*64)
    return plan,source,record


def test_user_selected_teacher_export_is_content_bound_and_never_formal(tmp_path):
    _,source,record=selection(tmp_path)
    assert record['selection_status']=='USER_SELECTED'
    assert record['full_dev_selection_performed'] is False
    assert record['selection_preregistration'] is None
    export_teacher(record,tmp_path/'teacher')
    contract=load_dense_teacher_contract(tmp_path/'teacher')
    assert contract.formal_teacher is False
    assert contract.manifest['teacher_source']=='dense_sft_user_selected'
    assert contract.manifest['dense_checkpoint_sha256']==record['selected_checkpoint_hash']
    with pytest.raises(RuntimeError,match='NOT_SELECTED'):
        dense_teacher_fingerprint(tmp_path/'teacher',expected_base_identity=contract.manifest['base_model_identity'],formal=True)


@pytest.mark.parametrize('field,value',[('formal_teacher',True),('selection_status','SELECTED'),
    ('selected_epoch',8),('full_dev_selection_performed',True),('selected_step',8)])
def test_user_selection_rejects_rehashed_false_claims(tmp_path,field,value):
    _,_,record=selection(tmp_path)
    record[field]=value
    record['record_sha256']=recipe_fingerprint({k:v for k,v in record.items() if k!='record_sha256'})
    with pytest.raises(RuntimeError):verify_selection_record(record,record['selected_checkpoint_hash'])


def test_user_selection_binds_checkpoint_request_and_data(tmp_path):
    plan,source,record=selection(tmp_path)
    with pytest.raises(RuntimeError,match='EPOCH_MISMATCH'):
        user_select(plan,source,epoch=8,instruction='Use epoch8',authorization_sha256='a'*64)
    with pytest.raises(RuntimeError,match='RECORD_INVALID'):
        verify_selection_record(record,'b'*64)
    with open(plan['train_manifest'],'a') as handle:handle.write('{}\n')
    with pytest.raises(RuntimeError):
        user_select(plan,source,epoch=7,instruction='Use epoch7',authorization_sha256='a'*64)


def test_user_selection_cannot_hide_missing_authorization(tmp_path):
    plan,source,_=selection(tmp_path)
    with pytest.raises(RuntimeError,match='RECORD_INVALID'):
        user_select(plan,source,epoch=7,instruction='',authorization_sha256='a'*64)
    with pytest.raises(RuntimeError,match='RECORD_INVALID'):
        user_select(plan,source,epoch=7,instruction='Use epoch7',authorization_sha256='not-a-hash')
