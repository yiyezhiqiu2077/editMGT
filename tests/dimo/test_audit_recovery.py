"""Recovery is bound to explicit consent; repeated launches stay forbidden."""
import json
import pytest
from src.explicit_region.contracts import sha256_file
from scripts.train.continue_magicbrush_dimo_after_audit import verify_recovery_request


def stopped(tmp_path):
    evidence={}
    values={
        'configs/dimo.yaml':{'unchanged':True},
        'TEACHER_SELECTION.json':{'selected_step':3843},
        'dimo/fresh20/failure_rank1.json':{'last_committed_step':19,'message':'LINEAR_SURROGATE_REAL_LOGIT_GRADIENT_FAILED: details'},
        'dimo/fresh20/failure_rank2.json':{'last_committed_step':19,'message':'LINEAR_SURROGATE_REAL_LOGIT_GRADIENT_FAILED: details'},
        'STATUS.json':{'status':'STOPPED_HARD_ERROR','phase':'dimo:fresh20:step20'}}
    for relative,value in values.items():
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));evidence[relative]=sha256_file(path)
    request={'schema':'magicbrush-dimo-audit-recovery-v1','user_instruction':'继续',
             'failed_status_sha256':evidence.pop('STATUS.json'),'evidence':evidence}
    return request


def test_recovery_preserves_failed_evidence_and_rejects_repeated_launch(tmp_path):
    request=stopped(tmp_path)
    assert verify_recovery_request(tmp_path,request)['status']=='STOPPED_HARD_ERROR'
    (tmp_path/'DIMO_AUDIT_RECOVERY_STARTED.json').write_text('{}')
    with pytest.raises(RuntimeError,match='ALREADY_STARTED'):
        verify_recovery_request(tmp_path,request)


@pytest.mark.parametrize('change',['consent','status','evidence','missing','failure'])
def test_recovery_rejects_mismatched_request(tmp_path,change):
    request=stopped(tmp_path)
    if change=='consent':request['user_instruction']=''
    elif change=='status':request['failed_status_sha256']='0'*64
    elif change=='evidence':(tmp_path/'configs/dimo.yaml').write_text('changed')
    elif change=='missing':del request['evidence']['dimo/fresh20/failure_rank1.json']
    elif change=='failure':
        path=tmp_path/'dimo/fresh20/failure_rank1.json'
        path.write_text(json.dumps({'last_committed_step':19,'message':'CUDA out of memory'}))
        request['evidence']['dimo/fresh20/failure_rank1.json']=sha256_file(path)
    with pytest.raises(RuntimeError,match='DIMO_AUDIT_RECOVERY'):
        verify_recovery_request(tmp_path,request)
