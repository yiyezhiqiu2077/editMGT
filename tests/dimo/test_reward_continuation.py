import copy
import json
from pathlib import Path
import pytest
import torch
from src.dimo.reward_continuation import continuation_manifest, validate_continuation, load_continuation
from src.dimo.full_checkpoint import save, load
from src.dimo.recipe import recipe_identity
from src.dimo.transaction import StepCommit
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.contracts import sha256_file
from tests.dimo.test_soft_reward import objective, step
from tests.dimo.test_v2_joint import bundle, config


def prepared(tmp_path):
    roles, opts, schedules, ema = bundle()
    engine = objective()
    step(roles, opts, schedules, ema, engine, 0)
    old = config()
    old.update(output_dir=str(tmp_path/'old'), checkpoint_interval_steps=2000)
    old['soft_reward']={'enabled':True,'calibration':{'batches':1,'relative_scale':.1,'max_weight':100.},
                        'max_new_steps':500,'inference_output_dir':str(tmp_path/'old_exports')}
    identity={'recipe':'dimo-v2','recipe_contract':recipe_identity(old,roles),
              'frozen_parameter_hashes':roles.frozen_parameter_hashes(),'reward_extension':old['soft_reward'],
              'config_sha256':recipe_fingerprint(old),'git_sha':'old-code'}
    root=tmp_path/'old/checkpoints/checkpoint-1'
    root.parent.parent.mkdir(parents=True)
    (root.parent.parent/'resolved_config.json').write_text(json.dumps(old))
    cursor=identity['recipe_contract']['budget']['samples_per_optimizer_step']
    save(root,roles=roles,optimizers=opts,schedulers=schedules,ema=ema,step=1,cursor=cursor,
         identity=identity,config_hash=identity['config_sha256'],commit=StepCommit(1,cursor),
         sampler_state={'epoch':0,'samples_consumed_in_epoch':cursor},
         loader_rng={'current':torch.Generator().get_state()},
         runtime_stats={'student_clip_count':1,'reward_calibration':engine.calibration.state})
    auth=tmp_path/'authorization.json'
    auth.write_text(json.dumps({'selected_complete_resume_checkpoint':str(root),
        'long_target_step':old['max_optimizer_steps'],'checkpoint_frequency':2000}))
    new=copy.deepcopy(old)
    new['output_dir']=str(tmp_path/'long')
    new['soft_reward']['inference_output_dir']=str(tmp_path/'long_exports')
    new['soft_reward_continuation']=continuation_manifest(root,auth)
    new_id=copy.deepcopy(identity)
    new_id.update(config_sha256=recipe_fingerprint(new),git_sha='new-code',
                  reward_extension=new['soft_reward'],reward_continuation=new['soft_reward_continuation'])
    return roles,opts,schedules,ema,engine,root,new,new_id


def test_authorized_extension_preserves_exact_next_update_and_future_resume(tmp_path):
    roles,opts,schedules,ema,engine,source,new,new_id=prepared(tmp_path)
    checksum=sha256_file(source/'COMPLETED.json')
    r2,o2,s2,e2=bundle();engine2=objective()
    engine2.decoder.load_state_dict(engine.decoder.state_dict())
    state=load_continuation(source,config=new,new_identity=new_id,roles=r2,optimizers=o2,schedulers=s2,ema=e2)
    assert state['step']==e2.update_count==1
    assert state['sampler_state']['samples_consumed_in_epoch']==state['cursor']
    assert all(v['step'].item()==1 for o in o2 for v in o.state.values())
    engine2.calibration.load_state_dict(state['runtime_stats']['reward_calibration'])
    step(roles,opts,schedules,ema,engine,1);step(r2,o2,s2,e2,engine2,1)
    for role in ['student','auxiliary']:
        assert all(torch.equal(v,r2.model_for(role).state_dict()[k])
                   for k,v in roles.model_for(role).state_dict().items())
    assert all(torch.equal(v,e2.shadow[k]) for k,v in ema.shadow.items())
    assert engine2.calibration.state==engine.calibration.state
    assert sha256_file(source/'COMPLETED.json')==checksum
    target=tmp_path/'long/checkpoints/checkpoint-2'
    cursor=state['cursor']*2
    save(target,roles=r2,optimizers=o2,schedulers=s2,ema=e2,step=2,cursor=cursor,identity=new_id,
         config_hash=new_id['config_sha256'],commit=StepCommit(2,cursor),
         sampler_state={'epoch':0,'samples_consumed_in_epoch':cursor},
         runtime_stats={'student_clip_count':2,'reward_calibration':engine2.calibration.state})
    r3,o3,s3,e3=bundle()
    resumed=load(target,roles=r3,optimizers=o3,schedulers=s3,ema=e3,
                 expected_identity=new_id,expected_config_hash=new_id['config_sha256'])
    assert resumed['step']==e3.update_count==2
    assert all(v['step'].item()==2 for o in o3 for v in o.state.values())


@pytest.mark.parametrize('change',['optimizer','seed','reward','output','authorization'])
def test_extension_rejects_recipe_drift_overlap_or_changed_authorization(tmp_path,change):
    *_,source,new,new_id=prepared(tmp_path)
    if change=='optimizer':new['optimizer']['student_learning_rate']=2e-6
    if change=='seed':new['seed']+=1
    if change=='reward':new['soft_reward']['calibration']['relative_scale']=.2
    if change=='output':new['output_dir']=str(source.parent.parent/'overwrite')
    if change=='authorization':
        p=Path(new['soft_reward_continuation']['authorization']);p.write_text(p.read_text()+'\n')
    with pytest.raises(RuntimeError):validate_continuation(new,new_id)
