"""Two-rank CPU reward transaction/checkpoint replay (invoked explicitly)."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tests.dimo.test_v2_joint import bundle, config
from tests.dimo.test_soft_reward import objective
from src.dimo.synthetic import SyntheticDataset, synthetic_v2_batch
from src.dimo.step import complete_dimo_step
from src.dimo.recipe import recipe_identity
from src.dimo.transaction import StepCommit
from src.dimo.full_checkpoint import save,load


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--authorized-continuation',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);dist.init_process_group('gloo');rank=dist.get_rank()
    c=config();c['soft_reward']={'enabled':True}
    roles,opts,sch,ema=bundle();roles.wrap_ddp();engine=objective(batches=2)
    def update(roles,opts,sch,ema,engine,index):
        a=synthetic_v2_batch([index*4+rank],SyntheticDataset())
        b=synthetic_v2_batch([index*4+2+rank],SyntheticDataset())
        return complete_dimo_step(roles,**a,student_optimizer=opts[0],auxiliary_optimizer=opts[1],
            student_scheduler=sch[0],auxiliary_scheduler=sch[1],student_ema=ema,base_seed=43,
            epoch=0,committed_dimo_step=index,mask_token_id=5,codebook_size=5,config=c,
            fresh_batch=b,reward_objective=engine,
            reward_batch={'target':torch.full((1,3,4,4),.8),'pixel_mask':a['edit_region_mask']})
    update(roles,opts,sch,ema,engine,0)
    ident={'recipe':'dimo-v2','recipe_contract':recipe_identity(config(),roles),
        'frozen_parameter_hashes':roles.frozen_parameter_hashes(),
        'reward_extension':{'calibration':{'batches':2,'relative_scale':.1,'max_weight':100.}}}
    root=Path(args.output);checkpoint=root/'checkpoint-1'
    state_hash='reward-ddp'
    if args.authorized_continuation:
        from src.explicit_region.checkpoint import recipe_fingerprint
        c.update(output_dir=str(root/'old'),checkpoint_interval_steps=2000)
        c['soft_reward']=dict(ident['reward_extension'],enabled=True,max_new_steps=500,
                             inference_output_dir=str(root/'old_exports'))
        ident.update(reward_extension=c['soft_reward'],config_sha256=recipe_fingerprint(c),git_sha='old-code')
        state_hash=ident['config_sha256'];checkpoint=root/'old/checkpoints/checkpoint-1'
        if rank==0:
            checkpoint.parent.parent.mkdir(parents=True)
            (checkpoint.parent.parent/'resolved_config.json').write_text(json.dumps(c))
        dist.barrier()
    save(checkpoint,roles=roles,optimizers=opts,schedulers=sch,ema=ema,step=1,cursor=4,
         identity=ident,config_hash=state_hash,commit=StepCommit(1,4),
         runtime_stats={'student_clip_count':0,'reward_calibration':engine.calibration.state},
         sampler_state={'epoch':0,'samples_consumed_in_epoch':4},loader_rng={'current':torch.Generator().get_state()})
    restored,ro,rs,re=bundle();restored.wrap_ddp();other=objective(batches=2)
    other.decoder.load_state_dict(engine.decoder.state_dict())
    if args.authorized_continuation:
        from src.dimo.reward_continuation import continuation_manifest,load_continuation
        from src.explicit_region.checkpoint import recipe_fingerprint
        auth=root/'authorization.json'
        if rank==0:
            auth.write_text(json.dumps({'selected_complete_resume_checkpoint':str(checkpoint.resolve()),
                'long_target_step':c['max_optimizer_steps'],'checkpoint_frequency':2000}))
        dist.barrier()
        new=copy.deepcopy(c);new['output_dir']=str(root/'long')
        new['soft_reward']['inference_output_dir']=str(root/'long_exports')
        new['soft_reward_continuation']=continuation_manifest(checkpoint,auth)
        new_ident=dict(ident,config_sha256=recipe_fingerprint(new),git_sha='new-code',
                       reward_extension=new['soft_reward'],reward_continuation=new['soft_reward_continuation'])
        state=load_continuation(checkpoint,config=new,new_identity=new_ident,
                               roles=restored,optimizers=ro,schedulers=rs,ema=re)
    else:
        state=load(checkpoint,roles=restored,optimizers=ro,schedulers=rs,ema=re,
            expected_identity=ident,expected_config_hash='reward-ddp')
    other.calibration.load_state_dict(state['runtime_stats']['reward_calibration'])
    update(roles,opts,sch,ema,engine,1);update(restored,ro,rs,re,other,1)
    for role in ['student','auxiliary']:
        for k,v in roles.model_for(role).state_dict().items():assert torch.equal(v,restored.model_for(role).state_dict()[k])
    assert engine.calibration.state==other.calibration.state
    assert all(torch.equal(v,re.shadow[k]) for k,v in ema.shadow.items())
    digest=hashlib.sha256(b''.join(v.detach().contiguous().numpy().tobytes() for r in ['student','auxiliary']
        for v in roles.model_for(r).state_dict().values())).hexdigest()
    evidence={'rank':rank,'parameters_sha256':digest,'calibration':engine.calibration.state}
    all_rows=[None]*dist.get_world_size();dist.all_gather_object(all_rows,evidence)
    assert len({r['parameters_sha256'] for r in all_rows})==1
    assert all(r['calibration']==all_rows[0]['calibration'] for r in all_rows)
    # A disconnected reward on one rank must collectively abort before any
    # student optimizer update, rather than stranding other ranks in backward.
    if rank==1:engine.rewards.scales[0]=0
    try:
        update(roles,opts,sch,ema,engine,2)
        raise AssertionError('missing reward gradient was not rejected')
    except RuntimeError as exc:
        assert 'REWARD_LOGITS_GRADIENT_MISSING_OR_NONFINITE' in str(exc)
    assert all(v['step'].item()==2 for o in opts for v in o.state.values())
    if rank==0:
        report={'status':'PASS','evidence':'CPU_SYNTHETIC_DDP_NOT_GPU',
            'world_size':2,'checkpoint_next_step_bitwise_replay':True,
            'rank_parameters_equal':True,'calibration_equal':True,'ranks':all_rows}
        report['single_rank_reward_failure_collective_abort']=True
        report['authorized_full_state_continuation']=args.authorized_continuation
        (root/'CPU_DDP_PASS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    dist.destroy_process_group()


if __name__=='__main__':main()
