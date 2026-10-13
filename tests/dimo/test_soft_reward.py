import copy
import json
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from src.dimo.soft_reward import (SoftEmbeddingEditDecoder, DifferentiableMaskedLPIPS,
    EditingRewards, RewardGradientCalibration, SoftRewardObjective, crop_preprocess, roi_boxes)
from src.dimo.soft_reward import deterministic_antialias_resize
from src.dimo.reward_fork import fork_manifest, fork_weights, protect_outputs
from src.dimo.full_checkpoint import save, load
from src.dimo.recipe import recipe_identity
from src.dimo.transaction import StepCommit
from src.dimo.step import complete_dimo_step
from src.dimo.synthetic import SyntheticDataset, synthetic_v2_batch
from tests.dimo.test_v2_joint import bundle, config


class TinyDecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 3, 1)
    def forward(self, z, spatial=None):
        return self.conv(z).sigmoid()


class TinyVQ(nn.Module):
    def __init__(self, size=5):
        super().__init__()
        self.quantize = nn.Module()
        self.quantize.embedding = nn.Embedding(size, 3)
        self.post_quant_conv = nn.Conv2d(3, 3, 1)
        self.decoder = TinyDecoder(3)
        self.config = SimpleNamespace(norm_type="group")


class TinyLPIPSFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
    def forward(self, x):
        return (self.conv(x),)


class TinyLPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = TinyLPIPSFeatures()
        self.lins = nn.ModuleList([nn.Conv2d(4, 1, 1, bias=False)])
        nn.init.constant_(self.lins[0].weight, .25)
        self.scaling_layer = nn.Identity()


class TinyFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 7)
    def get_image_features(self, pixel_values):
        return self.proj(pixel_values.mean((-1,-2)))
    def forward(self, pixel_values):
        return SimpleNamespace(last_hidden_state=self.get_image_features(pixel_values)[:,None])


class TinyRewards(nn.Module):
    """Small distinct smooth rewards for full transaction and resume tests."""
    def __init__(self):
        super().__init__()
        self.register_buffer('scales', torch.tensor([1.,2.,3.]))
    def forward(self, generated, target, mask):
        weight = mask[:,None] if mask.ndim == 3 else mask
        return {k:((generated-target).square()*weight).sum()/weight.sum()/3*self.scales[i]
                for i,k in enumerate(('lpips','dino','clip'))}


def objective(batches=1):
    return SoftRewardObjective(SoftEmbeddingEditDecoder(TinyVQ(), codebook_size=5), TinyRewards(),
        {'calibration': {'batches':batches,'relative_scale':.1,'max_weight':100.}, 'gap_interval':1})


def test_soft_decoder_numerical_gradient_and_outside_lock():
    torch.manual_seed(7)
    decoder = SoftEmbeddingEditDecoder(TinyVQ().double(), codebook_size=5)
    logits = torch.randn(1,2,2,6,dtype=torch.double,requires_grad=True)
    source = torch.tensor([[[0,1],[2,3]]]); region = torch.tensor([[[True,False],[True,False]]])
    assert torch.autograd.gradcheck(lambda z: decoder(z,source,region), (logits,), atol=1e-5)
    z = decoder.embeddings(logits, source, region).permute(0,2,3,1)
    expected = decoder.vqvae.quantize.embedding(source)
    assert torch.equal(z[~region],expected[~region])
    decoder(logits,source,region).sum().backward()
    assert logits.grad[region][...,:5].norm()>0
    assert torch.count_nonzero(logits.grad[~region])==0
    assert torch.count_nonzero(logits.grad[...,5])==0
    assert all(p.grad is None and not p.requires_grad for p in decoder.parameters())


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_precision_one_hot_limit_hard_consistency_and_invalid_codes(dtype):
    decoder = SoftEmbeddingEditDecoder(TinyVQ(),codebook_size=5)
    source = torch.tensor([[[0,1],[2,3]]]); region = torch.ones_like(source,dtype=torch.bool)
    logits = torch.full((1,2,2,6),-40.,dtype=dtype)
    logits.scatter_(-1,source[...,None],40.)
    logits.requires_grad_()
    with torch.autocast('cpu',dtype=torch.bfloat16): image = decoder(logits,source,region)
    assert image.dtype == torch.float32
    torch.testing.assert_close(image,decoder.hard(source),rtol=1e-5,atol=1e-6)
    image.sum().backward();assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError): decoder(logits,source+10,region)
    random=torch.randn(1,2,2,6,dtype=dtype,requires_grad=True)
    decoder(random,source,region).sum().backward()
    assert torch.isfinite(random.grad).all() and random.grad.norm()>0


def test_masked_lpips_matches_eval_aggregation_and_gradcheck():
    from src.explicit_region.metrics import MaskedLPIPS
    model = TinyLPIPS().double()
    differentiable = DifferentiableMaskedLPIPS(model)
    original = object.__new__(MaskedLPIPS); original.model=model
    a = torch.rand(1,3,4,4,dtype=torch.double,requires_grad=True)
    b = torch.rand_like(a); mask=torch.zeros(1,1,4,4,dtype=torch.double);mask[:,:,1:3,1:3]=1
    torch.testing.assert_close(differentiable(a,b,mask), original(a,b,mask)[0].mean())
    assert torch.autograd.gradcheck(lambda x:differentiable(x,b,mask),(a,),atol=1e-5)
    assert all(not p.requires_grad for p in model.parameters())


def test_shared_roi_preprocessing_identical_target_and_small_roi_semantics():
    a=torch.rand(2,3,40,60,requires_grad=True);mask=torch.zeros(2,1,40,60);mask[:,:,15:20,30:35]=1
    boxes=roi_boxes(mask)
    spec=dict(shortest_edge=24,crop_size=16,mean=[.4]*3,std=[.2]*3)
    torch.testing.assert_close(crop_preprocess(a,boxes,**spec),crop_preprocess(a.clone(),boxes,**spec))
    assert boxes[0] == (22,7,43,28)
    with pytest.raises(ValueError):roi_boxes(torch.zeros_like(mask))
    rewards=EditingRewards(TinyLPIPS(),TinyFeatures(),TinyFeatures(),lpips_resolution=32)
    state={k:v.clone() for k,v in rewards.state_dict().items()}
    values=rewards(a,torch.rand_like(a),mask)
    for v in values.values():
        grad=torch.autograd.grad(v,a,retain_graph=True)[0];assert torch.isfinite(grad).all() and grad.norm()>0
    sum(values.values()).backward()
    assert all(p.grad is None and not p.requires_grad for p in rewards.parameters())
    assert all(torch.equal(v,state[k]) for k,v in rewards.state_dict().items())


@pytest.mark.parametrize('mode', ['bilinear', 'bicubic'])
@pytest.mark.parametrize('size', [(3, 5), (13, 19)])
def test_deterministic_antialias_forward_backward_and_numerical_gradient(mode, size):
    image = torch.rand(1, 2, 7, 11, dtype=torch.double, requires_grad=True)
    reference = torch.nn.functional.interpolate(image, size=size, mode=mode,
        align_corners=False, antialias=True)
    actual = deterministic_antialias_resize(image, size, mode=mode)
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    upstream = torch.randn_like(actual)
    expected_grad = torch.autograd.grad(reference, image, upstream)[0]
    actual_grad = torch.autograd.grad(actual, image, upstream)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)
    assert torch.autograd.gradcheck(lambda x: deterministic_antialias_resize(x, size, mode=mode),
        (image,), atol=1e-6)


def test_gradient_calibration_uses_norms_not_loss_values_and_resume():
    c=RewardGradientCalibration(batches=2)
    weights,warmup=c.weights_for_step(torch.tensor([10.,2.,5.,1.]));assert warmup and not any(weights.values())
    other=RewardGradientCalibration(batches=2);other.load_state_dict(c.state)
    c.weights_for_step(torch.tensor([10.,2.,5.,1.]));other.weights_for_step(torch.tensor([10.,2.,5.,1.]))
    assert c.state==other.state
    assert c.state['weights']==pytest.approx({'lpips':.5,'dino':.2,'clip':1.})
    assert c.weights_for_step(torch.tensor([100.,1.,1.,1.]))[0]==c.state['weights']
    with pytest.raises(RuntimeError):RewardGradientCalibration().weights_for_step(torch.tensor([1.,0.,1.,1.]))
    with pytest.raises(ValueError):other.load_state_dict({'count':2,'norm_sums':[1.]*4,'weights':None})


def step(roles, opts, schedules, ema, engine, index):
    c=config();c['soft_reward']={'enabled':True}
    a=synthetic_v2_batch([index*2],SyntheticDataset());b=synthetic_v2_batch([index*2+1],SyntheticDataset())
    target=torch.full((1,3,4,4),.8)
    return complete_dimo_step(roles,**a,student_optimizer=opts[0],auxiliary_optimizer=opts[1],
        student_scheduler=schedules[0],auxiliary_scheduler=schedules[1],student_ema=ema,
        base_seed=43,epoch=0,committed_dimo_step=index,mask_token_id=5,codebook_size=5,
        config=c,fresh_batch=b,reward_objective=engine,reward_batch={'target':target,'pixel_mask':a['edit_region_mask']})


def test_reward_backward_single_student_step_fresh_b_and_ema():
    roles,opts,schedules,ema=bundle();engine=objective()
    state={k:v.clone() for k,v in engine.decoder.state_dict().items()}
    frozen=roles.frozen_parameter_hashes();teacher={k:v.clone() for k,v in roles.model_for('teacher').state_dict().items()}
    first,_=step(roles,opts,schedules,ema,engine,0)
    second,artifacts=step(roles,opts,schedules,ema,engine,1)
    assert first['reward_calibration_only'] and not second['reward_calibration_only']
    assert all(second['reward_logits_grad_norms'][k]>0 for k in ['linear','lpips','dino','clip'])
    assert second['student_grad_norm_postclip']<=10.00001
    assert second['student_sample_uids']!=second['auxiliary_sample_uids']
    assert ema.update_count==2
    assert all(v['step'].item()==2 for o in opts for v in o.state.values())
    roles.assert_frozen_unchanged(frozen);engine.assert_frozen()
    assert all(torch.equal(v,teacher[k]) for k,v in roles.model_for('teacher').state_dict().items())
    assert all(torch.equal(v,state[k]) for k,v in engine.decoder.state_dict().items())
    assert 'soft_hard_gap' in second and all(torch.isfinite(torch.tensor(x)) for x in second['soft_hard_gap'].values())


def test_reward_resume_preserves_calibration_and_exact_next_update(tmp_path):
    roles,opts,sch,ema=bundle();engine=objective(batches=2)
    step(roles,opts,sch,ema,engine,0)
    ident={'recipe':'dimo-v2','recipe_contract':recipe_identity(config(),roles),
        'frozen_parameter_hashes':roles.frozen_parameter_hashes(),
        'reward_extension':{'calibration':{'batches':2,'relative_scale':.1,'max_weight':100.}}}
    stats={'student_clip_count':0,'reward_calibration':engine.calibration.state}
    root=tmp_path/'state'
    cursor=ident['recipe_contract']['budget']['samples_per_optimizer_step']
    save(root,roles=roles,optimizers=opts,schedulers=sch,ema=ema,step=1,cursor=cursor,identity=ident,
        config_hash='reward',commit=StepCommit(1,cursor),runtime_stats=stats,
        sampler_state={'epoch':0,'samples_consumed_in_epoch':cursor},loader_rng={'current':torch.Generator().get_state()})
    next_engine=objective(batches=2)
    next_engine.decoder.load_state_dict(engine.decoder.state_dict())
    r2,o2,s2,e2=bundle()
    resumed=load(root,roles=r2,optimizers=o2,schedulers=s2,ema=e2,expected_identity=ident,expected_config_hash='reward')
    next_engine.calibration.load_state_dict(resumed['runtime_stats']['reward_calibration'])
    step(roles,opts,sch,ema,engine,1);step(r2,o2,s2,e2,next_engine,1)
    assert next_engine.calibration.state==engine.calibration.state
    for role in ['student','auxiliary']:
        for k,v in roles.model_for(role).state_dict().items():assert torch.equal(v,r2.model_for(role).state_dict()[k])
    assert all(torch.equal(v,e2.shadow[k]) for k,v in ema.shadow.items())
    with pytest.raises(RuntimeError,match='IDENTITY'):
        load(root,roles=r2,optimizers=o2,schedulers=s2,ema=e2,expected_identity={**ident,'reward_extension':{}},expected_config_hash='reward')


def test_weights_only_fork_retains_roles_and_ema_but_resets_optimizer(tmp_path):
    roles,opts,sch,ema=bundle();old={'recipe':'dimo-v2','recipe_contract':recipe_identity(config(),roles),
        'frozen_parameter_hashes':roles.frozen_parameter_hashes(),'teacher_bundle_fingerprint':'teacher','data_sha256':'data'}
    root=tmp_path/'old'/'checkpoints'/'checkpoint-0'
    save(root,roles=roles,optimizers=opts,schedulers=sch,ema=ema,step=0,cursor=0,identity=old,config_hash='old',
         commit=StepCommit(),runtime_stats={'student_clip_count':0},sampler_state={'epoch':0,'samples_consumed_in_epoch':0})
    before=(root/'COMPLETED.json').read_bytes();r2,o2,s2,e2=bundle()
    fork_weights(root,r2,o2,e2,fork_manifest(root),old)
    assert all(not o.state for o in o2) and e2.update_count==0
    assert (root/'COMPLETED.json').read_bytes()==before
    with pytest.raises(RuntimeError,match='OVERLAPS'):protect_outputs(root.parent.parent/'reward',root)
    protect_outputs(tmp_path/'new',root)
    with pytest.raises(RuntimeError,match='TEACHER_OR_DATA'):
        fork_weights(root,r2,o2,e2,fork_manifest(root),{**old,'data_sha256':'wrong'})


def test_soft_hard_gap_direction_and_shared_target_reference():
    from scripts.eval.evaluate_soft_dimo_reward import decode_gap_delta
    keys=['inside_masked_lpips','roi_clip_to_target','roi_dino_to_target']
    def metrics(value):
        return {'per_sample_after_seed_mean':[{'sample_uid':str(i),'cluster_id':str(i),
            'metrics':dict(zip(keys,value))} for i in range(8)]}
    # Hard stays constant; soft alone gets better. All three gaps increase.
    hard=metrics([.6,.8,.5]);soft0=metrics([.6,.8,.5]);soft1=metrics([.5,.9,.6])
    gap=decode_gap_delta(hard,soft1,hard,soft0,{'seed':42,'resamples':100})
    assert all(v['mean_delta']==pytest.approx(.1) for v in gap.values())
    same=decode_gap_delta(hard,soft0,hard,soft0,{'seed':42,'resamples':100})
    assert all(v['mean_delta']==0 for v in same.values())


def test_short_gate_refuses_failed_quality_or_mismatched_weights(tmp_path):
    from scripts.train.run_soft_dimo_reward import validate_short_gate
    from src.explicit_region.contracts import sha256_file
    root=tmp_path; (root/'experiment_manifest.json').write_text('{}')
    for kind in ['checkpoints','inference']:
        folder=root/kind/'checkpoint-20';folder.mkdir(parents=True);(folder/'COMPLETED.json').write_text('{}')
    proof={'status':'PASS','gpu':'REAL_GPU','step':20,'run_manifest_sha256':sha256_file(root/'experiment_manifest.json'),
        'checkpoint_completed_sha256':sha256_file(root/'checkpoints/checkpoint-20/COMPLETED.json'),
        'inference_completed_sha256':sha256_file(root/'inference/checkpoint-20/COMPLETED.json')}
    gate={'status':'STOP_REVIEW','checkpoint_step':20,'run_manifest_sha256':proof['run_manifest_sha256'],
        'checkpoint_completed_sha256':proof['inference_completed_sha256']}
    (root/'GPU_SMOKE_PASS.json').write_text(json.dumps(proof));(root/'evaluation/soft_reward').mkdir(parents=True)
    path=root/'evaluation/soft_reward/QUALITY_GATE_step20.json';path.write_text(json.dumps(gate))
    with pytest.raises(RuntimeError,match='GATE_FAILED'):validate_short_gate(root)
    gate['status']='PASS';path.write_text(json.dumps(gate));validate_short_gate(root)
    (root/'inference/checkpoint-20/COMPLETED.json').write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match='BINDING_CHANGED'):validate_short_gate(root)
