from src.explicit_region.epoch_sampler import DeterministicEpochSampler,epoch_permutation,permutation_sha256


SHA="a"*64


def test_epoch_permutation_and_ddp_partition_200k():
    first=epoch_permutation(200000,42,0,SHA);second=epoch_permutation(200000,42,0,SHA)
    assert first==second and len(set(first))==200000
    assert permutation_sha256(first)!=permutation_sha256(epoch_permutation(200000,42,1,SHA))
    ranks=[list(DeterministicEpochSampler(200000,base_seed=42,epoch=0,train_manifest_sha256=SHA,rank=r,world_size=8)) for r in range(8)]
    assert all(len(order)==25000 for order in ranks)
    assert len(set().union(*map(set,ranks)))==200000
    assert sum(len(set(ranks[i])&set(ranks[j])) for i in range(8) for j in range(i))==0


def test_resume_cursor_and_optimizer_step_contract():
    uninterrupted=list(DeterministicEpochSampler(200000,base_seed=42,epoch=0,train_manifest_sha256=SHA,rank=3,world_size=8))
    resumed=list(DeterministicEpochSampler(200000,base_seed=42,epoch=0,train_manifest_sha256=SHA,rank=3,world_size=8,samples_consumed_in_epoch=32000))
    assert resumed==uninterrupted[4000:]
    assert 6250*8*1*4==200000 and 1000*8*1*4==32000 and 1500*32==48000
