import torch
import pytest
from scripts.diagnostics.surrogate_precision import linear, squared, analytical, compare
from scripts.diagnostics.dimo_step0_runtime import state_digest


def test_linear_exact_roi_vocab_batch_reduction_and_detach():
    z=torch.full((2,4,7),10.,requires_grad=True)
    g=torch.full_like(z,1e-8,requires_grad=True)
    mask=torch.tensor([[True,False,False,False],[True,True,True,False]])
    actual=torch.autograd.grad(linear(z,g,mask),(z,g),allow_unused=True)
    assert actual[1] is None
    assert torch.allclose(actual[0].double(),analytical(g,mask),atol=0,rtol=1e-7)
    assert float(linear(z,g,mask)) == pytest.approx(7e-7)


def test_squared_cancels_nonzero_small_gradient_but_linear_preserves_it():
    z=torch.full((1,2,7),20.,requires_grad=True)
    g=torch.full_like(z,1e-8)
    mask=torch.tensor([[True,False]])
    a=torch.autograd.grad(squared(z,g,mask),z)[0]
    b=torch.autograd.grad(linear(z,g,mask),z)[0]
    assert torch.count_nonzero(a)==0
    assert compare(b,analytical(g,mask))['nonzero_reference_computed_zero_fraction']==0


def test_fp64_squared_matches_reference_and_empty_sample_rejected():
    z=torch.full((1,2,7),20.,dtype=torch.float64,requires_grad=True)
    g=torch.full_like(z,1e-8)
    mask=torch.tensor([[True,False]])
    a=torch.autograd.grad(squared(z,g,mask,double=True),z)[0]
    assert torch.allclose(a,analytical(g,mask),atol=1e-14,rtol=1e-6)
    with pytest.raises(ValueError):linear(z,g,torch.zeros_like(mask))


def test_weight_integrity_hash_handles_bf16_singleton_dimension_strides():
    transposed=torch.arange(8).to(torch.bfloat16).reshape(1,8).t()
    ordinary=torch.arange(8).to(torch.bfloat16).reshape(8,1)
    assert transposed.is_contiguous() and transposed.stride(-1)!=1
    assert state_digest({'weight':transposed})==state_digest({'weight':ordinary})
    assert state_digest({'weight':ordinary})!=state_digest({'weight':ordinary.float()})
