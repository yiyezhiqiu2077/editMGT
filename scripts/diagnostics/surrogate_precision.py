"""Diagnosis-only gradient injection; never imported by the training path."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import torch


def _reduce(values, mask):
    if mask.shape != values.shape:
        raise ValueError('mask must match every non-vocabulary dimension')
    mask = mask.bool().flatten(1)
    counts = mask.sum(1)
    if (counts == 0).any():
        raise ValueError('each sample requires a selected token')
    return (values.flatten(1).masked_fill(~mask, 0).sum(1) / counts).mean()


def linear(z, g, mask, *, double=False):
    """Vocabulary SUM, per-sample selected-token mean, then batch mean."""
    if z.shape != g.shape:
        raise ValueError('logit/gradient shapes differ')
    dtype = torch.float64 if double else torch.float32
    return _reduce((z.to(dtype) * g.detach().to(dtype)).sum(-1), mask)


def squared(z, g, mask, *, double=False):
    dtype = torch.float64 if double else torch.float32
    z, g = z.to(dtype), g.to(dtype)
    target = (z - g).detach()
    return _reduce(0.5 * (z - target).square().sum(-1), mask)


def analytical(g, mask):
    counts = mask.flatten(1).sum(1)
    shape = [g.shape[0]] + [1] * (g.ndim - 1)
    return g.detach().double().masked_fill(~mask.bool()[..., None], 0) / counts.reshape(shape) / g.shape[0]


def compare(actual, expected):
    actual, expected = actual.double(), expected.double()
    diff = actual - expected
    nz = expected != 0
    rn, an = float(expected.norm()), float(actual.norm())
    return {
        'expected_gradient_norm': rn, 'actual_gradient_norm': an,
        'gradient_norm_ratio': an / rn if rn else None,
        'max_absolute_error': float(diff.abs().max()),
        'relative_l2_error': float(diff.norm()) / rn if rn else None,
        'max_relative_error_nonzero': float((diff[nz].abs()/expected[nz].abs()).max()) if nz.any() else None,
        'gradient_cosine': float((actual*expected).sum())/(an*rn) if an and rn else None,
        'nonzero_reference_count': int(nz.sum()),
        'nonzero_reference_computed_zero_fraction': float((actual[nz] == 0).double().mean()) if nz.any() else None,
        'nonfinite_actual_count': int((~torch.isfinite(actual)).sum()),
        'zero_reference_nonzero_actual_count': int((actual[~nz] != 0).sum()),
    }


def audit(z, g, mask, context=None, include_bf16=True):
    """Detach real logits: autograd touches only independent logit leaves."""
    expected = analytical(g, mask)
    rows = []
    modes = [('fp64_squared', torch.float64, squared, True),
             ('fp32_squared', torch.float32, squared, False),
             ('fp32_linear', torch.float32, linear, False)]
    if include_bf16:
        modes += [('bf16_leaf_squared', torch.bfloat16, squared, False),
                  ('bf16_leaf_linear', torch.bfloat16, linear, False)]
    for name, dtype, objective, double in modes:
        leaf = z.detach().to(dtype).clone().requires_grad_(True)
        loss = objective(leaf, g, mask, double=double)
        actual = torch.autograd.grad(loss, leaf)[0]
        row = dict(context or {}) | {'objective': name, 'leaf_dtype': str(dtype),
              'raw_g_norm': float(g.double().norm()), 'loss_scalar': float(loss),
              'selected_tokens': mask.flatten(1).sum(1).tolist(), 'batch_size': len(mask),
              'reference_definition': 'sg(g)/(batch_size * selected_tokens_per_sample)'}
        row.update(compare(actual, expected))
        row['outside_roi_nonzero_actual_count']=int(actual.masked_select((~mask.bool())[...,None].expand_as(actual)).count_nonzero())
        # Quantization at a BF16 logit boundary is separate from FP32 cancellation.
        row['relative_l2_error_vs_dtype_rounded_reference'] = compare(actual, expected.to(dtype))['relative_l2_error']
        rows.append(row)
    return rows


def synthetic():
    rows = []
    for magnitude, gscale, ratio, batch in itertools.product(
            [1., 5., 10., 20.], [10.**(-i) for i in range(2,9)], [.03125,.25,1.], [1,2,4]):
        rng = torch.Generator().manual_seed(823)
        # Controlled absolute magnitudes, avoiding a near-zero logit tail.
        z = (0.5 + 0.5*torch.rand(batch,32,64,generator=rng)) * magnitude
        z *= torch.where(torch.rand(batch,32,64,generator=rng)<.5, -1., 1.)
        g = torch.randn(batch,32,64,generator=rng) * gscale
        mask = torch.zeros(batch,32,dtype=torch.bool)
        for b in range(batch):
            count = max(1, min(32, round(32*ratio) + b))
            mask[b, torch.randperm(32,generator=rng)[:count]] = True
        rows += audit(z,g,mask,{'kind':'synthetic','logit_magnitude':magnitude,
                    'g_magnitude':gscale,'roi_fraction_requested':ratio},include_bf16=False)
    return rows


def write_csv(path, rows):
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    rows=synthetic()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'surrogate_synthetic.json').write_text(json.dumps({'rows':rows},indent=2)+'\n')
    write_csv(args.output/'surrogate_synthetic.csv',rows)
    print(f'SYNTHETIC_COMPLETE {len(rows)} rows',flush=True)


if __name__=='__main__':main()
