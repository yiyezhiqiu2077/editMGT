"""Explicit role-gradient averaging and fail-closed rank consistency checks."""
from __future__ import annotations

import hashlib
import json

import torch
import torch.distributed as dist


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def collective_require(condition, message, device):
    valid = torch.tensor(int(bool(condition)), device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not valid.item():
        raise RuntimeError(message)


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    header = json.dumps([str(value.dtype), list(value.shape)]).encode()
    return hashlib.sha256(header + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def state_digest(value):
    def encode(v):
        if isinstance(v, torch.Tensor):
            return {'tensor_sha256': tensor_digest(v)}
        if isinstance(v, dict):
            return {str(k): encode(x) for k, x in sorted(v.items(), key=lambda item: str(item[0]))}
        if isinstance(v, (list, tuple)):
            return [encode(x) for x in v]
        return v
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class ExplicitGradientSync:
    """Average FP32 buckets before each role optimizer, not after updates.

    Globally unused parameters keep grad=None; rank-local unused parameters
    contribute zero when another rank has a gradient. Every rank uses the same
    ordered parameter metadata and collective schedule.
    """

    def __init__(self, roles, bucket_bytes=16 * 1024 * 1024):
        self.roles = roles
        self.world_size = rank_world()[1]
        self.bucket_bytes = int(bucket_bytes)
        if self.bucket_bytes <= 0:
            raise ValueError('bucket_bytes must be positive')
        self.named = {role: sorted(roles.role_named_parameters(role))
                      for role in ('student', 'auxiliary')}
        metadata = {role: [(n, list(p.shape), str(p.dtype)) for n, p in items]
                    for role, items in self.named.items()}
        if self.world_size > 1:
            all_metadata = [None] * self.world_size
            dist.all_gather_object(all_metadata, metadata)
            if any(m != metadata for m in all_metadata):
                raise RuntimeError('DIMO_DISTRIBUTED_PARAMETER_ORDER_MISMATCH')

    @torch.no_grad()
    def __call__(self, role):
        items = self.named[role]
        device = items[0][1].device
        other_roles = ['teacher', 'auxiliary' if role == 'student' else 'student']
        isolated = all(p.grad is None or not bool(p.grad.ne(0).any())
                       for other in other_roles for _, p in self.roles.role_named_parameters(other))
        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for _, p in items)
        collective_require(isolated and finite, 'DIMO_ROLE_ISOLATION_OR_GRADIENT_FINITE_FAILURE', device)
        present = torch.tensor([int(p.grad is not None) for _, p in items], device=device, dtype=torch.int32)
        if self.world_size > 1:
            dist.all_reduce(present, op=dist.ReduceOp.SUM)
        usage = present.cpu().tolist()
        buckets, current, size = [], [], 0
        for (_, parameter), used in zip(items, usage):
            if not used:
                parameter.grad = None
                continue
            if current and size + parameter.numel() * 4 > self.bucket_bytes:
                buckets.append(current); current, size = [], 0
            current.append(parameter); size += parameter.numel() * 4
        if current:
            buckets.append(current)
        for parameters in buckets:
            flat = torch.cat([p.grad.detach().float().reshape(-1) if p.grad is not None
                              else torch.zeros(p.numel(), device=device, dtype=torch.float32)
                              for p in parameters])
            if self.world_size > 1:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                flat.div_(self.world_size)
            collective_require(bool(torch.isfinite(flat).all()), 'DIMO_AVERAGED_GRADIENT_NONFINITE', device)
            offset = 0
            for parameter in parameters:
                averaged = flat[offset:offset + parameter.numel()].reshape_as(parameter)
                if parameter.grad is None:
                    parameter.grad = averaged.to(parameter.dtype).clone()
                else:
                    parameter.grad.copy_(averaged)
                offset += parameter.numel()
        return {'role': role, 'gradient_sync': 'explicit_allreduce',
                'world_size': self.world_size, 'buckets': len(buckets), 'role_isolation': True}


def assert_rank_consistency(roles, ema, student_optimizer, auxiliary_optimizer,
                            student_scheduler, auxiliary_scheduler, step):
    row = {'step': int(step), 'student': state_digest(roles.role_state_dict('student')),
           'auxiliary': state_digest(roles.role_state_dict('auxiliary')),
           'teacher': state_digest(roles.role_state_dict('teacher')),
           'ema': state_digest(ema.state_dict()),
           'student_optimizer': state_digest(student_optimizer.state_dict()),
           'auxiliary_optimizer': state_digest(auxiliary_optimizer.state_dict()),
           'student_scheduler': state_digest(student_scheduler.state_dict()),
           'auxiliary_scheduler': state_digest(auxiliary_scheduler.state_dict())}
    world = rank_world()[1]
    ranks = [None] * world
    if world > 1:
        dist.all_gather_object(ranks, row)
    else:
        ranks[0] = row
    if any(r != row for r in ranks):
        raise RuntimeError('DIMO_SILENT_RANK_DESYNCHRONIZATION')
    return {'status': 'PASS', 'equivalence': 'bitwise_exact', 'world_size': world,
            'step': int(step), 'ranks': ranks}
