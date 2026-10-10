"""Synchronized wall-clock module phases; warmup and repetitions are explicit."""
from contextlib import contextmanager
import functools
import time

import torch


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def phase_measurements(calls, device):
    data = {name: {"seconds": 0., "calls": 0} for name in calls}
    originals = []
    try:
        for name, (owner, method) in calls.items():
            original = getattr(owner, method)
            originals.append((owner, method, original))
            def wrap(function, label):
                @functools.wraps(function)
                def measured(*args, **kwargs):
                    sync(device); start = time.perf_counter()
                    result = function(*args, **kwargs)
                    sync(device)
                    data[label]["seconds"] += time.perf_counter() - start
                    data[label]["calls"] += 1
                    return result
                return measured
            setattr(owner, method, wrap(original, name))
        yield data
    finally:
        for owner, method, original in originals:
            setattr(owner, method, original)


def repeated_latency(operation, calls, device, *, warmup=1, repeats=3):
    if warmup < 0 or repeats < 1:
        raise ValueError("invalid latency repetition protocol")
    for _ in range(warmup):
        operation()
    totals, phases, outputs = [], [], []
    for _ in range(repeats):
        with phase_measurements(calls, device) as record:
            sync(device); start = time.perf_counter()
            output = operation()
            sync(device)
            totals.append(time.perf_counter() - start)
        phases.append(record); outputs.append(output)
    return outputs[-1], {"warmup": warmup, "repeats": repeats, "cuda_synchronized": torch.device(device).type == "cuda",
        "total_seconds": totals, "mean_seconds": sum(totals) / len(totals), "phases": phases,
        "bounds": "generation_call_includes_encoding_transformer_decoding_excludes_dataset_geometry_and_image_io"}
