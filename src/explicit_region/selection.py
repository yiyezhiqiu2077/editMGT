"""Pure deterministic checkpoint-ranking helpers shared by formal and dev flows."""
from __future__ import annotations

import math


def summarize_selection_metrics(evaluation: dict) -> dict[str, float | None]:
    rows = evaluation.get("per_sample_after_seed_mean", [])
    result = {}
    for name in ("inside_masked_lpips", "outside_masked_lpips", "inside_psnr", "inside_ssim"):
        values = [float(row["metrics"][name]) for row in rows
                  if row.get("metrics", {}).get(name) is not None
                  and math.isfinite(float(row["metrics"][name]))]
        result[name] = sum(values) / len(values) if values else None
    result["samples"] = len(rows)
    return result


def rank_candidates(candidates: list[dict], *, max_outside_masked_lpips: float | None) -> dict:
    if not candidates:
        raise ValueError("no checkpoint candidates")
    normalized = []
    for candidate in candidates:
        metrics = candidate["metrics"]
        inside, outside = metrics.get("inside_masked_lpips"), metrics.get("outside_masked_lpips")
        if inside is None or outside is None or not all(math.isfinite(float(x)) for x in (inside, outside)):
            continue
        passed = max_outside_masked_lpips is not None and float(outside) <= max_outside_masked_lpips
        normalized.append(dict(candidate, quality_gate_passed=passed))
    if not normalized:
        raise RuntimeError("no finite checkpoint candidates")
    eligible = [row for row in normalized if row["quality_gate_passed"]]
    pool = eligible or normalized
    ranked = sorted(pool, key=lambda row: (
        float(row["metrics"]["inside_masked_lpips"]),
        float(row["metrics"]["outside_masked_lpips"]),
        str(row["candidate_id"]),
    ))
    winner = ranked[0]
    return {
        "winner": winner,
        "quality_gate_passed": bool(eligible),
        "selection_reason": "quality_gate_rank" if eligible else "dev_pipeline_fallback",
        "ranked_candidate_ids": [row["candidate_id"] for row in ranked],
    }
