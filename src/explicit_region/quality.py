"""Stateful scientific-quality gate independent of process exit status."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics


@dataclass
class QualityGate:
    config: dict
    losses: list[tuple[int, float]] = field(default_factory=list)
    bad_windows: int = 0
    status: str = "INSUFFICIENT_BASELINE"
    reason: str | None = None
    clips: list[bool] = field(default_factory=list)

    @classmethod
    def from_state(cls, config, state):
        gate = cls(config)
        if state:
            gate.losses = [(int(a), float(b)) for a, b in state.get("losses", [])]
            gate.bad_windows = int(state.get("bad_windows", 0))
            gate.status = state.get("status", gate.status)
            gate.reason = state.get("reason")
            gate.clips = [bool(value) for value in state.get("clips", [])]
        return gate

    def update(self, step, *, loss, max_abs_logit, clip_applied, update_ratio, finite_parameters):
        if not math.isfinite(loss) or not finite_parameters:
            self.status, self.reason = "QUALITY_FAILED", "non_finite"
            return False
        if max_abs_logit > float(self.config.get("max_abs_logit", float("inf"))):
            self.status, self.reason = "QUALITY_FAILED", "max_abs_logit"
            return False
        if update_ratio > float(self.config.get("max_update_ratio", float("inf"))):
            self.status, self.reason = "QUALITY_FAILED", "max_update_ratio"
            return False
        self.losses.append((step, loss))
        self.clips.append(bool(clip_applied))
        clip_window = self.clips[-int(self.config.get("window_size", 50)):]
        if len(clip_window) == int(self.config.get("window_size", 50)) and sum(clip_window) / len(clip_window) > float(self.config.get("max_clip_fraction", 1.0)):
            self.status, self.reason = "QUALITY_FAILED", "clip_fraction"
            return False
        start, end = int(self.config["baseline_start"]), int(self.config["baseline_end"])
        baseline = [value for s, value in self.losses if start <= s <= end]
        if step <= end or len(baseline) < end - start + 1:
            self.status, self.reason = "INSUFFICIENT_BASELINE", None
            return True
        window_size = int(self.config["window_size"])
        recent = [value for _, value in self.losses[-window_size:]]
        if len(recent) == window_size and statistics.mean(recent) > statistics.mean(baseline) * float(self.config["max_loss_ratio"]):
            self.bad_windows += 1
        else:
            self.bad_windows = 0
        if self.bad_windows >= int(self.config["consecutive_bad_windows"]):
            self.status, self.reason = "QUALITY_FAILED", "loss_regression"
            return False
        self.status, self.reason = "QUALITY_OK", None
        return True

    def state_dict(self):
        return {"losses": self.losses, "clips": self.clips, "bad_windows": self.bad_windows, "status": self.status, "reason": self.reason}
