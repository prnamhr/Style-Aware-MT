"""
Register-drift stop rule: halt a run whose marker rate leaves the regime it started in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.eval.stylometrics import CENTROID_FEATURES


@dataclass
class DriftRule:
    """Where the drift band sits, in centroid standard deviations."""

    feature: str = "marker_rate"
    baseline_steps: int = 20
    window: int = 5
    k_sigma: float = 4.0
    min_delta: float = 0.23

    def __post_init__(self) -> None:
        if self.feature not in CENTROID_FEATURES:
            raise ValueError(f"feature must be one of {CENTROID_FEATURES}, got {self.feature!r}")
        if self.baseline_steps < 1:
            raise ValueError(f"baseline_steps must be >= 1, got {self.baseline_steps}")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if self.k_sigma <= 0:
            raise ValueError(f"k_sigma must be positive, got {self.k_sigma}")
        if self.min_delta <= 0:
            raise ValueError(
                f"min_delta must be positive, got {self.min_delta}: without a floor the rule "
                f"halts on any drift a long run can resolve, however small"
            )

    @property
    def first_testable_step(self) -> int:
        """Steps that must have been logged before the rule can fire at all."""
        return self.baseline_steps + self.window


@dataclass
class DriftVerdict:
    """One step's reading of the rule."""

    step: int
    tripped: bool
    reason: str
    baseline: float = float("nan")
    window_mean: float = float("nan")
    delta: float = float("nan")
    threshold: float = float("nan")

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "tripped": self.tripped,
            "reason": self.reason,
            "baseline": self.baseline,
            "window_mean": self.window_mean,
            "delta": self.delta,
            "threshold": self.threshold,
        }


def _pooled(values: list[float], ses: list[float]) -> tuple[float, float]:
    """Mean of a set of step means and its standard error, steps taken as independent."""
    n = len(values)
    mean = sum(values) / n
    return mean, math.sqrt(sum(se * se for se in ses)) / n


class DriftMonitor:
    """Watches one centroid feature against the baseline the run itself sets."""

    def __init__(self, rule: DriftRule | None = None) -> None:
        self.rule = rule or DriftRule()
        self._z: list[float] = []
        self._se: list[float] = []

    @property
    def baseline(self) -> tuple[float, float]:
        """Mean and standard error of the baseline window; nan until it is complete."""
        r = self.rule
        if len(self._z) < r.baseline_steps:
            return float("nan"), float("nan")
        return _pooled(self._z[: r.baseline_steps], self._se[: r.baseline_steps])

    def update(self, log) -> DriftVerdict:
        """Record a step's StepLog (or its dict form) and read the rule against it."""
        d = log.as_dict() if hasattr(log, "as_dict") else log
        if "z_se" not in d:
            raise KeyError(
                "the step log carries no z_se, so the rule has no error to size its band "
                "against and would never fire. Log written before z_step_stats?"
            )
        name = self.rule.feature
        self._z.append(float(d["z"][name]))
        self._se.append(float(d["z_se"][name]))
        return self._verdict(int(d.get("step", len(self._z) - 1)))

    def _verdict(self, step: int) -> DriftVerdict:
        r = self.rule
        if len(self._z) < r.first_testable_step:
            return DriftVerdict(
                step,
                False,
                f"{len(self._z)}/{r.first_testable_step} steps logged: baseline and window "
                f"are not complete",
            )

        base, base_se = self.baseline
        window_mean, window_se = _pooled(self._z[-r.window :], self._se[-r.window :])
        delta = window_mean - base
        se = math.hypot(base_se, window_se)
        if math.isnan(se):
            return DriftVerdict(
                step,
                False,
                f"no clustered error for {r.feature}: a step in the baseline or window has "
                f"fewer than two prompts",
                base,
                window_mean,
                delta,
            )

        threshold = max(r.min_delta, r.k_sigma * se)
        # Sustained: one spiking step cannot carry the window mean over on its own.
        sustained = all(z > base + r.min_delta for z in self._z[-r.window :])
        tripped = sustained and delta >= threshold
        if tripped:
            reason = (
                f"{r.feature} sat {delta:+.2f} above the run's opening {base:+.2f} for "
                f"{r.window} consecutive steps, past the {threshold:.2f} band"
            )
        elif not sustained:
            reason = f"{r.feature} is not above {base + r.min_delta:+.2f} in every window step"
        else:
            reason = (
                f"{r.feature} is {delta:+.2f} above the opening {base:+.2f}, inside the "
                f"{threshold:.2f} band"
            )
        return DriftVerdict(step, tripped, reason, base, window_mean, delta, threshold)
