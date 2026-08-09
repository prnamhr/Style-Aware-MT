"""
Operating characteristic of the register-drift stop rule.

Usage:
    python manage.py drift_oc                      # the candidates compared on 2026-08-09
    python manage.py drift_oc --runs 5000 --sigma_prompt 1.4
"""

from __future__ import annotations

import argparse

import numpy as np

from src.rlsf.config import drift_rule, load_config
from src.rlsf.stop import DriftMonitor, DriftRule

# Backed out of the smoke's clustered z_se for marker_rate: 0.295 over 20 prompts.
DEFAULT_SIGMA_PROMPT = 0.295 * 20**0.5
SHIFT_AT = 50

# (baseline_steps, window, k_sigma). The configured rule is prepended at run time.
CANDIDATES = ((5, 3, 3.0), (5, 3, 1.5), (20, 3, 4.0), (20, 5, 4.0), (30, 5, 4.0))


def _mu(t: int, delta: float, ramp: bool) -> float:
    """Drift path: a step change at SHIFT_AT, or a linear climb to delta by the last step."""
    if t < SHIFT_AT:
        return 0.0
    return delta * (t - SHIFT_AT) / (500 - SHIFT_AT) if ramp else delta


def trip_step(rule: DriftRule, rng, delta: float, *, steps: int, prompts: int,
              sigma_prompt: float, ramp: bool = False) -> int | None:
    """The step the rule halts a simulated run at, or None if it runs to the end."""
    monitor = DriftMonitor(rule)
    for t in range(steps):
        draws = rng.normal(_mu(t, delta, ramp), sigma_prompt, prompts)
        verdict = monitor.update({
            "step": t,
            "z": {rule.feature: draws.mean()},
            "z_se": {rule.feature: draws.std(ddof=1) / np.sqrt(prompts)},
        })
        if verdict.tripped:
            return t
    return None


def profile(rule: DriftRule, scenarios, *, runs: int, seed: int, **kw) -> list[tuple]:
    """Trip rate and median trip step per scenario, each scenario on the same seed."""
    out = []
    for delta, ramp in scenarios:
        rng = np.random.default_rng(seed)
        trips = [trip_step(rule, rng, delta, ramp=ramp, **kw) for _ in range(runs)]
        hit = [t for t in trips if t is not None]
        out.append((len(hit) / runs, int(np.median(hit)) if hit else None))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the register-drift stop rule.")
    parser.add_argument("--config", default="configs/rlsf.yaml")
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=500, help="rollouts per simulated run")
    parser.add_argument("--prompts", type=int, default=None, help="default: config rollout")
    parser.add_argument("--sigma_prompt", type=float, default=DEFAULT_SIGMA_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config, require_caps=False)
    configured = drift_rule(cfg)
    prompts = args.prompts or cfg["rlsf"]["rollout"]["prompts_per_step"]
    per_step_sd = args.sigma_prompt / np.sqrt(prompts)

    scenarios = [(0.0, False), (0.23, False), (0.35, False), (0.5, True)]
    labels = ["null", "step +0.23", "step +0.35", "ramp +0.50"]

    rules = [configured]
    rules += [
        DriftRule(feature=configured.feature, baseline_steps=b, window=w, k_sigma=k,
                  min_delta=configured.min_delta)
        for b, w, k in CANDIDATES
        if (b, w, k) != (configured.baseline_steps, configured.window, configured.k_sigma)
    ]

    print(f"{prompts} prompts/step, per-prompt sd {args.sigma_prompt:.3f} -> per-step sd "
          f"{per_step_sd:.3f}; {args.runs} runs of {args.steps} steps, seed {args.seed}")
    print(f"\n{'rule':<24} {'band':>6}  " + "  ".join(f"{name:>12}" for name in labels))
    for i, rule in enumerate(rules):
        se = per_step_sd * np.sqrt(1 / rule.baseline_steps + 1 / rule.window)
        cells = profile(rule, scenarios, runs=args.runs, seed=args.seed, steps=args.steps,
                        prompts=prompts, sigma_prompt=args.sigma_prompt)
        name = (f"b{rule.baseline_steps} w{rule.window} k{rule.k_sigma}"
                + (" (configured)" if i == 0 else ""))
        row = [f"{rate:>7.1%}" + (f"@{med:<4}" if med is not None else "     ")
               for rate, med in cells]
        print(f"{name:<24} {max(rule.min_delta, rule.k_sigma * se):>6.3f}  " + "  ".join(row))
    print("\ncell: halt rate @ median halt step. The drift starts at step "
          f"{SHIFT_AT}; a halt under 'null' is a false alarm.")


if __name__ == "__main__":
    main()
