"""
Paired-bootstrap confidence intervals for pairwise condition comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def paired_bootstrap(
    a: list[float],
    b: list[float],
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Paired bootstrap of ``mean(a) - mean(b)`` over aligned per-segment scores.

    ``a`` and ``b`` must be aligned segment-for-segment; pairs where either value
    is NaN (e.g. an unparsed judge verdict) are dropped before resampling.
    Returns observed difference, the ``[alpha/2, 1-alpha/2]`` CI, a two-sided
    bootstrap p-value, the paired sample size, and per-condition means.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"length mismatch: {arr_a.shape} vs {arr_b.shape}")
    keep = ~(np.isnan(arr_a) | np.isnan(arr_b))
    arr_a, arr_b = arr_a[keep], arr_b[keep]
    n = arr_a.shape[0]
    if n == 0:
        raise ValueError("no paired segments to bootstrap")

    observed = float(arr_a.mean() - arr_b.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = arr_a[idx].mean(axis=1) - arr_b[idx].mean(axis=1)

    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    # Two-sided bootstrap p-value: twice the smaller tail mass at 0.
    p = 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return {
        "n": n,
        "mean_a": float(arr_a.mean()),
        "mean_b": float(arr_b.mean()),
        "diff": observed,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": min(p, 1.0),
        "significant": not (lo <= 0.0 <= hi),
    }


def _nan_segments(raw: list) -> list[float]:
    """Map a stored per-segment list (which may contain nulls) to floats/NaN."""
    return [float("nan") if v is None else float(v) for v in raw]


def _load_segment_scores(metric: str, conditions: list[str], out_dir: Path, split: str) -> dict:
    """Return ``{condition: per-segment score list}`` for the chosen metric.

    Surface metrics are computed directly from the inference files; COMET/judge
    are read from the JSON that their own commands wrote.
    """
    metric = metric.lower()
    if metric in ("chrf", "bleu"):
        from src.eval._io import condition_path, load_condition
        from src.eval.quick import segment_scores

        scores = {}
        for cond in conditions:
            if not condition_path(out_dir, cond, split).exists():
                print(f"skip {cond}: inference file not found")
                continue
            _, preds, refs = load_condition(out_dir, cond, split)
            scores[cond] = segment_scores(preds, refs, metric)
        return scores

    if metric in ("comet", "judge"):
        path = Path("results") / f"{metric}_{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `manage.py {metric}` first")
        stored = json.loads(path.read_text(encoding="utf-8"))
        return {c: _nan_segments(stored[c]["segments"]) for c in conditions if c in stored}

    raise ValueError(f"unknown metric '{metric}' (expected chrf|bleu|comet|judge)")


def _pairs(conditions: list[str], baseline: str, adjacent: bool) -> list[tuple[str, str]]:
    """Comparison pairs as ``(a, b)`` reporting ``mean(a) - mean(b)``."""
    out = [(c, baseline) for c in conditions if c != baseline]
    if adjacent:
        for lo, hi in zip(conditions, conditions[1:]):
            if (hi, lo) not in out:
                out.append((hi, lo))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired-bootstrap CIs for condition pairs.")
    parser.add_argument("--metric", required=True, choices=["chrf", "bleu", "comet", "judge"])
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--split", default="val", help="output split tag (default: val)")
    parser.add_argument("--out_dir", default="outputs", help="inference output directory")
    parser.add_argument("--baseline", default=None, help="reference condition (default: first)")
    parser.add_argument(
        "--adjacent", action="store_true", help="also compare each consecutive ladder pair"
    )
    parser.add_argument("--n_resamples", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scores = _load_segment_scores(args.metric, args.conditions, Path(args.out_dir), args.split)
    present = [c for c in args.conditions if c in scores]
    if len(present) < 2:
        print("need at least two conditions with scores to compare")
        return

    # Alignment is required for paired resampling: identical eval order and count.
    counts = {c: len(scores[c]) for c in present}
    if len(set(counts.values())) != 1:
        raise ValueError(f"conditions differ in segment count: {counts}")

    baseline = args.baseline or present[0]
    if baseline not in scores:
        raise ValueError(f"baseline '{baseline}' has no scores")

    rows = []
    for a, b in _pairs(present, baseline, args.adjacent):
        res = paired_bootstrap(
            scores[a], scores[b], n_resamples=args.n_resamples, alpha=args.alpha, seed=args.seed
        )
        rows.append(
            {
                "comparison": f"{a} - {b}",
                "n": res["n"],
                "diff": round(res["diff"], 3),
                f"ci{int((1 - args.alpha) * 100)}": f"[{res['ci_low']:.3f}, {res['ci_high']:.3f}]",
                "p": round(res["p_value"], 4),
                "sig": "*" if res["significant"] else "",
            }
        )

    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print(f"\n{args.metric} paired bootstrap  (resamples={args.n_resamples}, split={args.split})")
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    print("\n* = 95% CI excludes 0 (difference significant at α={:.2f})".format(args.alpha))


if __name__ == "__main__":
    main()
