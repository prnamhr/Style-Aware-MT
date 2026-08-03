"""
Bootstrap confidence intervals on the register fit of the seven main conditions.

    python manage.py stylometrics_ci
    python manage.py stylometrics_ci --conditions zeroshot peft --split val
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from src.eval._io import condition_path
from src.eval.stylometrics import (
    CENTROID_FEATURES,
    FEATURE_NAMES,
    aggregate,
    bootstrap_draws,
    distance_to_centroid,
    draw_intervals,
    feature_vector,
    signed_z,
)

_CENTROID_PATH = Path("results/stylometrics_centroid.json")

# The headline ladder, in reporting order: prompt-only baselines, then retrieval,
# then AFSP, then the fine-tuned system, then the commercial reference point.
MAIN_CONDITIONS = [
    "zeroshot",
    "random_fewshot",
    "knn_fewshot",
    "afsp_margin",
    "afsp_full",
    "peft",
    "commercial_haiku",
]


def _load_condition(out_dir: Path, condition: str, split: str) -> tuple[list[str], list[str]]:
    """Return ``(sources, predictions)`` for one condition's inference file."""
    sources: list[str] = []
    predictions: list[str] = []
    with condition_path(out_dir, condition, split).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sources.append(row.get("input", ""))
            predictions.append(row.get("prediction", ""))
    return sources, predictions


def _assert_aligned(present: list[str], sources: dict[str, list[str]]) -> None:
    """Fail if conditions' segments don't line up index-for-index."""
    ref_cond = present[0]
    ref_src = sources[ref_cond]
    for cond in present[1:]:
        src = sources[cond]
        if src != ref_src:
            i = next(k for k, (x, y) in enumerate(zip(ref_src, src)) if x != y)
            raise ValueError(
                f"source mismatch between '{ref_cond}' and '{cond}' at segment {i}: the paired "
                f"bootstrap requires identical source order across conditions"
            )


def _feature_matrix(texts: list[str]) -> np.ndarray:
    return np.asarray([feature_vector(t) for t in texts if t.strip()], dtype=float)


def _assert_no_blank_drop(condition: str, texts: list[str], matrix: np.ndarray) -> None:
    """Blank predictions are dropped from the feature matrix, which would desync
    the shared resample indices. Refuse to pair rather than silently misalign."""
    if matrix.shape[0] != len(texts):
        dropped = len(texts) - matrix.shape[0]
        raise ValueError(
            f"'{condition}' has {dropped} blank prediction(s); they drop out of the feature "
            f"matrix and break index-for-index pairing across conditions"
        )


def score_condition(
    texts: list[str], centroid: dict, *, n_resamples: int, alpha: float, seed: int
) -> tuple[dict, np.ndarray]:
    """Point estimates plus percentile CIs for one condition, and its dist draws."""
    agg = aggregate(texts)
    dists, z = bootstrap_draws(_feature_matrix(texts), centroid, n_resamples=n_resamples, seed=seed)
    row = {
        "n": agg["n"],
        "stylo_dist": distance_to_centroid(agg["mean"], centroid),
        "z": signed_z(agg["mean"], centroid),
        "mean": {name: agg["mean"][name] for name in FEATURE_NAMES},
        **draw_intervals(dists, z, centroid, alpha=alpha),
    }
    return row, dists


def paired_diff(a_draws: np.ndarray, b_draws: np.ndarray, *, alpha: float) -> dict:
    """Paired-bootstrap interval for ``stylo_dist(a) - stylo_dist(b)``.

    Negative means ``a`` sits closer to the target register than ``b``.
    """
    d = a_draws - b_draws
    lo = float(np.percentile(d, 100 * alpha / 2))
    hi = float(np.percentile(d, 100 * (1 - alpha / 2)))
    # Two-sided bootstrap p-value: twice the smaller tail mass at 0 (matches
    # src/eval/bootstrap.py, so the two tables read on the same scale).
    p = 2.0 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0)))
    return {
        "diff": float(d.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "p_value": min(p, 1.0),
        "significant": not (lo <= 0.0 <= hi),
    }


def rank_distribution(draws: dict[str, np.ndarray], present: list[str]) -> dict[str, dict]:
    """Per-condition distribution of stylo_dist rank across the shared resamples."""
    matrix = np.stack([draws[c] for c in present], axis=1)  # (n_resamples, n_conditions)
    order = matrix.argsort(axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, len(present) + 1), axis=1)

    out: dict[str, dict] = {}
    for i, cond in enumerate(present):
        counts = np.bincount(ranks[:, i], minlength=len(present) + 2)[1 : len(present) + 1]
        probs = (counts / ranks.shape[0]).tolist()
        out[cond] = {
            "rank_probs": probs,
            "modal_rank": int(np.argmax(counts) + 1),
            "modal_prob": float(max(probs)),
            "mean_rank": float(ranks[:, i].mean()),
        }
    return out


def build(
    out_dir: Path,
    split: str,
    *,
    conditions: list[str],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    centroid = json.loads(_CENTROID_PATH.read_text(encoding="utf-8"))

    loaded: dict[str, tuple[list[str], list[str]]] = {}
    for cond in conditions:
        path = condition_path(out_dir, cond, split)
        if not path.exists():
            print(f"skip {cond}: {path} not found")
            continue
        loaded[cond] = _load_condition(out_dir, cond, split)

    present = [c for c in conditions if c in loaded]
    if not present:
        raise FileNotFoundError("no requested condition has an inference file")

    counts = {c: len(loaded[c][1]) for c in present}
    if len(set(counts.values())) != 1:
        raise ValueError(f"conditions differ in segment count: {counts}")
    _assert_aligned(present, {c: loaded[c][0] for c in present})

    rows: dict[str, dict] = {}
    draws: dict[str, np.ndarray] = {}
    for cond in present:
        texts = loaded[cond][1]
        _assert_no_blank_drop(cond, texts, _feature_matrix(texts))
        rows[cond], draws[cond] = score_condition(
            texts, centroid, n_resamples=n_resamples, alpha=alpha, seed=seed
        )

    ranked = sorted(present, key=lambda c: rows[c]["stylo_dist"])
    for i, cond in enumerate(ranked, start=1):
        rows[cond]["rank"] = i

    ranks = rank_distribution(draws, present)
    for cond in present:
        rows[cond]["rank_distribution"] = ranks[cond]

    # Every pair, ordered better-rank-first, so `diff` is negative when the
    # observed ordering holds in the resamples too.
    pairs = [
        {"a": a, "b": b, **paired_diff(draws[a], draws[b], alpha=alpha)}
        for a, b in itertools.combinations(ranked, 2)
    ]
    adjacent = [
        {"a": a, "b": b, **paired_diff(draws[a], draws[b], alpha=alpha)}
        for a, b in zip(ranked, ranked[1:])
    ]

    return {
        "split": split,
        "out_dir": str(out_dir),
        "centroid": {"features": centroid["features"], "n_segments": centroid["n_segments"]},
        "bootstrap": {
            "n_resamples": n_resamples,
            "seed": seed,
            "alpha": alpha,
            "paired": True,
            "n_segments": counts[present[0]],
        },
        "conditions": present,
        "ranking": ranked,
        "cells": {c: rows[c] for c in present},
        "paired_adjacent": adjacent,
        "paired_all": pairs,
    }


def _ci(bounds: list[float], places: int = 4) -> str:
    return f"[{bounds[0]:.{places}f}, {bounds[1]:.{places}f}]"


def _table(rows: list[dict], cols: list[str]) -> None:
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def _print_summary(report: dict) -> None:
    boot = report["bootstrap"]
    pct = int(round((1 - boot["alpha"]) * 100))
    ci_col = f"ci{pct}"
    ranked = report["ranking"]
    cells = report["cells"]

    print(
        f"\nRegister fit of the main conditions  (split={report['split']}, "
        f"n={boot['n_segments']} segments, resamples={boot['n_resamples']}, seed={boot['seed']})"
    )
    print("stylo_dist = standardized distance to the target-register centroid; lower is better.\n")

    rows = []
    for cond in ranked:
        c = cells[cond]
        rd = c["rank_distribution"]
        rows.append(
            {
                "rank": str(c["rank"]),
                "condition": cond,
                "stylo_dist": f"{c['stylo_dist']:.4f}",
                ci_col: _ci(c["stylo_dist_ci"]),
                "P(this rank)": f"{rd['rank_probs'][c['rank'] - 1]:.3f}",
                "modal rank": f"{rd['modal_rank']} ({rd['modal_prob']:.3f})",
                "mean rank": f"{rd['mean_rank']:.2f}",
            }
        )
    _table(rows, list(rows[0].keys()))

    print(f"\nSigned z per register feature ({pct}% CI; 0 = on target)")
    rows = []
    for cond in ranked:
        c = cells[cond]
        row = {"condition": cond}
        for name in CENTROID_FEATURES:
            row[name] = f"{c['z'][name]:+.3f} {_ci(c[f'z_{name}_ci'], 3)}"
        rows.append(row)
    _table(rows, list(rows[0].keys()))

    print(f"\nAdjacent ranks, paired bootstrap on the shared resamples  (a - b, {pct}% CI)")
    rows = []
    for rec in report["paired_adjacent"]:
        rows.append(
            {
                "comparison": f"{rec['a']} - {rec['b']}",
                "diff": f"{rec['diff']:+.4f}",
                ci_col: _ci([rec["ci_low"], rec["ci_high"]]),
                "p": f"{rec['p_value']:.4f}",
                "sig": "*" if rec["significant"] else "",
            }
        )
    _table(rows, list(rows[0].keys()))
    n_sig = sum(1 for r in report["paired_adjacent"] if r["significant"])
    print(
        f"\n* = {pct}% CI excludes 0. {n_sig}/{len(report['paired_adjacent'])} adjacent-rank gaps "
        f"are separated; unstarred neighbours are not distinguishable and their relative order "
        f"should not be reported as a finding."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap CIs on the register fit of the main conditions."
    )
    parser.add_argument("--conditions", nargs="+", default=MAIN_CONDITIONS)
    parser.add_argument("--split", default="val", help="output split tag (default: val)")
    parser.add_argument("--out_dir", default="outputs", help="inference output directory")
    parser.add_argument(
        "--results_path", default=None, help="default: results/stylometrics_ci_<split>.json"
    )
    parser.add_argument("--n_resamples", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = build(
        Path(args.out_dir),
        args.split,
        conditions=args.conditions,
        n_resamples=args.n_resamples,
        alpha=args.alpha,
        seed=args.seed,
    )
    _print_summary(report)

    results_path = Path(args.results_path or f"results/stylometrics_ci_{args.split}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    main()
