"""
Bootstrap confidence intervals on the judge score Phi of the seven main conditions.

    python manage.py judge_ci
    python manage.py judge_ci --tag gpt
    python manage.py judge_ci --tag gpt --conditions zeroshot peft --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.judge import judge_results_path
from src.eval.stylometrics_ci import paired_diff, rank_distribution

MAIN_CONDITIONS = [
    "zeroshot",
    "random_fewshot",
    "knn_fewshot",
    "afsp_margin",
    "afsp_full",
    "peft",
    "commercial_haiku",
]
REFERENCE_CONDITIONS = {"commercial_haiku"}

_SCALE = (1, 2, 3, 4, 5)


def load_scores(results_path: Path, conditions: list[str]) -> tuple[dict, dict]:
    """Return ``(scores, sources)`` per condition; unparsed scores become NaN."""
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found; run `manage.py judge` for this judge first"
        )
    stored = json.loads(results_path.read_text(encoding="utf-8"))
    scores, sources = {}, {}
    for cond in conditions:
        rec = stored.get(cond)
        if rec is None:
            print(f"skip {cond}: not scored in {results_path}")
            continue
        scores[cond] = np.asarray(
            [np.nan if v is None else float(v) for v in rec["segments"]], dtype=float
        )
        sources[cond] = rec.get("sources")
    return scores, sources


def assert_aligned(present: list[str], sources: dict) -> None:
    """Fail if conditions' segments do not line up index-for-index."""
    ref_cond = present[0]
    ref = sources.get(ref_cond)
    if ref is None:
        print(f"warning: '{ref_cond}' has no recorded sources; cannot verify paired alignment")
        return
    for cond in present[1:]:
        src = sources.get(cond)
        if src is None:
            print(f"warning: '{cond}' has no recorded sources; cannot verify paired alignment")
            continue
        if src != ref:
            i = next(k for k, (x, y) in enumerate(zip(ref, src)) if x != y)
            raise ValueError(
                f"source mismatch between '{ref_cond}' and '{cond}' at segment {i}: the shared "
                f"resample indices require identical source order across conditions"
            )


def shared_draws(
    scores: dict[str, np.ndarray],
    present: list[str],
    *,
    n_resamples: int,
    seed: int,
    chunk: int = 512,
) -> dict[str, np.ndarray]:
    """Bootstrap draws of each condition's mean Phi over ONE shared index stream.

    Resampling segment positions once and applying the same indices to every
    condition is what makes the rank distribution and the adjacent contrasts paired.
    Drawing per condition independently would inflate every gap by treating the
    common segment difficulty as if it were noise.

    A condition may be missing a score at a position -- coverage is not always 1.0 --
    so the mean over a resample ignores NaN at the drawn positions rather than
    dropping the position for every condition.
    """
    n = len(scores[present[0]])
    rng = np.random.default_rng(seed)
    out = {c: np.empty(n_resamples, dtype=float) for c in present}
    filled = 0
    while filled < n_resamples:
        size = min(chunk, n_resamples - filled)
        idx = rng.integers(0, n, size=(size, n))
        for cond in present:
            drawn = scores[cond][idx]
            with np.errstate(invalid="ignore"):
                # All-NaN row would warn and yield NaN; impossible at real coverage,
                # but guarded so a pathological condition cannot abort the report.
                out[cond][filled : filled + size] = np.nanmean(drawn, axis=1)
        filled += size
    return out


def score_histogram(values: np.ndarray) -> dict[str, float]:
    """Share of parsed segments at each rubric level, so the mean can be read in context."""
    parsed = values[~np.isnan(values)]
    if parsed.size == 0:
        return {str(v): 0.0 for v in _SCALE}
    return {str(v): float(np.mean(parsed == v)) for v in _SCALE}


def build(
    split: str,
    *,
    conditions: list[str],
    tag: str | None,
    results_dir: Path,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    results_path = judge_results_path(results_dir, split, tag)
    scores, sources = load_scores(results_path, conditions)
    present = [c for c in conditions if c in scores]
    if not present:
        raise FileNotFoundError(f"no requested condition is scored in {results_path}")

    counts = {c: len(scores[c]) for c in present}
    if len(set(counts.values())) != 1:
        raise ValueError(f"conditions differ in segment count: {counts}")
    assert_aligned(present, sources)

    draws = shared_draws(scores, present, n_resamples=n_resamples, seed=seed, chunk=512)

    stored = json.loads(results_path.read_text(encoding="utf-8"))
    cells: dict[str, dict] = {}
    for cond in present:
        v = scores[cond]
        parsed = v[~np.isnan(v)]
        d = draws[cond]
        cells[cond] = {
            "n": int(v.size),
            "n_scored": int(parsed.size),
            "coverage": float(parsed.size / v.size) if v.size else 0.0,
            "phi": float(parsed.mean()) if parsed.size else float("nan"),
            "sd": float(parsed.std(ddof=1)) if parsed.size > 1 else float("nan"),
            "phi_ci": [
                float(np.percentile(d, 100 * alpha / 2)),
                float(np.percentile(d, 100 * (1 - alpha / 2))),
            ],
            "score_histogram": score_histogram(v),
            "is_reference": cond in REFERENCE_CONDITIONS,
        }

    # Phi is higher-is-better, so rank on the negated draws: rank 1 = highest Phi.
    ranked = sorted(present, key=lambda c: -cells[c]["phi"])
    for i, cond in enumerate(ranked, start=1):
        cells[cond]["rank"] = i
    ranks = rank_distribution({c: -draws[c] for c in present}, present)
    for cond in present:
        cells[cond]["rank_distribution"] = ranks[cond]

    # Ordered better-rank-first, so `diff` is positive when the observed ordering
    # holds in the resamples too.
    adjacent = [
        {"a": a, "b": b, **paired_diff(draws[a], draws[b], alpha=alpha)}
        for a, b in zip(ranked, ranked[1:])
    ]

    judge_rec = next((stored[c] for c in present if isinstance(stored.get(c), dict)), {})
    return {
        "split": split,
        "judge": {
            "tag": tag,
            "model": judge_rec.get("model"),
            "template_sha256": judge_rec.get("template_sha256"),
            "results_path": str(results_path),
        },
        "bootstrap": {
            "n_resamples": n_resamples,
            "seed": seed,
            "alpha": alpha,
            "paired": True,
            "unit": "segment",
            "n_segments": counts[present[0]],
        },
        "evidence_class": {
            "study_conditions": [c for c in present if c not in REFERENCE_CONDITIONS],
            "external_reference": [c for c in present if c in REFERENCE_CONDITIONS],
            "note": (
                "commercial_haiku is a diagnostic external reference baseline, not a "
                "condition of the study, and is excluded from study tables. Phi is not "
                "seed-reproducible, so differences of order 0.05 are within measurement "
                "noise."
            ),
        },
        "conditions": present,
        "ranking": ranked,
        "cells": cells,
        "paired_adjacent": adjacent,
    }


def _ci(bounds: list[float], places: int = 4) -> str:
    return f"[{bounds[0]:.{places}f}, {bounds[1]:.{places}f}]"


def _table(rows: list[dict], cols: list[str]) -> None:
    if not rows:
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def _print_summary(report: dict) -> None:
    boot = report["bootstrap"]
    pct = int(round((1 - boot["alpha"]) * 100))
    ci_col = f"ci{pct}"
    ranked, cells = report["ranking"], report["cells"]
    j = report["judge"]

    print(
        f"\nJudge register fidelity Phi by condition  (split={report['split']}, "
        f"n={boot['n_segments']} segments, resamples={boot['n_resamples']}, seed={boot['seed']})"
    )
    print(f"judge: {j['model'] or 'unknown'}  [tag {j['tag'] or '(none)'}]")
    print("Phi = mean 1-5 rubric rating against the authorized reference; higher is better.\n")

    rows = []
    for cond in ranked:
        c = cells[cond]
        rd = c["rank_distribution"]
        rows.append(
            {
                "rank": str(c["rank"]),
                "condition": cond,
                "class": "reference" if c["is_reference"] else "study",
                "n": str(c["n_scored"]),
                "Phi": f"{c['phi']:.4f}",
                ci_col: _ci(c["phi_ci"]),
                "sd": f"{c['sd']:.3f}",
                "P(this rank)": f"{rd['rank_probs'][c['rank'] - 1]:.3f}",
                "modal rank": f"{rd['modal_rank']} ({rd['modal_prob']:.3f})",
                "mean rank": f"{rd['mean_rank']:.2f}",
            }
        )
    _table(rows, list(rows[0].keys()))

    print("\nScore distribution over the rubric (share of parsed segments)")
    rows = []
    for cond in ranked:
        c = cells[cond]
        row = {"condition": cond, "coverage": f"{c['coverage']:.4f}"}
        for level in _SCALE:
            row[f"={level}"] = f"{c['score_histogram'][str(level)]:.3f}"
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
    _table(rows, list(rows[0].keys()) if rows else [])

    n_sig = sum(1 for r in report["paired_adjacent"] if r["significant"])
    print(
        f"\n* = {pct}% CI excludes 0. {n_sig}/{len(report['paired_adjacent'])} adjacent-rank gaps "
        f"are separated; unstarred neighbours are not distinguishable and their relative order "
        f"must not be reported as a finding. These are uncorrected: apply a family-wise "
        f"correction before calling any of them significant."
    )
    print(f"\n{report['evidence_class']['note']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap CIs on per-condition judge Phi for one judge."
    )
    parser.add_argument("--conditions", nargs="+", default=MAIN_CONDITIONS)
    parser.add_argument("--split", default="val", help="output split tag (default: val)")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument(
        "--tag",
        default=None,
        help="which judge to report (default: none, the primary judge_<split>.json)",
    )
    parser.add_argument(
        "--results_path", default=None, help="default: results/judge_ci_[<tag>_]<split>.json"
    )
    parser.add_argument("--n_resamples", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = build(
        args.split,
        conditions=args.conditions,
        tag=args.tag,
        results_dir=Path(args.results_dir),
        n_resamples=args.n_resamples,
        alpha=args.alpha,
        seed=args.seed,
    )
    _print_summary(report)

    stem = f"judge_ci_{args.tag}_{args.split}" if args.tag else f"judge_ci_{args.split}"
    out_path = Path(args.results_path or Path(args.results_dir) / f"{stem}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
