"""
Judge-judge agreement between the primary and a cross-family second rater (RQ4).

Usage:
    python manage.py judge_agreement --tag_b gpt
    python manage.py judge_agreement --tag_b gpt --split val --n_resamples 10000
    python manage.py judge_agreement --tag_b gpt --no-reference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from src.eval._io import condition_path, load_condition
from src.eval.bootstrap import paired_bootstrap
from src.eval.judge import judge_results_path, judge_segment_dir
from src.eval.metric_agreement import holm_bonferroni, permutation_p_floor, spearman_ci

STUDY_CONDITIONS = [
    "zeroshot",
    "random_fewshot",
    "knn_fewshot",
    "afsp_margin",
    "afsp_full",
    "peft",
]
REFERENCE_CONDITION = "commercial_haiku"

PRIMARY_CONTRASTS = [
    ("random_fewshot", "zeroshot"),
    ("knn_fewshot", "zeroshot"),
    ("afsp_margin", "zeroshot"),
    ("afsp_full", "zeroshot"),
    ("peft", "zeroshot"),
]
SECONDARY_CONTRASTS = [
    ("afsp_full", "knn_fewshot"),
    ("sparse_knn", "knn_fewshot"),
    ("afsp_margin", "knn_fewshot"),
    ("afsp_full", "afsp_margin"),
    ("peft", "afsp_full"),
]

_SCALE = (1, 2, 3, 4, 5)


def load_judge_segments(segment_dir: Path, condition: str) -> tuple[list[str], list[int | None]]:
    """Return ``(sources, scores)`` for one condition; unparsed scores stay ``None``."""
    path = segment_dir / f"{condition}.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r.get("input", "") for r in rows], [r.get("score") for r in rows]


def _first_difference(a: list[str], b: list[str]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def _scale_index(x: np.ndarray, scale=_SCALE) -> np.ndarray:
    """Map rubric values onto 0..k-1, rejecting anything off the frozen scale."""
    lo, hi = min(scale), max(scale)
    arr = np.asarray(x, dtype=float)
    if arr.size and (np.any(arr < lo) or np.any(arr > hi) or np.any(arr != np.round(arr))):
        raise ValueError(f"scores outside the {lo}-{hi} integer rubric: {np.unique(arr)[:8]}")
    return (np.round(arr).astype(int) - lo).astype(np.int64)


def _kappa_from_counts(counts: np.ndarray, k: int) -> np.ndarray:
    n = counts.sum(axis=(-2, -1), keepdims=True)
    obs = np.divide(counts, n, out=np.zeros_like(counts, dtype=float), where=n > 0)
    hist_a = obs.sum(axis=-1)
    hist_b = obs.sum(axis=-2)
    exp = hist_a[..., :, None] * hist_b[..., None, :]
    i = np.arange(k)
    w = ((i[:, None] - i[None, :]) ** 2) / ((k - 1) ** 2)
    num = (w * obs).sum(axis=(-2, -1))
    den = (w * exp).sum(axis=(-2, -1))
    return np.where(den > 0, 1.0 - np.divide(num, np.where(den > 0, den, 1.0)), np.nan)


def _confusions(ia: np.ndarray, ib: np.ndarray, k: int) -> np.ndarray:
    ia = np.atleast_2d(ia)
    ib = np.atleast_2d(ib)
    rows = ia.shape[0]
    flat = (np.arange(rows)[:, None] * k * k) + ia * k + ib
    return np.bincount(flat.ravel(), minlength=rows * k * k).reshape(rows, k, k).astype(float)


def quadratic_weighted_kappa(a: np.ndarray, b: np.ndarray, scale=_SCALE) -> float:
    k = len(scale)
    if np.asarray(a).size == 0:
        return float("nan")
    counts = _confusions(_scale_index(a, scale), _scale_index(b, scale), k)
    return float(_kappa_from_counts(counts, k)[0])


def kappa_draws(
    a: np.ndarray, b: np.ndarray, *, n_resamples: int, seed: int, chunk: int = 256, scale=_SCALE
) -> np.ndarray:
    """Bootstrap draws of quadratic-weighted kappa, resampling observation pairs."""
    k = len(scale)
    ia, ib = _scale_index(a, scale), _scale_index(b, scale)
    rng = np.random.default_rng(seed)
    m = ia.size
    out = np.empty(n_resamples, dtype=float)
    filled = 0
    while filled < n_resamples:
        size = min(chunk, n_resamples - filled)
        idx = rng.integers(0, m, size=(size, m))
        out[filled : filled + size] = _kappa_from_counts(_confusions(ia[idx], ib[idx], k), k)
        filled += size
    return out


def kappa_ci(a: np.ndarray, b: np.ndarray, *, n_resamples: int, seed: int, alpha: float) -> dict:
    """Point quadratic-weighted kappa with a percentile bootstrap interval over pairs."""
    point = quadratic_weighted_kappa(a, b)
    m = int(np.asarray(a).size)
    if m == 0:
        return {
            "kappa": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0,
            "n_degenerate_resamples": 0,
        }
    draws = kappa_draws(a, b, n_resamples=n_resamples, seed=seed)
    finite = draws[np.isfinite(draws)]
    out = {"kappa": point, "n": m, "n_degenerate_resamples": int(n_resamples - finite.size)}
    if finite.size == 0:
        # Every resample was degenerate; report no interval rather than a fabricated one.
        return {**out, "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        **out,
        "ci_low": float(np.percentile(finite, 100 * alpha / 2)),
        "ci_high": float(np.percentile(finite, 100 * (1 - alpha / 2))),
    }


def _load_pair(
    out_dir: Path,
    split: str,
    condition: str,
    dir_a: Path,
    dir_b: Path,
) -> dict | None:
    """Align one condition's predictions with both judges' segment scores."""
    if not condition_path(out_dir, condition, split).exists():
        print(f"skip {condition}: no predictions for split {split}")
        return None
    missing = [d for d in (dir_a, dir_b) if not (d / f"{condition}.jsonl").exists()]
    if missing:
        print(f"skip {condition}: no judge segments in {', '.join(str(m) for m in missing)}")
        return None

    sources, _, _ = load_condition(out_dir, condition, split)
    src_a, scores_a = load_judge_segments(dir_a, condition)
    src_b, scores_b = load_judge_segments(dir_b, condition)
    for name, src in (("judge A", src_a), ("judge B", src_b)):
        if src != sources:
            i = _first_difference(sources, src)
            raise ValueError(
                f"'{condition}': {name} segments and predictions disagree at segment {i}; "
                f"the paired comparison requires them aligned index-for-index"
            )

    both = [i for i in range(len(sources)) if scores_a[i] is not None and scores_b[i] is not None]
    return {
        "condition": condition,
        "n_total": len(sources),
        "n_a": sum(s is not None for s in scores_a),
        "n_b": sum(s is not None for s in scores_b),
        "n_both": len(both),
        "index": np.asarray(both, dtype=int),
        # Full-length series with NaN where unparsed, so cross-condition contrasts
        # can intersect masks by index before dropping anything.
        "a_full": np.asarray([np.nan if s is None else float(s) for s in scores_a]),
        "b_full": np.asarray([np.nan if s is None else float(s) for s in scores_b]),
        "sources": sources,
    }


def rater_agreement(
    loaded: dict[str, dict],
    conditions: list[str],
    *,
    n_resamples: int,
    seed: int,
    alpha: float,
) -> dict:
    """Per-condition and pooled agreement between the two raters."""

    def _one(a: np.ndarray, b: np.ndarray, label: str) -> dict:
        exact = float(np.mean(a == b))
        adjacent = float(np.mean(np.abs(a - b) <= 1))
        row = {
            "n": int(a.size),
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "exact_agreement": exact,
            "adjacent_agreement": adjacent,
            "qwk": kappa_ci(a, b, n_resamples=n_resamples, seed=seed, alpha=alpha),
            "spearman": spearman_ci(a, b, n_resamples=n_resamples, seed=seed, alpha=alpha),
            "offset": paired_bootstrap(
                list(a), list(b), n_resamples=n_resamples, alpha=alpha, seed=seed
            ),
            "label": label,
        }
        return row

    present = [c for c in conditions if c in loaded]
    per_condition = {}
    for cond in present:
        d = loaded[cond]
        idx = d["index"]
        if idx.size == 0:
            print(f"skip {cond}: no segment scored by both judges")
            continue
        per_condition[cond] = _one(d["a_full"][idx], d["b_full"][idx], cond)

    pooled = None
    if per_condition:
        a = np.concatenate([loaded[c]["a_full"][loaded[c]["index"]] for c in per_condition])
        b = np.concatenate([loaded[c]["b_full"][loaded[c]["index"]] for c in per_condition])
        pooled = _one(a, b, "pooled")
    return {"conditions": list(per_condition), "per_condition": per_condition, "pooled": pooled}


def condition_ordering(loaded: dict[str, dict], conditions: list[str]) -> dict:
    present = [c for c in conditions if c in loaded and loaded[c]["index"].size]
    means_a, means_b = [], []
    for c in present:
        idx = loaded[c]["index"]
        means_a.append(float(loaded[c]["a_full"][idx].mean()))
        means_b.append(float(loaded[c]["b_full"][idx].mean()))
    order_a = [c for _, c in sorted(zip(means_a, present), reverse=True)]
    order_b = [c for _, c in sorted(zip(means_b, present), reverse=True)]
    out = {
        "conditions": present,
        "n": len(present),
        "mean_a": dict(zip(present, means_a)),
        "mean_b": dict(zip(present, means_b)),
        "ranking_a": order_a,
        "ranking_b": order_b,
        "identical_ranking": order_a == order_b,
        "p_floor": permutation_p_floor(len(present)),
    }
    if len(present) >= 3:
        res = stats.spearmanr(means_a, means_b)
        out["spearman"] = {
            "rho": float(res.statistic),
            "p_value": float(res.pvalue),
            "n": len(present),
        }
    return out


def contrast_replication(
    loaded: dict[str, dict],
    contrasts: list[tuple[str, str]],
    *,
    n_resamples: int,
    seed: int,
    alpha: float,
) -> dict:
    rows: dict[str, dict] = {}
    for hi, lo in contrasts:
        if hi not in loaded or lo not in loaded:
            continue
        d_hi, d_lo = loaded[hi], loaded[lo]
        if d_hi["sources"] != d_lo["sources"]:
            i = _first_difference(d_hi["sources"], d_lo["sources"])
            raise ValueError(
                f"contrast {hi} vs {lo}: the two conditions' segments diverge at {i}; "
                f"a paired contrast requires the same segments in the same order"
            )
        mask = (
            ~np.isnan(d_hi["a_full"])
            & ~np.isnan(d_hi["b_full"])
            & ~np.isnan(d_lo["a_full"])
            & ~np.isnan(d_lo["b_full"])
        )
        if not mask.any():
            continue
        res_a = paired_bootstrap(
            list(d_hi["a_full"][mask]),
            list(d_lo["a_full"][mask]),
            n_resamples=n_resamples,
            alpha=alpha,
            seed=seed,
        )
        res_b = paired_bootstrap(
            list(d_hi["b_full"][mask]),
            list(d_lo["b_full"][mask]),
            n_resamples=n_resamples,
            alpha=alpha,
            seed=seed,
        )
        rows[f"{hi} - {lo}"] = {
            "high": hi,
            "low": lo,
            "n": int(mask.sum()),
            "judge_a": res_a,
            "judge_b": res_b,
            "same_sign": (res_a["diff"] >= 0) == (res_b["diff"] >= 0),
            "both_separate": res_a["significant"] and res_b["significant"],
            "neither_separates": not res_a["significant"] and not res_b["significant"],
        }

    # Multiplicity lives inside each rater's family of contrasts.
    keys = list(rows)
    for judge in ("judge_a", "judge_b"):
        rejected = holm_bonferroni([rows[k][judge]["p_value"] for k in keys], alpha=alpha)
        for k, rej in zip(keys, rejected):
            rows[k][judge]["holm_significant"] = bool(rej)
    for k in keys:
        rows[k]["both_separate_holm"] = (
            rows[k]["judge_a"]["holm_significant"] and rows[k]["judge_b"]["holm_significant"]
        )
    return {"family_size": len(keys), "contrasts": rows}


def build(
    out_dir: Path,
    split: str,
    *,
    conditions: list[str],
    reference: str | None,
    dir_a: Path,
    dir_b: Path,
    tag_a: str | None,
    tag_b: str,
    results_dir: Path,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    requested = [*conditions, reference] if reference else list(conditions)
    loaded: dict[str, dict] = {}
    for cond in requested:
        entry = _load_pair(out_dir, split, cond, dir_a, dir_b)
        if entry is not None:
            loaded[cond] = entry
    if not loaded:
        raise FileNotFoundError(
            f"no requested condition has predictions plus segments in both {dir_a} and {dir_b}"
        )

    study = [c for c in conditions if c in loaded]
    with_ref = [c for c in requested if c in loaded]
    if not study:
        raise FileNotFoundError("no study condition loaded; cannot report judge agreement")

    models = _judge_models(results_dir, split, tag_a, tag_b, with_ref)
    report = {
        "split": split,
        "out_dir": str(out_dir),
        "judges": models,
        "bootstrap": {
            "n_resamples": n_resamples,
            "seed": seed,
            "alpha": alpha,
            "paired": True,
            "unit": "segment",
        },
        "evidence_class": {
            "study_conditions": study,
            "external_reference": reference if reference in loaded else None,
            "note": (
                "Descriptive. commercial_haiku is a diagnostic external reference "
                "baseline, not a condition of the study. Neither rater is a ground "
                "truth: agreement bounds how far a Phi-based claim depends on rater "
                "identity, it does not establish that either rater is correct."
            ),
            "self_family_caveat": (
                "The primary rater and the commercial_haiku condition are the same "
                "model family, so that condition's primary Phi carries a self-"
                "preference risk the second rater does not share. Compare the two "
                "raters' means for commercial_haiku against their means for the "
                "study conditions before reading anything into the reference point."
            ),
        },
        "coverage": {
            c: {k: loaded[c][k] for k in ("n_total", "n_a", "n_b", "n_both")} for c in with_ref
        },
        "rater_agreement": {
            "study_only": rater_agreement(
                loaded, study, n_resamples=n_resamples, seed=seed, alpha=alpha
            ),
            "with_reference": rater_agreement(
                loaded, with_ref, n_resamples=n_resamples, seed=seed, alpha=alpha
            ),
        },
        "condition_ordering": {
            "study_only": condition_ordering(loaded, study),
            "with_reference": condition_ordering(loaded, with_ref),
        },
        "contrast_replication": contrast_replication(
            loaded,
            [*PRIMARY_CONTRASTS, *SECONDARY_CONTRASTS],
            n_resamples=n_resamples,
            seed=seed,
            alpha=alpha,
        ),
        "contrast_classes": {
            "primary": [f"{a} - {b}" for a, b in PRIMARY_CONTRASTS],
            "secondary": [f"{a} - {b}" for a, b in SECONDARY_CONTRASTS],
        },
    }
    return report


def _judge_models(
    results_dir: Path, split: str, tag_a: str | None, tag_b: str, conditions: list[str]
) -> dict:
    """Identify each rater, and check both scored through the same frozen rubric."""
    out: dict[str, dict] = {}
    for key, tag in (("a", tag_a), ("b", tag_b)):
        path = judge_results_path(results_dir, split, tag)
        entry: dict = {"tag": tag, "results_path": str(path)}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = [data[c] for c in conditions if c in data and isinstance(data[c], dict)]
            entry["model"] = sorted({r.get("model") for r in rows if r.get("model")})
            entry["template_sha256"] = sorted(
                {r.get("template_sha256") for r in rows if r.get("template_sha256")}
            )
        out[key] = entry

    digests = {k: v.get("template_sha256") or [] for k, v in out.items()}
    if all(digests.values()):
        if set(digests["a"]) != set(digests["b"]):
            raise ValueError(
                f"the two judges scored through different rubrics (A: {digests['a']}, "
                f"B: {digests['b']}). A cross-family comparison requires the same frozen "
                f"template, or rater identity is confounded with rubric identity."
            )
        out["template_verified"] = True
    else:
        out["template_verified"] = False
        out["template_note"] = (
            "at least one judge's results predate template digest recording; that both "
            "raters read prompts/judge_eval.txt is asserted, not verified"
        )
    return out


def _fmt_ci(rec: dict, key_lo: str = "ci_low", key_hi: str = "ci_high", places: int = 3) -> str:
    lo, hi = rec.get(key_lo), rec.get(key_hi)
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return "[n/a]"
    return f"[{lo:+.{places}f}, {hi:+.{places}f}]"


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
    j = report["judges"]
    a_model = ", ".join(j["a"].get("model") or ["?"])
    b_model = ", ".join(j["b"].get("model") or ["?"])
    print(
        f"\nJudge-judge agreement  (split={report['split']}, resamples={boot['n_resamples']}, "
        f"seed={boot['seed']}, {pct}% percentile CIs)"
    )
    print(f"  judge A: {a_model}  [tag {j['a']['tag'] or '(none)'}]")
    print(f"  judge B: {b_model}  [tag {j['b']['tag']}]")
    print(f"  same frozen rubric verified by digest: {j['template_verified']}")
    if not j["template_verified"]:
        print(f"  ! {j['template_note']}")

    print("\nCoverage (segments parsed by each rater)")
    _table(
        [
            {"condition": c, **{k: str(v) for k, v in m.items()}}
            for c, m in report["coverage"].items()
        ],
        ["condition", "n_total", "n_a", "n_b", "n_both"],
    )

    for scope in ("study_only", "with_reference"):
        ra = report["rater_agreement"][scope]
        print(f"\nRater agreement -- {scope}")
        rows = []
        for cond, m in list(ra["per_condition"].items()) + (
            [("POOLED", ra["pooled"])] if ra["pooled"] else []
        ):
            rows.append(
                {
                    "condition": cond,
                    "n": str(m["n"]),
                    "Phi_A": f"{m['mean_a']:.3f}",
                    "Phi_B": f"{m['mean_b']:.3f}",
                    "A-B": f"{m['offset']['diff']:+.3f}",
                    f"ci{pct}": _fmt_ci(m["offset"]),
                    "qwk": f"{m['qwk']['kappa']:+.3f}",
                    "qwk_ci": _fmt_ci(m["qwk"]),
                    "rho": f"{m['spearman']['rho']:+.3f}",
                    "exact": f"{m['exact_agreement']:.1%}",
                    "adj": f"{m['adjacent_agreement']:.1%}",
                }
            )
        _table(rows, list(rows[0].keys()) if rows else [])

    for scope in ("study_only", "with_reference"):
        co = report["condition_ordering"][scope]
        sp = co.get("spearman")
        print(
            f"\nSystem ordering -- {scope}  (n={co['n']} systems, "
            f"two-sided p floor at this n = {co['p_floor']:.4f})"
        )
        print(f"  judge A: {' > '.join(co['ranking_a'])}")
        print(f"  judge B: {' > '.join(co['ranking_b'])}")
        print(f"  identical ranking: {co['identical_ranking']}")
        if sp:
            print(f"  Spearman rho = {sp['rho']:+.4f}  (p = {sp['p_value']:.4f}, point only)")

    cr = report["contrast_replication"]
    classes = report["contrast_classes"]
    print(
        f"\nContrast replication under each rater  (Holm family = {cr['family_size']} "
        f"per rater; same segments for both)"
    )
    rows = []
    for name, m in cr["contrasts"].items():
        cls = "primary" if name in classes["primary"] else "secondary"
        rows.append(
            {
                "contrast": name,
                "class": cls,
                "n": str(m["n"]),
                "dA": f"{m['judge_a']['diff']:+.3f}",
                "ciA": _fmt_ci(m["judge_a"]),
                "hA": "*" if m["judge_a"]["holm_significant"] else "",
                "dB": f"{m['judge_b']['diff']:+.3f}",
                "ciB": _fmt_ci(m["judge_b"]),
                "hB": "*" if m["judge_b"]["holm_significant"] else "",
                "sign": "same" if m["same_sign"] else "FLIPPED",
                "verdict": (
                    "both separate"
                    if m["both_separate"]
                    else ("neither separates" if m["neither_separates"] else "RATER-DEPENDENT")
                ),
            }
        )
    _table(rows, list(rows[0].keys()) if rows else [])
    print(
        f"\n* = survives Holm-Bonferroni within that rater's family of {cr['family_size']} "
        f"contrasts at alpha = {boot['alpha']}."
    )
    print(
        "A contrast marked RATER-DEPENDENT separates under one rater and not the other: "
        "it is not reportable as a finding without stating which rater it depends on. "
        "'neither separates' is a failure to separate, not evidence of no difference."
    )
    print(f"\n{report['evidence_class']['self_family_caveat']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agreement between the primary judge and a cross-family second judge."
    )
    parser.add_argument("--conditions", nargs="+", default=STUDY_CONDITIONS)
    parser.add_argument(
        "--reference",
        default=REFERENCE_CONDITION,
        help="external reference baseline, reported separately (default: commercial_haiku)",
    )
    parser.add_argument(
        "--no-reference",
        dest="reference",
        action="store_const",
        const=None,
        help="omit the external reference baseline entirely",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument(
        "--tag_a", default=None, help="primary judge tag (default: none, the unsuffixed artefacts)"
    )
    parser.add_argument("--tag_b", required=True, help="second judge tag, e.g. gpt")
    parser.add_argument("--results_path", default=None)
    parser.add_argument("--n_resamples", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.tag_a == args.tag_b:
        raise ValueError("--tag_a and --tag_b are the same judge; there is nothing to compare")

    results_dir = Path(args.results_dir)
    report = build(
        Path(args.out_dir),
        args.split,
        conditions=args.conditions,
        reference=args.reference,
        dir_a=judge_segment_dir(results_dir, args.split, args.tag_a),
        dir_b=judge_segment_dir(results_dir, args.split, args.tag_b),
        tag_a=args.tag_a,
        tag_b=args.tag_b,
        results_dir=results_dir,
        n_resamples=args.n_resamples,
        alpha=args.alpha,
        seed=args.seed,
    )
    _print_summary(report)

    out_path = Path(
        args.results_path or results_dir / f"judge_agreement_{args.tag_b}_{args.split}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
