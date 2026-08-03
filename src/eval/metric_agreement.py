"""
Usage:
    python manage.py metric_agreement
    python manage.py metric_agreement --n_resamples 10000 --split val
    python manage.py metric_agreement --conditions zeroshot peft --no-reference
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from src.eval._io import condition_path, load_condition
from src.eval.stylometrics import (
    CENTROID_FEATURES,
    aggregate,
    distance_to_centroid,
    features,
    register_band_distance,
    signed_z,
)

_CENTROID_PATH = Path("results/stylometrics_centroid.json")
_JUDGE_SEGMENT_DIR = Path("results/judge_val_segments")

# The six conditions of the study, in ladder order. commercial_haiku is deliberately
# absent: it is a diagnostic external reference, handled separately.
STUDY_CONDITIONS = [
    "zeroshot",
    "random_fewshot",
    "knn_fewshot",
    "afsp_margin",
    "afsp_full",
    "peft",
]
REFERENCE_CONDITION = "commercial_haiku"

_DISTANCE_KEYS = ("centroid_dist", "band_dist")


def _load_register_params(config_path: Path) -> tuple[float, list[float]]:
    """Read the band-pass sigma and register direction from a condition config."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    block = cfg.get("afsp") or cfg.get("register") or {}
    direction_map = block.get("style_register_direction")
    if not direction_map:
        raise ValueError(f"{config_path} has no style_register_direction")
    missing = set(CENTROID_FEATURES) - set(direction_map)
    if missing:
        raise ValueError(f"{config_path} register direction is missing {sorted(missing)}")
    sigma = block.get("select_target_sigma")
    if sigma is None:
        raise ValueError(f"{config_path} has no select_target_sigma")
    # Ordered to match CENTROID_FEATURES, which is the order register_band_distance
    # indexes the z-vector by. A dict-order mismatch here would silently reweight.
    return float(sigma), [float(direction_map[name]) for name in CENTROID_FEATURES]


def _load_judge_segments(condition: str, judge_dir: Path) -> tuple[list[str], list]:
    """Return ``(sources, scores)`` for one condition; unparseable scores stay ``None``."""
    path = judge_dir / f"{condition}.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r.get("input", "") for r in rows], [r.get("score") for r in rows]


def spearman_draws(
    x: np.ndarray, y: np.ndarray, *, n_resamples: int, seed: int, chunk: int = 512
) -> np.ndarray:
    """Bootstrap draws of Spearman rho, resampling observation *pairs*."""
    if x.shape != y.shape:
        raise ValueError(f"x and y differ in shape: {x.shape} vs {y.shape}")
    rng = np.random.default_rng(seed)
    m = x.size
    out = np.empty(n_resamples, dtype=float)
    filled = 0
    while filled < n_resamples:
        k = min(chunk, n_resamples - filled)
        idx = rng.integers(0, m, size=(k, m))
        xr = stats.rankdata(x[idx], axis=1)
        yr = stats.rankdata(y[idx], axis=1)
        xr = xr - xr.mean(axis=1, keepdims=True)
        yr = yr - yr.mean(axis=1, keepdims=True)
        den = np.sqrt((xr**2).sum(axis=1) * (yr**2).sum(axis=1))
        num = (xr * yr).sum(axis=1)
        # A resample can be constant in one variable (all-identical Phi); rho is
        # undefined there, so it contributes 0 rather than a nan that poisons the
        # percentiles.
        out[filled : filled + k] = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
        filled += k
    return out


def spearman_ci(x: np.ndarray, y: np.ndarray, *, n_resamples: int, seed: int, alpha: float) -> dict:
    """Point Spearman rho with a percentile bootstrap interval."""
    res = stats.spearmanr(x, y)
    draws = spearman_draws(x, y, n_resamples=n_resamples, seed=seed)
    lo = float(np.percentile(draws, 100 * alpha / 2))
    hi = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return {
        "rho": float(res.statistic),
        "ci_low": lo,
        "ci_high": hi,
        "p_value": float(res.pvalue),
        "n": int(x.size),
        "separates": not (lo <= 0.0 <= hi),
    }


def permutation_p_floor(n: int) -> float:
    """Smallest attainable two-sided p for a perfect monotone relation over ``n`` points."""
    if n < 3:
        return float("nan")
    return 2.0 / float(math.factorial(n))


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down: True where the hypothesis is rejected."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    for rank, i in enumerate(order):
        if p_values[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break  # step-down: once one fails, all larger p-values fail too
    return reject


def _segment_register(preds: list[str], centroid: dict, sigma: float, direction: list[float]):
    """Per-segment register quantities for one condition's predictions."""
    out: dict[str, list[float]] = {k: [] for k in (*_DISTANCE_KEYS, *CENTROID_FEATURES)}
    for text in preds:
        f = features(text)
        out["centroid_dist"].append(distance_to_centroid(f, centroid))
        out["band_dist"].append(register_band_distance(f, centroid, sigma, direction))
        for name in CENTROID_FEATURES:
            out[name].append(f[name])
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _load_all(
    out_dir: Path,
    split: str,
    conditions: list[str],
    judge_dir: Path,
    comet: dict,
    centroid: dict,
    sigma: float,
    direction: list[float],
) -> dict[str, dict]:
    """Load, align, and derive every per-segment series needed, per condition."""
    loaded: dict[str, dict] = {}
    for cond in conditions:
        path = condition_path(out_dir, cond, split)
        if not path.exists():
            print(f"skip {cond}: {path} not found")
            continue
        if not (judge_dir / f"{cond}.jsonl").exists():
            print(f"skip {cond}: no judge segments in {judge_dir}")
            continue
        sources, preds, _ = load_condition(out_dir, cond, split)
        jsources, scores = _load_judge_segments(cond, judge_dir)
        if jsources != sources:
            i = next(
                (k for k, (a, b) in enumerate(zip(sources, jsources)) if a != b),
                min(len(sources), len(jsources)),
            )
            raise ValueError(
                f"'{cond}': judge segments and predictions disagree at segment {i}; "
                f"the per-segment correlation requires them aligned index-for-index"
            )

        keep = [i for i, s in enumerate(scores) if s is not None]
        reg = _segment_register(preds, centroid, sigma, direction)
        agg = aggregate(preds)
        entry = {
            "n_total": len(sources),
            "n_scored": len(keep),
            "coverage": len(keep) / len(sources) if sources else 0.0,
            "phi": np.asarray([float(scores[i]) for i in keep]),
            "phi_mean": float(np.mean([scores[i] for i in keep])) if keep else float("nan"),
            "stylo_dist": distance_to_centroid(agg["mean"], centroid),
            "z": signed_z(agg["mean"], centroid),
            **{k: v[keep] for k, v in reg.items()},
        }
        if cond in comet:
            segs = comet[cond]["segments"]
            if len(segs) != len(sources):
                raise ValueError(
                    f"'{cond}': COMET has {len(segs)} segments, predictions have "
                    f"{len(sources)}; refusing to correlate unaligned series"
                )
            entry["comet"] = np.asarray([segs[i] for i in keep], dtype=float)
            entry["comet_system"] = float(comet[cond]["system"])
        loaded[cond] = entry
    return loaded


def condition_level(loaded: dict[str, dict], conditions: list[str]) -> dict:
    """Pairwise Spearman across conditions on the three system-level metrics."""
    present = [c for c in conditions if c in loaded]
    metrics = {
        "phi": np.asarray([loaded[c]["phi_mean"] for c in present]),
        "stylo_dist": np.asarray([loaded[c]["stylo_dist"] for c in present]),
    }
    if all("comet_system" in loaded[c] for c in present):
        metrics["comet"] = np.asarray([loaded[c]["comet_system"] for c in present])

    pairs = {}
    for a, b in itertools.combinations(metrics, 2):
        res = stats.spearmanr(metrics[a], metrics[b])
        pairs[f"{a}~{b}"] = {
            "rho": float(res.statistic),
            "p_value": float(res.pvalue),
            "n": len(present),
        }
    n = len(present)
    return {"conditions": present, "n": n, "p_floor": permutation_p_floor(n), "pairs": pairs}


def segment_level(
    loaded: dict[str, dict],
    conditions: list[str],
    *,
    n_resamples: int,
    seed: int,
    alpha: float,
) -> dict:
    """Per-condition and pooled Spearman of Phi against register and adequacy."""
    present = [c for c in conditions if c in loaded]
    targets = [*_DISTANCE_KEYS, "marker_rate", "comet"]

    per_condition: dict[str, dict] = {}
    for cond in present:
        d = loaded[cond]
        row = {}
        for t in targets:
            if t in d:
                row[t] = spearman_ci(
                    d["phi"], d[t], n_resamples=n_resamples, seed=seed, alpha=alpha
                )
        per_condition[cond] = row

    pooled: dict[str, dict] = {}
    phi_all = np.concatenate([loaded[c]["phi"] for c in present])
    for t in targets:
        if all(t in loaded[c] for c in present):
            v = np.concatenate([loaded[c][t] for c in present])
            pooled[t] = spearman_ci(phi_all, v, n_resamples=n_resamples, seed=seed, alpha=alpha)
    # Adequacy against the register proxy, the third README pair.
    if all("comet" in loaded[c] for c in present):
        for key in _DISTANCE_KEYS:
            pooled[f"comet~{key}"] = spearman_ci(
                np.concatenate([loaded[c]["comet"] for c in present]),
                np.concatenate([loaded[c][key] for c in present]),
                n_resamples=n_resamples,
                seed=seed,
                alpha=alpha,
            )

    keys = sorted(pooled)
    rejected = holm_bonferroni([pooled[k]["p_value"] for k in keys], alpha=alpha)
    for k, rej in zip(keys, rejected):
        pooled[k]["holm_significant"] = bool(rej)

    return {
        "conditions": present,
        "n_pooled": int(phi_all.size),
        "per_condition": per_condition,
        "pooled": pooled,
        "holm_family_size": len(keys),
    }


def component_level(
    loaded: dict[str, dict],
    conditions: list[str],
    *,
    n_resamples: int,
    seed: int,
    alpha: float,
) -> dict:
    """Phi against each centroid feature separately, pooled over conditions.

    The composite distance mixes four features. If the judge tracks one of them and
    ignores the rest, the composite correlation is diluted toward zero and the
    component table is the only place that shows it.
    """
    present = [c for c in conditions if c in loaded]
    phi = np.concatenate([loaded[c]["phi"] for c in present])
    out = {}
    for name in CENTROID_FEATURES:
        v = np.concatenate([loaded[c][name] for c in present])
        out[name] = spearman_ci(phi, v, n_resamples=n_resamples, seed=seed, alpha=alpha)
    keys = list(out)
    for k, rej in zip(keys, holm_bonferroni([out[k]["p_value"] for k in keys], alpha=alpha)):
        out[k]["holm_significant"] = bool(rej)
    return {"conditions": present, "n_pooled": int(phi.size), "features": out}


def build(
    out_dir: Path,
    split: str,
    *,
    conditions: list[str],
    reference: str | None,
    config_path: Path,
    judge_dir: Path,
    comet_path: Path,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    centroid = json.loads(_CENTROID_PATH.read_text(encoding="utf-8"))
    comet = json.loads(comet_path.read_text(encoding="utf-8")) if comet_path.exists() else {}
    sigma, direction = _load_register_params(config_path)

    requested = [*conditions, reference] if reference else list(conditions)
    loaded = _load_all(out_dir, split, requested, judge_dir, comet, centroid, sigma, direction)
    if not loaded:
        raise FileNotFoundError("no requested condition has both predictions and judge segments")

    study = [c for c in conditions if c in loaded]
    with_ref = [c for c in requested if c in loaded]
    if not study:
        raise FileNotFoundError("no study condition loaded; cannot report agreement")

    report = {
        "split": split,
        "out_dir": str(out_dir),
        "register_params": {
            "config": str(config_path),
            "select_target_sigma": sigma,
            "direction": dict(zip(CENTROID_FEATURES, direction)),
        },
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
                "commercial_haiku is a diagnostic external reference baseline, not a "
                "condition of the study; correlations are reported both without it "
                "(study_only) and with it (with_reference), and the two differ materially"
            ),
        },
        "coverage": {
            c: {
                "n_total": loaded[c]["n_total"],
                "n_scored": loaded[c]["n_scored"],
                "coverage": loaded[c]["coverage"],
            }
            for c in with_ref
        },
        "system_metrics": {
            c: {
                "phi": loaded[c]["phi_mean"],
                "stylo_dist": loaded[c]["stylo_dist"],
                "comet": loaded[c].get("comet_system"),
                "z": loaded[c]["z"],
            }
            for c in with_ref
        },
        "condition_level": {
            "study_only": condition_level(loaded, study),
            "with_reference": condition_level(loaded, with_ref),
        },
        "segment_level": {
            "study_only": segment_level(
                loaded, study, n_resamples=n_resamples, seed=seed, alpha=alpha
            ),
            "with_reference": segment_level(
                loaded, with_ref, n_resamples=n_resamples, seed=seed, alpha=alpha
            ),
        },
        "component_level": {
            "study_only": component_level(
                loaded, study, n_resamples=n_resamples, seed=seed, alpha=alpha
            ),
            "with_reference": component_level(
                loaded, with_ref, n_resamples=n_resamples, seed=seed, alpha=alpha
            ),
        },
    }
    return report


def _ci(rec: dict, places: int = 4) -> str:
    return f"[{rec['ci_low']:+.{places}f}, {rec['ci_high']:+.{places}f}]"


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
    print(
        f"\nRQ4 evaluation-component agreement  (split={report['split']}, "
        f"resamples={boot['n_resamples']}, seed={boot['seed']}, {pct}% percentile CIs)"
    )
    print(
        "Descriptive only. stylo_dist and band_dist are DISTANCES, so agreement with Phi\n"
        "(higher = better) is a NEGATIVE rho. Positive rho against a distance means the\n"
        "two measures point in opposite directions."
    )

    print("\nSystem-level metrics")
    rows = []
    for cond, m in report["system_metrics"].items():
        cls = "reference" if cond == report["evidence_class"]["external_reference"] else "study"
        rows.append(
            {
                "condition": cond,
                "class": cls,
                "Phi": f"{m['phi']:.4f}",
                "stylo_dist": f"{m['stylo_dist']:.4f}",
                "COMET": f"{m['comet']:.4f}" if m["comet"] is not None else "-",
                "z_marker": f"{m['z']['marker_rate']:+.3f}",
                "z_lex": f"{m['z']['lex_density']:+.3f}",
            }
        )
    _table(rows, list(rows[0].keys()))

    for scope in ("study_only", "with_reference"):
        cl = report["condition_level"][scope]
        print(
            f"\nCondition level -- {scope}  (n={cl['n']} systems, "
            f"two-sided p floor at this n = {cl['p_floor']:.4f})"
        )
        rows = [
            {"pair": k, "rho": f"{v['rho']:+.4f}", "p": f"{v['p_value']:.4f}"}
            for k, v in cl["pairs"].items()
        ]
        _table(rows, ["pair", "rho", "p"])

    for scope in ("study_only", "with_reference"):
        sl = report["segment_level"][scope]
        print(
            f"\nSegment level pooled -- {scope}  (n={sl['n_pooled']} scored segments, "
            f"Holm family = {sl['holm_family_size']})"
        )
        rows = []
        for k, v in sorted(sl["pooled"].items()):
            rows.append(
                {
                    "pair": k if "~" in k else f"phi~{k}",
                    "rho": f"{v['rho']:+.4f}",
                    f"ci{pct}": _ci(v),
                    "p": f"{v['p_value']:.3e}",
                    "holm": "*" if v.get("holm_significant") else "",
                }
            )
        _table(rows, list(rows[0].keys()))

    sl = report["segment_level"]["with_reference"]
    print("\nSegment level per condition -- phi vs each register quantity (rho)")
    rows = []
    for cond, row in sl["per_condition"].items():
        entry = {"condition": cond, "n": str(row[_DISTANCE_KEYS[0]]["n"])}
        for k in (*_DISTANCE_KEYS, "marker_rate", "comet"):
            entry[k] = f"{row[k]['rho']:+.4f}" if k in row else "-"
        rows.append(entry)
    _table(rows, list(rows[0].keys()))

    for scope in ("study_only", "with_reference"):
        comp = report["component_level"][scope]
        print(
            f"\nComponent level -- {scope}: phi vs each centroid feature "
            f"(n={comp['n_pooled']}, pooled)"
        )
        rows = []
        for name, v in comp["features"].items():
            rows.append(
                {
                    "feature": name,
                    "rho": f"{v['rho']:+.4f}",
                    f"ci{pct}": _ci(v),
                    "p": f"{v['p_value']:.3e}",
                    "holm": "*" if v.get("holm_significant") else "",
                }
            )
        _table(rows, list(rows[0].keys()))

    print(
        "\n* = survives Holm-Bonferroni within its family at alpha = "
        f"{boot['alpha']}. Correlations without a star are reported as failing to "
        "separate from zero, not as zero."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RQ4 agreement between COMET, the stylometric register proxy, and judge Phi."
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
    parser.add_argument(
        "--config",
        default="configs/afsp_sweep.yaml",
        help="config supplying select_target_sigma and the register direction",
    )
    parser.add_argument("--judge_dir", default=str(_JUDGE_SEGMENT_DIR))
    parser.add_argument("--comet_path", default=None, help="default: results/comet_<split>.json")
    parser.add_argument("--results_path", default=None)
    parser.add_argument("--n_resamples", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = build(
        Path(args.out_dir),
        args.split,
        conditions=args.conditions,
        reference=args.reference,
        config_path=Path(args.config),
        judge_dir=Path(args.judge_dir),
        comet_path=Path(args.comet_path or f"results/comet_{args.split}.json"),
        n_resamples=args.n_resamples,
        alpha=args.alpha,
        seed=args.seed,
    )
    _print_summary(report)

    results_path = Path(args.results_path or f"results/metric_agreement_{args.split}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    main()
