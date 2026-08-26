"""
Usage:
    python manage.py metric_agreement
    python manage.py metric_agreement --n_resamples 10000 --split val
    python manage.py metric_agreement --raters phi_a --conditions zeroshot peft --no-reference
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
from src.eval.register_direction import configured_direction
from src.eval.stylometrics import (
    CENTROID_FEATURES,
    aggregate,
    centroid_provenance,
    distance_to_centroid,
    features,
    register_band_distance,
    signed_z,
)

_CENTROID_PATH = Path("results/stylometrics_centroid.json")

# Both raters scored every condition below from the same prompt template, so they differ
# only in generator family; that is what makes the rater contrast interpretable.
RATERS = {
    "phi_a": Path("results/judge_val_segments"),
    "phi_b": Path("results/judge_gpt_val_segments"),
}

# The study conditions in ladder order, grouped by method family. commercial_haiku is
# deliberately absent: it is a diagnostic external reference, handled separately.
STUDY_CONDITIONS = [
    "zeroshot",
    "random_fewshot",
    "knn_fewshot",
    "sparse_knn",
    "afsp_margin",
    "afsp_full",
    "peft",
    "peft_knn",
    "peft_afsp",
    "rlsf_w3_0.0",
    "rlsf_w3_2.0",
    "rlsf_w3_6.0",
]
REFERENCE_CONDITION = "commercial_haiku"

_DISTANCE_KEYS = ("centroid_dist", "band_dist")


def _load_register_params(config_path: Path) -> tuple[float, list[float]]:
    """Read the band-pass sigma and register direction from a condition config."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    block = cfg.get("afsp") or cfg.get("register") or {}
    centroid = None
    centroid_path = block.get("centroid_file")
    if centroid_path:
        centroid = json.loads(Path(centroid_path).read_text(encoding="utf-8"))
    direction_map = configured_direction(block, centroid)
    if not direction_map:
        raise ValueError(
            f"{config_path} has no style_register_direction_file or style_register_direction"
        )
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


def _rho_rows(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Spearman rho of ``x`` against ``y`` for each resample row of ``idx``."""
    xr = stats.rankdata(x[idx], axis=1)
    yr = stats.rankdata(y[idx], axis=1)
    xr = xr - xr.mean(axis=1, keepdims=True)
    yr = yr - yr.mean(axis=1, keepdims=True)
    den = np.sqrt((xr**2).sum(axis=1) * (yr**2).sum(axis=1))
    num = (xr * yr).sum(axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


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
        out[filled : filled + k] = _rho_rows(x, y, idx)
        filled += k
    return out


def spearman_diff_ci(
    xa: np.ndarray,
    ya: np.ndarray,
    xb: np.ndarray,
    yb: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    alpha: float,
    chunk: int = 512,
) -> dict:
    if not (xa.shape == ya.shape == xb.shape == yb.shape):
        raise ValueError(
            f"all four series must be aligned: {xa.shape}, {ya.shape}, {xb.shape}, {yb.shape}"
        )
    rho_a = float(stats.spearmanr(xa, ya).statistic)
    rho_b = float(stats.spearmanr(xb, yb).statistic)

    rng = np.random.default_rng(seed)
    m = xa.size
    draws = np.empty(n_resamples, dtype=float)
    filled = 0
    while filled < n_resamples:
        k = min(chunk, n_resamples - filled)
        idx = rng.integers(0, m, size=(k, m))
        draws[filled : filled + k] = _rho_rows(xa, ya, idx) - _rho_rows(xb, yb, idx)
        filled += k

    lo = float(np.percentile(draws, 100 * alpha / 2))
    hi = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    # Two-sided bootstrap p by interval inversion: the smallest alpha whose interval
    # still excludes zero, doubled.
    frac = float(np.mean(draws <= 0.0))
    p = 2.0 * min(frac, 1.0 - frac)
    return {
        "rho_a": rho_a,
        "rho_b": rho_b,
        "delta": rho_a - rho_b,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p,
        "n": int(m),
        "separates": not (lo <= 0.0 <= hi),
    }


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
    keep_index: dict[str, list[int]] | None = None,
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

        if keep_index is not None:
            keep = list(keep_index.get(cond, []))
            bad = [i for i in keep if scores[i] is None]
            if bad:
                raise ValueError(
                    f"'{cond}': forced segment {bad[0]} has no parseable score in {judge_dir}"
                )
        else:
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
    """Phi against each centroid feature separately, pooled over conditions."""
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


def _common_keep(conditions: list[str], rater_dirs: dict[str, Path]) -> dict[str, list[int]]:
    """Segment indices every rater scored, per condition."""
    out: dict[str, list[int]] = {}
    for cond in conditions:
        masks = []
        for jdir in rater_dirs.values():
            _, scores = _load_judge_segments(cond, jdir)
            masks.append([sc is not None for sc in scores])
        n = min(len(m) for m in masks)
        out[cond] = [i for i in range(n) if all(m[i] for m in masks)]
    return out


def rater_comparison(
    paired: dict[str, dict[str, dict]],
    conditions: list[str],
    *,
    n_resamples: int,
    seed: int,
    alpha: float,
) -> dict:
    """Whether each RQ4 correlation depends on which rater supplied Phi."""
    raters = list(paired)
    present = [c for c in conditions if all(c in paired[r] for r in raters)]
    if len(raters) < 2 or not present:
        return {"conditions": present, "raters": raters, "pairs": {}}

    phi = {r: np.concatenate([paired[r][c]["phi"] for c in present]) for r in raters}
    targets = [*_DISTANCE_KEYS, "marker_rate", "comet"]

    pairs: dict[str, dict] = {}
    for ra, rb in itertools.combinations(raters, 2):
        key = f"{ra}~{rb}"
        deltas: dict[str, dict] = {}
        for t in targets:
            if not all(t in paired[r][c] for r in (ra, rb) for c in present):
                continue
            # The target series is a property of the translations, not of the rater,
            # so only Phi differs between the two correlations being differenced.
            v = np.concatenate([paired[ra][c][t] for c in present])
            deltas[t] = spearman_diff_ci(
                phi[ra], v, phi[rb], v, n_resamples=n_resamples, seed=seed, alpha=alpha
            )
        keys = sorted(deltas)
        for k, rej in zip(keys, holm_bonferroni([deltas[k]["p_value"] for k in keys], alpha=alpha)):
            deltas[k]["holm_significant"] = bool(rej)

        means_a = np.asarray([paired[ra][c]["phi_mean"] for c in present])
        means_b = np.asarray([paired[rb][c]["phi_mean"] for c in present])
        ordering = stats.spearmanr(means_a, means_b)
        pairs[key] = {
            "segment_phi_agreement": spearman_ci(
                phi[ra], phi[rb], n_resamples=n_resamples, seed=seed, alpha=alpha
            ),
            "condition_ordering": {
                "rho": float(ordering.statistic),
                "p_value": float(ordering.pvalue),
                "n": len(present),
                "p_floor": permutation_p_floor(len(present)),
            },
            "correlation_deltas": deltas,
            "holm_family_size": len(keys),
        }

    return {
        "conditions": present,
        "raters": raters,
        "n_pooled": int(next(iter(phi.values())).size),
        "note": (
            "Pooled over segments scored by every rater. A delta whose interval excludes "
            "zero means the RQ4 answer for that pair is rater-dependent."
        ),
        "pairs": pairs,
    }


def _rater_meta(judge_dir: Path) -> dict:
    meta_path = judge_dir / "_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "judge_dir": str(judge_dir),
        "model": meta.get("model"),
        "template_sha256": meta.get("template_sha256"),
    }


def build(
    out_dir: Path,
    split: str,
    *,
    conditions: list[str],
    reference: str | None,
    config_path: Path,
    rater_dirs: dict[str, Path],
    comet_path: Path,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    centroid = json.loads(_CENTROID_PATH.read_text(encoding="utf-8"))
    comet = json.loads(comet_path.read_text(encoding="utf-8")) if comet_path.exists() else {}
    sigma, direction = _load_register_params(config_path)

    requested = [*conditions, reference] if reference else list(conditions)
    per_rater_loaded = {
        rater: _load_all(out_dir, split, requested, jdir, comet, centroid, sigma, direction)
        for rater, jdir in rater_dirs.items()
    }
    per_rater_loaded = {r: d for r, d in per_rater_loaded.items() if d}
    if not per_rater_loaded:
        raise FileNotFoundError("no requested condition has both predictions and judge segments")

    scoped = {}
    for rater, loaded in per_rater_loaded.items():
        study = [c for c in conditions if c in loaded]
        if not study:
            raise FileNotFoundError(f"{rater}: no study condition loaded; cannot report agreement")
        scoped[rater] = {"study": study, "with_ref": [c for c in requested if c in loaded]}

    analyses = {}
    for rater, loaded in per_rater_loaded.items():
        study, with_ref = scoped[rater]["study"], scoped[rater]["with_ref"]
        analyses[rater] = {
            "conditions": {"study_only": study, "with_reference": with_ref},
            "coverage": {
                c: {
                    "n_total": loaded[c]["n_total"],
                    "n_scored": loaded[c]["n_scored"],
                    "coverage": loaded[c]["coverage"],
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

    # Rater contrast: reload the shared conditions on the segments every rater scored.
    live_dirs = {r: rater_dirs[r] for r in per_rater_loaded}
    common = [c for c in requested if all(c in per_rater_loaded[r] for r in live_dirs)]
    common_study = [c for c in conditions if c in common]
    rater_block: dict = {}
    if len(live_dirs) > 1 and common:
        keep = _common_keep(common, live_dirs)
        paired = {
            rater: _load_all(
                out_dir, split, common, jdir, comet, centroid, sigma, direction, keep_index=keep
            )
            for rater, jdir in live_dirs.items()
        }
        rater_block = {
            "study_only": rater_comparison(
                paired, common_study, n_resamples=n_resamples, seed=seed, alpha=alpha
            ),
            "with_reference": rater_comparison(
                paired, common, n_resamples=n_resamples, seed=seed, alpha=alpha
            ),
        }

    all_conds = [c for c in requested if any(c in d for d in per_rater_loaded.values())]
    system_metrics = {}
    for c in all_conds:
        src = next(d[c] for d in per_rater_loaded.values() if c in d)
        system_metrics[c] = {
            "phi": {r: d[c]["phi_mean"] for r, d in per_rater_loaded.items() if c in d},
            "stylo_dist": src["stylo_dist"],
            "comet": src.get("comet_system"),
            "z": src["z"],
        }

    return {
        "split": split,
        "out_dir": str(out_dir),
        "centroid": centroid_provenance(centroid, _CENTROID_PATH),
        "register_params": {
            "config": str(config_path),
            "select_target_sigma": sigma,
            "direction": dict(zip(CENTROID_FEATURES, direction)),
        },
        "raters": {r: _rater_meta(rater_dirs[r]) for r in per_rater_loaded},
        "bootstrap": {
            "n_resamples": n_resamples,
            "seed": seed,
            "alpha": alpha,
            "paired": True,
            "unit": "segment",
        },
        "evidence_class": {
            "study_conditions": scoped[next(iter(scoped))]["study"],
            "external_reference": (
                reference if any(reference in d for d in per_rater_loaded.values()) else None
            ),
            "note": (
                "commercial_haiku is a diagnostic external reference baseline, not a "
                "condition of the study; correlations are reported both without it "
                "(study_only) and with it (with_reference), and the two differ materially. "
                "Neither rater is a ground truth: the rater_comparison block bounds how far "
                "an RQ4 claim depends on rater identity, not which rater is correct."
            ),
        },
        "system_metrics": system_metrics,
        "per_rater": analyses,
        "rater_comparison": rater_block,
    }


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
    raters = list(report["per_rater"])
    print(
        f"\nRQ4 evaluation-component agreement  (split={report['split']}, "
        f"resamples={boot['n_resamples']}, seed={boot['seed']}, {pct}% percentile CIs)"
    )
    for r, meta in report["raters"].items():
        print(f"  {r}: {meta.get('model')}  ({meta['judge_dir']})")
    print(
        "Descriptive only. stylo_dist and band_dist are DISTANCES, so agreement with Phi\n"
        "(higher = better) is a NEGATIVE rho. Positive rho against a distance means the\n"
        "two measures point in opposite directions."
    )

    print("\nSystem-level metrics")
    rows = []
    for cond, m in report["system_metrics"].items():
        cls = "reference" if cond == report["evidence_class"]["external_reference"] else "study"
        row = {"condition": cond, "class": cls}
        for r in raters:
            row[r] = f"{m['phi'][r]:.4f}" if m["phi"].get(r) is not None else "-"
        row["stylo_dist"] = f"{m['stylo_dist']:.4f}"
        row["COMET"] = f"{m['comet']:.4f}" if m["comet"] is not None else "-"
        row["z_marker"] = f"{m['z']['marker_rate']:+.3f}"
        rows.append(row)
    _table(rows, list(rows[0].keys()))

    for rater in raters:
        an = report["per_rater"][rater]
        print(f"\n{'=' * 78}\nRATER {rater} -- {report['raters'][rater].get('model')}")

        for scope in ("study_only", "with_reference"):
            cl = an["condition_level"][scope]
            print(
                f"\nCondition level -- {scope}  (n={cl['n']} systems, "
                f"two-sided p floor at this n = {cl['p_floor']:.2e})"
            )
            _table(
                [
                    {"pair": k, "rho": f"{v['rho']:+.4f}", "p": f"{v['p_value']:.4f}"}
                    for k, v in cl["pairs"].items()
                ],
                ["pair", "rho", "p"],
            )

        for scope in ("study_only", "with_reference"):
            sl = an["segment_level"][scope]
            print(
                f"\nSegment level pooled -- {scope}  (n={sl['n_pooled']} scored segments, "
                f"Holm family = {sl['holm_family_size']})"
            )
            rows = [
                {
                    "pair": k if "~" in k else f"phi~{k}",
                    "rho": f"{v['rho']:+.4f}",
                    f"ci{pct}": _ci(v),
                    "p": f"{v['p_value']:.3e}",
                    "holm": "*" if v.get("holm_significant") else "",
                }
                for k, v in sorted(sl["pooled"].items())
            ]
            _table(rows, list(rows[0].keys()))

        comp = an["component_level"]["study_only"]
        print(
            f"\nComponent level -- study_only: phi vs each centroid feature "
            f"(n={comp['n_pooled']}, pooled)"
        )
        rows = [
            {
                "feature": name,
                "rho": f"{v['rho']:+.4f}",
                f"ci{pct}": _ci(v),
                "p": f"{v['p_value']:.3e}",
                "holm": "*" if v.get("holm_significant") else "",
            }
            for name, v in comp["features"].items()
        ]
        _table(rows, list(rows[0].keys()))

    rc = report.get("rater_comparison") or {}
    if rc:
        print(f"\n{'=' * 78}\nRATER DEPENDENCE OF THE RQ4 ANSWER")
        for scope in ("study_only", "with_reference"):
            block = rc.get(scope) or {}
            for key, pair in block.get("pairs", {}).items():
                agree = pair["segment_phi_agreement"]
                order = pair["condition_ordering"]
                print(
                    f"\n{key} -- {scope}  (n={block['n_pooled']} segments both raters scored, "
                    f"{order['n']} conditions)"
                )
                print(
                    f"  segment Phi agreement   rho={agree['rho']:+.4f} {_ci(agree)}\n"
                    f"  condition ordering      rho={order['rho']:+.4f} "
                    f"p={order['p_value']:.4f}"
                )
                rows = [
                    {
                        "target": f"phi~{t}",
                        f"rho[{key.split('~')[0]}]": f"{v['rho_a']:+.4f}",
                        f"rho[{key.split('~')[1]}]": f"{v['rho_b']:+.4f}",
                        "delta": f"{v['delta']:+.4f}",
                        f"ci{pct}": _ci(v),
                        "differs": "*" if v.get("holm_significant") else "",
                    }
                    for t, v in sorted(pair["correlation_deltas"].items())
                ]
                _table(rows, list(rows[0].keys()))
        print(
            "\nA starred delta means the two raters give materially different answers to "
            "that\nRQ4 pair, on the same segments. Direct rater-rater agreement and contrast "
            "\nreplication are reported by manage.py judge_agreement."
        )

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
    parser.add_argument(
        "--raters",
        nargs="+",
        default=list(RATERS),
        choices=list(RATERS),
        help="judge segment sets to analyse (default: both)",
    )
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
        rater_dirs={r: RATERS[r] for r in args.raters},
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
