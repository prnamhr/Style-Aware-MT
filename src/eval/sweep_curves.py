"""
Dose-response curves for the AFSP register-rerank weight lambda (RQ3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from src.eval._io import condition_path
from src.eval.stylometrics import (
    CENTROID_FEATURES,
    FEATURE_NAMES,
    aggregate,
    distance_to_centroid,
    feature_vector,
)

_CENTROID_PATH = Path("results/stylometrics_centroid.json")
_VERIFY_PATH = Path("results/afsp_verify_val.json")
_JUDGE_SEG_DIR = Path("results/judge_val_segments")

K_VALUES = [4, 8, 16]
LAMBDA_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


REFERENCE_CONDITIONS = ["knn_fewshot", "afsp_margin", "afsp_full"]


def cell_tag(k: int, lam: float) -> str:
    """Sweep filenames use the shortest decimal form: k8_l0, k8_l0.1, k8_l1."""
    text = f"{lam:g}"
    return f"afsp_k{k}_l{text}"


def _feature_matrix(texts: list[str]) -> np.ndarray:
    return np.asarray([feature_vector(t) for t in texts if t.strip()], dtype=float)


def signed_z(mean_by_feature: dict[str, float], centroid: dict) -> dict[str, float]:
    """Signed z-deviation of each centroid feature from the target register."""
    mean = np.asarray(centroid["mean"], dtype=float)
    std = np.asarray(centroid["std"], dtype=float)
    vec = np.asarray([mean_by_feature[name] for name in centroid["features"]], dtype=float)
    return dict(zip(centroid["features"], ((vec - mean) / std).tolist()))


def bootstrap_cell(
    matrix: np.ndarray,
    centroid: dict,
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Percentile CIs for stylo_dist and each signed z, resampling segments."""
    idx_features = [FEATURE_NAMES.index(name) for name in centroid["features"]]
    c_mean = np.asarray(centroid["mean"], dtype=float)
    c_std = np.asarray(centroid["std"], dtype=float)

    n = matrix.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    # (n_resamples, n_features) means of the centroid features across resamples
    means = matrix[:, idx_features][idx].mean(axis=1)
    z = (means - c_mean) / c_std
    dists = np.linalg.norm(z, axis=1)

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {"stylo_dist_ci": [float(np.percentile(dists, lo)), float(np.percentile(dists, hi))]}
    for i, name in enumerate(centroid["features"]):
        out[f"z_{name}_ci"] = [
            float(np.percentile(z[:, i], lo)),
            float(np.percentile(z[:, i], hi)),
        ]
    return out


def _load_predictions(out_dir: Path, condition: str, split: str) -> list[str]:
    path = condition_path(out_dir, condition, split)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line).get("prediction", "") for line in f if line.strip()]


def score_cell(
    texts: list[str], centroid: dict, *, bootstrap: bool, seed: int, n_resamples: int
) -> dict:
    agg = aggregate(texts)
    row: dict = {
        "n": agg["n"],
        "stylo_dist": distance_to_centroid(agg["mean"], centroid),
        "z": signed_z(agg["mean"], centroid),
        "mean": {name: agg["mean"][name] for name in FEATURE_NAMES},
    }
    if bootstrap:
        row.update(
            bootstrap_cell(_feature_matrix(texts), centroid, n_resamples=n_resamples, seed=seed)
        )
    return row


def trend_test(lams: list[float], values: list[float]) -> dict:
    """Spearman rank correlation of a statistic against lambda."""
    rho, p = stats.spearmanr(lams, values)
    return {"spearman_rho": float(rho), "p": float(p), "n_points": len(lams)}


def _judge_points() -> dict[str, dict]:
    """Judge means for the sweep cells that were judged during AFSP verification."""
    if not _VERIFY_PATH.exists():
        return {}
    verify = json.loads(_VERIFY_PATH.read_text(encoding="utf-8"))
    points: dict[str, dict] = {}
    for cell in verify.get("cells", []):
        if "judge_mean" not in cell:
            continue
        entry = {
            "judge_mean": cell["judge_mean"],
            "judge_coverage": cell.get("judge_coverage"),
            "k": cell["k"],
            "lambda": cell["lambda"],
        }
        seg_path = _JUDGE_SEG_DIR / f"{cell['tag']}.jsonl"
        if seg_path.exists():
            scores = [
                json.loads(line)["score"]
                for line in seg_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line).get("score") is not None
            ]
            if scores:
                arr = np.asarray(scores, dtype=float)
                se = float(arr.std(ddof=1) / np.sqrt(arr.size))
                entry["judge_ci"] = [float(arr.mean() - 1.96 * se), float(arr.mean() + 1.96 * se)]
        points[cell["tag"]] = entry
    return points


def _comet_by_tag(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {tag: float(res["system"]) for tag, res in data.items() if "system" in res}


def build(
    out_dir: Path,
    split: str,
    *,
    comet_path: Path,
    bootstrap: bool = True,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict:
    centroid = json.loads(_CENTROID_PATH.read_text(encoding="utf-8"))
    comet = _comet_by_tag(comet_path)
    judged = _judge_points()

    cells = []
    for k in K_VALUES:
        for lam in LAMBDA_VALUES:
            tag = cell_tag(k, lam)
            path = condition_path(out_dir, tag, split)
            if not path.exists():
                print(f"skip {tag}: {path} not found")
                continue
            row = score_cell(
                _load_predictions(out_dir, tag, split),
                centroid,
                bootstrap=bootstrap,
                seed=seed,
                n_resamples=n_resamples,
            )
            row = {"tag": tag, "k": k, "lambda": lam, **row}
            if tag in comet:
                row["comet_system"] = comet[tag]
            if tag in judged:
                row["judge_mean"] = judged[tag]["judge_mean"]
                if "judge_ci" in judged[tag]:
                    row["judge_ci"] = judged[tag]["judge_ci"]
            cells.append(row)

    references = {}
    for cond in REFERENCE_CONDITIONS:
        if condition_path(Path("outputs"), cond, split).exists():
            references[cond] = score_cell(
                _load_predictions(Path("outputs"), cond, split),
                centroid,
                bootstrap=bootstrap,
                seed=seed,
                n_resamples=n_resamples,
            )

    trends: dict[str, dict] = {}
    for k in K_VALUES:
        rows = [c for c in cells if c["k"] == k]
        if len(rows) < 3:
            continue
        lams = [c["lambda"] for c in rows]
        entry = {"stylo_dist": trend_test(lams, [c["stylo_dist"] for c in rows])}
        for name in CENTROID_FEATURES:
            entry[f"z_{name}"] = trend_test(lams, [c["z"][name] for c in rows])
        comet_rows = [c for c in rows if "comet_system" in c]
        if len(comet_rows) >= 3:
            entry["comet"] = trend_test(
                [c["lambda"] for c in comet_rows], [c["comet_system"] for c in comet_rows]
            )
        trends[f"k{k}"] = entry

    pooled_rows = [c for c in cells]
    pooled = {
        "stylo_dist": trend_test(
            [c["lambda"] for c in pooled_rows], [c["stylo_dist"] for c in pooled_rows]
        )
    }
    for name in CENTROID_FEATURES:
        pooled[f"z_{name}"] = trend_test(
            [c["lambda"] for c in pooled_rows], [c["z"][name] for c in pooled_rows]
        )

    return {
        "split": split,
        "out_dir": str(out_dir),
        "centroid": {"features": centroid["features"], "n_segments": centroid["n_segments"]},
        "bootstrap": {"n_resamples": n_resamples, "seed": seed} if bootstrap else None,
        "comet_source": str(comet_path) if comet else None,
        "cells": cells,
        "references": references,
        "trends_by_k": trends,
        "trends_pooled": pooled,
        "judged_cells": sorted(judged),
    }


_K_COLORS = {4: "#4c72b0", 8: "#dd8452", 16: "#55a868"}
_FEATURE_LABELS = {
    "lex_density": "lexical density",
    "ttr": "type-token ratio",
    "root_ttr": "root TTR",
    "marker_rate": "archaic marker rate",
}


def _series(cells: list[dict], k: int, key: str, feature: str | None = None):
    rows = sorted((c for c in cells if c["k"] == k), key=lambda c: c["lambda"])
    if feature is not None:
        rows = [c for c in rows if feature in c.get("z", {})]
        return [c["lambda"] for c in rows], [c["z"][feature] for c in rows]
    rows = [c for c in rows if key in c]
    return [c["lambda"] for c in rows], [c[key] for c in rows]


def _ci_band(ax, cells, k, ci_key, lo_hi_from_z=None):
    rows = sorted((c for c in cells if c["k"] == k), key=lambda c: c["lambda"])
    rows = [c for c in rows if ci_key in c]
    if not rows:
        return
    lam = [c["lambda"] for c in rows]
    lo = [c[ci_key][0] for c in rows]
    hi = [c[ci_key][1] for c in rows]
    ax.fill_between(lam, lo, hi, color=_K_COLORS[k], alpha=0.12, linewidth=0)


def figure_stylo_dist(report: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    cells, refs = report["cells"], report["references"]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for k in K_VALUES:
        lam, val = _series(cells, k, "stylo_dist")
        if not lam:
            continue
        _ci_band(ax, cells, k, "stylo_dist_ci")
        rho = report["trends_by_k"].get(f"k{k}", {}).get("stylo_dist", {}).get("spearman_rho")
        label = f"k={k}" + (f"  (ρ={rho:+.2f})" if rho is not None else "")
        ax.plot(lam, val, marker="o", color=_K_COLORS[k], label=label, linewidth=1.8)

    for cond, style in (
        ("knn_fewshot", (":", "#666666")),
        ("afsp_full", ("--", "#999999")),
    ):
        if cond in refs:
            ax.axhline(
                refs[cond]["stylo_dist"],
                linestyle=style[0],
                color=style[1],
                linewidth=1.2,
                label=f"{cond} (frozen)",
            )

    ax.set_xlabel("λ  (register-rerank weight)")
    ax.set_ylabel("stylo_dist  →  target register (lower is better)")
    ax.set_title("AFSP register rerank: stylistic distance vs λ")
    ax.set_xticks(LAMBDA_VALUES)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_z_deviations(report: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    cells, refs = report["cells"], report["references"]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8), sharex=True)
    for ax, name in zip(axes.ravel(), CENTROID_FEATURES):
        for k in K_VALUES:
            lam, val = _series(cells, k, "z", feature=name)
            if not lam:
                continue
            _ci_band(ax, cells, k, f"z_{name}_ci")
            ax.plot(lam, val, marker="o", color=_K_COLORS[k], label=f"k={k}", linewidth=1.7)
        if "knn_fewshot" in refs:
            ax.axhline(
                refs["knn_fewshot"]["z"][name],
                linestyle=":",
                color="#666666",
                linewidth=1.1,
                label="knn_fewshot",
            )
        ax.axhline(0.0, color="black", linewidth=1.0)
        rho = report["trends_pooled"].get(f"z_{name}", {}).get("spearman_rho")
        title = _FEATURE_LABELS[name] + (f"   (pooled ρ={rho:+.2f})" if rho is not None else "")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("signed z vs target")
        ax.grid(alpha=0.25, linewidth=0.6)
    for ax in axes[1]:
        ax.set_xlabel("λ  (register-rerank weight)")
        ax.set_xticks(LAMBDA_VALUES)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Signed deviation from the target register, by feature (0 = on target)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_tradeoff(report: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    cells = report["cells"]
    if not any("comet_system" in c for c in cells):
        return

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for k in K_VALUES:
        lam, val = _series(cells, k, "comet_system")
        if lam:
            ax_l.plot(lam, val, marker="o", color=_K_COLORS[k], label=f"k={k}", linewidth=1.8)
    ax_l.set_xlabel("λ  (register-rerank weight)")
    ax_l.set_ylabel("COMET (system)")
    ax_l.set_title("Adequacy vs λ")
    ax_l.set_xticks(LAMBDA_VALUES)
    ax_l.legend(frameon=False, fontsize=9)
    ax_l.grid(alpha=0.25, linewidth=0.6)

    for k in K_VALUES:
        rows = sorted(
            (c for c in cells if c["k"] == k and "comet_system" in c), key=lambda c: c["lambda"]
        )
        if not rows:
            continue
        ax_r.plot(
            [c["stylo_dist"] for c in rows],
            [c["comet_system"] for c in rows],
            marker="o",
            color=_K_COLORS[k],
            label=f"k={k}",
            linewidth=1.5,
            alpha=0.9,
        )
        for c in rows:
            ax_r.annotate(
                f"{c['lambda']:g}",
                (c["stylo_dist"], c["comet_system"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color=_K_COLORS[k],
            )
    ax_r.invert_xaxis()  # style improves to the right
    ax_r.set_xlabel("stylo_dist  (improving →)")
    ax_r.set_ylabel("COMET (system)")
    ax_r.set_title("Style–adequacy trade-off (points labelled by λ)")
    ax_r.legend(frameon=False, fontsize=9)
    ax_r.grid(alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_judge_overlay(report: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    cells = report["cells"]
    judged = [c for c in cells if "judge_mean" in c]
    if not judged:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for k in K_VALUES:
        lam, val = _series(cells, k, "stylo_dist")
        if lam:
            ax.plot(
                lam, val, marker="o", color=_K_COLORS[k], label=f"stylo_dist k={k}", linewidth=1.6
            )
    ax.set_xlabel("λ  (register-rerank weight)")
    ax.set_ylabel("stylo_dist (lower is better)")
    ax.set_xticks(LAMBDA_VALUES)
    ax.grid(alpha=0.25, linewidth=0.6)

    ax_j = ax.twinx()
    for k in K_VALUES:
        rows = sorted((c for c in judged if c["k"] == k), key=lambda c: c["lambda"])
        if not rows:
            continue
        lam = [c["lambda"] for c in rows]
        phi = [c["judge_mean"] for c in rows]
        yerr = None
        if all("judge_ci" in c for c in rows):
            yerr = np.array(
                [
                    [c["judge_mean"] - c["judge_ci"][0] for c in rows],
                    [c["judge_ci"][1] - c["judge_mean"] for c in rows],
                ]
            )
        ax_j.errorbar(
            lam,
            phi,
            yerr=yerr,
            marker="s",
            markersize=7,
            linestyle="--",
            color=_K_COLORS[k],
            alpha=0.65,
            capsize=3,
            label=f"Φ (judge) k={k}",
        )
    ax_j.set_ylabel("Φ  LLM-as-Judge mean (higher is better)")

    handles = ax.get_legend_handles_labels()
    handles_j = ax_j.get_legend_handles_labels()
    ax.legend(handles[0] + handles_j[0], handles[1] + handles_j[1], frameon=False, fontsize=8)
    ax.set_title("Judge score overlaid on stylo_dist at the judged λ points")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _print_summary(report: dict) -> None:
    cols = ["tag", "k", "lambda", "stylo_dist", "comet_system", "judge_mean"] + [
        f"z_{n}" for n in CENTROID_FEATURES
    ]
    rows = []
    for c in report["cells"]:
        row = {
            "tag": c["tag"],
            "k": str(c["k"]),
            "lambda": f"{c['lambda']:g}",
            "stylo_dist": f"{c['stylo_dist']:.4f}",
            "comet_system": f"{c['comet_system']:.4f}" if "comet_system" in c else "-",
            "judge_mean": f"{c['judge_mean']:.3f}" if "judge_mean" in c else "-",
        }
        for n in CENTROID_FEATURES:
            row[f"z_{n}"] = f"{c['z'][n]:+.3f}"
        rows.append(row)
    for name, ref in report["references"].items():
        row = {
            "tag": name,
            "k": "-",
            "lambda": "-",
            "stylo_dist": f"{ref['stylo_dist']:.4f}",
            "comet_system": "-",
            "judge_mean": "-",
        }
        for n in CENTROID_FEATURES:
            row[f"z_{n}"] = f"{ref['z'][n]:+.3f}"
        rows.append(row)

    widths = {c: max(len(c), *(len(r[c]) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(r[c].ljust(widths[c]) for c in cols))

    print("\nMonotone trend in λ (Spearman ρ; negative ρ on stylo_dist = rerank helps)")
    for k_label, entry in report["trends_by_k"].items():
        sd = entry["stylo_dist"]
        parts = [f"stylo_dist ρ={sd['spearman_rho']:+.2f} p={sd['p']:.3f}"]
        if "comet" in entry:
            cm = entry["comet"]
            parts.append(f"COMET ρ={cm['spearman_rho']:+.2f} p={cm['p']:.3f}")
        print(f"  {k_label:<5} " + "   ".join(parts))
    pooled = report["trends_pooled"]
    print(
        f"  pooled stylo_dist ρ={pooled['stylo_dist']['spearman_rho']:+.2f} "
        f"p={pooled['stylo_dist']['p']:.4f}  (n={pooled['stylo_dist']['n_points']} cells)"
    )
    for n in CENTROID_FEATURES:
        t = pooled[f"z_{n}"]
        print(f"  pooled z_{n:<13} ρ={t['spearman_rho']:+.2f} p={t['p']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AFSP λ dose-response curves (RQ3).")
    parser.add_argument("--out_dir", default="outputs/sweep", help="sweep inference outputs")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--comet_results",
        default="results/sweep/comet_val.json",
        help="per-cell COMET map; omit/absent to build the style curves alone",
    )
    parser.add_argument("--results_path", default="results/sweep_curves_val.json")
    parser.add_argument("--fig_dir", default="docs/figures")
    parser.add_argument("--no-figures", dest="figures", action="store_false")
    parser.add_argument("--no-bootstrap", dest="bootstrap", action="store_false")
    parser.add_argument("--n_resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = build(
        Path(args.out_dir),
        args.split,
        comet_path=Path(args.comet_results),
        bootstrap=args.bootstrap,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    _print_summary(report)

    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {results_path}")

    if args.figures:
        fig_dir = Path(args.fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        figure_stylo_dist(report, fig_dir / "lambda_stylo_dist.png")
        figure_z_deviations(report, fig_dir / "lambda_z_deviations.png")
        figure_tradeoff(report, fig_dir / "lambda_style_adequacy.png")
        figure_judge_overlay(report, fig_dir / "lambda_judge_overlay.png")
        print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
