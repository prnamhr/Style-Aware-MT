"""
Does LoRA rank or epoch count move *register*, as distinct from adequacy?

Usage:
    python manage.py peft_register
    python manage.py peft_register --n_resamples 10000
    python manage.py peft_register --sweep_path results/peft_sweep_val.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

from src.eval._io import condition_path
from src.eval.metric_agreement import holm_bonferroni, permutation_p_floor
from src.eval.stylometrics import (
    aggregate,
    bootstrap_draws,
    distance_to_centroid,
    draw_intervals,
    feature_vector,
    signed_z,
)
from src.eval.stylometrics_ci import paired_diff

_CENTROID_PATH = Path("results/stylometrics_centroid.json")
_SWEEP_PATH = Path("results/peft_sweep_val.json")
_JUDGE_SEGMENT_DIR = Path("results/judge_val_segments")


def _load_predictions(path: Path) -> tuple[list[str], list[str]]:
    """Return ``(sources, predictions)`` for one sweep cell."""
    sources: list[str] = []
    preds: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sources.append(row.get("input", ""))
            preds.append(row.get("prediction", ""))
    return sources, preds


def _feature_matrix(texts: list[str]) -> np.ndarray:
    return np.asarray([feature_vector(t) for t in texts if t.strip()], dtype=float)


def _assert_pairable(tag: str, texts: list[str], matrix: np.ndarray) -> None:
    """Blank predictions drop out of the feature matrix and would desync the shared
    resample indices, so refuse to pair rather than silently misalign."""
    if matrix.shape[0] != len(texts):
        raise ValueError(
            f"'{tag}' has {len(texts) - matrix.shape[0]} blank prediction(s); they drop "
            f"out of the feature matrix and break index-for-index pairing across cells"
        )


def _judge_mean(tag: str, judge_dir: Path) -> tuple[float | None, int, float]:
    """Mean Phi for a cell if it was judged; ``(None, 0, 0.0)`` otherwise."""
    path = judge_dir / f"{tag}.jsonl"
    if not path.exists():
        return None, 0, 0.0
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    scores = [r.get("score") for r in rows if r.get("score") is not None]
    if not scores:
        return None, 0, 0.0
    return float(np.mean(scores)), len(scores), len(scores) / len(rows)


def build(
    sweep_path: Path,
    out_dir: Path,
    split: str,
    *,
    judge_dir: Path,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    centroid = json.loads(_CENTROID_PATH.read_text(encoding="utf-8"))
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    cells_meta = {c["tag"]: c for c in sweep["cells"]}

    loaded: dict[str, dict] = {}
    sources_ref: list[str] | None = None
    ref_tag = ""
    for tag, meta in cells_meta.items():
        path = condition_path(out_dir, tag, split)
        if not path.exists():
            print(f"skip {tag}: {path} not found")
            continue
        sources, preds = _load_predictions(path)
        if sources_ref is None:
            sources_ref, ref_tag = sources, tag
        elif sources != sources_ref:
            i = next(
                (k for k, (a, b) in enumerate(zip(sources_ref, sources)) if a != b),
                min(len(sources_ref), len(sources)),
            )
            raise ValueError(
                f"source mismatch between '{ref_tag}' and '{tag}' at segment {i}: the "
                f"paired bootstrap requires identical source order across cells"
            )
        matrix = _feature_matrix(preds)
        _assert_pairable(tag, preds, matrix)
        agg = aggregate(preds)
        dists, z = bootstrap_draws(matrix, centroid, n_resamples=n_resamples, seed=seed)
        phi, phi_n, phi_cov = _judge_mean(tag, judge_dir)
        loaded[tag] = {
            "draws": dists,
            "row": {
                "tag": tag,
                "r": meta["r"],
                "alpha_lora": meta["alpha"],
                "lr": meta["lr"],
                "epoch": meta["epoch"],
                "n": agg["n"],
                "eval_loss": meta["eval_loss"],
                "chrF": meta["chrF"],
                "BLEU": meta["BLEU"],
                "stylo_dist": distance_to_centroid(agg["mean"], centroid),
                "register_fit": meta.get("register_fit"),
                "z": signed_z(agg["mean"], centroid),
                "phi": phi,
                "phi_n": phi_n,
                "phi_coverage": phi_cov,
                **draw_intervals(dists, z, centroid, alpha=alpha),
            },
        }

    if not loaded:
        raise FileNotFoundError(f"no sweep cell in {sweep_path} has predictions under {out_dir}")

    rows = {t: loaded[t]["row"] for t in loaded}
    draws = {t: loaded[t]["draws"] for t in loaded}

    # --- epoch axis: paired within each (r, lr), which is the clean factorial contrast
    epoch_pairs = []
    by_config: dict[tuple[int, float], dict[int, str]] = {}
    for tag, row in rows.items():
        by_config.setdefault((row["r"], row["lr"]), {})[row["epoch"]] = tag
    for (r, lr), by_epoch in sorted(by_config.items()):
        if len(by_epoch) < 2:
            continue
        for e_lo, e_hi in itertools.combinations(sorted(by_epoch), 2):
            a, b = by_epoch[e_hi], by_epoch[e_lo]
            epoch_pairs.append(
                {
                    "r": r,
                    "lr": lr,
                    "a": a,
                    "b": b,
                    "epochs": f"{e_hi} vs {e_lo}",
                    "d_chrF": round(rows[a]["chrF"] - rows[b]["chrF"], 4),
                    "d_BLEU": round(rows[a]["BLEU"] - rows[b]["BLEU"], 4),
                    "d_eval_loss": round(rows[a]["eval_loss"] - rows[b]["eval_loss"], 6),
                    **paired_diff(draws[a], draws[b], alpha=alpha),
                }
            )

    # --- rank axis: paired within each (lr, epoch)
    rank_pairs = []
    by_le: dict[tuple[float, int], dict[int, str]] = {}
    for tag, row in rows.items():
        by_le.setdefault((row["lr"], row["epoch"]), {})[row["r"]] = tag
    for (lr, epoch), by_rank in sorted(by_le.items()):
        for r_lo, r_hi in itertools.combinations(sorted(by_rank), 2):
            a, b = by_rank[r_hi], by_rank[r_lo]
            rank_pairs.append(
                {
                    "lr": lr,
                    "epoch": epoch,
                    "a": a,
                    "b": b,
                    "ranks": f"r{r_hi} vs r{r_lo}",
                    "d_chrF": round(rows[a]["chrF"] - rows[b]["chrF"], 4),
                    "d_BLEU": round(rows[a]["BLEU"] - rows[b]["BLEU"], 4),
                    **paired_diff(draws[a], draws[b], alpha=alpha),
                }
            )

    # Multiplicity is live (protocol 5). The epoch and rank contrasts are one family of
    # register tests over the same eight cells, so Holm runs across both together rather
    # than pretending each axis was the only thing asked.
    family = epoch_pairs + rank_pairs
    for rec, rej in zip(family, holm_bonferroni([r["p_value"] for r in family], alpha=alpha)):
        rec["holm_significant"] = bool(rej)

    # --- selection validity: eval_loss against register and adequacy across cells.
    # Confirms DEVLOG 2026-07-25 on this grid rather than discovering it.
    tags = sorted(rows)
    loss = np.asarray([rows[t]["eval_loss"] for t in tags])
    axes = {}
    for name in ("stylo_dist", "chrF", "BLEU"):
        v = np.asarray([rows[t][name] for t in tags], dtype=float)
        res = stats.spearmanr(loss, v)
        axes[name] = {"rho": float(res.statistic), "p_value": float(res.pvalue), "n": len(tags)}
    judged = [t for t in tags if rows[t]["phi"] is not None]
    judged_epochs = sorted({rows[t]["epoch"] for t in judged})

    return {
        "split": split,
        "sweep_path": str(sweep_path),
        "out_dir": str(out_dir),
        "evidence_class": "exploratory",
        "evidence_note": (
            "sweep cells, no pre-specified contrast; register is bootstrapped per segment, "
            "adequacy is aggregate-only from the sweep record and has no interval, and Phi "
            "covers only cells "
            + ", ".join(judged)
            + " (epoch(s) "
            + ", ".join(str(e) for e in judged_epochs)
            + "), so the judge cannot address the epoch axis"
        ),
        "bootstrap": {
            "n_resamples": n_resamples,
            "seed": seed,
            "alpha": alpha,
            "paired": True,
            "n_segments": rows[tags[0]]["n"],
        },
        "cells": rows,
        "epoch_axis": epoch_pairs,
        "rank_axis": rank_pairs,
        "selection_validity": {
            "eval_loss_vs": axes,
            "n_cells": len(tags),
            "p_floor": permutation_p_floor(len(tags)),
            "note": (
                "eval_loss is a generation pre-filter, not a selection criterion "
                "(protocol 6, DEVLOG 2026-07-25). A NEGATIVE rho against stylo_dist means "
                "higher eval_loss goes with better register fit -- equivalently, picking "
                "the lowest-loss checkpoint picks the worst register."
            ),
        },
        "judged_cells": judged,
    }


def _ci(bounds, places: int = 4) -> str:
    return f"[{bounds[0]:+.{places}f}, {bounds[1]:+.{places}f}]"


def _table(rows: list[dict], cols: list[str]) -> None:
    if not rows:
        print("  (none)")
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
    cells = report["cells"]

    print(
        f"\nPEFT sweep: register vs adequacy  (split={report['split']}, "
        f"n={boot['n_segments']} segments, resamples={boot['n_resamples']}, "
        f"seed={boot['seed']})"
    )
    print("EXPLORATORY -- sweep cells, no pre-specified contrast. Leads, not corroboration.")
    print("stylo_dist: lower is better. chrF/BLEU are aggregate-only, no intervals.\n")

    order = sorted(cells, key=lambda t: (cells[t]["lr"], cells[t]["r"], cells[t]["epoch"]))
    rows = []
    for t in order:
        c = cells[t]
        rows.append(
            {
                "cell": t,
                "r": str(c["r"]),
                "lr": f"{c['lr']:.0e}",
                "ep": str(c["epoch"]),
                "eval_loss": f"{c['eval_loss']:.4f}",
                "stylo_dist": f"{c['stylo_dist']:.4f}",
                ci_col: _ci(c["stylo_dist_ci"]),
                "chrF": f"{c['chrF']:.2f}",
                "BLEU": f"{c['BLEU']:.2f}",
                "Phi": f"{c['phi']:.4f}" if c["phi"] is not None else "-",
            }
        )
    _table(rows, list(rows[0].keys()))

    print(f"\nEpoch axis, paired bootstrap on register  (a - b, {pct}% CI)")
    rows = []
    for rec in report["epoch_axis"]:
        rows.append(
            {
                "r": str(rec["r"]),
                "lr": f"{rec['lr']:.0e}",
                "epochs": rec["epochs"],
                "d_stylo": f"{rec['diff']:+.4f}",
                ci_col: _ci([rec["ci_low"], rec["ci_high"]]),
                "p": f"{rec['p_value']:.4f}",
                "sig": "*" if rec["significant"] else "",
                "holm": "*" if rec.get("holm_significant") else "",
                "d_chrF": f"{rec['d_chrF']:+.2f}",
                "d_BLEU": f"{rec['d_BLEU']:+.2f}",
                "d_loss": f"{rec['d_eval_loss']:+.4f}",
            }
        )
    _table(rows, list(rows[0].keys()))

    print(f"\nRank axis, paired bootstrap on register  (a - b, {pct}% CI)")
    rows = []
    for rec in report["rank_axis"]:
        rows.append(
            {
                "lr": f"{rec['lr']:.0e}",
                "ep": str(rec["epoch"]),
                "ranks": rec["ranks"],
                "d_stylo": f"{rec['diff']:+.4f}",
                ci_col: _ci([rec["ci_low"], rec["ci_high"]]),
                "p": f"{rec['p_value']:.4f}",
                "sig": "*" if rec["significant"] else "",
                "holm": "*" if rec.get("holm_significant") else "",
                "d_chrF": f"{rec['d_chrF']:+.2f}",
                "d_BLEU": f"{rec['d_BLEU']:+.2f}",
            }
        )
    _table(rows, list(rows[0].keys()))

    sv = report["selection_validity"]
    print(
        f"\nSelection validity: Spearman(eval_loss, axis) over {sv['n_cells']} cells "
        f"(two-sided p floor at this n = {sv['p_floor']:.4g})"
    )
    rows = [
        {"axis": k, "rho": f"{v['rho']:+.4f}", "p": f"{v['p_value']:.4f}"}
        for k, v in sv["eval_loss_vs"].items()
    ]
    _table(rows, ["axis", "rho", "p"])
    print(f"  {sv['note']}")

    n_ep = sum(1 for r in report["epoch_axis"] if r["significant"])
    n_rk = sum(1 for r in report["rank_axis"] if r["significant"])
    n_holm = sum(1 for r in report["epoch_axis"] + report["rank_axis"] if r.get("holm_significant"))
    family = len(report["epoch_axis"]) + len(report["rank_axis"])
    print(
        f"\nsig = {pct}% CI excludes 0; holm = survives Holm-Bonferroni across all "
        f"{family} register contrasts. Epoch axis: {n_ep}/{len(report['epoch_axis'])} "
        f"separate. Rank axis: {n_rk}/{len(report['rank_axis'])}. "
        f"{n_holm}/{family} survive the correction."
    )
    print(
        f"  Phi coverage: {', '.join(report['judged_cells']) or 'none'} "
        f"-- the epoch axis is unresolvable on the primary stylistic metric."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register vs adequacy across the PEFT sweep cells (exploratory)."
    )
    parser.add_argument("--sweep_path", default=str(_SWEEP_PATH))
    parser.add_argument("--out_dir", default="outputs/peft_sweep")
    parser.add_argument("--split", default="val")
    parser.add_argument("--judge_dir", default=str(_JUDGE_SEGMENT_DIR))
    parser.add_argument("--results_path", default=None)
    parser.add_argument("--n_resamples", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = build(
        Path(args.sweep_path),
        Path(args.out_dir),
        args.split,
        judge_dir=Path(args.judge_dir),
        n_resamples=args.n_resamples,
        alpha=args.alpha,
        seed=args.seed,
    )
    _print_summary(report)

    results_path = Path(args.results_path or f"results/peft_register_{args.split}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    main()
