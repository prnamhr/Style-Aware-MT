"""
Per-feature decomposition of the held-out register distance, by judge weight.

    python manage.py heldout_decomp
    python manage.py heldout_decomp --conditions peft rlsf_w3_2.0 --no-figure
    python manage.py heldout_decomp --trajectory
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from src.eval._io import condition_path
from src.eval.stylometrics import (
    HELDOUT_FEATURES,
    REWARD_FEATURES,
    SPLIT_FEATURES,
    feature_vector,
    subcentroid,
)

_SPLIT_CENTROID_PATH = Path("results/stylometrics_centroid_split.json")
_SELECT_PATH = "results/rlsf_select_{cell}.json"
_FIGURE_PATH = Path("docs/figures/heldout_decomp_by_omega.png")
_TRAJ_FIGURE_PATH = Path("docs/figures/heldout_traj_by_step.png")

# The frozen-PEFT initialization every RLSF arm starts from: the arms are read as
# movement away from it, not against each other.
REFERENCE = "peft"

# Judge weight omega_3 per condition. The reference is the arm's initialization and
# carries no weight of its own; it is placed at 0 so the figure has an anchor.
OMEGA = {"peft": 0.0, "rlsf_w3_0.0": 0.0, "rlsf_w3_2.0": 2.0, "rlsf_w3_6.0": 6.0}

CONDITIONS = ["peft", "rlsf_w3_0.0", "rlsf_w3_2.0", "rlsf_w3_6.0"]

# cell -> the val condition trained from it, for the checkpoint ladder
CELLS = {"w3_0.0": "rlsf_w3_0.0", "w3_2.0": "rlsf_w3_2.0", "w3_6.0": "rlsf_w3_6.0"}

# The checkpoints regenerated on val by notebooks/rlsf_trajectory_gpu.ipynb, and where they live.
STEPS = [100, 200, 400, 800, 1200]
TRAJ_DIR = Path("outputs/rlsf_traj")
REF_DIR = Path("outputs")

_TRAJ_RE = re.compile(r"^rlsf_w3_(?P<omega>\d+(?:\.\d+)?)(?:_step(?P<step>\d+))?$")


def omega_of(condition: str) -> float:
    """Judge weight of a condition, read from its name so trajectory tags need no table entry."""
    if condition in OMEGA:
        return OMEGA[condition]
    match = _TRAJ_RE.match(condition)
    if match is None:
        raise KeyError(f"no judge weight for condition '{condition}'")
    return float(match["omega"])


def traj_condition(cell: str, step: int) -> str:
    """Condition name of one checkpoint on the ladder, matching the generated file names."""
    return f"rlsf_{cell}_step{step}"


def _predictions(out_dir: Path, condition: str, split: str) -> list[str]:
    with condition_path(out_dir, condition, split).open(encoding="utf-8") as f:
        return [json.loads(line)["prediction"] for line in f if line.strip()]


def _feature_matrix(texts: list[str], features: list[str]) -> np.ndarray:
    """Segment x feature matrix in ``features`` order."""
    from src.eval.stylometrics import FEATURE_NAMES

    idx = [FEATURE_NAMES.index(name) for name in features]
    return np.asarray([feature_vector(t) for t in texts], dtype=float)[:, idx]


def z_draws(matrix: np.ndarray, centroid: dict, idx: np.ndarray, chunk: int = 1000) -> np.ndarray:
    """Bootstrap draws of the signed z vector, one row per resample."""
    mean = np.asarray(centroid["mean"], dtype=float)
    std = np.asarray(centroid["std"], dtype=float)
    blocks = np.array_split(idx, max(1, len(idx) // chunk))
    means = np.concatenate([matrix[block].mean(axis=1) for block in blocks])
    return (means - mean) / std


def decompose(z: dict[str, float], features: list[str]) -> dict:
    """Squared-distance contribution and share of the total, per feature."""
    squares = {name: float(z[name]) ** 2 for name in features}
    total = sum(squares.values())
    return {
        "dist": float(np.sqrt(total)),
        "contribution": squares,
        # A zero vector has no shares to report; it never occurs on real text.
        "share": {name: (squares[name] / total if total > 0 else 0.0) for name in features},
    }


def _interval(draws: np.ndarray, alpha: float) -> list[float]:
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return [float(np.percentile(draws, lo)), float(np.percentile(draws, hi))]


def _p_value(draws: np.ndarray) -> float:
    """Two-sided bootstrap p, matching src/eval/stylometrics_ci.py's convention."""
    return min(2.0 * min(float(np.mean(draws <= 0)), float(np.mean(draws >= 0))), 1.0)


def paired_delta(a: np.ndarray, b: np.ndarray, alpha: float) -> dict:
    """Interval on ``a - b`` over the shared resamples."""
    d = a - b
    lo, hi = _interval(d, alpha)
    return {
        "delta": float(d.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "p_value": _p_value(d),
        "significant": not (lo <= 0.0 <= hi),
    }


def checkpoint_ladder(cells: dict[str, str] = CELLS) -> dict:
    """Held-out distance per saved checkpoint per arm, and which tags rank in omega order."""
    rows: dict[str, dict] = {}
    selected: dict[str, str] = {}
    for cell in cells:
        path = Path(_SELECT_PATH.format(cell=cell))
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        rows[cell] = {r["tag"]: r for r in record["rows"]}
        selected[cell] = record["selected"]
    if not rows:
        return {}

    shared = [tag for tag in next(iter(rows.values())) if all(tag in r for r in rows.values())]
    order = sorted(cells, key=lambda c: omega_of(cells[c]))
    ladder = []
    for tag in shared:
        dists = [rows[cell][tag]["dist_heldout"] for cell in order]
        ladder.append(
            {
                "tag": tag,
                "dist_heldout": dict(zip(order, dists)),
                "marker_rate": {cell: rows[cell][tag]["marker_rate"] for cell in order},
                "z_marker_rate": {cell: rows[cell][tag]["z"]["marker_rate"] for cell in order},
                "monotone_in_omega": dists == sorted(dists),
            }
        )
    return {
        "cells": order,
        "selected": selected,
        "selected_matched": len(set(selected.values())) == 1,
        "steps": ladder,
        "n_monotone": sum(1 for row in ladder if row["monotone_in_omega"]),
    }


def build(
    out_dir: Path,
    split: str,
    *,
    conditions: list[str] = CONDITIONS,
    reference: str = REFERENCE,
    ref_dir: Path | None = None,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    centroid = json.loads(_SPLIT_CENTROID_PATH.read_text(encoding="utf-8"))
    full = subcentroid(centroid, SPLIT_FEATURES)

    dirs = {c: (Path(ref_dir) if c == reference and ref_dir else out_dir) for c in conditions}
    texts = {}
    for cond in conditions:
        path = condition_path(dirs[cond], cond, split)
        if not path.exists():
            print(f"skip {cond}: {path} not found")
            continue
        texts[cond] = _predictions(dirs[cond], cond, split)
    present = [c for c in conditions if c in texts]
    if reference not in present:
        raise FileNotFoundError(f"reference condition '{reference}' has no {split} inference file")

    counts = {c: len(texts[c]) for c in present}
    if len(set(counts.values())) != 1:
        raise ValueError(f"conditions differ in segment count: {counts}")
    n = counts[reference]

    # One index array shared by every condition: the differences below are paired
    # segment-for-segment, as in src/eval/stylometrics_ci.py.
    idx = np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))

    z_point: dict[str, dict[str, float]] = {}
    z_boot: dict[str, np.ndarray] = {}
    for cond in present:
        matrix = _feature_matrix(texts[cond], SPLIT_FEATURES)
        mean = np.asarray(full["mean"], dtype=float)
        std = np.asarray(full["std"], dtype=float)
        z_point[cond] = dict(zip(SPLIT_FEATURES, ((matrix.mean(axis=0) - mean) / std).tolist()))
        z_boot[cond] = z_draws(matrix, full, idx)

    held_cols = [SPLIT_FEATURES.index(name) for name in HELDOUT_FEATURES]
    reward_cols = [SPLIT_FEATURES.index(name) for name in REWARD_FEATURES]
    dist_boot = {
        cond: {
            "heldout": np.linalg.norm(z_boot[cond][:, held_cols], axis=1),
            "reward": np.linalg.norm(z_boot[cond][:, reward_cols], axis=1),
        }
        for cond in present
    }

    cells: dict[str, dict] = {}
    for cond in present:
        held_split = decompose(z_point[cond], HELDOUT_FEATURES)
        reward_split = decompose(z_point[cond], REWARD_FEATURES)
        row = {
            "n": counts[cond],
            "omega_3": omega_of(cond),
            "dist_heldout": held_split["dist"],
            "dist_reward": reward_split["dist"],
            "dist_heldout_ci": _interval(dist_boot[cond]["heldout"], alpha),
            "dist_reward_ci": _interval(dist_boot[cond]["reward"], alpha),
            "features": {},
        }
        for name in SPLIT_FEATURES:
            j = SPLIT_FEATURES.index(name)
            side = "heldout" if name in HELDOUT_FEATURES else "reward"
            block = held_split if side == "heldout" else reward_split
            row["features"][name] = {
                "side": side,
                "z": z_point[cond][name],
                "z_ci": _interval(z_boot[cond][:, j], alpha),
                "contribution": block["contribution"][name],
                "share": block["share"][name],
                "delta_vs_reference": paired_delta(
                    z_boot[cond][:, j], z_boot[reference][:, j], alpha
                ),
            }
        if cond != reference:
            for side in ("heldout", "reward"):
                row[f"dist_{side}_delta"] = paired_delta(
                    dist_boot[cond][side], dist_boot[reference][side], alpha
                )
        cells[cond] = row

    # Adjacent in omega_3, so the "is the damage monotone in judge weight" question is
    # answered by an interval rather than by reading the point estimates in order.
    arms = sorted((c for c in present if c != reference), key=lambda c: omega_of(c))
    adjacent = [
        {
            "a": b,
            "b": a,
            **paired_delta(dist_boot[b]["heldout"], dist_boot[a]["heldout"], alpha),
        }
        for a, b in zip(arms, arms[1:])
    ]

    return {
        "split": split,
        "out_dir": str(out_dir),
        "reference": reference,
        "reference_dir": str(dirs[reference]),
        "centroid": {"path": str(_SPLIT_CENTROID_PATH), "n_segments": centroid["n_segments"]},
        "heldout_features": HELDOUT_FEATURES,
        "reward_features": REWARD_FEATURES,
        "bootstrap": {
            "n_resamples": n_resamples,
            "seed": seed,
            "alpha": alpha,
            "paired": True,
            "n_segments": n,
        },
        "conditions": present,
        "cells": cells,
        "adjacent_in_omega": adjacent,
        "checkpoint_ladder": checkpoint_ladder(),
    }


def _slope_draws(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Least-squares slope of each row of ``y`` on ``x``, one slope per row."""
    xc = x - x.mean()
    return ((y - y.mean(axis=1, keepdims=True)) @ xc) / float(xc @ xc)


def trajectory(
    traj_dir: Path = TRAJ_DIR,
    split: str = "val",
    *,
    cells: dict[str, str] = CELLS,
    steps: list[int] = STEPS,
    ref_dir: Path = REF_DIR,
    reference: str = REFERENCE,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Held-out distance and marker inflation along the checkpoint ladder, on the held-out split."""
    centroid = json.loads(_SPLIT_CENTROID_PATH.read_text(encoding="utf-8"))
    full = subcentroid(centroid, SPLIT_FEATURES)

    texts = {reference: _predictions(Path(ref_dir), reference, split)}
    for cell in cells:
        for step in steps:
            cond = traj_condition(cell, step)
            path = condition_path(traj_dir, cond, split)
            if not path.exists():
                print(f"skip {cond}: {path} not found")
                continue
            texts[cond] = _predictions(traj_dir, cond, split)

    counts = {c: len(t) for c, t in texts.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"conditions differ in segment count: {counts}")
    n = counts[reference]

    # The same paired scheme as build(): one index array, so a difference between two
    # checkpoints is taken segment-for-segment rather than between two independent resamples.
    idx = np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))
    held_cols = [SPLIT_FEATURES.index(name) for name in HELDOUT_FEATURES]
    marker_col = SPLIT_FEATURES.index("marker_rate")

    z_point: dict[str, dict[str, float]] = {}
    z_boot: dict[str, np.ndarray] = {}
    dist_boot: dict[str, np.ndarray] = {}
    mean = np.asarray(full["mean"], dtype=float)
    std = np.asarray(full["std"], dtype=float)
    for cond, text in texts.items():
        matrix = _feature_matrix(text, SPLIT_FEATURES)
        z_point[cond] = dict(zip(SPLIT_FEATURES, ((matrix.mean(axis=0) - mean) / std).tolist()))
        z_boot[cond] = z_draws(matrix, full, idx)
        dist_boot[cond] = np.linalg.norm(z_boot[cond][:, held_cols], axis=1)

    def dist(cond: str) -> float:
        return decompose(z_point[cond], HELDOUT_FEATURES)["dist"]

    order = [c for c in sorted(cells, key=lambda c: omega_of(cells[c]))]
    quantities = ("dist_heldout", "z_marker_rate")
    slope_boot: dict[str, dict[str, np.ndarray]] = {q: {} for q in quantities}
    arms: dict[str, dict] = {}

    for cell in order:
        present = [s for s in steps if traj_condition(cell, s) in texts]
        if len(present) < 2:
            continue
        conds = [traj_condition(cell, s) for s in present]
        # log2 of the step ratio: the ladder doubles, so a slope reads as "per doubling of
        # training" rather than per optimizer step, where the late steps would dominate.
        x = np.log2(np.asarray(present, dtype=float) / present[0])

        draws = {
            "dist_heldout": np.column_stack([dist_boot[c] for c in conds]),
            "z_marker_rate": np.column_stack([z_boot[c][:, marker_col] for c in conds]),
        }
        point = {
            "dist_heldout": np.asarray([dist(c) for c in conds]),
            "z_marker_rate": np.asarray([z_point[c]["marker_rate"] for c in conds]),
        }
        ref_draw = {
            "dist_heldout": dist_boot[reference],
            "z_marker_rate": z_boot[reference][:, marker_col],
        }

        rows = []
        for i, (step, cond) in enumerate(zip(present, conds)):
            rows.append(
                {
                    "step": step,
                    "condition": cond,
                    "dist_heldout": point["dist_heldout"][i],
                    "dist_heldout_ci": _interval(draws["dist_heldout"][:, i], alpha),
                    "dist_heldout_delta": paired_delta(
                        draws["dist_heldout"][:, i], ref_draw["dist_heldout"], alpha
                    ),
                    "z": {name: z_point[cond][name] for name in SPLIT_FEATURES},
                    "z_marker_rate_ci": _interval(draws["z_marker_rate"][:, i], alpha),
                    "z_marker_rate_delta": paired_delta(
                        draws["z_marker_rate"][:, i], ref_draw["z_marker_rate"], alpha
                    ),
                }
            )

        growth = {}
        for q in quantities:
            slope_boot[q][cell] = _slope_draws(draws[q], x)
            lo, hi = _interval(slope_boot[q][cell], alpha)
            growth[q] = {
                "slope_per_doubling": float(_slope_draws(point[q][None, :], x)[0]),
                "ci_low": lo,
                "ci_high": hi,
                "p_value": _p_value(slope_boot[q][cell]),
                "significant": not (lo <= 0.0 <= hi),
                "endpoint_delta": paired_delta(draws[q][:, -1], draws[q][:, 0], alpha),
            }

        arms[cell] = {
            "condition": cells[cell],
            "omega_3": omega_of(cells[cell]),
            "steps": present,
            "rows": rows,
            "growth": growth,
        }

    # Growth rates compared between arms adjacent in omega_3: whether the ordering of the three
    # lines is readable at all, rather than three point estimates printed in order.
    ranked = [c for c in order if c in arms]
    adjacent = [
        {
            "quantity": q,
            "a": b,
            "b": a,
            **paired_delta(slope_boot[q][b], slope_boot[q][a], alpha),
        }
        for q in quantities
        for a, b in zip(ranked, ranked[1:])
    ]
    ordered_in_omega = {
        q: [arms[c]["growth"][q]["slope_per_doubling"] for c in ranked]
        == sorted(arms[c]["growth"][q]["slope_per_doubling"] for c in ranked)
        for q in quantities
    }

    by_step = []
    for step in steps:
        conds = [traj_condition(c, step) for c in ranked]
        if not all(c in texts for c in conds):
            continue
        dists = [dist(c) for c in conds]
        by_step.append(
            {
                "step": step,
                "dist_heldout": dict(zip(ranked, dists)),
                "z_marker_rate": {
                    c: z_point[traj_condition(c, step)]["marker_rate"] for c in ranked
                },
                "monotone_in_omega": dists == sorted(dists),
            }
        )

    return {
        "split": split,
        "traj_dir": str(traj_dir),
        "reference": reference,
        "reference_dir": str(ref_dir),
        "centroid": {"path": str(_SPLIT_CENTROID_PATH), "n_segments": centroid["n_segments"]},
        "heldout_features": HELDOUT_FEATURES,
        "bootstrap": {
            "n_resamples": n_resamples,
            "seed": seed,
            "alpha": alpha,
            "paired": True,
            "n_segments": n,
        },
        "reference_cell": {
            "dist_heldout": dist(reference),
            "dist_heldout_ci": _interval(dist_boot[reference], alpha),
            "z": {name: z_point[reference][name] for name in SPLIT_FEATURES},
            "z_marker_rate_ci": _interval(z_boot[reference][:, marker_col], alpha),
        },
        "cells": ranked,
        "arms": arms,
        "by_step": by_step,
        "adjacent_in_omega": adjacent,
        "ordered_in_omega": ordered_in_omega,
        "selected": {
            cell: json.loads(Path(_SELECT_PATH.format(cell=cell)).read_text(encoding="utf-8"))[
                "selected"
            ]
            for cell in ranked
            if Path(_SELECT_PATH.format(cell=cell)).exists()
        },
    }


def _table(rows: list[dict], cols: list[str]) -> None:
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def _ci(bounds: list[float], places: int = 3) -> str:
    return f"[{bounds[0]:+.{places}f}, {bounds[1]:+.{places}f}]"


def print_report(report: dict) -> None:
    boot = report["bootstrap"]
    pct = int(round((1 - boot["alpha"]) * 100))
    ref = report["reference"]
    cells = report["cells"]
    order = sorted(report["conditions"], key=lambda c: (c != ref, omega_of(c)))

    print(
        f"\nHeld-out register distance, decomposed  (split={report['split']}, "
        f"n={boot['n_segments']} segments, resamples={boot['n_resamples']}, seed={boot['seed']})"
    )
    print(
        f"distance over {', '.join(report['heldout_features'])}; "
        f"delta and share are against {ref}. Lower distance is better.\n"
    )

    rows = []
    for cond in order:
        c = cells[cond]
        for name in report["heldout_features"]:
            f = c["features"][name]
            d = f["delta_vs_reference"]
            rows.append(
                {
                    "condition": cond,
                    "w3": "-" if c["omega_3"] is None else f"{c['omega_3']:g}",
                    "feature": name,
                    "z": f"{f['z']:+.4f}",
                    f"z ci{pct}": _ci(f["z_ci"]),
                    f"d vs {ref}": f"{d['delta']:+.4f}" + ("*" if d["significant"] else " "),
                    "z^2": f"{f['contribution']:.5f}",
                    "share": f"{f['share']:6.1%}",
                    "dist": f"{c['dist_heldout']:.4f}",
                }
            )
    _table(rows, list(rows[0].keys()))

    print("\nReward-side features, same decomposition against the reward distance")
    rows = []
    for cond in order:
        c = cells[cond]
        for name in report["reward_features"]:
            f = c["features"][name]
            d = f["delta_vs_reference"]
            rows.append(
                {
                    "condition": cond,
                    "feature": name,
                    "z": f"{f['z']:+.4f}",
                    f"z ci{pct}": _ci(f["z_ci"]),
                    f"d vs {ref}": f"{d['delta']:+.4f}" + ("*" if d["significant"] else " "),
                    "share": f"{f['share']:6.1%}",
                    "dist": f"{c['dist_reward']:.4f}",
                }
            )
    _table(rows, list(rows[0].keys()))

    print(f"\nHeld-out distance against {ref}, paired bootstrap ({pct}% CI)")
    rows = []
    for cond in order:
        if cond == ref:
            continue
        d = cells[cond]["dist_heldout_delta"]
        rows.append(
            {
                "comparison": f"{cond} - {ref}",
                "delta": f"{d['delta']:+.4f}",
                f"ci{pct}": _ci([d["ci_low"], d["ci_high"]], 4),
                "p": f"{d['p_value']:.4f}",
                "sig": "*" if d["significant"] else "",
            }
        )
    for rec in report["adjacent_in_omega"]:
        rows.append(
            {
                "comparison": f"{rec['a']} - {rec['b']}",
                "delta": f"{rec['delta']:+.4f}",
                f"ci{pct}": _ci([rec["ci_low"], rec["ci_high"]], 4),
                "p": f"{rec['p_value']:.4f}",
                "sig": "*" if rec["significant"] else "",
            }
        )
    _table(rows, list(rows[0].keys()))

    ladder = report.get("checkpoint_ladder")
    if not ladder:
        return
    print("\nDev-slice checkpoint ladder: held-out distance per saved checkpoint")
    picks = ", ".join(f"{cell} -> {tag}" for cell, tag in ladder["selected"].items())
    rows = []
    for step in ladder["steps"]:
        row = {"tag": step["tag"]}
        for cell in ladder["cells"]:
            row[cell] = f"{step['dist_heldout'][cell]:.4f}"
        row["omega order"] = "yes" if step["monotone_in_omega"] else "NO"
        rows.append(row)
    _table(rows, list(rows[0].keys()))
    print(
        f"\nselected: {picks}. "
        f"{ladder['n_monotone']}/{len(ladder['steps'])} matched checkpoints rank in omega_3 order."
    )
    if not ladder["selected_matched"]:
        print(
            "The val arms are NOT matched on training length: the arm-level comparison above "
            "confounds judge weight with the checkpoint each arm's selection rule picked."
        )


def figure(report: dict, path: Path = _FIGURE_PATH) -> None:
    """Two panels: per-feature z by judge weight on val, and the dev checkpoint ladder."""
    import matplotlib.pyplot as plt

    cells = report["cells"]
    arms = sorted(
        (c for c in report["conditions"] if c != report["reference"]), key=lambda c: omega_of(c)
    )
    x = [omega_of(c) for c in arms]
    ref = report["reference"]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    for name in report["heldout_features"] + report["reward_features"]:
        held = name in report["heldout_features"]
        f = [cells[c]["features"][name] for c in arms]
        y = [cell["z"] for cell in f]
        err = np.array(
            [
                [cell["z"] - cell["z_ci"][0] for cell in f],
                [cell["z_ci"][1] - cell["z"] for cell in f],
            ]
        )
        ax_l.errorbar(
            x,
            y,
            yerr=err if held else None,
            marker="o" if held else "s",
            ms=5,
            lw=1.9 if held else 1.1,
            ls="-" if held else "--",
            capsize=3,
            alpha=1.0 if held else 0.55,
            label=name + ("" if held else " (reward)"),
        )
        ax_l.axhline(cells[ref]["features"][name]["z"], lw=0.6, ls=":", color="0.7")

    ax_l.axhline(0.0, lw=0.8, color="0.3")
    ax_l.set_xlabel(r"judge weight $\omega_3$")
    ax_l.set_ylabel("signed z from the target register")
    ax_l.set_title("val, at each arm's selected checkpoint")
    ax_l.legend(fontsize=7.5, ncol=2)

    ladder = report.get("checkpoint_ladder")
    if ladder:
        steps = [s for s in ladder["steps"] if s["tag"].startswith("step")]
        xs = [int(s["tag"].removeprefix("step")) for s in steps]
        tags = [s["tag"] for s in steps]
        for cell in ladder["cells"]:
            ys = [s["dist_heldout"][cell] for s in steps]
            (line,) = ax_r.plot(xs, ys, marker="o", ms=3.5, lw=1.6, label=cell)
            pick = ladder["selected"][cell]
            if pick in tags:
                i = tags.index(pick)
                ax_r.scatter(
                    xs[i], ys[i], s=110, facecolors="none", edgecolors=line.get_color(), lw=1.8
                )
        ax_r.set_xlabel("optimizer step")
        ax_r.set_ylabel("held-out register distance (dev)")
        ax_r.set_title("dev ladder; circled = the checkpoint evaluated on val")
        ax_r.legend(fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def print_trajectory(report: dict) -> None:
    boot = report["bootstrap"]
    pct = int(round((1 - boot["alpha"]) * 100))
    ref = report["reference"]
    ref_cell = report["reference_cell"]

    print(
        f"\nRegister trajectory across training  (split={report['split']}, "
        f"n={boot['n_segments']} segments, resamples={boot['n_resamples']}, seed={boot['seed']})"
    )
    print(
        f"distance over {', '.join(report['heldout_features'])}; deltas are against {ref}, "
        f"which sits at dist {ref_cell['dist_heldout']:.4f} and "
        f"z(marker_rate) {ref_cell['z']['marker_rate']:+.4f}.\n"
    )

    rows = []
    for cell in report["cells"]:
        arm = report["arms"][cell]
        for r in arm["rows"]:
            d, m = r["dist_heldout_delta"], r["z_marker_rate_delta"]
            rows.append(
                {
                    "cell": cell,
                    "w3": f"{arm['omega_3']:g}",
                    "step": str(r["step"]),
                    "dist": f"{r['dist_heldout']:.4f}",
                    f"dist ci{pct}": _ci(r["dist_heldout_ci"], 4),
                    f"d vs {ref}": f"{d['delta']:+.4f}" + ("*" if d["significant"] else " "),
                    "z marker": f"{r['z']['marker_rate']:+.4f}",
                    f"z ci{pct}": _ci(r["z_marker_rate_ci"], 4),
                    f"dz vs {ref}": f"{m['delta']:+.4f}" + ("*" if m["significant"] else " "),
                }
            )
    _table(rows, list(rows[0].keys()))

    first = report["arms"][report["cells"][0]]["steps"][0]
    print(f"\nGrowth per doubling of training, anchored at step {first}")
    rows = []
    for q in ("dist_heldout", "z_marker_rate"):
        for cell in report["cells"]:
            g = report["arms"][cell]["growth"][q]
            e = g["endpoint_delta"]
            rows.append(
                {
                    "quantity": q,
                    "cell": cell,
                    "w3": f"{report['arms'][cell]['omega_3']:g}",
                    "slope": f"{g['slope_per_doubling']:+.4f}",
                    f"ci{pct}": _ci([g["ci_low"], g["ci_high"]], 4),
                    "p": f"{g['p_value']:.4f}",
                    "sig": "*" if g["significant"] else "",
                    "last - first": f"{e['delta']:+.4f}" + ("*" if e["significant"] else " "),
                }
            )
    _table(rows, list(rows[0].keys()))

    print(f"\nGrowth rates between arms adjacent in omega_3, paired bootstrap ({pct}% CI)")
    rows = []
    for rec in report["adjacent_in_omega"]:
        rows.append(
            {
                "quantity": rec["quantity"],
                "comparison": f"{rec['a']} - {rec['b']}",
                "delta": f"{rec['delta']:+.4f}",
                f"ci{pct}": _ci([rec["ci_low"], rec["ci_high"]], 4),
                "p": f"{rec['p_value']:.4f}",
                "sig": "*" if rec["significant"] else "",
            }
        )
    _table(rows, list(rows[0].keys()))
    for q, ok in report["ordered_in_omega"].items():
        verdict = "rank in omega_3 order" if ok else "do NOT rank in omega_3 order"
        print(f"  {q}: the three growth rates {verdict}")

    print("\nHeld-out distance at matched training length")
    rows = []
    for step in report["by_step"]:
        row = {"step": str(step["step"])}
        for cell in report["cells"]:
            row[cell] = f"{step['dist_heldout'][cell]:.4f}"
        row["omega order"] = "yes" if step["monotone_in_omega"] else "NO"
        rows.append(row)
    _table(rows, list(rows[0].keys()))
    n_mono = sum(1 for step in report["by_step"] if step["monotone_in_omega"])
    print(f"\n{n_mono}/{len(report['by_step'])} matched steps rank in omega_3 order.")
    if report.get("selected"):
        picks = ", ".join(f"{cell} -> {tag}" for cell, tag in report["selected"].items())
        print(f"dev-slice selection picked {picks}; those points are circled in the figure.")


def trajectory_figure(report: dict, path: Path = _TRAJ_FIGURE_PATH) -> None:
    """Held-out distance and marker inflation against optimizer step, one line per arm."""
    import matplotlib.pyplot as plt

    ref_cell = report["reference_cell"]
    panels = (
        ("dist_heldout", "held-out register distance", ref_cell["dist_heldout"]),
        ("z_marker_rate", "signed z, marker_rate", ref_cell["z"]["marker_rate"]),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharex=True)
    for ax, (key, ylabel, ref_value) in zip(axes, panels):
        for cell in report["cells"]:
            arm = report["arms"][cell]
            rows = arm["rows"]
            xs = [r["step"] for r in rows]
            ys = [
                r["dist_heldout"] if key == "dist_heldout" else r["z"]["marker_rate"] for r in rows
            ]
            slope = arm["growth"][key]["slope_per_doubling"]
            (line,) = ax.plot(
                xs,
                ys,
                marker="o",
                ms=4,
                lw=1.7,
                label=rf"$\omega_3$={arm['omega_3']:g}  ({slope:+.3f}/doubling)",
            )
            delta = [r[f"{key}_delta"] for r in rows]
            ax.fill_between(
                xs,
                [ref_value + d["ci_low"] for d in delta],
                [ref_value + d["ci_high"] for d in delta],
                color=line.get_color(),
                alpha=0.16,
                lw=0,
            )
            pick = report.get("selected", {}).get(cell, "")
            if pick.startswith("step") and int(pick.removeprefix("step")) in xs:
                i = xs.index(int(pick.removeprefix("step")))
                ax.scatter(
                    xs[i], ys[i], s=110, facecolors="none", edgecolors=line.get_color(), lw=1.8
                )
        ax.axhline(ref_value, lw=0.9, ls="--", color="0.4", label=report["reference"])
        ax.set_xscale("log", base=2)
        ax.set_xticks([r["step"] for r in report["arms"][report["cells"][0]]["rows"]])
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    axes[0].set_title("drift from the target register")
    axes[1].set_title("marker inflation, the held-out feature the reward never saw")
    fig.suptitle(
        f"bands: 95% paired interval on the gap to {report['reference']}, drawn about its level",
        y=0.03,
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose the held-out register distance per feature, by judge weight."
    )
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--split", default="val", help="output split tag (default: val)")
    parser.add_argument("--out_dir", default=None, help="inference output directory")
    parser.add_argument(
        "--ref_dir", default=None, help=f"where the reference lives (default: {REF_DIR})"
    )
    parser.add_argument(
        "--trajectory",
        action="store_true",
        help="report the checkpoint ladder regenerated on the split instead of the arms",
    )
    parser.add_argument("--steps", nargs="+", type=int, default=STEPS)
    parser.add_argument("--results_path", default=None)
    parser.add_argument("--figure_path", default=None)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--n_resamples", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.trajectory:
        report = trajectory(
            Path(args.out_dir or TRAJ_DIR),
            args.split,
            steps=args.steps,
            ref_dir=Path(args.ref_dir or REF_DIR),
            reference=args.reference,
            n_resamples=args.n_resamples,
            alpha=args.alpha,
            seed=args.seed,
        )
        print_trajectory(report)
    else:
        report = build(
            Path(args.out_dir or REF_DIR),
            args.split,
            conditions=args.conditions,
            reference=args.reference,
            ref_dir=Path(args.ref_dir) if args.ref_dir else None,
            n_resamples=args.n_resamples,
            alpha=args.alpha,
            seed=args.seed,
        )
        print_report(report)

    stem = "heldout_traj" if args.trajectory else "heldout_decomp"
    results_path = Path(args.results_path or f"results/{stem}_{args.split}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {results_path}")

    if not args.no_figure:
        default = _TRAJ_FIGURE_PATH if args.trajectory else _FIGURE_PATH
        draw = trajectory_figure if args.trajectory else figure
        draw(report, Path(args.figure_path or default))


if __name__ == "__main__":
    main()
