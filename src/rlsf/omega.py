"""
Re-argmax the cached best-of-N pool under each omega cell. Free: no GPU, no paid calls.

Ranking is scale-invariant, so a pool paid for once ranks every weighting in the grid.

Usage:
    python manage.py rlsf_omega
    python manage.py rlsf_omega --pool outputs/rlsf/pool.jsonl --subgroup 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.stylometrics import aggregate, distance_to_centroid, features
from src.rlsf.config import drift_rule, grid_reward_configs, load_config
from src.rlsf.pool import read_pool, sidecar
from src.rlsf.reward import compute_rewards, group_normalize, load_centroid

# The same quarter-of-groups threshold the reward-path smoke fails on, declared once there.
from src.rlsf.smoke import _MAX_DEGENERATE


def flatten(rows: list[dict]) -> tuple[int, list[str], list[str], list[str], dict]:
    """The pool as the reward path wants it: N, then flat arrays in segment-major order."""
    if not rows:
        raise ValueError("the pool is empty")
    n = len(rows[0]["hyps"])
    sources: list[str] = []
    refs: list[str] = []
    hyps: list[str] = []
    raw: dict[str, list[float]] = {}
    for row in rows:
        if len(row["hyps"]) != n:
            raise ValueError(
                f"pool row {row['idx']} carries {len(row['hyps'])} completions, not {n}; "
                f"a ragged pool cannot be grouped"
            )
        sources += [row["source"]] * n
        refs += [row["reference"]] * n
        hyps += row["hyps"]
        for name, values in row["scores"].items():
            if len(values) != n:
                raise ValueError(f"pool row {row['idx']} has {len(values)} {name} scores, not {n}")
            raw.setdefault(name, []).extend(values)
    return n, sources, refs, hyps, raw


def z_matrix(hyps: list[str], centroid: dict) -> np.ndarray:
    """Per-sample z-scores of the centroid features, in centroid order."""
    names = centroid["features"]
    matrix = np.asarray([[features(h)[name] for name in names] for h in hyps], dtype=float)
    mean = np.asarray(centroid["mean"], dtype=float)
    std = np.asarray(centroid["std"], dtype=float)
    return (matrix - mean) / std


def argmax_picks(rewards: np.ndarray, feasible: np.ndarray, n: int) -> list[int | None]:
    """Index of each group's highest-reward feasible sample; None where none was feasible."""
    picks: list[int | None] = []
    for start in range(0, len(rewards), n):
        candidates = [
            start + i
            for i in range(n)
            if feasible[start + i] and np.isfinite(rewards[start + i])
        ]
        picks.append(max(candidates, key=lambda j: rewards[j]) if candidates else None)
    return picks


def random_picks(feasible: np.ndarray, n: int, seed: int) -> list[int | None]:
    """One feasible sample per group, drawn uniformly: the noise floor reranking beats."""
    rng = np.random.default_rng(seed)
    picks: list[int | None] = []
    for start in range(0, len(feasible), n):
        candidates = [start + i for i in range(n) if feasible[start + i]]
        picks.append(int(rng.choice(candidates)) if candidates else None)
    return picks


def marker_shift(z_all: np.ndarray, picks: list[int | None], n: int, column: int) -> dict:
    """Feature z of each pick minus its own group's mean, averaged over segments.

    Paired within the segment, so the register spread between one source segment and the
    next drops out of the error and what is left is what picking the argmax did.
    """
    deltas = [
        float(z_all[pick, column] - z_all[g * n : (g + 1) * n, column].mean())
        for g, pick in enumerate(picks)
        if pick is not None
    ]
    if not deltas:
        return {"segments": 0, "delta": float("nan"), "se": float("nan")}
    d = np.asarray(deltas, dtype=float)
    se = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else float("nan")
    return {"segments": int(d.size), "delta": float(d.mean()), "se": se}


def pick_reading(
    hyps: list[str],
    refs: list[str],
    raw: dict[str, list[float]],
    picks: list[int | None] | None,
    *,
    n: int,
    centroid: dict,
    z_all: np.ndarray,
    feature: str,
) -> dict:
    """What one selection rule's picks look like against yardsticks outside the reward.

    ``picks`` of None reads every sample: the pool is the baseline the shift is measured
    from, so its own shift is zero by construction rather than estimated.
    """
    whole = picks is None
    chosen = list(range(len(hyps))) if whole else [p for p in picks if p is not None]
    if not chosen:
        raise ValueError("no group had a feasible sample, so there is nothing to read")
    texts = [hyps[p] for p in chosen]
    lengths = np.asarray([len(t.split()) for t in texts], dtype=float)
    ref_lengths = np.asarray([max(len(refs[p].split()), 1) for p in chosen], dtype=float)
    column = centroid["features"].index(feature)
    return {
        "picked": len(chosen),
        "unpicked_groups": 0 if whole else len(picks) - len(chosen),
        "components": {
            name: float(np.nanmean(np.asarray(values, dtype=float)[chosen]))
            for name, values in raw.items()
        },
        # Distance of the picked set's mean feature vector to the target register. Not a
        # reward term, so it reads the picks rather than restating what selected them.
        "stylo_dist": float(distance_to_centroid(aggregate(texts)["mean"], centroid)),
        "length_mean": float(lengths.mean()),
        "length_ratio_mean": float((lengths / ref_lengths).mean()),
        f"{feature}_shift": (
            {"segments": len(hyps) // n, "delta": 0.0, "se": 0.0}
            if whole
            else marker_shift(z_all, picks, n, column)
        ),
    }


def component_degeneracy(raw: dict[str, list[float]], feasible: np.ndarray, n: int) -> dict:
    """Per component, the fraction of groups it cannot separate on its own."""
    out = {}
    for name, values in raw.items():
        arr = np.asarray(values, dtype=float)
        flat = 0
        for start in range(0, arr.size, n):
            sl = slice(start, start + n)
            if not np.any(group_normalize(arr[sl], feasible[sl])):
                flat += 1
        groups = arr.size // n
        out[name] = {"groups": groups, "degenerate": flat, "degenerate_frac": flat / groups}
    return out


def cell_reading(
    name: str,
    rc,
    sources: list[str],
    hyps: list[str],
    refs: list[str],
    raw: dict[str, list[float]],
    *,
    n: int,
    subgroup: int | None,
    centroid: dict,
    z_all: np.ndarray,
    feature: str,
) -> dict:
    """One omega cell: how well its reward separates a group, and what its argmax picks."""
    rewards, feasible, log = compute_rewards(
        sources, hyps, refs, cfg=rc, group_size=n, component_scores=raw, centroid=centroid
    )
    reading = {
        "cell": name,
        "weights": {k: round(v, 4) for k, v in rc.weights.items()},
        "n": n,
        "feasible": log.n_feasible,
        "unmeasured": log.n_unmeasured,
        "degenerate_frac": log.degenerate_frac,
        "degenerate_groups": log.degenerate_groups,
        "groups": log.n_groups,
        "picks": pick_reading(
            hyps, refs, raw, argmax_picks(rewards, feasible, n),
            n=n, centroid=centroid, z_all=z_all, feature=feature,
        ),
    }
    if subgroup and subgroup < n and n % subgroup == 0:
        # The pool's N samples are exchangeable draws from one segment, so splitting them
        # into groups of the training size estimates the degeneracy training will see.
        _, _, sub = compute_rewards(
            sources, hyps, refs,
            cfg=rc, group_size=subgroup, component_scores=raw, centroid=centroid,
        )
        reading["subgroup"] = {
            "group_size": subgroup,
            "groups": sub.n_groups,
            "degenerate_groups": sub.degenerate_groups,
            "degenerate_frac": sub.degenerate_frac,
        }
    return reading


def select(
    readings: list[dict],
    cells: dict[str, dict],
    *,
    max_degenerate: float,
    warn_shift: float,
    feature: str,
) -> dict:
    """The cell that feeds the training run, and why each other one does not."""
    rejected, kept = {}, []
    for r in readings:
        # One hard gate. A cell whose groups have no reward spread trains nothing, whatever
        # its argmax would have picked, so it is out before register fit is compared.
        deg = r.get("subgroup", r)["degenerate_frac"]
        at = r.get("subgroup", {}).get("group_size", r["n"])
        if deg > max_degenerate:
            rejected[r["cell"]] = (
                f"{deg:.0%} of groups have no reward spread at G={at}, past {max_degenerate:.0%}"
            )
        else:
            kept.append(r)

    if not kept:
        return {
            "cell": None,
            "reason": (
                "no cell gives most groups a gradient. A reward that cannot separate a group "
                "of samples is a finding about the reward, not a blocked run"
            ),
            "rejected": rejected,
        }
    # Register distance already counts `feature` among its four components, so a cell that
    # wins by inflating markers pays for it here; the shift is reported, not vetoed on.
    winner = min(kept, key=lambda r: (r["picks"]["stylo_dist"], cells[r["cell"]]["w_judge"]))
    shift = winner["picks"][f"{feature}_shift"]
    return {
        "cell": winner["cell"],
        "weights": cells[winner["cell"]],
        "reason": (
            f"lowest register distance ({winner['picks']['stylo_dist']:.3f}) of the "
            f"{len(kept)} cell{'s' if len(kept) != 1 else ''} that give most groups a gradient"
        ),
        "goodhart": {
            "feature": feature,
            "delta": shift["delta"],
            "se": shift["se"],
            "threshold": warn_shift,
            # Best-of-N is the ceiling on what this reward can pull the policy toward, so a
            # pick already past the drift band says GRPO would trip the stop rule.
            "over_threshold": bool(shift["delta"] > warn_shift),
        },
        "rejected": rejected,
    }


def _row(reading: dict, feature: str) -> str:
    p, shift = reading["picks"], reading["picks"][f"{feature}_shift"]
    sub = reading.get("subgroup")
    comp = p["components"]
    return (
        f"  {reading['cell']:9s} {reading['degenerate_frac']:7.0%} "
        f"{(sub['degenerate_frac'] if sub else float('nan')):7.0%} "
        f"{p['picked']:6d} "
        + " ".join(f"{comp[k]:7.3f}" for k in sorted(comp))
        + f" {p['stylo_dist']:10.3f} {shift['delta']:+8.3f} +/- {shift['se']:.3f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank the omega grid over a cached pool.")
    parser.add_argument("--config", default="configs/rlsf.yaml")
    parser.add_argument("--pool", default=None, help="default: output.pool")
    parser.add_argument("--manifest", default=None, help="default: <pool>_manifest.json")
    parser.add_argument(
        "--subgroup",
        type=int,
        default=None,
        help="also report degeneracy at this group size; default: rollout.group_size",
    )
    parser.add_argument("--seed", type=int, default=None, help="random anchor; default: rlsf.seed")
    parser.add_argument("--out", default=None, help="default: <pool>_omega.json")
    parser.add_argument(
        "--allow_flat_judge",
        action="store_true",
        help="rank a pool whose judge term was held flat; the grid then varies nothing",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # No caps needed: this command spends nothing, and refusing to read a pool that is
    # already paid for because a training cap is null would help no one.
    cfg = load_config(args.config, require_caps=False)
    pool_path = Path(args.pool or cfg["output"]["pool"])
    man_path = Path(args.manifest) if args.manifest else sidecar(pool_path, "manifest.json")
    feature = drift_rule(cfg).feature
    subgroup = args.subgroup or cfg["rlsf"]["rollout"]["group_size"]

    if man_path.exists():
        held = json.loads(man_path.read_text(encoding="utf-8"))["pool"]["held_flat"]
        if "judge" in held and not args.allow_flat_judge:
            raise SystemExit(
                f"{man_path} records the judge term held flat, so every cell in the grid "
                f"reduces to the same ranking and omega cannot be selected from this pool. "
                f"Rebuild it with paid judge calls, or pass --allow_flat_judge to look anyway."
            )
    else:
        print(f"note: no manifest at {man_path}; this pool's provenance is unrecorded")

    rows = read_pool(pool_path)
    n, sources, refs, hyps, raw = flatten(rows)
    centroid = load_centroid(cfg["data"]["centroid_file"])
    z_all = z_matrix(hyps, centroid)
    print(f"{pool_path}: {len(rows)} segments x N={n} = {len(hyps)} samples")

    grid = grid_reward_configs(cfg)
    cells = {c["name"]: c for c in cfg["rlsf"]["weight_grid"]["cells"]}
    readings = [
        cell_reading(
            name, rc, sources, hyps, refs, raw,
            n=n, subgroup=subgroup, centroid=centroid, z_all=z_all, feature=feature,
        )
        for name, rc in grid
    ]

    # Feasibility does not depend on the weights, so any cell's mask reads the components.
    _, feasible, _ = compute_rewards(
        sources, hyps, refs, cfg=grid[0][1], group_size=n,
        component_scores=raw, centroid=centroid,
    )
    per_component = component_degeneracy(raw, feasible, n)
    anchors = {
        key: pick_reading(
            hyps, refs, raw, picks, n=n, centroid=centroid, z_all=z_all, feature=feature
        )
        for key, picks in (
            ("random", random_picks(feasible, n, args.seed or cfg["rlsf"]["seed"])),
            ("pool", None),
        )
    }

    verdict = select(
        readings, cells,
        max_degenerate=_MAX_DEGENERATE, warn_shift=drift_rule(cfg).min_delta, feature=feature,
    )

    print(f"\nper-component degeneracy at N={n}: what each term can separate on its own")
    for name, stats in per_component.items():
        print(f"  {name:8s} {stats['degenerate']}/{stats['groups']} flat groups "
              f"({stats['degenerate_frac']:.0%})")

    comps = sorted(readings[0]["picks"]["components"])
    print(
        f"\n{'cell':11s} {'deg@' + str(n):>7s} {'deg@' + str(subgroup):>7s} {'picks':>6s} "
        + " ".join(f"{c:>7s}" for c in comps)
        + f" {'stylo':>10s} {feature + ' dz':>8s}"
    )
    for reading in readings:
        print(_row(reading, feature))
    for key, anchor in anchors.items():
        shift = anchor[f"{feature}_shift"]
        print(
            f"  {key:9s} {'':7s} {'':7s} {anchor['picked']:6d} "
            + " ".join(f"{anchor['components'][c]:7.3f}" for c in comps)
            + f" {anchor['stylo_dist']:10.3f} {shift['delta']:+8.3f} +/- {shift['se']:.3f}"
        )

    print(
        f"\nDegeneracy at N={n} and at G={subgroup} are not comparable: a larger group "
        f"degenerates less by construction, because more draws is more chances to differ. "
        f"G={subgroup} is the training-time figure."
    )
    if verdict["cell"]:
        g = verdict["goodhart"]
        print(f"\nselected {verdict['cell']}: {verdict['reason']}")
        print(f"  set rlsf.reward to {verdict['weights']}")
        print(
            f"  {feature} of its picks sits {g['delta']:+.2f} +/- {g['se']:.2f} over their own "
            f"groups, against the {g['threshold']:.2f} drift band"
        )
        if g["over_threshold"]:
            print(
                "  best-of-N is the ceiling on what this reward can pull toward, and it is "
                "already past the band: expect GRPO to trip the drift stop"
            )
    else:
        print(f"\nno cell selected: {verdict['reason']}")
    for name, why in verdict["rejected"].items():
        print(f"  rejected {name}: {why}")

    out_path = Path(args.out) if args.out else sidecar(pool_path, "omega.json")
    gates = {"max_degenerate": _MAX_DEGENERATE, "max_shift": drift_rule(cfg).min_delta}
    out_path.write_text(
        json.dumps(
            {
                "pool": str(pool_path),
                "segments": len(rows),
                "n": n,
                "subgroup": subgroup,
                "feature": feature,
                "gates": gates,
                "per_component_degeneracy": per_component,
                "cells": readings,
                "anchors": anchors,
                "selection": verdict,
                "caveats": [
                    "Dev-slice figures select the reward weights and are never reported as a "
                    "result; val remains the reported split.",
                    f"Degeneracy at N={n} and at G={subgroup} are not comparable; larger "
                    f"groups degenerate less by construction.",
                    "Best-of-N reranking of a frozen policy is not GRPO. It bounds what the "
                    "reward can select for, not what training will do with it.",
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
