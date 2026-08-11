"""
Re-argmax the cached best-of-N pool under each omega cell.
Usage:
    python manage.py rlsf_omega
    python manage.py rlsf_omega --pool outputs/rlsf/pool.jsonl --subgroup 4
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np

from src.eval.stylometrics import (
    HELDOUT_FEATURES,
    REWARD_FEATURES,
    aggregate,
    bootstrap_draws,
    distance_to_centroid,
    feature_vector,
    features,
    subcentroid,
)
from src.eval.stylometrics_ci import paired_diff
from src.rlsf.config import (
    drift_rule,
    exploratory_reward_configs,
    grid_reward_configs,
    load_config,
)
from src.rlsf.pool import read_pool, sidecar
from src.rlsf.reward import (
    compute_rewards,
    group_normalize,
    load_centroid,
    overlap_scores,
    stylo_scores,
)

# The same quarter-of-groups threshold the reward-path smoke fails on, declared once there.
from src.rlsf.smoke import _MAX_DEGENERATE

_PRE_REGISTERED = "pre-registered"

_ALPHA = 0.05
_RESAMPLES = 2000

_DIST_LABELS = {
    "stylo_dist": "stylo",
    "reward_dist": "reward",
    "reward_per_sample": "per-samp",
    "heldout_dist": "heldout",
}


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


def argmax_picks(
    rewards: np.ndarray, feasible: np.ndarray, n: int, *, seed: int = 0
) -> list[int | None]:
    # A stream of its own, so a tie-break is not one of the draws the random anchor makes.
    rng = np.random.default_rng([seed, 1])
    picks: list[int | None] = []
    for start in range(0, len(rewards), n):
        candidates = [
            start + i
            for i in range(n)
            if feasible[start + i] and np.isfinite(rewards[start + i])
        ]
        if not candidates:
            picks.append(None)
            continue
        best = max(rewards[j] for j in candidates)
        top = [j for j in candidates if rewards[j] == best]
        picks.append(top[0] if len(top) == 1 else int(rng.choice(top)))
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
    dists: dict[str, dict] | None = None,
) -> dict:
    """What one selection rule's picks look like against yardsticks outside the reward.

    ``picks`` of None reads every sample: the pool is the baseline the shift is measured
    from, so its own shift is zero by construction rather than estimated.
    ``dists`` names further centroids to measure against, each reported as ``<name>_dist``.
    """
    whole = picks is None
    chosen = list(range(len(hyps))) if whole else [p for p in picks if p is not None]
    if not chosen:
        raise ValueError("no group had a feasible sample, so there is nothing to read")
    texts = [hyps[p] for p in chosen]
    mean_vector = aggregate(texts)["mean"]
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
        "stylo_dist": float(distance_to_centroid(mean_vector, centroid)),
        **{
            f"{name}_dist": float(distance_to_centroid(mean_vector, sub))
            for name, sub in (dists or {}).items()
        },
        **{
            f"{name}_per_sample": float(
                np.mean([distance_to_centroid(features(t), sub) for t in texts])
            )
            for name, sub in (dists or {}).items()
        },
        "length_mean": float(lengths.mean()),
        "length_ratio_mean": float((lengths / ref_lengths).mean()),
        f"{feature}_shift": (
            {"segments": len(hyps) // n, "delta": 0.0, "se": 0.0}
            if whole
            else marker_shift(z_all, picks, n, column)
        ),
    }


def derive_components(
    raw: dict[str, list[float]],
    hyps: list[str],
    refs: list[str],
    needed: set[str],
    stylo_centroid: dict,
) -> list[str]:
    """Fill in components the pool did not cache. All of them local, none of them paid for."""
    added = []
    for name in sorted(needed - set(raw)):
        if name == "chrf":
            raw[name] = overlap_scores(hyps, refs, "chrf")
        elif name == "stylo":
            raw[name] = stylo_scores(hyps, stylo_centroid)
        else:
            raise ValueError(
                f"a cell needs a {name!r} score that {sorted(raw)} does not carry and that "
                f"cannot be derived from the pool's text. Rebuild the pool with it."
            )
        added.append(name)
    return added


def cell_picks(
    rc, sources, hyps, refs, raw, *, n: int, centroid: dict, seed: int = 0
) -> list[int | None]:
    """One cell's argmax picks, without the rest of its reading."""
    rewards, feasible, _ = compute_rewards(
        sources, hyps, refs, cfg=rc, group_size=n, component_scores=raw, centroid=centroid
    )
    return argmax_picks(rewards, feasible, n, seed=seed)


def paired_stylo(
    baseline: list[int | None], cell: list[int | None], hyps: list[str], centroid: dict
) -> dict:
    """Register distance of two selection rules over only the groups where both picked."""
    pairs = [(b, c) for b, c in zip(baseline, cell) if b is not None and c is not None]
    if not pairs:
        return {"groups": 0, "baseline_stylo": float("nan"), "cell_stylo": float("nan")}
    return {
        "groups": len(pairs),
        "baseline_stylo": float(
            distance_to_centroid(aggregate([hyps[b] for b, _ in pairs])["mean"], centroid)
        ),
        "cell_stylo": float(
            distance_to_centroid(aggregate([hyps[c] for _, c in pairs])["mean"], centroid)
        ),
    }


def heldout_draws(
    picks: list[int | None], hyps: list[str], centroid: dict, *, n_resamples: int, seed: int
) -> np.ndarray | None:
    texts = [hyps[p] for p in picks if p is not None]
    matrix = np.asarray([feature_vector(t) for t in texts if t.strip()], dtype=float)
    if matrix.shape[0] != len(texts):
        return None
    return bootstrap_draws(matrix, centroid, n_resamples=n_resamples, seed=seed)[0]


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
    dists: dict[str, dict] | None = None,
    evidence_class: str = _PRE_REGISTERED,
    seed: int = 0,
) -> tuple[dict, list[int | None]]:
    """One omega cell: how well its reward separates a group, and what its argmax picks."""
    rewards, feasible, log = compute_rewards(
        sources, hyps, refs, cfg=rc, group_size=n, component_scores=raw, centroid=centroid
    )
    picks = argmax_picks(rewards, feasible, n, seed=seed)
    reading = {
        "cell": name,
        "evidence_class": evidence_class,
        "weights": {k: round(v, 4) for k, v in rc.weights.items()},
        "normalization": rc.normalization,
        "judge_gate": rc.judge_gate,
        "n": n,
        "feasible": log.n_feasible,
        "unmeasured": log.n_unmeasured,
        "degenerate_frac": log.degenerate_frac,
        "degenerate_groups": log.degenerate_groups,
        "groups": log.n_groups,
        # A gate strict enough to empty every group is a reading about the gate, not a crash.
        "picks": None if all(p is None for p in picks) else pick_reading(
            hyps, refs, raw, picks,
            n=n, centroid=centroid, z_all=z_all, feature=feature, dists=dists,
        ),
    }
    if subgroup and subgroup < n and n % subgroup == 0:
        # The pool's N samples are exchangeable draws from one segment, so splitting them
        # into groups of the training size estimates the degeneracy training will see.
        sub_rewards, sub_feasible, sub = compute_rewards(
            sources, hyps, refs,
            cfg=rc, group_size=subgroup, component_scores=raw, centroid=centroid,
        )
        sub_picks = argmax_picks(sub_rewards, sub_feasible, subgroup, seed=seed)
        reading["subgroup"] = {
            "group_size": subgroup,
            "groups": sub.n_groups,
            "degenerate_groups": sub.degenerate_groups,
            "degenerate_frac": sub.degenerate_frac,
            # Selection ranks on this reading, not the one at N: the degeneracy gate is
            # already read here, and a best-of-N pick is not one the run at G ever makes.
            "picks": None if all(p is None for p in sub_picks) else pick_reading(
                hyps, refs, raw, sub_picks,
                n=subgroup, centroid=centroid, z_all=z_all, feature=feature, dists=dists,
            ),
        }
    return reading, picks


def adequacy_floor(readings: list[dict], anchor: dict, component: str = "kiwi") -> dict:
    """Which cells translate at least as well as a uniform draw from the same pool."""
    floor = anchor["components"][component]
    clears, below = [], {}
    for reading in readings:
        if reading["picks"] is None:
            continue
        value = reading["picks"]["components"][component]
        if value >= floor:
            clears.append(reading["cell"])
        else:
            below[reading["cell"]] = value
    return {
        "component": component,
        "floor": floor,
        "source": "random anchor over the same pool",
        "clears": clears,
        "below": below,
        "unread": [r["cell"] for r in readings if r["picks"] is None],
    }


def ranked_at(reading: dict) -> tuple[int, dict | None]:
    sub = reading.get("subgroup") or {}
    if "picks" in sub:
        return sub["group_size"], sub["picks"]
    return reading["n"], reading["picks"]


def select(
    readings: list[dict],
    cells: dict[str, dict],
    *,
    max_degenerate: float,
    warn_shift: float,
    feature: str,
) -> dict:
    """The cell that feeds the training run, and why each other one does not."""
    exploratory = [
        r["cell"] for r in readings if r.get("evidence_class", _PRE_REGISTERED) != _PRE_REGISTERED
    ]
    if exploratory:
        raise ValueError(
            f"{', '.join(exploratory)} are exploratory cells. Selecting one would make the "
            f"choice of reward post hoc; they are read off the pool, never selected from."
        )
    rejected, kept = {}, []
    for r in readings:
        # One hard gate. A cell whose groups have no reward spread trains nothing, whatever
        # its argmax would have picked, so it is out before register fit is compared.
        deg = r.get("subgroup", r)["degenerate_frac"]
        deg_at = r.get("subgroup", {}).get("group_size", r["n"])
        rank_at, picks = ranked_at(r)
        if deg > max_degenerate:
            rejected[r["cell"]] = (
                f"{deg:.0%} of groups have no reward spread at G={deg_at}, "
                f"past {max_degenerate:.0%}"
            )
        elif picks is None:
            rejected[r["cell"]] = (
                f"no group had a feasible sample at G={rank_at}, so it ranks on nothing"
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
    winner = min(kept, key=lambda r: (ranked_at(r)[1]["stylo_dist"], cells[r["cell"]]["w_judge"]))
    at, picks = ranked_at(winner)
    shift = picks[f"{feature}_shift"]
    return {
        "cell": winner["cell"],
        "ranked_at": at,
        "weights": cells[winner["cell"]],
        "reason": (
            f"lowest register distance ({picks['stylo_dist']:.3f}) at G={at} of the "
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


def _row(reading: dict, feature: str, comps: list[str], dist_keys: list[str]) -> str:
    sub = reading.get("subgroup")
    head = (
        f"  {reading['cell']:11s} {reading['degenerate_frac']:7.0%} "
        f"{(sub['degenerate_frac'] if sub else float('nan')):7.0%} "
    )
    p = reading["picks"]
    if p is None:
        return head + f"{0:6d}   no sample cleared the gate in any group"
    shift = p[f"{feature}_shift"]
    return (
        head
        + f"{p['picked']:6d} "
        + " ".join(f"{p['components'][k]:7.2f}" for k in comps)
        + " "
        + " ".join(f"{p[k]:8.4f}" for k in dist_keys)
        + f" {shift['delta']:+7.3f}"
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
    parser.add_argument(
        "--resamples",
        type=int,
        default=_RESAMPLES,
        help=f"paired-bootstrap resamples for the held-out reading (default: {_RESAMPLES})",
    )
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
    seed = args.seed or cfg["rlsf"]["seed"]

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
    explore = exploratory_reward_configs(cfg)
    cells = {c["name"]: c for c in cfg["rlsf"]["weight_grid"]["cells"]}

    split = load_centroid(cfg["data"]["split_centroid_file"])
    stylo_centroid = subcentroid(split, REWARD_FEATURES)
    dists = {"reward": stylo_centroid, "heldout": subcentroid(split, HELDOUT_FEATURES)}

    needed = {c for _, rc in [*grid, *explore] for c in rc.required_components}
    added = derive_components(raw, hyps, refs, needed, stylo_centroid)
    if added:
        print(f"derived locally, no calls: {', '.join(added)}")

    read = partial(
        cell_reading,
        sources=sources, hyps=hyps, refs=refs, raw=raw,
        n=n, subgroup=subgroup, centroid=centroid, z_all=z_all, feature=feature, dists=dists,
        seed=seed,
    )
    scored = [read(name, rc) for name, rc in grid]
    scored += [read(name, rc, evidence_class="exploratory") for name, rc in explore]
    picks_by_cell = {r["cell"]: p for r, p in scored}
    readings = [r for r, _ in scored[: len(grid)]]
    exploratory = [r for r, _ in scored[len(grid) :]]

    # A gate empties groups, so its picks cover fewer segments than the cell it is read
    # against. Without pairing, a gate is credited for the segments it dropped.
    gated = [(r, rc) for r, (_, rc) in zip(exploratory, explore) if rc.gated]
    if gated:
        base_name = grid[0][0]
        base_picks = picks_by_cell[base_name]
        for reading, _ in gated:
            reading["paired"] = {
                "baseline": base_name,
                **paired_stylo(base_picks, picks_by_cell[reading["cell"]], hyps, centroid),
            }

    # Feasibility does not depend on the weights, so any cell's mask reads the components.
    _, feasible, _ = compute_rewards(
        sources, hyps, refs, cfg=grid[0][1], group_size=n,
        component_scores=raw, centroid=centroid,
    )
    per_component = component_degeneracy(raw, feasible, n)
    anchor_picks = random_picks(feasible, n, seed)
    anchors = {
        key: pick_reading(
            hyps, refs, raw, picks,
            n=n, centroid=centroid, z_all=z_all, feature=feature, dists=dists,
        )
        for key, picks in (("random", anchor_picks), ("pool", None))
    }

    boot = {"n_resamples": args.resamples, "alpha": _ALPHA, "seed": seed, "anchor": "random"}
    anchor_draws = heldout_draws(
        anchor_picks, hyps, dists["heldout"], n_resamples=args.resamples, seed=seed
    )
    covered = sum(p is not None for p in anchor_picks)
    for reading in [*readings, *exploratory]:
        cell_pick = picks_by_cell[reading["cell"]]
        if anchor_draws is None or sum(p is not None for p in cell_pick) != covered:
            continue
        draws = heldout_draws(
            cell_pick, hyps, dists["heldout"], n_resamples=args.resamples, seed=seed
        )
        if draws is not None:
            reading["heldout_vs_random"] = paired_diff(draws, anchor_draws, alpha=_ALPHA)

    verdict = select(
        readings, cells,
        max_degenerate=_MAX_DEGENERATE, warn_shift=drift_rule(cfg).min_delta, feature=feature,
    )
    floor = adequacy_floor([*readings, *exploratory], anchors["random"])

    print(f"\nper-component degeneracy at N={n}: what each term can separate on its own")
    for name, stats in per_component.items():
        print(f"  {name:8s} {stats['degenerate']}/{stats['groups']} flat groups "
              f"({stats['degenerate_frac']:.0%})")

    comps = sorted(readings[0]["picks"]["components"])
 
    dist_keys = ["stylo_dist", "reward_dist", "reward_per_sample", "heldout_dist"]
    print(
        f"\n{'cell':13s} {'deg@' + str(n):>7s} {'deg@' + str(subgroup):>7s} {'picks':>6s} "
        + " ".join(f"{c:>7s}" for c in comps)
        + " "
        + " ".join(f"{_DIST_LABELS[k]:>8s}" for k in dist_keys)
        + f" {feature[:7] + ' dz':>7s}"
    )
    for reading in readings:
        print(_row(reading, feature, comps, dist_keys))
    for key, anchor in anchors.items():
        shift = anchor[f"{feature}_shift"]
        print(
            f"  {key:11s} {'':7s} {'':7s} {anchor['picked']:6d} "
            + " ".join(f"{anchor['components'][c]:7.2f}" for c in comps)
            + " "
            + " ".join(f"{anchor[k]:8.4f}" for k in dist_keys)
            + f" {shift['delta']:+7.3f}"
        )

    if exploratory:
        print("\nexploratory -- read off the pool after it was built, selected from by nothing:")
        for reading in exploratory:
            print(_row(reading, feature, comps, dist_keys))
        print(
            "  Groups a gate empties make no pick, so a gated cell's component means are read "
            "over fewer segments than the cells above and are not paired with them."
        )
    scored_vs_random = [
        r for r in [*readings, *exploratory] if "heldout_vs_random" in r
    ]
    if scored_vs_random:
        pct = int(round(100 * (1 - _ALPHA)))
        print(
            f"\nheld-out features {HELDOUT_FEATURES}, which no reward here is fitted on."
            f"\ncell minus the random anchor, paired over {args.resamples} resamples "
            f"({pct}% CI). Negative is closer to the target register:"
        )
        for r in sorted(scored_vs_random, key=lambda r: r["heldout_vs_random"]["diff"]):
            d = r["heldout_vs_random"]
            print(
                f"  {r['cell']:11s} {d['diff']:+8.4f}  "
                f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  p={d['p_value']:.3f}"
                f"{'  *' if d['significant'] else ''}"
            )
        beat = [r["cell"] for r in scored_vs_random if r["heldout_vs_random"]["significant"]
                and r["heldout_vs_random"]["diff"] < 0]
        print(f"  * = CI excludes 0. Beating the anchor: {', '.join(beat) if beat else 'none'}")

    if gated:
        base_name = gated[0][0]["paired"]["baseline"]
        print(f"\ngated cells against {base_name}, on the groups both pick:")
        for reading, _ in gated:
            p = reading["paired"]
            print(
                f"  {reading['cell']:11s} {p['groups']:3d} groups  "
                f"gate {p['cell_stylo']:.3f} vs {p['baseline_stylo']:.3f} "
                f"({p['cell_stylo'] - p['baseline_stylo']:+.3f})"
            )

    print(
        f"\nDegeneracy at N={n} and at G={subgroup} are not comparable: a larger group "
        f"degenerates less by construction, because more draws is more chances to differ. "
        f"G={subgroup} is the training-time figure."
    )
    print(
        f"\nregister distance selection ranks on, at the training group size. The table "
        f"above reads at N={n}; the run trains at G={subgroup}:"
    )
    for reading in readings:
        at, p = ranked_at(reading)
        print(f"  {reading['cell']:11s} G={at}  "
              + (f"{p['stylo_dist']:.4f} over {p['picked']} picks" if p else "no pick"))
    print(
        f"\nadequacy floor for Phase 2 arms: {floor['component']} >= {floor['floor']:.4f}, "
        f"the {floor['source']}. Read alongside `selected` below, never into it:"
    )
    for cell, value in sorted(floor["below"].items(), key=lambda kv: kv[1]):
        print(f"  below the floor  {cell:11s} {value:.4f} ({value - floor['floor']:+.4f})")
    print(f"  clears it: {', '.join(floor['clears']) if floor['clears'] else 'no cell'}")

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
                "seed": seed,
                "feature": feature,
                "feature_split": {
                    "centroid": cfg["data"]["split_centroid_file"],
                    "reward": REWARD_FEATURES,
                    "heldout": HELDOUT_FEATURES,
                },
                "heldout_bootstrap": boot,
                "gates": gates,
                "per_component_degeneracy": per_component,
                "cells": readings,
                "exploratory_cells": exploratory,
                "anchors": anchors,
                "selection": verdict,
                "adequacy_floor": floor,
                "caveats": [
                    "Dev-slice figures select the reward weights and are never reported as a "
                    "result; val remains the reported split.",
                    "`exploratory_cells` were added after the pool was built and after the "
                    "pre-registered grid was read. They are leads for Phase 2's design, not "
                    "results, and `selection` cannot draw from them.",
                    "A gated cell's picks are read over only the groups its gate leaves "
                    "non-empty, so its component means are not paired with the other cells'.",
                    "`adequacy_floor` constrains which arms Phase 2 may consider. It is not "
                    "part of `selection`, which stays the pre-registered register-distance "
                    "rule; a cell below the floor is still reported, not deleted.",
                    "A cell ranked at `normalization: fixed` is not comparable with a "
                    "group_z cell's reward values, only with its picks.",
                    f"Degeneracy at N={n} and at G={subgroup} are not comparable; larger "
                    f"groups degenerate less by construction.",
                    f"`cells` and `anchors` read at N={n}. `selection` ranks on the "
                    f"`subgroup.picks` reading at G={subgroup}, the size the run trains at.",
                    f"A group whose reward is flat has its pick drawn uniformly at seed "
                    f"{seed} rather than taken in sampling order, so those groups are noise "
                    f"in every cell instead of the same sample in all of them.",
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
