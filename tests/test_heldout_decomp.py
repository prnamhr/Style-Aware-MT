from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.heldout_decomp import (  # noqa: E402
    _slope_draws,
    checkpoint_ladder,
    decompose,
    omega_of,
    paired_delta,
    traj_condition,
    z_draws,
)
from src.eval.stylometrics import HELDOUT_FEATURES, SPLIT_FEATURES  # noqa: E402

CENTROID = {
    "features": SPLIT_FEATURES,
    "mean": [0.43, 12.0, 8.0, 0.85, 4.04, 0.033],
    "std": [0.11, 5.0, 6.0, 0.109, 1.04, 0.057],
}

Z = {"ttr": -0.02, "root_ttr": -0.10, "marker_rate": 0.14}


def _matrix(seed: int, n: int = 60, shift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        np.asarray(CENTROID["mean"]) + shift + rng.normal(0.0, 0.01, size=(n, len(SPLIT_FEATURES)))
    )


def test_shares_sum_to_one_and_reconstruct_the_norm() -> None:
    out = decompose(Z, HELDOUT_FEATURES)
    assert math.isclose(sum(out["share"].values()), 1.0, rel_tol=1e-12)
    assert math.isclose(out["dist"], math.hypot(*(Z[f] for f in HELDOUT_FEATURES)), rel_tol=1e-12)
    assert math.isclose(sum(out["contribution"].values()), out["dist"] ** 2, rel_tol=1e-12)


def test_share_ordering_follows_squared_z_not_signed_z() -> None:
    """root_ttr is further from target than ttr despite both being negative."""
    out = decompose(Z, HELDOUT_FEATURES)
    assert out["share"]["marker_rate"] > out["share"]["root_ttr"] > out["share"]["ttr"]


def test_all_zero_z_yields_no_shares_rather_than_a_division_by_zero() -> None:
    out = decompose(dict.fromkeys(HELDOUT_FEATURES, 0.0), HELDOUT_FEATURES)
    assert out["dist"] == 0.0
    assert set(out["share"].values()) == {0.0}


def test_draws_are_paired_on_shared_indices() -> None:
    """Two conditions differing by a constant shift must give that shift back exactly."""
    idx = np.random.default_rng(42).integers(0, 60, size=(200, 60))
    a = z_draws(_matrix(1), CENTROID, idx)
    b = z_draws(_matrix(1, shift=0.01), CENTROID, idx)
    expected = 0.01 / np.asarray(CENTROID["std"])
    assert np.allclose(b - a, expected)


def test_unpaired_indices_lose_that_exactness() -> None:
    matrix = _matrix(1)
    a = z_draws(matrix, CENTROID, np.random.default_rng(42).integers(0, 60, size=(200, 60)))
    b = z_draws(matrix, CENTROID, np.random.default_rng(43).integers(0, 60, size=(200, 60)))
    assert not np.allclose(b - a, 0.0)


def test_paired_delta_flags_a_separated_difference() -> None:
    idx = np.random.default_rng(42).integers(0, 60, size=(2000, 60))
    a = z_draws(_matrix(1, shift=0.02), CENTROID, idx)[:, 0]
    b = z_draws(_matrix(1), CENTROID, idx)[:, 0]
    rec = paired_delta(a, b, 0.05)
    assert rec["significant"] and rec["p_value"] == 0.0
    assert rec["ci_low"] > 0.0


def test_paired_delta_does_not_flag_a_condition_against_itself() -> None:
    idx = np.random.default_rng(42).integers(0, 60, size=(500, 60))
    a = z_draws(_matrix(1), CENTROID, idx)[:, 0]
    rec = paired_delta(a, a, 0.05)
    assert rec == {
        "delta": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "p_value": 1.0,
        "significant": False,
    }


def test_checkpoint_ladder_reports_the_selection_mismatch() -> None:
    """Skipped where the select records are absent; they are the point of the check."""
    ladder = checkpoint_ladder()
    if not ladder:
        print("  (skipped: no results/rlsf_select_*.json)")
        return
    assert set(ladder["selected"]) == set(ladder["cells"])
    assert ladder["selected_matched"] == (len(set(ladder["selected"].values())) == 1)
    for step in ladder["steps"]:
        dists = [step["dist_heldout"][cell] for cell in ladder["cells"]]
        assert step["monotone_in_omega"] == (dists == sorted(dists))


def test_omega_is_read_off_a_trajectory_tag() -> None:
    """A step-indexed condition carries its judge weight in its name, not in the OMEGA table."""
    assert omega_of(traj_condition("w3_6.0", 800)) == 6.0
    assert omega_of("rlsf_w3_2.0") == omega_of(traj_condition("w3_2.0", 100)) == 2.0
    assert omega_of("peft") == 0.0


def test_an_unnamed_condition_has_no_judge_weight() -> None:
    for name in ("knn_fewshot", "rlsf_w3_2.0_step", "rlsf_step800"):
        try:
            omega_of(name)
        except KeyError:
            continue
        raise AssertionError(f"'{name}' should carry no judge weight")


def test_chunking_leaves_the_draws_unchanged() -> None:
    idx = np.random.default_rng(42).integers(0, 60, size=(200, 60))
    whole = z_draws(_matrix(1), CENTROID, idx, chunk=len(idx))
    assert np.array_equal(z_draws(_matrix(1), CENTROID, idx, chunk=7), whole)


def test_slope_recovers_a_planted_growth_rate() -> None:
    """The design is log2 of the step ratio, so the slope reads per doubling of training."""
    steps = np.asarray([100, 200, 400, 800], dtype=float)
    x = np.log2(steps / steps[0])
    y = 0.17 + 0.05 * x
    assert np.allclose(_slope_draws(y[None, :], x), 0.05)
    assert np.allclose(_slope_draws(np.tile(y, (3, 1)) + np.arange(3)[:, None], x), 0.05)


def test_a_flat_arm_has_no_slope() -> None:
    x = np.log2(np.asarray([100.0, 200.0, 400.0]) / 100.0)
    assert np.allclose(_slope_draws(np.full((4, 3), 0.2), x), 0.0)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(tests)} checks passed")
