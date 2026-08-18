from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.heldout_decomp import (  # noqa: E402
    checkpoint_ladder,
    decompose,
    paired_delta,
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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(tests)} checks passed")
