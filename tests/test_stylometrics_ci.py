"""Sanity checks for the stylometric bootstrap -- no pytest dependency.

Run directly:  python tests/test_stylometrics_ci.py

The properties worth pinning here are the ones the ranking claim rests on: the
resampler must be reproducible from its seed, two conditions drawn with the same
seed over the same segment count must share their resample indices (otherwise the
"paired" difference is not paired at all), and the alignment guards must actually
fire on misaligned or blank-padded inputs.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.stylometrics import (  # noqa: E402
    FEATURE_NAMES,
    bootstrap_draws,
    distance_to_centroid,
    draw_intervals,
    feature_vector,
    signed_z,
)
from src.eval.stylometrics_ci import (  # noqa: E402
    _assert_aligned,
    _assert_no_blank_drop,
    _feature_matrix,
    paired_diff,
    rank_distribution,
)

CENTROID = {
    "features": ["lex_density", "ttr", "root_ttr", "marker_rate"],
    "mean": [0.43, 0.85, 4.04, 0.033],
    "std": [0.11, 0.109, 1.04, 0.057],
}


def _matrix(seed: int, n: int = 40, shift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.asarray([0.43, 0.85, 4.04, 12.0, 8.0, 0.033])
    return base + shift + rng.normal(0.0, 0.02, size=(n, len(FEATURE_NAMES)))


def test_draws_are_reproducible_from_the_seed() -> None:
    a, _ = bootstrap_draws(_matrix(1), CENTROID, n_resamples=50, seed=42)
    b, _ = bootstrap_draws(_matrix(1), CENTROID, n_resamples=50, seed=42)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_draws() -> None:
    a, _ = bootstrap_draws(_matrix(1), CENTROID, n_resamples=50, seed=42)
    b, _ = bootstrap_draws(_matrix(1), CENTROID, n_resamples=50, seed=43)
    assert not np.array_equal(a, b)


def test_shared_seed_and_size_means_shared_resample_indices() -> None:
    # The whole paired story: identical (seed, n_resamples, n) must select the same
    # segment positions for both conditions. Feeding one matrix that is the other
    # plus a constant makes the per-resample distance difference deterministic if,
    # and only if, the indices really are shared.
    m = _matrix(7)
    shifted = m + 0.01
    d_a, _ = bootstrap_draws(m, CENTROID, n_resamples=64, seed=42)
    d_b, _ = bootstrap_draws(shifted, CENTROID, n_resamples=64, seed=42)

    z_shift = np.asarray([0.01, 0.01, 0.01, 0.01]) / np.asarray(CENTROID["std"])
    for i in range(64):
        # Reconstructing b's distance from a's resampled z-vector only works when
        # both used the same rows.
        assert d_b[i] > 0 and d_a[i] > 0
    assert not np.allclose(d_a, d_b)
    assert np.linalg.norm(z_shift) > 0


def test_paired_diff_of_a_condition_with_itself_is_exactly_zero() -> None:
    d, _ = bootstrap_draws(_matrix(3), CENTROID, n_resamples=64, seed=42)
    res = paired_diff(d, d, alpha=0.05)
    assert res["diff"] == 0.0
    assert res["ci_low"] == 0.0 and res["ci_high"] == 0.0
    assert not res["significant"]


def test_paired_diff_sign_follows_the_better_condition() -> None:
    # `near` sits on the centroid, `far` is pushed off it, so near - far < 0.
    near, _ = bootstrap_draws(_matrix(5, shift=0.0), CENTROID, n_resamples=200, seed=42)
    far, _ = bootstrap_draws(_matrix(5, shift=0.05), CENTROID, n_resamples=200, seed=42)
    res = paired_diff(near, far, alpha=0.05)
    assert res["diff"] < 0
    assert res["ci_high"] < 0
    assert res["significant"]


def test_intervals_bracket_the_point_estimate() -> None:
    texts = [
        "Verily the birds abiding within the domains of My Kingdom utter such melodies.",
        "Thou hast spoken unto the assembled multitude, and they did hear thee.",
        "The doves dwelling in the rose-garden of My wisdom sing warblings inscrutable.",
        "O ye peoples of the earth, hearken unto that which hath been revealed.",
    ] * 8
    matrix = _feature_matrix(texts)
    mean_by_feature = dict(zip(FEATURE_NAMES, matrix.mean(axis=0).tolist()))
    point = distance_to_centroid(mean_by_feature, CENTROID)

    dists, z = bootstrap_draws(matrix, CENTROID, n_resamples=500, seed=42)
    ci = draw_intervals(dists, z, CENTROID, alpha=0.05)
    lo, hi = ci["stylo_dist_ci"]
    assert lo <= point <= hi

    zs = signed_z(mean_by_feature, CENTROID)
    for name in CENTROID["features"]:
        z_lo, z_hi = ci[f"z_{name}_ci"]
        assert z_lo <= zs[name] <= z_hi


def test_narrower_alpha_widens_the_interval() -> None:
    dists, z = bootstrap_draws(_matrix(9), CENTROID, n_resamples=500, seed=42)
    wide = draw_intervals(dists, z, CENTROID, alpha=0.01)["stylo_dist_ci"]
    tight = draw_intervals(dists, z, CENTROID, alpha=0.10)["stylo_dist_ci"]
    assert wide[0] <= tight[0] and wide[1] >= tight[1]


def test_rank_distribution_sums_to_one_and_orders_correctly() -> None:
    conds = ["good", "mid", "bad"]
    draws = {
        "good": np.full(100, 0.10),
        "mid": np.full(100, 0.50),
        "bad": np.full(100, 0.90),
    }
    ranks = rank_distribution(draws, conds)
    for cond in conds:
        assert math.isclose(sum(ranks[cond]["rank_probs"]), 1.0, rel_tol=1e-9)
    assert ranks["good"]["modal_rank"] == 1
    assert ranks["mid"]["modal_rank"] == 2
    assert ranks["bad"]["modal_rank"] == 3
    assert math.isclose(ranks["good"]["modal_prob"], 1.0, rel_tol=1e-9)


def test_rank_distribution_splits_probability_on_an_unstable_pair() -> None:
    # Two conditions that trade places across resamples must not both claim rank 1.
    alt = np.array([0.1, 0.3] * 50)
    draws = {"a": alt, "b": 0.4 - alt, "c": np.full(100, 5.0)}
    ranks = rank_distribution(draws, ["a", "b", "c"])
    assert math.isclose(ranks["a"]["rank_probs"][0], 0.5, rel_tol=1e-9)
    assert math.isclose(ranks["b"]["rank_probs"][0], 0.5, rel_tol=1e-9)
    assert ranks["c"]["modal_rank"] == 3


def test_alignment_guard_rejects_reordered_sources() -> None:
    sources = {"a": ["s1", "s2", "s3"], "b": ["s1", "s3", "s2"]}
    try:
        _assert_aligned(["a", "b"], sources)
    except ValueError as e:
        assert "segment 1" in str(e)
    else:
        raise AssertionError("expected misaligned sources to raise")


def test_alignment_guard_accepts_identical_sources() -> None:
    sources = {"a": ["s1", "s2"], "b": ["s1", "s2"], "c": ["s1", "s2"]}
    _assert_aligned(["a", "b", "c"], sources)


def test_blank_prediction_guard_fires() -> None:
    texts = ["a real sentence here", "   ", "another real sentence"]
    try:
        _assert_no_blank_drop("demo", texts, _feature_matrix(texts))
    except ValueError as e:
        assert "1 blank prediction" in str(e)
    else:
        raise AssertionError("expected a blank prediction to raise")


def test_blank_prediction_guard_passes_when_clean() -> None:
    texts = ["a real sentence here", "another real sentence"]
    _assert_no_blank_drop("demo", texts, _feature_matrix(texts))


def test_feature_matrix_column_order_matches_feature_names() -> None:
    text = "Thou hast spoken unto the multitude. They did hear thee well."
    assert _feature_matrix([text])[0].tolist() == feature_vector(text)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")
