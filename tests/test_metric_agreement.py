from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.metric_agreement import (  # noqa: E402
    _common_keep,
    _load_register_params,
    _segment_register,
    condition_level,
    holm_bonferroni,
    permutation_p_floor,
    rater_comparison,
    segment_level,
    spearman_ci,
    spearman_diff_ci,
    spearman_draws,
)
from src.eval.peft_register import _assert_pairable, _feature_matrix, _judge_mean  # noqa: E402

CENTROID = {
    "features": ["lex_density", "ttr", "root_ttr", "marker_rate"],
    "mean": [0.43, 0.85, 4.04, 0.033],
    "std": [0.11, 0.109, 1.04, 0.057],
}
DIRECTION = [0.355, 0.098, -0.296, 0.514]


def _phi_like(seed: int, n: int = 300) -> np.ndarray:
    """Integer 1-5 scores: the tie structure Phi actually has."""
    return np.random.default_rng(seed).integers(1, 6, size=n).astype(float)


def test_spearman_draws_reproducible_from_seed() -> None:
    x, y = _phi_like(1), _phi_like(2)
    a = spearman_draws(x, y, n_resamples=200, seed=42)
    b = spearman_draws(x, y, n_resamples=200, seed=42)
    assert np.array_equal(a, b)


def test_spearman_draws_differ_across_seeds() -> None:
    x, y = _phi_like(1), _phi_like(2)
    a = spearman_draws(x, y, n_resamples=200, seed=42)
    b = spearman_draws(x, y, n_resamples=200, seed=43)
    assert not np.array_equal(a, b)


def test_spearman_draws_invariant_to_chunk_size() -> None:
    x, y = _phi_like(3), _phi_like(4)
    one = spearman_draws(x, y, n_resamples=150, seed=42, chunk=150)
    many = spearman_draws(x, y, n_resamples=150, seed=42, chunk=7)
    assert np.allclose(one, many)


def test_point_estimate_matches_scipy_under_heavy_ties() -> None:
    # Phi has five distinct values, so tie-averaged ranks are the norm, not the corner.
    x, y = _phi_like(5), _phi_like(6)
    got = spearman_ci(x, y, n_resamples=50, seed=42, alpha=0.05)
    assert math.isclose(got["rho"], stats.spearmanr(x, y).statistic, rel_tol=1e-12)


def test_draws_reproduce_scipy_on_a_degenerate_resample() -> None:
    # A resample that repeats one index is constant in both variables; scipy yields nan
    # there, and the draw must contribute 0 instead of poisoning the percentiles.
    x = np.array([1.0, 1.0, 1.0, 1.0])
    y = np.array([2.0, 2.0, 2.0, 2.0])
    draws = spearman_draws(x, y, n_resamples=32, seed=42)
    assert np.all(np.isfinite(draws))
    assert np.all(draws == 0.0)


def test_interval_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=400)
    y = x + rng.normal(0.0, 0.5, size=400)
    got = spearman_ci(x, y, n_resamples=400, seed=42, alpha=0.05)
    assert got["ci_low"] <= got["rho"] <= got["ci_high"]
    assert got["separates"]


def test_exactly_uncorrelated_series_fail_to_separate() -> None:
    # A full 4x4 factorial repeated: every (x, y) combination occurs equally often, so
    # the rank correlation is exactly 0 by construction. Drawing two independent normal
    # samples instead would put 0 outside the interval 5% of the time and make this
    # check flaky rather than wrong.
    grid = [(i, j) for i in range(4) for j in range(4)] * 20
    x = np.asarray([g[0] for g in grid], dtype=float)
    y = np.asarray([g[1] for g in grid], dtype=float)
    got = spearman_ci(x, y, n_resamples=400, seed=42, alpha=0.05)
    assert abs(got["rho"]) < 1e-12
    assert got["ci_low"] <= 0.0 <= got["ci_high"]
    assert not got["separates"]


def test_narrower_alpha_widens_the_interval() -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=300)
    y = x + rng.normal(0.0, 1.0, size=300)
    wide = spearman_ci(x, y, n_resamples=400, seed=42, alpha=0.01)
    tight = spearman_ci(x, y, n_resamples=400, seed=42, alpha=0.10)
    assert wide["ci_low"] <= tight["ci_low"] and wide["ci_high"] >= tight["ci_high"]


def test_shape_mismatch_is_rejected() -> None:
    try:
        spearman_draws(np.zeros(5), np.zeros(6), n_resamples=10, seed=42)
    except ValueError as e:
        assert "differ in shape" in str(e)
    else:
        raise AssertionError("expected mismatched shapes to raise")


# multiplicity


def test_holm_rejects_nothing_when_all_p_are_large() -> None:
    assert holm_bonferroni([0.4, 0.6, 0.9], alpha=0.05) == [False, False, False]


def test_holm_rejects_everything_when_all_p_are_tiny() -> None:
    assert holm_bonferroni([1e-9, 1e-8, 1e-7], alpha=0.05) == [True, True, True]


def test_holm_is_a_step_down_not_plain_bonferroni() -> None:
    # 0.02 clears plain Bonferroni only at 0.05/3 = .0167 -> no; but after the smallest
    # is rejected the threshold rises to 0.05/2 = .025, which 0.02 does clear.
    assert holm_bonferroni([0.001, 0.02, 0.9], alpha=0.05) == [True, True, False]


def test_holm_stops_at_the_first_failure() -> None:
    # A large p must block every larger p even if one of them would pass alone.
    got = holm_bonferroni([0.03, 0.04, 0.045], alpha=0.05)
    assert got == [False, False, False]


def test_holm_preserves_input_order() -> None:
    got = holm_bonferroni([0.9, 1e-9, 0.5], alpha=0.05)
    assert got == [False, True, False]


def test_permutation_p_floor_matches_the_devlog_convention() -> None:
    assert math.isclose(permutation_p_floor(6), 2 / 720, rel_tol=1e-12)
    assert f"{permutation_p_floor(6):.4f}" == "0.0028"
    assert permutation_p_floor(7) < permutation_p_floor(6)
    assert math.isnan(permutation_p_floor(2))


def _write_config(body: str) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "cfg.yaml"
    tmp.write_text(body, encoding="utf-8")
    return tmp


def test_register_params_ordered_by_centroid_features() -> None:
    cfg = _write_config(
        "afsp:\n"
        "  select_target_sigma: 0.5\n"
        "  style_register_direction:\n"
        "    marker_rate: 0.514\n"
        "    lex_density: 0.355\n"
        "    root_ttr: -0.296\n"
        "    ttr: 0.098\n"
    )
    sigma, direction = _load_register_params(cfg)
    assert sigma == 0.5
    assert direction == DIRECTION


def test_register_params_reject_a_missing_feature() -> None:
    cfg = _write_config(
        "afsp:\n"
        "  select_target_sigma: 0.5\n"
        "  style_register_direction:\n"
        "    marker_rate: 0.5\n"
        "    lex_density: 0.3\n"
        "    ttr: 0.1\n"
    )
    try:
        _load_register_params(cfg)
    except ValueError as e:
        assert "root_ttr" in str(e)
    else:
        raise AssertionError("expected a missing register feature to raise")


def test_register_params_reject_a_missing_sigma() -> None:
    cfg = _write_config(
        "afsp:\n"
        "  style_register_direction:\n"
        "    marker_rate: 0.5\n"
        "    lex_density: 0.3\n"
        "    root_ttr: -0.2\n"
        "    ttr: 0.1\n"
    )
    try:
        _load_register_params(cfg)
    except ValueError as e:
        assert "select_target_sigma" in str(e)
    else:
        raise AssertionError("expected a missing sigma to raise")


def test_segment_register_handles_a_single_segment() -> None:
    reg = _segment_register(["Thou hast spoken unto them."], CENTROID, 0.5, DIRECTION)
    for key, arr in reg.items():
        assert arr.shape == (1,), key
        assert np.all(np.isfinite(arr)), key


def test_segment_register_handles_wordless_text() -> None:
    # features() returns all zeros for wordless input; the distances must stay finite.
    reg = _segment_register(["...", "!!!"], CENTROID, 0.5, DIRECTION)
    assert np.all(np.isfinite(reg["centroid_dist"]))
    assert np.all(np.isfinite(reg["band_dist"]))


def _fake_loaded(n: int, phi_seed: int) -> dict:
    """One condition's per-segment series. System-level values vary with ``phi_seed`` so
    condition-level correlations over several of these are not constant-input degenerate."""
    rng = np.random.default_rng(phi_seed)
    return {
        "n_total": n,
        "n_scored": n,
        "coverage": 1.0,
        "phi": rng.integers(1, 6, size=n).astype(float),
        "phi_mean": 2.5 + 0.1 * phi_seed,
        "stylo_dist": 0.6 - 0.05 * phi_seed,
        "z": dict.fromkeys(CENTROID["features"], 0.0),
        "centroid_dist": rng.normal(1.0, 0.2, size=n),
        "band_dist": rng.normal(1.0, 0.2, size=n),
        "comet": rng.normal(0.7, 0.1, size=n),
        "comet_system": 0.65 + 0.01 * phi_seed,
        "lex_density": rng.normal(0.4, 0.05, size=n),
        "ttr": rng.normal(0.85, 0.05, size=n),
        "root_ttr": rng.normal(4.0, 0.5, size=n),
        "marker_rate": rng.normal(0.03, 0.01, size=n),
    }


def test_segment_level_pools_unequal_condition_sizes() -> None:
    # knn_fewshot really is one segment short; pooling must not pad or truncate.
    loaded = {"a": _fake_loaded(50, 1), "b": _fake_loaded(49, 2)}
    got = segment_level(loaded, ["a", "b"], n_resamples=100, seed=42, alpha=0.05)
    assert got["n_pooled"] == 99
    assert got["per_condition"]["b"]["centroid_dist"]["n"] == 49


def test_segment_level_marks_holm_on_every_pooled_pair() -> None:
    loaded = {"a": _fake_loaded(80, 3)}
    got = segment_level(loaded, ["a"], n_resamples=100, seed=42, alpha=0.05)
    assert got["holm_family_size"] == len(got["pooled"])
    for rec in got["pooled"].values():
        assert "holm_significant" in rec


def test_condition_level_reports_the_p_floor_for_its_n() -> None:
    loaded = {k: _fake_loaded(20, i) for i, k in enumerate("abcdef")}
    got = condition_level(loaded, list("abcdef"))
    assert got["n"] == 6
    assert math.isclose(got["p_floor"], 2 / 720, rel_tol=1e-12)
    assert "phi~stylo_dist" in got["pairs"]


def test_condition_level_omits_comet_when_a_condition_lacks_it() -> None:
    loaded = {"a": _fake_loaded(20, 1), "b": _fake_loaded(20, 2), "c": _fake_loaded(20, 3)}
    del loaded["c"]["comet_system"]
    got = condition_level(loaded, ["a", "b", "c"])
    assert not any("comet" in k for k in got["pairs"])


def test_blank_prediction_guard_fires() -> None:
    texts = ["a real sentence here", "   ", "another real sentence"]
    try:
        _assert_pairable("cell", texts, _feature_matrix(texts))
    except ValueError as e:
        assert "1 blank prediction" in str(e)
    else:
        raise AssertionError("expected a blank prediction to raise")


def test_blank_prediction_guard_passes_when_clean() -> None:
    texts = ["a real sentence here", "another real sentence"]
    _assert_pairable("cell", texts, _feature_matrix(texts))


def test_judge_mean_is_none_for_an_unjudged_cell() -> None:
    phi, n, cov = _judge_mean("no_such_cell", Path(tempfile.mkdtemp()))
    assert phi is None and n == 0 and cov == 0.0


def test_judge_mean_ignores_unparseable_scores() -> None:
    d = Path(tempfile.mkdtemp())
    (d / "cell.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {"input": "a", "score": 3},
                {"input": "b", "score": None},
                {"input": "c", "score": 5},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    phi, n, cov = _judge_mean("cell", d)
    assert phi == 4.0  # mean of 3 and 5, the None dropped rather than imputed
    assert n == 2
    assert math.isclose(cov, 2 / 3, rel_tol=1e-12)


def test_judge_mean_is_none_when_every_score_failed() -> None:
    d = Path(tempfile.mkdtemp())
    (d / "cell.jsonl").write_text(json.dumps({"input": "a", "score": None}) + "\n", "utf-8")
    phi, n, cov = _judge_mean("cell", d)
    assert phi is None and n == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


def test_diff_ci_delta_matches_the_two_point_estimates() -> None:
    xa, xb, y = _phi_like(11), _phi_like(12), _phi_like(13)
    got = spearman_diff_ci(xa, y, xb, y, n_resamples=200, seed=42, alpha=0.05)
    assert math.isclose(got["rho_a"], stats.spearmanr(xa, y).statistic, rel_tol=1e-12)
    assert math.isclose(got["rho_b"], stats.spearmanr(xb, y).statistic, rel_tol=1e-12)
    assert math.isclose(got["delta"], got["rho_a"] - got["rho_b"], rel_tol=1e-12)


def test_diff_ci_of_a_rater_against_itself_is_exactly_zero() -> None:
    # The shared resample must cancel; independent bootstraps would leave noise here.
    x, y = _phi_like(21), _phi_like(22)
    got = spearman_diff_ci(x, y, x, y, n_resamples=200, seed=42, alpha=0.05)
    assert got["delta"] == 0.0
    assert got["ci_low"] == 0.0 and got["ci_high"] == 0.0
    assert not got["separates"]


def test_diff_ci_separates_when_one_rater_tracks_the_target_and_the_other_does_not() -> None:
    rng = np.random.default_rng(7)
    y = rng.normal(size=400)
    xa = y + rng.normal(scale=0.3, size=400)  # tracks y
    xb = rng.normal(size=400)  # does not
    got = spearman_diff_ci(xa, y, xb, y, n_resamples=400, seed=42, alpha=0.05)
    assert got["separates"]
    assert got["ci_low"] > 0


def test_diff_ci_rejects_unaligned_series() -> None:
    x = _phi_like(1, n=10)
    try:
        spearman_diff_ci(x, x, x, _phi_like(2, n=9), n_resamples=10, seed=1, alpha=0.05)
    except ValueError as e:
        assert "aligned" in str(e)
    else:
        raise AssertionError("expected unaligned series to raise")


def _write_judge(d: Path, cond: str, scores: list) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{cond}.jsonl").write_text(
        "\n".join(json.dumps({"input": f"s{i}", "score": v}) for i, v in enumerate(scores)) + "\n",
        encoding="utf-8",
    )


def test_common_keep_is_the_intersection_of_parseable_scores() -> None:
    root = Path(tempfile.mkdtemp())
    a, b = root / "a", root / "b"
    _write_judge(a, "cond", [1, None, 3, 4])
    _write_judge(b, "cond", [2, 2, None, 5])
    assert _common_keep(["cond"], {"phi_a": a, "phi_b": b}) == {"cond": [0, 3]}


def test_common_keep_stops_at_the_shorter_rater() -> None:
    root = Path(tempfile.mkdtemp())
    a, b = root / "a", root / "b"
    _write_judge(a, "cond", [1, 2, 3])
    _write_judge(b, "cond", [1, 2])
    assert _common_keep(["cond"], {"phi_a": a, "phi_b": b}) == {"cond": [0, 1]}


def test_rater_comparison_needs_two_raters() -> None:
    got = rater_comparison(
        {"phi_a": {"x": _fake_loaded(30, 1)}}, ["x"], n_resamples=50, seed=42, alpha=0.05
    )
    assert got["pairs"] == {}


def test_rater_comparison_reports_a_delta_per_target_with_holm() -> None:
    paired = {
        "phi_a": {"x": _fake_loaded(60, 1), "y": _fake_loaded(60, 2)},
        "phi_b": {"x": _fake_loaded(60, 3), "y": _fake_loaded(60, 4)},
    }
    got = rater_comparison(paired, ["x", "y"], n_resamples=100, seed=42, alpha=0.05)
    pair = got["pairs"]["phi_a~phi_b"]
    assert got["n_pooled"] == 120
    assert set(pair["correlation_deltas"]) == {"band_dist", "centroid_dist", "marker_rate", "comet"}
    for rec in pair["correlation_deltas"].values():
        assert "holm_significant" in rec
        assert math.isclose(rec["delta"], rec["rho_a"] - rec["rho_b"], rel_tol=1e-12)
    assert pair["condition_ordering"]["n"] == 2


def test_rater_comparison_uses_one_shared_target_series() -> None:
    # phi_b's own target arrays differ; the delta must still come from Phi alone, so a
    # rater whose phi equals the other's must produce exactly zero delta.
    a, b = _fake_loaded(60, 1), _fake_loaded(60, 5)
    b["phi"] = a["phi"].copy()
    got = rater_comparison(
        {"phi_a": {"x": a}, "phi_b": {"x": b}}, ["x"], n_resamples=100, seed=42, alpha=0.05
    )
    for rec in got["pairs"]["phi_a~phi_b"]["correlation_deltas"].values():
        assert rec["delta"] == 0.0
