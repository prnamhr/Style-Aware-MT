"""Degenerate-case tests for the reward-path smoke checks."""

from __future__ import annotations

import numpy as np
import pytest

from src.rlsf.smoke import group_variance_report, plan, verdicts


def _log(n_samples=16, n_feasible=16, ratio=1.0):
    return {"n_samples": n_samples, "n_feasible": n_feasible, "length_ratio_mean": ratio}


def _ok_variance(degenerate_frac=0.0):
    return {"judge": {"groups": 4, "degenerate": 0, "degenerate_frac": degenerate_frac,
                      "min_group_sd": 1.0}}


# --- plan ------------------------------------------------------------------------------


def test_plan_counts_one_judge_call_per_sample():
    assert plan(20, 4)["judge_calls"] == 80


def test_plan_price_is_positive_and_scales():
    assert plan(40, 4)["est_usd"] == pytest.approx(2 * plan(20, 4)["est_usd"], rel=1e-3)


# --- variance --------------------------------------------------------------------------


def test_identical_scores_in_a_group_are_flagged_degenerate():
    # The case the smoke exists to catch: no spread, so no advantage, so no gradient.
    raw = {"judge": [3.0, 3.0, 3.0, 3.0]}
    rep = group_variance_report(raw, np.ones(4, dtype=bool), 4)["judge"]
    assert rep == {"groups": 1, "degenerate": 1, "degenerate_frac": 1.0, "min_group_sd": 0.0}


def test_spread_scores_are_not_degenerate():
    raw = {"judge": [1.0, 2.0, 4.0, 5.0]}
    rep = group_variance_report(raw, np.ones(4, dtype=bool), 4)["judge"]
    assert rep["degenerate"] == 0 and rep["min_group_sd"] > 0


def test_a_group_with_one_feasible_sample_is_degenerate():
    # group_normalize needs 2 valid entries; below that it returns zeros.
    feasible = np.array([True, False, False, False])
    rep = group_variance_report({"judge": [1.0, 2.0, 3.0, 4.0]}, feasible, 4)["judge"]
    assert rep["degenerate"] == 1


def test_variance_is_reported_per_group_not_pooled():
    # Two groups, each internally flat but differing from each other. Pooled sd would be
    # non-zero and hide that neither group carries any advantage.
    raw = {"judge": [2.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0]}
    rep = group_variance_report(raw, np.ones(8, dtype=bool), 4)["judge"]
    assert rep == {"groups": 2, "degenerate": 2, "degenerate_frac": 1.0, "min_group_sd": 0.0}


def test_every_weighted_component_is_reported():
    raw = {"bleu": [1.0, 2.0], "kiwi": [3.0, 4.0], "judge": [5.0, 5.0]}
    rep = group_variance_report(raw, np.ones(2, dtype=bool), 2)
    assert set(rep) == {"bleu", "kiwi", "judge"}
    assert rep["judge"]["degenerate"] == 1 and rep["bleu"]["degenerate"] == 0


# --- verdicts --------------------------------------------------------------------------


def test_all_four_checks_are_reported():
    names = [n for n, _, _ in verdicts(True, _ok_variance(), True, _log())]
    assert names == ["kiwi handshake", "within-group variance", "steplog written", "length band"]


def test_a_failed_handshake_fails_its_check_only():
    out = dict((n, ok) for n, ok, _ in verdicts(False, _ok_variance(), True, _log()))
    assert out["kiwi handshake"] is False
    assert out["steplog written"] and out["length band"]


def test_mostly_degenerate_groups_fail():
    ok = dict((n, o) for n, o, _ in verdicts(True, _ok_variance(0.5), True, _log()))
    assert ok["within-group variance"] is False


def test_a_few_degenerate_groups_pass():
    ok = dict((n, o) for n, o, _ in verdicts(True, _ok_variance(0.1), True, _log()))
    assert ok["within-group variance"] is True


def test_length_band_flooring_most_of_the_batch_fails():
    ok = dict((n, o) for n, o, _ in verdicts(True, _ok_variance(), True, _log(16, 4)))
    assert ok["length band"] is False


def test_length_band_passes_when_most_samples_clear_it():
    ok = dict((n, o) for n, o, _ in verdicts(True, _ok_variance(), True, _log(16, 15)))
    assert ok["length band"] is True


def test_an_empty_batch_does_not_divide_by_zero():
    ok = dict((n, o) for n, o, _ in verdicts(True, _ok_variance(), True, _log(0, 0)))
    assert ok["length band"] is False
