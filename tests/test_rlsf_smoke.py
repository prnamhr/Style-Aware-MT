"""Degenerate-case tests for the reward-path smoke checks."""

from __future__ import annotations

import numpy as np
import pytest

from src.rlsf.reward import RewardConfig, compute_rewards
from src.rlsf.smoke import (
    _dump_hyps,
    _load_hyps,
    group_variance_report,
    plan,
    reward_degeneracy,
    verdicts,
)

CENTROID = {
    "features": ["lex_density", "ttr", "root_ttr", "marker_rate"],
    "mean": [0.4344, 0.8540, 4.0437, 0.0327],
    "std": [0.1101, 0.1085, 1.0426, 0.0567],
}


def _log(n_samples=16, n_feasible=16, ratio=1.0):
    return {"n_samples": n_samples, "n_feasible": n_feasible, "length_ratio_mean": ratio}




def test_plan_counts_one_judge_call_per_sample():
    assert plan(20, 4)["judge_calls"] == 80


def test_plan_price_is_positive_and_scales():
    assert plan(40, 4)["est_usd"] == pytest.approx(2 * plan(20, 4)["est_usd"], rel=1e-3)


def test_hyps_dump_round_trips_in_sampling_order(tmp_path):
    # Scoring reads back the group a sample belongs to; a reordered flatten would pair
    # completions with the wrong source.
    sources, refs = ["s0", "s1"], ["r0", "r1"]
    hyps = ["h00", "h01", "h10", "h11"]
    path = tmp_path / "smoke_hyps.jsonl"
    _dump_hyps(path, sources, refs, hyps, 2)
    assert _load_hyps(path, 2, 2) == (sources, refs, hyps)


def test_loading_a_dump_under_the_wrong_group_size_is_refused(tmp_path):
    path = tmp_path / "smoke_hyps.jsonl"
    _dump_hyps(path, ["s0", "s1"], ["r0", "r1"], ["h00", "h01", "h10", "h11"], 2)
    with pytest.raises(ValueError, match="completions per segment"):
        _load_hyps(path, 2, 4)



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



def test_a_floored_sample_does_not_manufacture_spread():
    # The case the check exists for. Three feasible samples scored identically, so the
    rewards = np.array([0.0, 0.0, 0.0, -1.0])
    feasible = np.array([True, True, True, False])
    assert reward_degeneracy(rewards, feasible, 4)["degenerate"] == 1
    # Pooled over all four entries the sd is 0.43, which would have looked healthy.
    assert rewards.std(ddof=0) > 0.4


def test_the_false_pass_is_prevented_through_compute_rewards():
    # The same case built by the real reward path rather than by hand: three feasible
    # samples with identical component scores, one rejected by the length band.

    ref = " ".join(["word"] * 10)
    hyps = [ref, ref, ref, "short"]
    rewards, feasible, _ = compute_rewards(
        ["src"] * 4,
        hyps,
        [ref] * 4,
        cfg=RewardConfig(),
        group_size=4,
        component_scores={n: [5.0, 5.0, 5.0, 5.0] for n in ("bleu", "kiwi", "judge")},
        centroid=CENTROID,
    )
    assert list(feasible) == [True, True, True, False]
    assert rewards[3] == pytest.approx(-1.0)  # floored to worst - margin
    assert reward_degeneracy(rewards, feasible, 4)["degenerate"] == 1
    assert np.asarray(rewards).std(ddof=0) > 0.4  # what pooling would have reported


def test_spread_across_feasible_samples_is_not_degenerate():
    rewards = np.array([-1.2, -0.3, 0.4, 1.1])
    assert reward_degeneracy(rewards, np.ones(4, dtype=bool), 4)["degenerate"] == 0


def test_identical_feasible_rewards_are_degenerate():
    rewards = np.array([0.0, 0.0, 0.0, 0.0])
    rep = reward_degeneracy(rewards, np.ones(4, dtype=bool), 4)
    assert rep == {"groups": 1, "degenerate": 1, "degenerate_frac": 1.0, "min_group_sd": 0.0}


def test_a_group_below_two_feasible_samples_is_degenerate():
    # group_normalize returns zeros below two valid entries, so there is nothing to spread.
    rewards = np.array([0.0, -1.0, -1.0, -1.0])
    feasible = np.array([True, False, False, False])
    assert reward_degeneracy(rewards, feasible, 4)["degenerate"] == 1


def test_a_wholly_infeasible_group_is_degenerate():
    rewards = np.array([-1.0, -1.0])
    assert reward_degeneracy(rewards, np.zeros(2, dtype=bool), 2)["degenerate"] == 1


def test_degeneracy_is_per_group_not_pooled():
    # Two flat groups at different levels: pooled sd is healthy, neither group is usable.
    rewards = np.array([0.0, 0.0, 5.0, 5.0])
    rep = reward_degeneracy(rewards, np.ones(4, dtype=bool), 2)
    assert rep["groups"] == 2 and rep["degenerate"] == 2


def test_near_zero_sd_counts_as_degenerate():
    rewards = np.array([0.0, 1e-12, 0.0, 1e-12])
    assert reward_degeneracy(rewards, np.ones(4, dtype=bool), 4)["degenerate"] == 1


def test_all_four_checks_are_reported():
    names = [n for n, _, _ in verdicts(True, 0.0, True, _log())]
    assert names == ["kiwi handshake", "reward variance", "steplog written", "length band"]


def test_a_failed_handshake_fails_its_check_only():
    out = dict((n, ok) for n, ok, _ in verdicts(False, 0.0, True, _log()))
    assert out["kiwi handshake"] is False
    assert out["steplog written"] and out["length band"]


def test_mostly_degenerate_groups_fail():
    ok = dict((n, o) for n, o, _ in verdicts(True, 0.5, True, _log()))
    assert ok["reward variance"] is False


def test_a_few_degenerate_groups_pass():
    ok = dict((n, o) for n, o, _ in verdicts(True, 0.1, True, _log()))
    assert ok["reward variance"] is True


def test_length_band_flooring_most_of_the_batch_fails():
    ok = dict((n, o) for n, o, _ in verdicts(True, 0.0, True, _log(16, 4)))
    assert ok["length band"] is False


def test_length_band_passes_when_most_samples_clear_it():
    ok = dict((n, o) for n, o, _ in verdicts(True, 0.0, True, _log(16, 15)))
    assert ok["length band"] is True


def test_an_empty_batch_does_not_divide_by_zero():
    ok = dict((n, o) for n, o, _ in verdicts(True, 0.0, True, _log(0, 0)))
    assert ok["length band"] is False
