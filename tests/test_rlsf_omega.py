"""Degenerate-case tests for the offline omega grid over a cached pool."""

from __future__ import annotations

import numpy as np
import pytest

from src.rlsf.omega import (
    argmax_picks,
    component_degeneracy,
    flatten,
    marker_shift,
    random_picks,
    select,
)
from src.rlsf.pool import pack_rows

CELLS = {
    "w3_0.0": {"name": "w3_0.0", "w_kiwi": 1.0, "w_bleu": 1.0, "w_judge": 0.0},
    "w3_1.0": {"name": "w3_1.0", "w_kiwi": 1.0, "w_bleu": 1.0, "w_judge": 1.0},
}


def _reading(cell, *, deg=0.0, stylo=1.0, shift=0.0, se=0.05):
    return {
        "cell": cell,
        "n": 8,
        "degenerate_frac": 0.0,
        "subgroup": {"group_size": 4, "degenerate_frac": deg},
        "picks": {"stylo_dist": stylo, "marker_rate_shift": {"delta": shift, "se": se}},
    }


def test_a_ragged_pool_cannot_be_grouped():
    rows = pack_rows(["s0"], ["r0"], ["a", "b"], {"bleu": [1.0, 2.0]}, 2)
    rows.append({"idx": 1, "source": "s1", "reference": "r1", "hyps": ["c"], "scores": {}})
    with pytest.raises(ValueError, match="a ragged pool"):
        flatten(rows)


def test_flatten_repeats_each_segment_across_its_own_samples():
    rows = pack_rows(["s0", "s1"], ["r0", "r1"], list("abcd"), {"bleu": [1.0, 2.0, 3.0, 4.0]}, 2)
    n, sources, refs, hyps, raw = flatten(rows)
    assert n == 2
    assert sources == ["s0", "s0", "s1", "s1"] and refs == ["r0", "r0", "r1", "r1"]
    assert hyps == list("abcd") and raw["bleu"] == [1.0, 2.0, 3.0, 4.0]


def test_the_pick_is_the_best_feasible_sample_not_the_best_sample():
    # The floored infeasible sample sits below the group, but an argmax over the raw array
    # would still have to be told to ignore it.
    rewards = np.array([0.5, 9.0, -1.0, 0.1])
    feasible = np.array([True, False, True, True])
    assert argmax_picks(rewards, feasible, 4) == [0]


def test_a_group_with_nothing_feasible_makes_no_pick():
    rewards = np.array([np.nan, np.nan])
    assert argmax_picks(rewards, np.array([False, False]), 2) == [None]


def test_the_random_anchor_only_draws_feasible_samples():
    feasible = np.array([False, False, True, False])
    assert random_picks(feasible, 4, seed=0) == [2]


def test_the_marker_shift_is_paired_within_the_group():
    # Group mean 1.0; picking the 3.0 is +2.0 whatever the next segment's level is.
    z = np.array([[1.0], [1.0], [1.0], [3.0], [10.0], [10.0], [10.0], [12.0]])
    shift = marker_shift(z, [3, 7], 4, 0)
    assert shift["segments"] == 2
    assert shift["delta"] == pytest.approx(1.5)
    assert shift["se"] == pytest.approx(0.0)


def test_a_component_that_scores_a_group_flat_is_degenerate():
    raw = {"judge": [3.0, 3.0, 3.0, 3.0, 1.0, 2.0, 3.0, 4.0]}
    report = component_degeneracy(raw, np.ones(8, dtype=bool), 4)["judge"]
    assert report == {"groups": 2, "degenerate": 1, "degenerate_frac": 0.5}


def test_a_cell_with_no_gradient_is_rejected_however_well_it_picks():
    readings = [_reading("w3_0.0", deg=0.0, stylo=2.0), _reading("w3_1.0", deg=0.9, stylo=0.1)]
    verdict = select(readings, CELLS, max_degenerate=0.25, warn_shift=0.23, feature="marker_rate")
    assert verdict["cell"] == "w3_0.0"
    assert "no reward spread at G=4" in verdict["rejected"]["w3_1.0"]


def test_every_cell_degenerate_is_a_reading_not_a_crash():
    readings = [_reading("w3_0.0", deg=1.0), _reading("w3_1.0", deg=1.0)]
    verdict = select(readings, CELLS, max_degenerate=0.25, warn_shift=0.23, feature="marker_rate")
    assert verdict["cell"] is None
    assert "finding about the reward" in verdict["reason"]


def test_the_marker_shift_flags_the_winner_rather_than_vetoing_it():
    readings = [_reading("w3_1.0", stylo=0.1, shift=0.9), _reading("w3_0.0", stylo=2.0)]
    verdict = select(readings, CELLS, max_degenerate=0.25, warn_shift=0.23, feature="marker_rate")
    assert verdict["cell"] == "w3_1.0"
    assert verdict["goodhart"]["over_threshold"] is True


def test_cells_that_tie_on_register_fit_take_the_weaker_style_pressure():
    readings = [_reading("w3_1.0", stylo=1.0), _reading("w3_0.0", stylo=1.0)]
    verdict = select(readings, CELLS, max_degenerate=0.25, warn_shift=0.23, feature="marker_rate")
    assert verdict["cell"] == "w3_0.0"
