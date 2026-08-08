"""Degenerate-case tests for the register-drift stop rule."""

from __future__ import annotations

import math

import pytest

from src.rlsf.config import drift_rule
from src.rlsf.reward import z_step_stats
from src.rlsf.stop import DriftMonitor, DriftRule

CENTROID = {
    "features": ["lex_density", "ttr", "root_ttr", "marker_rate"],
    "mean": [0.4344, 0.8540, 4.0437, 0.0327],
    "std": [0.1101, 0.1085, 1.0426, 0.0567],
}

# Wide enough that k_sigma alone never binds, so tests choose which arm they exercise.
RULE = DriftRule(baseline_steps=2, window=2, k_sigma=3.0, min_delta=0.5)


def _step(z, se=0.05, step=0):
    return {"step": step, "z": {"marker_rate": z}, "z_se": {"marker_rate": se}}


def _run(monitor, values, se=0.05):
    out = None
    for i, z in enumerate(values):
        out = monitor.update(_step(z, se, i))
    return out


def test_a_flat_run_far_above_the_val_constant_never_trips():
    # The bug this rule replaces: 0.577 read against zeroshot's 0.547 fires at step 0,
    # though the run has not moved at all.
    verdict = _run(DriftMonitor(RULE), [0.577] * 6)
    assert verdict.tripped is False
    assert verdict.baseline == pytest.approx(0.577)


def test_nothing_can_trip_before_the_baseline_and_window_are_complete():
    monitor = DriftMonitor(RULE)
    for i, z in enumerate([0.5, 9.0, 9.0]):
        verdict = monitor.update(_step(z, 0.05, i))
        assert verdict.tripped is False
    assert "not complete" in verdict.reason


def test_a_sustained_excursion_trips():
    verdict = _run(DriftMonitor(RULE), [0.5, 0.5, 2.0, 2.0])
    assert verdict.tripped is True
    assert verdict.delta == pytest.approx(1.5)


def test_a_single_spike_does_not_trip():
    # One step at +4 carries the window mean well past the band; the rule wants both
    # steps out, so a decoding fluke on one batch cannot stop a run.
    verdict = _run(DriftMonitor(RULE), [0.5, 0.5, 4.0, 0.5])
    assert verdict.tripped is False
    assert "not above" in verdict.reason


def test_an_excursion_inside_the_noise_does_not_trip():
    # Same +1.5 excursion, but the steps are too noisy to separate it from the opening.
    verdict = _run(DriftMonitor(RULE), [0.5, 0.5, 2.0, 2.0], se=1.0)
    assert verdict.tripped is False
    assert verdict.threshold > verdict.delta


def test_a_small_but_clean_excursion_does_not_trip():
    # Resolvable at any error, and still not a reason to halt: min_delta is the floor.
    verdict = _run(DriftMonitor(RULE), [0.5, 0.5, 0.7, 0.7], se=1e-6)
    assert verdict.tripped is False
    assert verdict.threshold == pytest.approx(RULE.min_delta)


def test_the_rule_is_one_sided():
    # Losing register is a failed reward, not a run to halt; the weight grid answers it.
    verdict = _run(DriftMonitor(RULE), [0.5, 0.5, -3.0, -3.0])
    assert verdict.tripped is False


def test_the_baseline_is_frozen_after_its_window():
    # Drift must not be absorbed into the reference it is measured against.
    monitor = DriftMonitor(RULE)
    _run(monitor, [0.5, 0.5, 2.0, 2.0, 2.0, 2.0])
    assert monitor.baseline[0] == pytest.approx(0.5)


def test_a_step_without_a_clustered_error_cannot_trip():
    verdict = _run(DriftMonitor(RULE), [0.5, 0.5, 9.0, 9.0], se=float("nan"))
    assert verdict.tripped is False
    assert "fewer than two prompts" in verdict.reason


def test_a_step_log_without_z_se_is_refused():
    # Silently never firing is the failure mode worth being loud about.
    monitor = DriftMonitor(RULE)
    with pytest.raises(KeyError, match="no z_se"):
        monitor.update({"step": 0, "z": {"marker_rate": 0.5}})


def test_pooling_the_baseline_narrows_its_error():
    one = DriftMonitor(DriftRule(baseline_steps=1, window=1))
    five = DriftMonitor(DriftRule(baseline_steps=5, window=1))
    _run(one, [0.5])
    _run(five, [0.5] * 5)
    assert five.baseline[1] == pytest.approx(one.baseline[1] / math.sqrt(5))


def test_a_rule_watching_a_feature_the_centroid_lacks_is_refused():
    with pytest.raises(ValueError, match="feature must be one of"):
        DriftRule(feature="sent_len_var")


def test_a_rule_without_a_practical_floor_is_refused():
    with pytest.raises(ValueError, match="min_delta"):
        DriftRule(min_delta=0.0)


def test_the_rule_is_read_from_the_config_block():
    cfg = {"rlsf": {"stop": {"feature": "ttr", "window": 4, "unused_key": 1}}}
    rule = drift_rule(cfg)
    assert rule.feature == "ttr" and rule.window == 4


def test_a_config_without_a_stop_block_gets_the_defaults():
    assert drift_rule({"rlsf": {}}) == DriftRule()


def test_clustered_error_exceeds_the_independent_one():
    # Two prompts answered four ways each: the within-group agreement is not evidence
    # about the next prompt, and pooling all eight as independent pretends it is.
    archaic = "Thou art the Lord of all being, and unto Thee do we turn."
    plain = "You are the master of everything, and we turn to you."
    hyps = [archaic] * 4 + [plain] * 4
    _, clustered = z_step_stats(hyps, CENTROID, group_size=4)
    _, independent = z_step_stats(hyps, CENTROID, group_size=1)
    assert clustered["marker_rate"] > independent["marker_rate"]


def test_a_single_group_step_has_no_clustered_error():
    _, se = z_step_stats(["Thou art here."] * 4, CENTROID, group_size=4)
    assert math.isnan(se["marker_rate"])


def test_ragged_group_sizes_are_refused():
    with pytest.raises(ValueError, match="whole number of groups"):
        z_step_stats(["a", "b", "c"], CENTROID, group_size=2)
