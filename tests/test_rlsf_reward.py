"""Degenerate-case tests for the RLSF reward and the dev-slice carver."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from src.data.rlsf_dev import partition, select_works, work_sizes
from src.eval.judge import build_prompt, template_digest
from src.eval.stylometrics import (
    HELDOUT_FEATURES,
    REWARD_FEATURES,
    SPLIT_FEATURES,
    features,
    subcentroid,
)
from src.rlsf.reward import (
    JudgeTiming,
    RewardConfig,
    compute_rewards,
    frozen_digest,
    group_normalize,
    judge_scores,
    length_feasible,
    load_centroid,
    load_train_template,
    overlap_scores,
    stylo_scores,
    z_deviations,
)
from src.rlsf.stop import DriftRule

CENTROID = {
    "features": ["lex_density", "ttr", "root_ttr", "marker_rate"],
    "mean": [0.4344, 0.8540, 4.0437, 0.0327],
    "std": [0.1101, 0.1085, 1.0426, 0.0567],
}


def _components(n, **overrides):
    base = {name: [1.0] * n for name in ("bleu", "kiwi", "judge")}
    base.update(overrides)
    return base



def test_group_normalize_standardizes_valid_entries():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    valid = np.ones(4, dtype=bool)
    out = group_normalize(values, valid)
    assert out.mean() == pytest.approx(0.0)
    assert out.std(ddof=0) == pytest.approx(1.0)


def test_group_normalize_identical_scores_gives_zeros():
    out = group_normalize(np.array([2.5, 2.5, 2.5]), np.ones(3, dtype=bool))
    assert np.all(out == 0.0)


def test_group_normalize_single_valid_sample_gives_zeros():
    out = group_normalize(np.array([1.0, 9.0]), np.array([True, False]))
    assert np.all(out == 0.0)


def test_group_normalize_no_valid_samples_gives_zeros():
    out = group_normalize(np.array([1.0, 9.0]), np.zeros(2, dtype=bool))
    assert np.all(out == 0.0)


def test_group_normalize_ignores_invalid_in_statistics():
    # The invalid entry is an extreme value; it must not move the mean or the sd.
    values = np.array([1.0, 3.0, 1000.0])
    valid = np.array([True, True, False])
    out = group_normalize(values, valid)
    assert out[2] == 0.0
    assert out[0] == pytest.approx(-1.0)
    assert out[1] == pytest.approx(1.0)



def test_length_feasible_band():
    cfg = RewardConfig(len_min_ratio=0.5, len_max_ratio=2.0)
    refs = ["a b c d"] * 4
    hyps = ["a b", "a b c d", "a b c d e f g h", "a b c d e f g h i"]
    assert list(length_feasible(hyps, refs, cfg)) == [True, True, True, False]


def test_length_feasible_rejects_empty_hypothesis():
    cfg = RewardConfig()
    assert list(length_feasible(["", "  "], ["a b", "a b"], cfg)) == [False, False]


def test_length_feasible_accepts_when_reference_empty():
    cfg = RewardConfig()
    assert list(length_feasible(["a b c"], [""], cfg)) == [True]



def test_single_sample_group_yields_zero_reward():
    # One sample per prompt has no within-group contrast, so no advantage.
    rewards, feasible, log = compute_rewards(
        ["s"], ["a b c"], ["a b c"],
        cfg=RewardConfig(), group_size=1,
        component_scores=_components(1), centroid=CENTROID,
    )
    assert rewards[0] == pytest.approx(0.0)
    assert feasible.all()
    assert log.n_samples == 1 and log.n_feasible == 1


def test_identical_component_scores_yield_zero_rewards():
    n = 4
    rewards, _, log = compute_rewards(
        ["s"] * n, ["a b c"] * n, ["a b c"] * n,
        cfg=RewardConfig(), group_size=n,
        component_scores=_components(n), centroid=CENTROID,
    )
    assert np.allclose(rewards, 0.0)
    assert log.reward_sd == pytest.approx(0.0)


def test_weights_select_the_ranking():
    n = 3
    scores = _components(n, bleu=[3.0, 2.0, 1.0], judge=[1.0, 2.0, 3.0])
    cfg = RewardConfig(w_bleu=1.0, w_kiwi=0.0, w_judge=0.0)
    rewards, _, _ = compute_rewards(
        ["s"] * n, ["a b c"] * n, ["a b c"] * n,
        cfg=cfg, group_size=n, component_scores=scores, centroid=CENTROID,
    )
    assert rewards[0] > rewards[1] > rewards[2]

    cfg = RewardConfig(w_bleu=0.0, w_kiwi=0.0, w_judge=1.0)
    rewards, _, _ = compute_rewards(
        ["s"] * n, ["a b c"] * n, ["a b c"] * n,
        cfg=cfg, group_size=n, component_scores=scores, centroid=CENTROID,
    )
    assert rewards[0] < rewards[1] < rewards[2]


def test_infeasible_sample_gets_group_floor():
    n = 3
    # Third hypothesis is far too long for its reference.
    hyps = ["a b c", "a b c d", " ".join(["x"] * 40)]
    refs = ["a b c d"] * n
    rewards, feasible, log = compute_rewards(
        ["s"] * n, hyps, refs,
        cfg=RewardConfig(on_violation="floor"), group_size=n,
        component_scores=_components(n, bleu=[1.0, 2.0, 99.0]), centroid=CENTROID,
    )
    assert list(feasible) == [True, True, False]
    assert rewards[2] < rewards[0] and rewards[2] < rewards[1]
    assert np.isfinite(rewards).all()
    assert log.n_feasible == 2
    assert log.n_unmeasured == 0


def test_infeasible_sample_dropped_as_nan():
    n = 2
    hyps = ["a b c d", " ".join(["x"] * 40)]
    refs = ["a b c d"] * n
    rewards, feasible, _ = compute_rewards(
        ["s"] * n, hyps, refs,
        cfg=RewardConfig(on_violation="drop"), group_size=n,
        component_scores=_components(n), centroid=CENTROID,
    )
    assert list(feasible) == [True, False]
    assert math.isnan(rewards[1])


def test_all_infeasible_group_does_not_raise():
    n = 2
    hyps = ["", ""]
    refs = ["a b c d"] * n
    rewards, feasible, log = compute_rewards(
        ["s"] * n, hyps, refs,
        cfg=RewardConfig(on_violation="floor"), group_size=n,
        component_scores=_components(n), centroid=CENTROID,
    )
    assert not feasible.any()
    assert log.n_feasible == 0
    assert np.isfinite(rewards).all()


class _StubJudge:
    """Replies keyed by hypothesis, so a verdict stays identifiable when calls are fanned out."""

    def __init__(self, replies, delays=None):
        self.replies = replies
        self.delays = delays or {}

    def complete(self, system, prompt):
        # The stub templates below end in the hypothesis.
        hyp = prompt.rsplit("|", 1)[-1]
        time.sleep(self.delays.get(hyp, 0.0))
        return self.replies[hyp]


def test_unreadable_verdict_scores_nan_not_the_rubric_minimum():
    # 1 is a verdict the judge can return; a refusal or a stray format is not one.
    out = judge_scores(
        _StubJudge({"h0": "Score: 4", "h1": "I cannot rate this."}),
        "{source}|{reference}|{prediction}",
        ["s"] * 2, ["r"] * 2, ["h0", "h1"],
    )
    assert out[0] == pytest.approx(4.0)
    assert math.isnan(out[1])


def test_fanned_out_verdicts_stay_aligned_with_their_samples():
    n = 8
    hyps = [f"h{i}" for i in range(n)]
    scores = [(i % 5) + 1 for i in range(n)]
    # Earliest sample replies last, so a result-ordered collection would misattribute verdicts.
    judge = _StubJudge(
        {h: f"Score: {s}" for h, s in zip(hyps, scores)},
        delays={h: 0.01 * (n - i) for i, h in enumerate(hyps)},
    )
    out = judge_scores(
        judge, "{source}|{reference}|{prediction}", ["s"] * n, ["r"] * n, hyps, max_workers=n
    )
    assert out == pytest.approx([float(s) for s in scores])


def test_timing_separates_the_wall_clock_from_the_call_time():
    n = 8
    hyps = [f"h{i}" for i in range(n)]
    judge = _StubJudge({h: "Score: 3" for h in hyps}, delays={h: 0.02 for h in hyps})
    timing = JudgeTiming()
    judge_scores(
        judge, "{source}|{reference}|{prediction}", ["s"] * n, ["r"] * n, hyps,
        max_workers=n, timing=timing,
    )
    assert timing.calls == n
    assert timing.call_s > timing.wall_s
    # The claim the number exists to check: workers asked for is not fan-out obtained.
    assert 2.0 < timing.achieved_parallelism <= n


def test_a_serial_judge_block_reads_as_no_fan_out():
    hyps = ["h0", "h1"]
    judge = _StubJudge({h: "Score: 3" for h in hyps}, delays={h: 0.02 for h in hyps})
    timing = JudgeTiming()
    judge_scores(
        judge, "{source}|{reference}|{prediction}", ["s"] * 2, ["r"] * 2, hyps, timing=timing
    )
    assert timing.achieved_parallelism == pytest.approx(1.0, abs=0.15)


def test_unmeasured_component_is_infeasible_not_a_low_score():
    n = 3
    scores = _components(n, judge=[5.0, 4.0, float("nan")])
    cfg = RewardConfig(w_bleu=0.0, w_kiwi=0.0, w_judge=1.0, on_violation="floor")
    rewards, feasible, log = compute_rewards(
        ["s"] * n, ["a b c"] * n, ["a b c"] * n,
        cfg=cfg, group_size=n, component_scores=scores, centroid=CENTROID,
    )
    assert list(feasible) == [True, True, False]
    assert log.n_feasible == 2 and log.n_unmeasured == 1
    # The unmeasured sample neither joins the group's statistics nor ranks above one that did.
    assert rewards[0] == pytest.approx(1.0) and rewards[1] == pytest.approx(-1.0)
    assert rewards[2] < rewards[1]


def test_unmeasured_component_dropped_as_nan():
    n = 2
    scores = _components(n, judge=[4.0, float("nan")])
    rewards, feasible, log = compute_rewards(
        ["s"] * n, ["a b c"] * n, ["a b c"] * n,
        cfg=RewardConfig(on_violation="drop"), group_size=n,
        component_scores=scores, centroid=CENTROID,
    )
    assert list(feasible) == [True, False]
    assert math.isnan(rewards[1])
    # The logged raw mean is over what was measured, so one failure cannot make it nan.
    assert log.raw["judge"] == pytest.approx(4.0)
    assert log.n_unmeasured == 1


def test_multiple_groups_normalized_independently():
    # Group B is uniformly higher than group A on the raw scale; after per-group
    # normalization both groups must span the same range.
    scores = _components(4, bleu=[1.0, 2.0, 101.0, 102.0])
    rewards, _, _ = compute_rewards(
        ["s"] * 4, ["a b c"] * 4, ["a b c"] * 4,
        cfg=RewardConfig(w_bleu=1.0, w_kiwi=0.0, w_judge=0.0),
        group_size=2, component_scores=scores, centroid=CENTROID,
    )
    assert rewards[0] == pytest.approx(rewards[2])
    assert rewards[1] == pytest.approx(rewards[3])


def test_ragged_batch_raises():
    with pytest.raises(ValueError, match="ragged batch"):
        compute_rewards(
            ["s"], ["a", "b"], ["a"],
            cfg=RewardConfig(), group_size=1,
            component_scores=_components(2), centroid=CENTROID,
        )


def test_batch_not_whole_groups_raises():
    with pytest.raises(ValueError, match="whole number of groups"):
        compute_rewards(
            ["s"] * 3, ["a b c"] * 3, ["a b c"] * 3,
            cfg=RewardConfig(), group_size=2,
            component_scores=_components(3), centroid=CENTROID,
        )


def test_missing_component_raises():
    with pytest.raises(ValueError, match="missing component scores"):
        compute_rewards(
            ["s"], ["a b c"], ["a b c"],
            cfg=RewardConfig(), group_size=1,
            component_scores={"bleu": [1.0], "kiwi": [1.0]}, centroid=CENTROID,
        )


def test_component_length_mismatch_raises():
    with pytest.raises(ValueError, match="has 1 scores for 2 samples"):
        compute_rewards(
            ["s"] * 2, ["a b c"] * 2, ["a b c"] * 2,
            cfg=RewardConfig(), group_size=2,
            component_scores=_components(2, bleu=[1.0]), centroid=CENTROID,
        )


def test_config_rejects_bad_settings():
    with pytest.raises(ValueError, match="on_violation"):
        RewardConfig(on_violation="penalize")
    with pytest.raises(ValueError, match="overlap_metric"):
        RewardConfig(overlap_metric="comet")
    with pytest.raises(ValueError, match="len_min_ratio"):
        RewardConfig(len_min_ratio=2.0, len_max_ratio=1.0)


# judge as a veto rather than a summand


def _gated(gate=4.0, **overrides):
    return RewardConfig(w_bleu=1.0, w_kiwi=1.0, w_judge=0.0, judge_gate=gate, **overrides)


def test_a_gate_vetoes_below_the_band_without_ranking_the_survivors():
    # Ranked on bleu alone: the judge decides who competes, never who wins.
    n = 4
    scores = _components(n, bleu=[1.0, 2.0, 3.0, 4.0], judge=[5.0, 5.0, 3.0, 3.0])
    rewards, feasible, _ = compute_rewards(
        ["s"] * n, ["a b c"] * n, ["a b c"] * n,
        cfg=_gated(), group_size=n, component_scores=scores, centroid=CENTROID,
    )
    assert list(feasible) == [True, True, False, False]
    # The 4.0 has the best overlap but was vetoed, so the 2.0 is the best feasible sample.
    assert rewards[1] > rewards[0]
    assert max(range(n), key=lambda i: rewards[i]) == 1


def test_a_gated_judge_is_not_also_a_summand():
    # Same weights, same scores; only the judge's role differs. Under the gate the judge
    # cannot reorder samples that all clear it.
    n = 3
    scores = _components(n, bleu=[3.0, 2.0, 1.0], judge=[3.0, 4.0, 5.0])
    common = dict(group_size=n, component_scores=scores, centroid=CENTROID)
    gated, _, _ = compute_rewards(
        ["s"] * n, ["a b c"] * n, ["a b c"] * n, cfg=_gated(gate=3.0), **common
    )
    summed, _, _ = compute_rewards(
        ["s"] * n, ["a b c"] * n, ["a b c"] * n,
        cfg=RewardConfig(w_bleu=1.0, w_kiwi=1.0, w_judge=2.0), **common,
    )
    assert gated[0] > gated[1] > gated[2]
    assert summed[0] < summed[2]


def test_a_gate_still_requires_a_judge_score():
    # The veto has to read the judge, so a pool without one is a missing component, not an
    # unweighted term that can be skipped.
    with pytest.raises(ValueError, match="missing component scores: \\['judge'\\]"):
        compute_rewards(
            ["s"] * 2, ["a b c"] * 2, ["a b c"] * 2,
            cfg=_gated(), group_size=2,
            component_scores={"bleu": [1.0, 2.0], "kiwi": [1.0, 2.0]}, centroid=CENTROID,
        )


def test_an_unreadable_verdict_is_vetoed_not_waved_through():
    n = 2
    scores = _components(n, judge=[5.0, float("nan")])
    _, feasible, log = compute_rewards(
        ["s"] * n, ["a b c"] * n, ["a b c"] * n,
        cfg=_gated(), group_size=n, component_scores=scores, centroid=CENTROID,
    )
    assert list(feasible) == [True, False]
    assert log.n_unmeasured == 1


def test_a_gate_that_empties_a_group_leaves_it_without_a_gradient():
    n = 3
    scores = _components(n, bleu=[1.0, 2.0, 3.0], judge=[1.0, 2.0, 3.0])
    _, feasible, log = compute_rewards(
        ["s"] * n, ["a b c"] * n, ["a b c"] * n,
        cfg=_gated(gate=5.0), group_size=n, component_scores=scores, centroid=CENTROID,
    )
    assert not feasible.any()
    assert log.degenerate_frac == 1.0


def test_a_gated_cell_reports_the_judge_it_gated_on():
    # Unweighted but still measured: the reading has to show what the veto read.
    assert "judge" not in _gated().weights
    assert "judge" in _gated().required_components
    _, _, log = compute_rewards(
        ["s"] * 2, ["a b c"] * 2, ["a b c"] * 2,
        cfg=_gated(), group_size=2,
        component_scores=_components(2, judge=[5.0, 4.0]), centroid=CENTROID,
    )
    assert log.raw["judge"] == pytest.approx(4.5)


def test_a_judge_that_both_vetoes_and_ranks_is_refused():
    with pytest.raises(ValueError, match="both veto a sample and rank"):
        RewardConfig(w_judge=1.0, judge_gate=4.0)


def test_a_gate_off_the_rubric_band_is_refused():
    with pytest.raises(ValueError, match="rubric band"):
        RewardConfig(w_judge=0.0, judge_gate=6.0)
    with pytest.raises(ValueError, match="rubric band"):
        RewardConfig(w_judge=0.0, judge_gate=0.0)


def test_the_reward_and_heldout_feature_sets_are_disjoint():
    assert not set(REWARD_FEATURES) & set(HELDOUT_FEATURES)
    assert set(SPLIT_FEATURES) == set(REWARD_FEATURES) | set(HELDOUT_FEATURES)


def test_the_drift_monitors_feature_is_never_rewarded():
    assert DriftRule().feature == "marker_rate"
    assert DriftRule().feature in HELDOUT_FEATURES
    assert DriftRule().feature not in REWARD_FEATURES


def test_the_split_centroid_extends_the_committed_one_rather_than_refitting_it():
    committed = load_centroid()
    split = load_centroid("results/stylometrics_centroid_split.json")
    assert split["n_segments"] == committed["n_segments"]
    shared = subcentroid(split, committed["features"])
    assert shared["mean"] == committed["mean"]
    assert shared["std"] == committed["std"]


def test_subcentroid_refuses_a_feature_it_has_no_statistics_for():
    with pytest.raises(KeyError, match="sent_len_mean"):
        subcentroid(CENTROID, ["lex_density", "sent_len_mean"])


def test_a_closer_sample_scores_higher_because_the_distance_is_negated():
    # Every other component is better when larger; the sum would pull the wrong way otherwise.
    centroid = {"features": ["lex_density"], "mean": [0.5], "std": [0.1]}
    on_target, off_target = "the of and cat", "cat dog bird fish"
    scores = stylo_scores([on_target, off_target], centroid)
    assert all(s <= 0 for s in scores)
    near = [abs(features(t)["lex_density"] - 0.5) for t in (on_target, off_target)]
    assert (scores[0] > scores[1]) == (near[0] < near[1])


def test_stylo_joins_the_weights_only_when_it_is_weighted():
    # Derived from the text rather than bought, so a cell that does not use it should not
    # have to carry a score for it -- unlike the judge, which the pool pays for regardless.
    assert "stylo" not in RewardConfig(w_stylo=0.0).weights
    assert "stylo" not in RewardConfig(w_stylo=0.0).required_components
    weighted = RewardConfig(w_stylo=1.0)
    assert weighted.weights["stylo"] == 1.0
    assert "stylo" in weighted.required_components


def test_the_stylo_weight_counts_toward_the_unit_norm():
    rc = RewardConfig(w_bleu=1.0, w_kiwi=1.0, w_judge=1.0, w_stylo=1.0).unit_omega()
    assert math.hypot(rc.w_bleu, rc.w_kiwi, rc.w_judge, rc.w_stylo) == pytest.approx(1.0)
    assert rc.w_stylo == pytest.approx(0.5)
    # A cell that does not use it normalizes over three, exactly as before.
    three = RewardConfig(w_bleu=1.0, w_kiwi=1.0, w_judge=1.0).unit_omega()
    assert three.w_bleu == pytest.approx(1 / math.sqrt(3))
    assert three.w_stylo == 0.0


def test_a_stylo_weighted_cell_ranks_on_the_stylo_score():
    n = 3
    scores = _components(n, bleu=[1.0, 1.0, 1.0], kiwi=[1.0, 1.0, 1.0], stylo=[-3.0, -2.0, -1.0])
    rewards, _, log = compute_rewards(
        ["s"] * n, ["a b c"] * n, ["a b c"] * n,
        cfg=RewardConfig(w_bleu=0.0, w_kiwi=0.0, w_judge=0.0, w_stylo=1.0),
        group_size=n, component_scores=scores, centroid=CENTROID,
    )
    # Least negative is closest to the register, so it takes the highest reward.
    assert rewards[2] > rewards[1] > rewards[0]
    assert log.raw["stylo"] == pytest.approx(-2.0)


def test_gating_leaves_the_summand_at_the_ablation_cell_s_weights():
    # The gated variant differs from w3_0.0 by the veto alone, so their summands must match
    # or the comparison reads a reweighting as well.
    gated = _gated().unit_omega()
    ablation = RewardConfig(w_bleu=1.0, w_kiwi=1.0, w_judge=0.0).unit_omega()
    assert gated.w_bleu == pytest.approx(ablation.w_bleu)
    assert gated.w_kiwi == pytest.approx(ablation.w_kiwi)


# logging 


def test_z_deviations_reports_all_four_features():
    z = z_deviations(["Thou art the Lord of all being."], CENTROID)
    assert set(z) == set(CENTROID["features"])
    assert all(math.isfinite(v) for v in z.values())


def test_z_deviations_empty_batch_is_nan_not_zero():
    # Zero would read as "on centroid"; nan is the honest value for no samples.
    z = z_deviations([], CENTROID)
    assert all(math.isnan(v) for v in z.values())


def test_step_log_records_components_length_and_z():
    n = 2
    _, _, log = compute_rewards(
        ["s"] * n, ["a b c", "a b c d"], ["a b c d"] * n,
        cfg=RewardConfig(), group_size=n,
        component_scores=_components(n, bleu=[10.0, 20.0]), centroid=CENTROID,
    )
    d = log.as_dict()
    assert d["raw"]["bleu"] == pytest.approx(15.0)
    assert set(d["normalized"]) == {"bleu", "kiwi", "judge"}
    assert d["length_mean"] == pytest.approx(3.5)
    assert d["length_ratio_mean"] == pytest.approx(0.875)
    assert set(d["z"]) == set(CENTROID["features"])


def test_group_degeneracy_is_persisted_not_just_printed():
    # Two groups, one flat and one spread: over 500 steps the trend in this fraction is
    # the reading that says whether the policy still has a gradient to learn from.
    n = 4
    _, _, log = compute_rewards(
        ["s"] * n, ["a b c"] * n, ["a b c"] * n,
        cfg=RewardConfig(), group_size=2,
        component_scores=_components(n, bleu=[10.0, 10.0, 10.0, 20.0]), centroid=CENTROID,
    )
    d = log.as_dict()
    assert (d["n_groups"], d["degenerate_groups"], d["degenerate_frac"]) == (2, 1, 0.5)
    assert d["min_group_sd"] == 0.0


def test_the_overlap_term_is_logged_under_the_metric_in_use():
    # The chrF grid cell must not write its scores under a "bleu" key in steps.jsonl.
    n = 2
    _, _, log = compute_rewards(
        ["s"] * n, ["a b c"] * n, ["a b c"] * n,
        cfg=RewardConfig(overlap_metric="chrf"), group_size=n,
        component_scores=_components(n, chrf=[10.0, 20.0]), centroid=CENTROID,
    )
    d = log.as_dict()
    assert set(d["raw"]) == {"chrf", "kiwi", "judge"}
    assert d["raw"]["chrf"] == pytest.approx(15.0)


# overlap and template 


def test_overlap_scores_smoothed_bleu_is_positive_without_4gram_match():
    # Shares unigrams and a bigram with the reference but no 4-gram. Unsmoothed
    # segment BLEU would be 0 here and flatten the group; smoothing keeps it ranked.
    scores = overlap_scores(
        ["the Lord of glory sitteth"], ["the Lord of all being is exalted"], "bleu"
    )
    assert scores[0] > 0.0


def test_overlap_scores_zero_on_no_shared_tokens():
    # Smoothing rescues missing high-order n-grams, not a total absence of overlap.
    assert overlap_scores(["alpha beta"], ["gamma delta"], "bleu")[0] == 0.0


def test_overlap_scores_perfect_match_is_maximal():
    ref = "Thou art the Lord of all being."
    assert overlap_scores([ref], [ref], "bleu")[0] == pytest.approx(100.0)
    assert overlap_scores([ref], [ref], "chrf")[0] == pytest.approx(100.0)


def test_load_train_template_refuses_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="circular"):
        load_train_template(tmp_path / "judge_train.txt")


def test_frozen_train_template_matches_its_freeze_record():
    # The committed rubric against the committed digest. This is the check that makes
    # rubric identity verifiable from the artefacts rather than asserted from a config.
    assert load_train_template().startswith("Rate the register")


def test_frozen_train_template_fills_and_is_brace_safe():

    filled = build_prompt(load_train_template(), "src {x}", "ref", "cand")
    assert "src {x}" in filled and "cand" in filled
    assert "{source}" not in filled and "{prediction}" not in filled


def test_the_two_rubrics_are_distinct_templates():
    # Circularity control: the reward must not be scored by the rubric that scores Phi.
    eval_digest = frozen_digest("judge_eval.txt")
    train_digest = frozen_digest("judge_train.txt")
    assert eval_digest != train_digest


def _freeze(tmp_path, text, digest):
    (tmp_path / "judge_train.txt").write_text(text, encoding="utf-8")
    (tmp_path / "hashes.json").write_text(
        json.dumps({"templates": {"judge_train.txt": {"digest": digest}}}), encoding="utf-8"
    )
    return tmp_path / "judge_train.txt", tmp_path / "hashes.json"


def test_load_train_template_refuses_a_drifted_rubric(tmp_path):

    path, hashes = _freeze(tmp_path, "rubric v1", template_digest("rubric v1"))
    assert load_train_template(path, hashes) == "rubric v1"
    path.write_text("rubric v2", encoding="utf-8")
    with pytest.raises(ValueError, match="drifted"):
        load_train_template(path, hashes)


def test_load_train_template_refuses_an_unrecorded_rubric(tmp_path):
    path, hashes = _freeze(tmp_path, "rubric", "deadbeefdeadbeef")
    hashes.write_text(json.dumps({"templates": {}}), encoding="utf-8")
    with pytest.raises(KeyError, match="no freeze record"):
        load_train_template(path, hashes)


def test_load_train_template_refuses_a_missing_freeze_record(tmp_path):
    path, hashes = _freeze(tmp_path, "rubric", "deadbeefdeadbeef")
    hashes.unlink()
    with pytest.raises(FileNotFoundError, match="which rubric it read"):
        load_train_template(path, hashes)



def _rec(work, text="a b c"):
    return {"input": text, "output": text, "metadata": {"source": work, "type": "sentence"}}


def test_work_sizes_counts_unlabelled_as_empty_key():
    recs = [_rec("A"), _rec("A"), _rec(""), _rec("B")]
    assert work_sizes(recs) == {"A": 2, "": 1, "B": 1}


def test_select_works_hits_target_exactly_when_possible():
    sizes = {"A": 300, "B": 200, "C": 180, "D": 120, "": 50}
    picked = select_works(sizes, 500)
    assert sum(sizes[w] for w in picked) == 500
    assert "" not in picked


def test_select_works_minimizes_absolute_error_when_target_unreachable():
    sizes = {"A": 300, "B": 220, "C": 180, "D": 120}
    picked = select_works(sizes, 500)
    total = sum(sizes[w] for w in picked)
    # No subset sums to 500; the best achievable error is 20 either side.
    assert abs(total - 500) == 20


def test_select_works_never_returns_unlabelled_or_everything():
    sizes = {"A": 10, "B": 10, "": 500}
    picked = select_works(sizes, 500)
    assert "" not in picked
    assert len(picked) < 2 or set(picked) != {"A", "B"}


def test_select_works_may_overshoot_to_get_closer():
    # 260 overshoots by 10; 240 undershoots by 20. Closest wins.
    sizes = {"A": 260, "B": 240}
    assert select_works(sizes, 250) == ["A"]


def test_select_works_is_deterministic():
    sizes = {"A": 300, "B": 220, "C": 180, "D": 120}
    assert select_works(sizes, 400) == select_works(sizes, 400)


def test_select_works_needs_two_labelled_works():
    with pytest.raises(ValueError, match="at least 2 labelled works"):
        select_works({"A": 100, "": 50}, 100)


def test_select_works_rejects_nonpositive_target():
    with pytest.raises(ValueError, match="target must be >= 1"):
        select_works({"A": 10, "B": 10}, 0)


def test_select_works_honours_exclude():
    sizes = {"A": 500, "B": 250, "C": 250}
    assert "A" not in select_works(sizes, 500, exclude={"A"})


def test_partition_is_disjoint_and_order_preserving():
    recs = [_rec("A", "1"), _rec("B", "2"), _rec("A", "3"), _rec("", "4")]
    dev, rest = partition(recs, {"A"})
    assert [r["input"] for r in dev] == ["1", "3"]
    assert [r["input"] for r in rest] == ["2", "4"]
    assert len(dev) + len(rest) == len(recs)
