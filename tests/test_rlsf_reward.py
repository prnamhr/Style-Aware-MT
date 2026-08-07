"""Degenerate-case tests for the RLSF reward and the dev-slice carver."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.data.rlsf_dev import partition, select_works, work_sizes
from src.eval.judge import build_prompt, template_digest
from src.rlsf.reward import (
    RewardConfig,
    compute_rewards,
    frozen_digest,
    group_normalize,
    length_feasible,
    load_train_template,
    overlap_scores,
    z_deviations,
)

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
