"""Degenerate-case tests for the best-of-N pool builder."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.infer.usage import Usage
from src.rlsf.pool import (
    append_rows,
    measured_per_call_usd,
    pack_rows,
    plan,
    pool_settings,
    read_pool,
    resume_point,
    score_chunk,
)

CFG = {
    "rlsf": {
        "rollout": {"group_size": 4},
        "caps": {"group_size_ceiling": 8},
        "pool": {"n": 8, "chunk": 25, "judge_calls": 3992},
    }
}


def _cfg(**pool):
    out = json.loads(json.dumps(CFG))
    out["rlsf"]["pool"].update(pool)
    return out


def _client(pricing):
    return SimpleNamespace(usage=Usage(pricing=pricing))


def test_plan_counts_one_judge_call_per_sample():
    assert plan(499, 8, 1e-4)["judge_calls"] == 3992


def test_plan_prices_at_the_rate_it_is_given():
    assert plan(10, 8, 1e-3)["est_usd"] == pytest.approx(0.08)


def test_a_sample_count_over_the_ceiling_is_refused():
    with pytest.raises(ValueError, match="group-size ceiling"):
        pool_settings(_cfg(n=16))


def test_an_undeclared_pool_ceiling_is_refused():
    with pytest.raises(ValueError, match="undeclared"):
        pool_settings(_cfg(judge_calls=None))


def test_the_per_call_rate_follows_the_current_price_not_the_recorded_cost(tmp_path):
    # The point of measuring it: the recorded cost is stale the moment the price moves,
    # but the token counts it was measured over are not.
    record = tmp_path / "usage.json"
    record.write_text(
        json.dumps(
            {"model": "m", "calls": 100, "prompt_tokens": 1_000_000,
             "completion_tokens": 0, "cost_usd": 0.15}
        )
    )
    doubled = measured_per_call_usd(_client({"m": (0.30, 0.0)}), "m", (record,))
    assert doubled == pytest.approx(0.30 / 100)


def test_a_usage_record_for_another_model_does_not_price_this_one(tmp_path):
    record = tmp_path / "usage.json"
    record.write_text(
        json.dumps({"model": "other", "calls": 1, "prompt_tokens": 1, "completion_tokens": 1})
    )
    with pytest.raises(FileNotFoundError, match="budget rule 1"):
        measured_per_call_usd(_client({"m": (1.0, 1.0)}), "m", (record,))


def test_pool_round_trips_in_sampling_order(tmp_path):
    # Re-ranking reads back the group a sample belongs to; a reordered flatten would pair
    # completions with the wrong source.
    raw = {"bleu": [1.0, 2.0, 3.0, 4.0], "kiwi": [0.1, 0.2, 0.3, 0.4]}
    rows = pack_rows(["s0", "s1"], ["r0", "r1"], ["h00", "h01", "h10", "h11"], raw, 2)
    path = tmp_path / "pool.jsonl"
    append_rows(path, rows)
    back = read_pool(path)
    assert [r["idx"] for r in back] == [0, 1]
    assert back[1]["hyps"] == ["h10", "h11"]
    assert back[1]["scores"]["bleu"] == [3.0, 4.0]


def test_an_unmeasured_score_survives_the_round_trip_as_nan(tmp_path):
    # An unparsed verdict is an unmeasured sample. Written as null so the file stays valid
    # JSON, read back as nan so compute_rewards marks it infeasible rather than scoring it.
    rows = pack_rows(["s"], ["r"], ["h0", "h1"], {"judge": [float("nan"), 3.0]}, 2)
    assert rows[0]["scores"]["judge"] == [None, 3.0]
    path = tmp_path / "pool.jsonl"
    append_rows(path, rows)
    first, second = read_pool(path)[0]["scores"]["judge"]
    assert first != first and second == 3.0


def test_resume_counts_the_segments_already_scored(tmp_path):
    rows = pack_rows(["s0", "s1"], ["r0", "r1"], list("abcd"), {"bleu": [1.0] * 4}, 2)
    assert resume_point(rows, ["s0", "s1", "s2"], 2) == 2


def test_resuming_into_a_pool_built_on_other_data_is_refused():
    rows = pack_rows(["s0"], ["r0"], ["a", "b"], {"bleu": [1.0, 2.0]}, 2)
    with pytest.raises(ValueError, match="rebuild rather than resume"):
        resume_point(rows, ["different"], 2)


def test_resuming_a_pool_sampled_at_another_n_is_refused():
    rows = pack_rows(["s0"], ["r0"], ["a", "b"], {"bleu": [1.0, 2.0]}, 2)
    with pytest.raises(ValueError, match="completions, not 8"):
        resume_point(rows, ["s0"], 8)


def test_skipped_components_are_held_flat_rather_than_invented():
    # A constant normalizes to zeros, so the term drops out of the combined reward.
    raw = score_chunk(
        ["s"], ["one two three"], ["h0", "h1"],
        n=2, metric="bleu", kiwi=None, judge=None, template=None, workers=1,
    )
    assert raw["kiwi"] == [1.0, 1.0] and raw["judge"] == [1.0, 1.0]
    assert len(raw["bleu"]) == 2
