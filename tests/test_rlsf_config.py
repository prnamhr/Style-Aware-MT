"""Validation tests for configs/rlsf.yaml and its loader."""

from __future__ import annotations

import copy
import math

import pytest
import yaml

from src.rlsf.config import (
    assert_caps_declared,
    assert_group_size_within_ceiling,
    drift_rule,
    exploratory_reward_configs,
    grid_reward_configs,
    judge_concurrency,
    load_config,
    make_judge_client,
    priced_worst_case,
    reward_config,
    worst_case_judge_calls,
)

_JUDGE_CONFIGS = ("configs/judge_eval.yaml", "configs/judge_eval_gpt.yaml")

# outputs/rlsf/smoke_usage.json, 2026-08-08: 80 calls, $0.0059.
_MEASURED_USD_PER_CALL = 7.375e-5


def _declared(**overrides):
    """The committed config, with any cap overridden for the case under test."""
    cfg = copy.deepcopy(load_config(require_caps=False))
    cfg["rlsf"]["caps"].update(overrides)
    return cfg


# caps


def test_committed_config_loads_with_its_caps_declared():
    # Declared 2026-08-08 from the smoke's measured per-call rate; see docs/budget.md.
    caps = load_config()["rlsf"]["caps"]
    assert caps["max_steps"] == 600
    assert caps["max_judge_spend_usd"] == 25.0


def test_nulling_any_one_cap_puts_the_config_back_out_of_bounds():
    with pytest.raises(ValueError, match="undeclared"):
        assert_caps_declared(_declared(max_judge_calls=None))


def test_every_spend_cap_is_reported_not_just_the_first():
    cfg = _declared(max_steps=None, max_judge_spend_usd=None)
    with pytest.raises(ValueError) as exc:
        assert_caps_declared(cfg)
    assert "max_steps" in str(exc.value) and "max_judge_spend_usd" in str(exc.value)


def test_declared_caps_pass():
    assert_caps_declared(_declared()) is None


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_cap_is_rejected(bad):
    with pytest.raises(ValueError, match="must be positive"):
        assert_caps_declared(_declared(max_steps=bad))


# authorized envelope 


def test_committed_group_size_is_within_the_ceiling():
    assert_group_size_within_ceiling(load_config(require_caps=False))


def test_group_size_above_the_ceiling_is_a_widening_not_a_tweak():
    cfg = copy.deepcopy(load_config(require_caps=False))
    cfg["rlsf"]["rollout"]["group_size"] = cfg["rlsf"]["caps"]["group_size_ceiling"] + 1
    with pytest.raises(ValueError, match="rule 3"):
        assert_group_size_within_ceiling(cfg)


def test_worst_case_calls_default_to_the_ceiling_not_the_operative_group_size():
    cfg = load_config()
    at_ceiling = worst_case_judge_calls(cfg)
    operative = worst_case_judge_calls(cfg, group_size=cfg["rlsf"]["rollout"]["group_size"])
    # 800 capped steps x 16 prompts x 8, and the same at the operative group size of 4.
    assert at_ceiling == 102_400
    assert operative == 51_200
    assert at_ceiling > operative


def test_priced_worst_case_matches_hand_arithmetic():
    # gpt-4o-mini at [0.15, 0.60] per Mtok, 600 prompt + 40 completion tokens.
    cost = priced_worst_case(44_800, (0.15, 0.60), prompt_tokens=600, completion_tokens=40)
    assert cost == pytest.approx(5.1072)


def test_the_step_caps_bind_before_the_call_and_dollar_caps():
    # docs/budget.md, 2026-08-08: the call and spend caps are a backstop against the
    # per-call rate moving, not a second opinion on the step count.
    caps = load_config()["rlsf"]["caps"]
    calls = worst_case_judge_calls(load_config())
    assert calls * _MEASURED_USD_PER_CALL == pytest.approx(7.55, abs=0.01)
    assert calls < caps["max_judge_calls"]
    assert calls * _MEASURED_USD_PER_CALL < caps["max_judge_spend_usd"]


# reward block 


def test_reward_config_reads_the_weights_and_ignores_sub_blocks():
    cfg = load_config(require_caps=False)
    rc = reward_config(cfg)
    # Delivered at unit ||omega||, so the three equal declared weights arrive equal, not at 1.
    assert rc.w_kiwi == rc.w_bleu == rc.w_judge == pytest.approx(1 / math.sqrt(3))
    assert rc.overlap_metric == "bleu"
    assert rc.on_violation == "floor"


def test_weight_grid_varies_only_omega_3_and_includes_the_ablation():
    cells = grid_reward_configs(load_config(require_caps=False))
    assert [name for name, _ in cells] == ["w3_0.0", "w3_0.5", "w3_1.0", "w3_2.0"]
    assert all(rc.w_kiwi == pytest.approx(rc.w_bleu) for _, rc in cells)
    # The cells are delivered at unit ||omega||, so the name is the ratio, not the weight.
    assert [rc.w_judge / rc.w_bleu for _, rc in cells] == pytest.approx([0.0, 0.5, 1.0, 2.0])


def test_the_reward_and_every_grid_cell_are_delivered_at_unit_omega_norm():
    # Both constructors, or the weights double as a step size: an online grid would read a
    # learning-rate sweep as a weighting sweep, and the final run would train the cell that
    # won at a different effective rate from the one it won at.
    cfg = load_config(require_caps=False)
    for rc in [reward_config(cfg)] + [rc for _, rc in grid_reward_configs(cfg)]:
        assert math.hypot(rc.w_bleu, rc.w_kiwi, rc.w_judge) == pytest.approx(1.0)


def test_the_exploratory_cells_are_not_part_of_the_pre_registered_grid():
    # Separate keys, or a cell added after the pool was read would sit in the grid the
    # selection rule draws from and the choice of reward would become post hoc.
    cfg = load_config(require_caps=False)
    pre_registered = {name for name, _ in grid_reward_configs(cfg)}
    exploratory = {name for name, _ in exploratory_reward_configs(cfg)}
    assert exploratory == {
        "lex_only", "sem_only", "judge_only",
        "gate_j3", "gate_j4", "gate_j5",
        "stylo_only", "mix_stylo", "chrf_only", "mix_chrf",
        "fixed_scale",
    }
    assert not pre_registered & exploratory


def test_each_single_component_cell_carries_exactly_one_term():
    cells = dict(exploratory_reward_configs(load_config(require_caps=False)))
    for name, term in (("lex_only", "w_bleu"), ("sem_only", "w_kiwi"), ("judge_only", "w_judge")):
        weights = {k: getattr(cells[name], k) for k in ("w_bleu", "w_kiwi", "w_judge")}
        assert weights.pop(term) == pytest.approx(1.0)
        assert set(weights.values()) == {0.0}


def test_the_gated_cells_sweep_the_band_over_one_unchanged_summand():
    cells = dict(exploratory_reward_configs(load_config(require_caps=False)))
    gates = [cells[f"gate_j{i}"] for i in (3, 4, 5)]
    assert [c.judge_gate for c in gates] == [3.0, 4.0, 5.0]
    # The veto is the only thing that varies, and the only thing separating them from w3_0.0.
    ablation = dict(grid_reward_configs(load_config(require_caps=False)))["w3_0.0"]
    for cell in gates:
        assert cell.gated and not cell.w_judge
        assert (cell.w_bleu, cell.w_kiwi) == pytest.approx((ablation.w_bleu, ablation.w_kiwi))


def test_the_chrf_cells_swap_the_overlap_metric_and_key_their_weight_by_it():
    cells = dict(exploratory_reward_configs(load_config(require_caps=False)))
    for name in ("chrf_only", "mix_chrf"):
        assert cells[name].overlap_metric == "chrf"
        assert "chrf" in cells[name].weights and "bleu" not in cells[name].weights
    # mix_chrf is w3_0.0 with the metric swapped and nothing else moved.
    ablation = dict(grid_reward_configs(load_config(require_caps=False)))["w3_0.0"]
    assert cells["mix_chrf"].w_kiwi == pytest.approx(ablation.w_kiwi)
    assert cells["mix_chrf"].w_judge == ablation.w_judge == 0.0


def test_the_stylo_cells_weight_stylo_and_carry_no_judge():
    cells = dict(exploratory_reward_configs(load_config(require_caps=False)))
    for name in ("stylo_only", "mix_stylo"):
        assert cells[name].w_stylo > 0
        assert cells[name].w_judge == 0.0
        assert "stylo" in cells[name].required_components
    # stylo_only is the term on its own; mix_stylo adds it to the w3_0.0 summand.
    assert cells["stylo_only"].w_bleu == cells["stylo_only"].w_kiwi == 0.0
    assert cells["mix_stylo"].w_bleu > 0 and cells["mix_stylo"].w_kiwi > 0


def test_the_fixed_scale_cell_moves_the_scaling_and_nothing_else():
    cfg = load_config(require_caps=False)
    cell = dict(exploratory_reward_configs(cfg))["fixed_scale"]
    twin = dict(grid_reward_configs(cfg))["w3_1.0"]
    assert cell.normalization == "fixed" and twin.normalization == "group_z"
    for term in ("w_bleu", "w_kiwi", "w_judge", "w_stylo"):
        assert getattr(cell, term) == pytest.approx(getattr(twin, term))


def test_every_other_cell_is_read_under_the_normalization_the_grid_was_registered_at():
    cfg = load_config(require_caps=False)
    for name, rc in [*grid_reward_configs(cfg), *exploratory_reward_configs(cfg)]:
        assert rc.normalization == ("fixed" if name == "fixed_scale" else "group_z")


def test_grid_cells_inherit_the_base_feasibility_band():
    # A cell that silently changed the length band would not be a weight ablation.
    base = reward_config(load_config(require_caps=False))
    for _, rc in grid_reward_configs(load_config(require_caps=False)):
        assert (rc.len_min_ratio, rc.len_max_ratio) == (base.len_min_ratio, base.len_max_ratio)


# circularity control 


def test_reward_reads_the_training_rubric_not_the_evaluation_one():
    assert load_config(require_caps=False)["template_file"] == "prompts/judge_train.txt"


def test_reward_judge_is_model_distinct_from_both_evaluation_raters():
    # The reward judge must not be either rater: training against one spends it on the
    # single condition that most needs two independent raters.
    reward_model = load_config(require_caps=False)["judge"]["model"]
    raters = {
        yaml.safe_load(open(p, encoding="utf-8"))["judge"]["model"] for p in _JUDGE_CONFIGS
    }
    assert reward_model not in raters


def test_make_judge_client_refuses_a_model_that_is_an_evaluation_rater():
    cfg = load_config(require_caps=False)
    cfg["judge"]["model"] = yaml.safe_load(open(_JUDGE_CONFIGS[0], encoding="utf-8"))["judge"][
        "model"
    ]
    with pytest.raises(ValueError, match="also an evaluation rater"):
        make_judge_client(cfg)


def test_reward_judge_is_reproducible():
    # Under group normalization a rater flipping a 3 to a 4 inverts an advantage sign.
    judge = load_config(require_caps=False)["judge"]
    assert judge["temperature"] == 0.0
    assert judge["seed"] == 42


def test_judge_concurrency_defaults_to_serial_when_unset():
    cfg = load_config(require_caps=False)
    del cfg["judge"]["concurrency"]
    assert judge_concurrency(cfg) == 1


@pytest.mark.parametrize("bad", [0, -4, 1.5, True, "8"])
def test_non_positive_judge_concurrency_is_rejected(bad):
    cfg = load_config(require_caps=False)
    cfg["judge"]["concurrency"] = bad
    with pytest.raises(ValueError, match="positive integer"):
        judge_concurrency(cfg)


def test_committed_judge_concurrency_fans_out():
    # Serial, a step's 64 calls are 64 round trips the rented GPU waits through.
    assert judge_concurrency(load_config(require_caps=False)) > 1


def test_policy_is_the_locked_base_unquantized():
    gen = load_config(require_caps=False)["generator"]
    assert gen["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert gen["load_in_4bit"] is False


def test_rollout_sampling_is_stochastic():
    # Greedy rollouts give a group zero variance, which group_normalize turns into all
    # zeros -- the reward would be identically flat and no update would be informative.
    assert load_config(require_caps=False)["rlsf"]["rollout"]["temperature"] > 0.0


def test_the_drift_stop_is_the_operating_point_it_was_priced_at():
    rule = drift_rule(load_config(require_caps=False))
    assert (rule.feature, rule.baseline_steps, rule.window, rule.k_sigma, rule.min_delta) == (
        "marker_rate",
        20,
        5,
        4.0,
        0.23,
    )
