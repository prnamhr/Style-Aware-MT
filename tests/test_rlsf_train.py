"""GRPO wiring: the normalization contract, the step accounting, and the loop's helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.rlsf.config import (
    assert_single_normalization,
    grpo_args,
    load_config,
    optimizer_steps,
    reward_config,
    rollout_batch,
)
from src.rlsf.reward import compute_rewards
from src.rlsf.train import BudgetExceeded, JudgeBudget, completion_text

CENTROID = {
    "features": ["lex_density", "ttr", "root_ttr", "marker_rate"],
    "mean": [0.4344, 0.8540, 4.0437, 0.0327],
    "std": [0.1101, 0.1085, 1.0426, 0.0567],
}


@pytest.fixture(scope="module")
def cfg():
    return load_config(require_caps=False)


# GRPOConfig defaults to bf16 and rejects it before any guard here runs unless use_cpu is
# set, which would make every config-building test in this file green only on a bf16 GPU.
CPU = {"use_cpu": True}


@pytest.fixture(scope="module")
def args(cfg, tmp_path_factory):
    return grpo_args(cfg, output_dir=tmp_path_factory.mktemp("grpo"), rollout_steps=10, **CPU)


# the normalization contract


def test_the_trainer_does_not_rescale_an_already_normalized_reward(args):
    # compute_rewards z-scores each component within its group; TRL scaling it again
    # divides by a second, unrelated standard deviation. See DEVLOG 2026-08-08.
    assert args.scale_rewards == "none"


@pytest.mark.parametrize("bad", ["group", "batch", True])
def test_re_enabling_the_trainer_scaling_is_refused(cfg, tmp_path, bad):
    # `True` included because GRPOConfig maps it to "group" before the guard sees it.
    with pytest.raises(ValueError, match="already z-scores"):
        grpo_args(cfg, output_dir=tmp_path, rollout_steps=1, scale_rewards=bad, **CPU)


def test_the_guard_is_readable_without_building_a_trainer_config():
    assert_single_normalization(SimpleNamespace(scale_rewards="none"))
    with pytest.raises(ValueError, match="omega grid"):
        assert_single_normalization(SimpleNamespace(scale_rewards="group"))


def test_the_grid_cells_reach_the_optimizer_at_one_advantage_scale():
    """As declared, the cells separate by 1.68x in advantage magnitude, ordered by the norm
    of their weight vector -- an online grid would spend that as a larger optimizer step
    rather than as a different weighting. grid_reward_configs holds ||omega|| at 1."""
    from src.rlsf.config import grid_reward_configs
    from src.rlsf.reward import RewardConfig

    rng = np.random.default_rng(0)
    prompts, group_size = 200, 4
    n = prompts * group_size
    hyps = [" ".join(["w"] * int(rng.integers(8, 16))) for _ in range(n)]
    refs = [" ".join(["w"] * 12)] * n
    components = {
        "bleu": rng.normal(30, 8, n).tolist(),
        "kiwi": rng.normal(0.7, 0.1, n).tolist(),
        "judge": rng.integers(1, 6, n).astype(float).tolist(),
    }

    def advantages(rc) -> np.ndarray:
        rewards, _, _ = compute_rewards(
            ["s"] * n, hyps, refs,
            cfg=rc,
            group_size=group_size,
            component_scores=components,
            centroid=CENTROID,
        )
        grouped = rewards.reshape(prompts, group_size)
        return grouped - grouped.mean(axis=1, keepdims=True)

    cfg = load_config(require_caps=False)
    base = cfg["rlsf"]["reward"]
    omega = ("w_kiwi", "w_bleu", "w_judge")
    declared = [
        advantages(RewardConfig(**{k: cell.get(k, base[k]) for k in omega})).std(axis=1).mean()
        for cell in cfg["rlsf"]["weight_grid"]["cells"]
    ]
    assert declared == sorted(declared)
    assert declared[-1] / declared[0] == pytest.approx(1.68, abs=0.02)

    # The 1.68x is gone; the 3.5% left is the judge's integer scale going flat inside a
    # group, which varies with the weight direction rather than with ||omega||.
    delivered = [advantages(rc).std(axis=1).mean() for _, rc in grid_reward_configs(cfg)]
    assert max(delivered) / min(delivered) < 1.05


def test_trainer_scaling_would_rescale_every_group_to_unit_advantage():
    """Why the trainer's own scaling stays off: it divides each group by that group's own
    reward spread, so a group the policy has already flattened is amplified to unit scale."""
    rng = np.random.default_rng(0)
    group_size = 4
    wide = rng.normal(30, 8, group_size)
    flat = np.full(group_size, 30.0) + rng.normal(0, 1e-3, group_size)
    hyps = [" ".join(["w"] * 12)] * (2 * group_size)

    rewards, _, _ = compute_rewards(
        ["s"] * len(hyps), hyps, hyps,
        cfg=reward_config(load_config(require_caps=False)),
        group_size=group_size,
        component_scores={
            "bleu": np.concatenate([wide, flat]).tolist(),
            "kiwi": [1.0] * len(hyps),
            "judge": [1.0] * len(hyps),
        },
        centroid=CENTROID,
    )
    grouped = rewards.reshape(2, group_size)
    advantage = grouped - grouped.mean(axis=1, keepdims=True)
    scaled = advantage / (advantage.std(axis=1, keepdims=True) + 1e-4)
    assert scaled.std(axis=1) == pytest.approx([1.0, 1.0], abs=1e-3)


def test_the_reward_reaching_the_trainer_is_a_combined_z_score():
    # One live component, so the combined reward is that component's z: unit group sd.
    rc = reward_config(load_config(require_caps=False))
    n, group_size = 16, 4
    rng = np.random.default_rng(1)
    hyps = [" ".join(["w"] * 12) for _ in range(n)]
    rewards, _, _ = compute_rewards(
        ["s"] * n, hyps, hyps,
        cfg=rc,
        group_size=group_size,
        component_scores={
            "bleu": rng.normal(30, 8, n).tolist(),
            "kiwi": [1.0] * n,
            "judge": [1.0] * n,
        },
        centroid=CENTROID,
    )
    assert rewards.reshape(-1, group_size).std(axis=1) == pytest.approx(1.0)


# step accounting


def test_caps_count_rollouts_and_the_trainer_is_given_optimizer_steps(cfg, args):
    # docs/budget.md prices rollouts: one is 16 prompts x G = 64 judge calls. GRPO reuses
    # each rollout mu times, so the number handed to TRL is not the number in the caps.
    mu = cfg["rlsf"]["train"]["num_iterations"]
    assert mu > 1
    assert optimizer_steps(cfg, 600) == 600 * mu
    assert args.max_steps == 10 * mu


def test_a_generation_batch_is_exactly_one_rollout(cfg, args):
    assert rollout_batch(cfg) == 64
    assert args.generation_batch_size == 64
    assert args.num_generations == cfg["rlsf"]["rollout"]["group_size"]
    # Equal, or a rollout stops being mu optimizer steps and the cap stops counting them.
    assert args.steps_per_generation == args.gradient_accumulation_steps


def test_a_micro_batch_that_does_not_divide_the_rollout_is_refused(cfg, tmp_path):
    import copy

    cfg = copy.deepcopy(cfg)
    cfg["rlsf"]["train"]["per_device_train_batch_size"] = 5
    with pytest.raises(ValueError, match="micro-batches"):
        grpo_args(cfg, output_dir=tmp_path, rollout_steps=1, **CPU)


def test_the_config_carries_grpo_field_names_not_ppo_ones(cfg):
    train = cfg["rlsf"]["train"]
    assert not {"ppo_epochs", "clip_range", "kl_coef"} & set(train)
    assert {"num_iterations", "epsilon", "scale_rewards"} <= set(train)
    assert "beta" in cfg["rlsf"]["reference"]


def test_the_sampling_and_kl_settings_reach_the_trainer(cfg, args):
    rollout = cfg["rlsf"]["rollout"]
    assert (args.temperature, args.top_p) == (rollout["temperature"], rollout["top_p"])
    assert args.beta == cfg["rlsf"]["reference"]["beta"]
    assert args.epsilon == cfg["rlsf"]["train"]["epsilon"]
    assert args.max_completion_length == cfg["generator"]["max_tokens"]


# loop helpers


def test_completion_text_reads_both_shapes():
    assert completion_text("plain") == "plain"
    assert completion_text([{"role": "assistant", "content": "chat"}]) == "chat"


def test_a_completion_with_no_content_is_the_empty_string():
    # length_feasible rejects it rather than the reward path raising on None.
    assert completion_text([{"role": "assistant", "content": None}]) == ""


def test_the_budget_refuses_the_block_that_would_cross_the_call_cap():
    budget = JudgeBudget(max_calls=100, max_spend_usd=25.0)
    budget.reserve(64)
    with pytest.raises(BudgetExceeded, match="max_judge_calls"):
        budget.reserve(64)
    assert budget.calls == 64


def test_asking_for_more_rollouts_than_the_cap_is_refused(cfg, monkeypatch):
    import sys

    from src.rlsf import train

    cap = cfg["rlsf"]["caps"]["max_steps"]
    monkeypatch.setattr(sys, "argv", ["rlsf_train", "--steps", str(cap + 1), "--yes"])
    with pytest.raises(SystemExit, match="rule 3"):
        train.main()


def test_the_budget_refuses_once_the_measured_spend_reaches_the_dollar_cap():
    budget = JudgeBudget(max_calls=10_000, max_spend_usd=1.0)
    budget.reserve(64)
    budget.spend_usd = 1.0
    with pytest.raises(BudgetExceeded, match="max_judge_spend_usd"):
        budget.reserve(64)
