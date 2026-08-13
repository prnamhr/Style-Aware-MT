"""GRPO wiring: the normalization contract, the step accounting, and the loop's helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
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
from src.rlsf.train import (
    BudgetExceeded,
    JudgeBudget,
    LoopState,
    _kiwi_or_none,
    arm_path,
    arm_reward_config,
    completion_text,
    make_reward_fn,
    prune_ref_adapter,
    run_manifest,
)
from src.rlsf.config import grid_reward_configs
from src.rlsf.reward import RewardConfig
import sys

from src.rlsf import train
import copy
import torch
from src.rlsf.train import make_optim_callback
from src.rlsf.train import adapter_delta, trainable_snapshot
from src.rlsf.stop import DriftMonitor
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
    assert declared[-1] / declared[0] == pytest.approx(4.44, abs=0.02)

    # The spread is gone; the few percent left is the judge's integer scale going flat inside
    # a group, which varies with the weight direction rather than with ||omega||.
    delivered = [advantages(rc).std(axis=1).mean() for _, rc in grid_reward_configs(cfg)]
    assert max(delivered) / min(delivered) < 1.10


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
    # One live component, so the combined reward is that component's z at its own weight.
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
    assert rewards.reshape(-1, group_size).std(axis=1) == pytest.approx(rc.w_bleu)


# step accounting


def test_caps_count_rollouts_and_the_trainer_is_given_optimizer_steps(cfg, args):
    # docs/budget.md prices rollouts: one is 16 prompts x G = 64 judge calls. GRPO reuses
    # each rollout mu times, so the number handed to TRL is not the number in the caps.
    mu = cfg["rlsf"]["train"]["num_iterations"]
    assert mu > 1
    assert optimizer_steps(cfg, 600) == 600 * mu
    assert args.max_steps == 10 * mu


def test_a_generation_batch_is_exactly_one_rollout(cfg, args):
    assert rollout_batch(cfg) == 32
    assert args.generation_batch_size == 32
    assert args.num_generations == cfg["rlsf"]["rollout"]["group_size"]
    # Equal, or a rollout stops being mu optimizer steps and the cap stops counting them.
    assert args.steps_per_generation == args.gradient_accumulation_steps


def test_a_micro_batch_that_does_not_divide_the_rollout_is_refused(cfg, tmp_path):

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


def test_pruning_a_checkpoint_takes_the_ref_copy_and_nothing_else(tmp_path):
    ckpt = tmp_path / "checkpoint-100"
    (ckpt / "ref").mkdir(parents=True)
    (ckpt / "ref" / "adapter_model.safetensors").write_bytes(b"frozen")
    (ckpt / "adapter_model.safetensors").write_bytes(b"trained")
    (ckpt / "adapter_config.json").write_text("{}")
    assert prune_ref_adapter(ckpt)
    assert not (ckpt / "ref").exists()
    assert (ckpt / "adapter_model.safetensors").read_bytes() == b"trained"
    # A save with no ref adapter -- the dry run, or beta 0 -- is not an error.
    assert not prune_ref_adapter(ckpt)


def test_the_budget_refuses_the_block_that_would_cross_the_call_cap():
    budget = JudgeBudget(max_calls=100, max_spend_usd=25.0)
    budget.reserve(64)
    with pytest.raises(BudgetExceeded, match="max_judge_calls"):
        budget.reserve(64)
    assert budget.calls == 64


def test_asking_for_more_rollouts_than_the_cap_is_refused(cfg, monkeypatch):

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


# the pre-registered arms


# docs/preregistration_rlsf.md and its 2026-08-13 addendum, which adds the third arm:
# omega at unit norm, (bleu, kiwi, judge).
PREREGISTERED = {
    "w3_0.0": (0.7071, 0.7071, 0.0),
    "w3_2.0": (0.4082, 0.4082, 0.8165),
    "w3_6.0": (0.1622, 0.1622, 0.9733),
}


@pytest.mark.parametrize("cell,omega", PREREGISTERED.items())
def test_the_trained_arms_reach_the_optimizer_at_the_omega_pre_registered(cfg, cell, omega):
    rc = arm_reward_config(cfg, cell)
    assert (rc.w_bleu, rc.w_kiwi, rc.w_judge) == pytest.approx(omega, abs=5e-5)
    assert math.hypot(rc.w_bleu, rc.w_kiwi, rc.w_judge) == pytest.approx(1.0)


def test_the_arms_are_not_the_reward_block_the_config_declares(cfg):
    """Without --cell the loop trains omega = (1,1,1)/sqrt(3), which is neither arm. The
    difference is not only the judge term: the metric weights move by 1.22x, and under a
    weighted sum that is an effective step size."""
    base = reward_config(cfg)
    metric = arm_reward_config(cfg, "w3_0.0")
    assert base.w_judge > 0
    assert metric.w_bleu / base.w_bleu == pytest.approx(math.sqrt(3) / math.sqrt(2), abs=1e-6)


def test_an_exploratory_cell_cannot_be_trained(cfg):
    # They are re-ranked on the cached pool, after it was read. Training one would put a
    # post hoc cell into the grid the pre-registered selection rule draws from.
    with pytest.raises(SystemExit, match="not a pre-registered grid cell"):
        arm_reward_config(cfg, "mix_stylo")


def test_the_arms_do_not_write_to_one_anothers_paths(cfg):
    logs = {arm_path(cfg["output"]["step_log"], cell) for cell in PREREGISTERED}
    adapters = {arm_path(cfg["output"]["adapter_dir"], cell) for cell in PREREGISTERED}
    assert len(logs) == len(adapters) == len(PREREGISTERED)
    assert arm_path(cfg["output"]["step_log"], None) == Path(cfg["output"]["step_log"])


def test_holding_the_judge_flat_leaves_the_ablation_arms_reward_untouched(cfg):
    """Why RL-Metric may run --skip_judge and still be the arm as declared: at omega_3 = 0
    a constant judge and a real one give the same reward, so the arm buys only the matched
    feasibility mask docs/preregistration_rlsf.md sizes on the pool."""
    rc = arm_reward_config(cfg, "w3_0.0")
    n, group_size = 32, 4
    rng = np.random.default_rng(3)
    hyps = [" ".join(["w"] * 12) for _ in range(n)]
    components = {"bleu": rng.normal(30, 8, n).tolist(), "kiwi": rng.normal(0.7, 0.1, n).tolist()}

    def rewards(judge):
        out, _, _ = compute_rewards(
            ["s"] * n, hyps, hyps,
            cfg=rc,
            group_size=group_size,
            component_scores={**components, "judge": judge},
            centroid=CENTROID,
        )
        return out

    flat = rewards([1.0] * n)
    scored = rewards(rng.integers(1, 6, n).astype(float).tolist())
    assert flat == pytest.approx(scored)


def test_a_step_line_carries_the_drift_verdict_the_run_acted_on(cfg, tmp_path):
    # The pre-registration reports the verdict at every step. Replaying it later reads
    # whatever rule is on disk then, which is not necessarily the one that halted the run.

    step_log = tmp_path / "steps.jsonl"
    state = LoopState()
    reward_fn = make_reward_fn(
        rc=arm_reward_config(cfg, "w3_0.0"),
        cell="w3_0.0",
        group_size=4,
        kiwi=None,
        judge=None,
        template=None,
        centroid=CENTROID,
        monitor=DriftMonitor(),
        step_log=step_log,
        judge_workers=1,
        budget=JudgeBudget(0, 0.0),
        state=state,
    )
    hyps = [f"however the {'long ' * i}fox jumps" for i in range(8)]
    reward_fn(None, hyps, source=["s"] * 8, reference=["the fox jumps over"] * 8)

    line = json.loads(step_log.read_text(encoding="utf-8").splitlines()[0])
    assert line["cell"] == "w3_0.0"
    assert line["drift"]["tripped"] is False
    assert "not complete" in line["drift"]["reason"]
    assert line["z_se"], "the monitor needs a clustered error to size its band"


def test_the_configs_kiwi_placement_reaches_the_worker(cfg, monkeypatch):
    """Left unset, the worker auto-detects the card GRPO is training on and allocates
    inside the first reward call, next to the rollout's logits."""

    monkeypatch.setenv("RLSF_COMET_PYTHON", sys.executable)
    kiwi_cfg = dict(cfg["rlsf"]["reward"]["kiwi"], python=None)
    assert kiwi_cfg["gpus"] == 0
    assert _kiwi_or_none(kiwi_cfg, False).gpus == 0
    assert _kiwi_or_none({**kiwi_cfg, "gpus": None}, False).gpus is None


def test_the_manifest_names_the_arm_and_what_was_held_flat(cfg):
    man = run_manifest(
        cfg, cell="w3_0.0", rc=arm_reward_config(cfg, "w3_0.0"), steps=500, held_flat=["judge"]
    )
    assert man["cell"] == "w3_0.0"
    assert man["omega"]["judge"] == 0.0
    assert man["omega_norm"] == pytest.approx(1.0)
    assert man["held_flat"] == ["judge"]
    # The comparison is only between arms trained under one drift rule and one initialization.
    assert man["drift_rule"]["min_delta"] == 0.23
    assert man["generator"]["adapter_path"] == cfg["rlsf"]["reference"]["adapter_path"]


def test_the_manifest_pins_the_libraries_the_update_depends_on(cfg):
    """A GRPO step is defined by trl and peft as much as by the config, so a run's numbers
    are only attributable to a build if that build is recorded with them."""
    man = run_manifest(cfg, cell="w3_0.0", rc=arm_reward_config(cfg, "w3_0.0"), steps=1,
                       held_flat=[])
    versions = man["versions"]
    for name in ("torch", "transformers", "trl", "peft", "accelerate"):
        assert versions[name], f"{name} version not recorded"
    assert "cuda" in versions and "device" in versions


def test_the_optimizer_block_names_the_rollout_that_produced_it():
    """The mu passes log after the rollout's reward call, so attributing them to the line
    they are written on would shift every kl and grad_norm forward by one step."""
    state = LoopState()
    assert state.take_optim() == {}, "nothing has been optimized before the first rollout"
    state.rollout = 1
    state.optim_block = [{"kl": 0.0, "grad_norm": 2.0}, {"kl": 0.4, "grad_norm": 4.0}]
    block = state.take_optim()
    assert block == {"rollout": 0, "passes": 2, "kl": 0.2, "grad_norm": 3.0}
    assert state.take_optim() == {}, "a block is reported once, not carried into the next step"


def test_the_callback_keeps_the_optimizer_keys_and_drops_the_rest():
    """What the trainer logs is wider than what the step log carries, and `epoch` or a
    string-valued entry reaching the JSON line would break the schema consumers read."""

    mu = 4
    state = LoopState()
    callback = make_optim_callback(state)
    for _ in range(mu):
        callback.on_log(None, None, None, logs={
            "kl": 0.5, "grad_norm": 1.0, "learning_rate": 1e-6, "loss": 0.25,
            "epoch": 0.01, "num_tokens": 4096, "mode": "train",
        })
    state.rollout = 1
    block = state.take_optim()
    assert block["passes"] == mu, "every mu pass should be collected, not just the last"
    assert set(block) == {"rollout", "passes", "kl", "grad_norm", "learning_rate", "loss"}

    callback.on_log(None, None, None, logs={"epoch": 0.02, "mode": "eval"})
    callback.on_log(None, None, None, logs=None)
    assert not state.optim_block, "a log carrying no optimizer key should add nothing"


def test_adapter_delta_is_zero_at_the_initialization_and_positive_after_a_step():
    """A flat style score is not evidence about the reward if the adapter never moved."""

    model = torch.nn.Linear(4, 4, bias=False)
    torch.nn.init.constant_(model.weight, 0.5)
    snapshot = trainable_snapshot(model)
    assert adapter_delta(model, snapshot) == {"l2": 0.0, "rel": 0.0}

    with torch.no_grad():
        model.weight += 0.1
    moved = adapter_delta(model, snapshot)
    assert moved["l2"] == pytest.approx(0.4)      # 16 entries, each 0.1
    assert moved["rel"] == pytest.approx(0.2)     # ||0.5|| over 16 entries is 2.0
