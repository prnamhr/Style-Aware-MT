
from __future__ import annotations

from pathlib import Path

import yaml

from src.infer.run import make_client
from src.rlsf.reward import RewardConfig
from src.rlsf.stop import DriftRule

_CONFIG = Path("configs/rlsf.yaml")

# `group_size_ceiling` is excluded: it is the authorized envelope, not a pending declaration.
_SPEND_CAPS = ("max_steps", "max_grid_steps", "max_judge_calls", "max_judge_spend_usd")


def load_config(path: str | Path = _CONFIG, *, require_caps: bool = True) -> dict:
    """Read the RLSF config, validated. ``require_caps=False`` inspects without spending."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert_group_size_within_ceiling(cfg)
    if require_caps:
        assert_caps_declared(cfg, path)
    return cfg


def assert_caps_declared(cfg: dict, path: str | Path = _CONFIG) -> None:
    """Refuse to proceed while any spend cap is null."""
    caps = cfg["rlsf"]["caps"]
    undeclared = [key for key in _SPEND_CAPS if caps.get(key) is None]
    if undeclared:
        raise ValueError(
            f"{path} leaves {', '.join(undeclared)} undeclared. RLSF spends one judge call "
            f"per sampled completion, so an uncapped run has no upper bound. Record the step "
            f"cap, rollout batch cap, worst-case call volume and priced worst case in "
            f"docs/budget.md first, then set them here (budget rule 1). The declared cap is "
            f"still untranscribed from docs/proposal.pdf; an estimate is not a cap."
        )
    for key in _SPEND_CAPS:
        if caps[key] <= 0:
            raise ValueError(f"cap {key} must be positive, got {caps[key]!r}")


def assert_group_size_within_ceiling(cfg: dict) -> None:
    """Keep the rollout inside the envelope priced in docs/budget.md."""
    rlsf = cfg["rlsf"]
    group_size = rlsf["rollout"]["group_size"]
    ceiling = rlsf["caps"]["group_size_ceiling"]
    if group_size > ceiling:
        raise ValueError(
            f"rollout.group_size {group_size} exceeds the authorized ceiling {ceiling}. "
            f"Judge calls scale one-for-one with it, so this is a widening of the "
            f"authorised run under budget rule 3, not a config tweak: re-price the arm in "
            f"docs/budget.md and raise the ceiling deliberately."
        )


def make_judge_client(cfg: dict):
    """Build the training-time judge client from the config's `judge:` block."""
    judge = cfg["judge"]
    if judge["model"] in _rater_models():
        raise ValueError(
            f"the reward judge is {judge['model']}, which is also an evaluation rater. "
            f"Training against a rater spends it on the one condition that most needs a "
            f"rater it was not trained against; pick a model distinct from both."
        )

    return make_client(judge)


def judge_concurrency(cfg: dict) -> int:
    """Threads the judge fan-out may use. Provider-side retries absorb the 429s it invites."""
    n = cfg["judge"].get("concurrency", 1)
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"judge.concurrency must be a positive integer, got {n!r}")
    return n


def _rater_models() -> set[str]:
    """Models used as evaluation raters, read from their configs."""
    out = set()
    for path in (Path("configs/judge_eval.yaml"), Path("configs/judge_eval_gpt.yaml")):
        if path.exists():
            out.add(yaml.safe_load(path.read_text(encoding="utf-8"))["judge"]["model"])
    return out


def reward_config(cfg: dict) -> RewardConfig:
    """Build the reward from the config's `reward:` block at unit ||omega||, ignoring sub-blocks."""
    block = cfg["rlsf"]["reward"]
    fields = RewardConfig.__dataclass_fields__
    # Normalized for the same reason the grid cells are: unnormalized, the weights set here
    # double as a step size, so learning_rate means something different under each omega.
    return RewardConfig(**{k: v for k, v in block.items() if k in fields}).unit_omega()


def drift_rule(cfg: dict) -> DriftRule:
    """Build the register-drift stop rule from the config's `stop:` block."""
    block = cfg["rlsf"].get("stop") or {}
    fields = DriftRule.__dataclass_fields__
    return DriftRule(**{k: v for k, v in block.items() if k in fields})


def grid_reward_configs(cfg: dict) -> list[tuple[str, RewardConfig]]:
    """The RQ3 weight grid as (cell name, reward config) pairs, each at unit ||omega||.
    """
    base = cfg["rlsf"]["reward"]
    fields = RewardConfig.__dataclass_fields__
    out = []
    for cell in cfg["rlsf"]["weight_grid"]["cells"]:
        merged = {k: v for k, v in {**base, **cell}.items() if k in fields}
        # As declared the cells' weight vectors differ in norm by 1.68x, which an online grid
        # would spend as a larger optimizer step rather than as a different weighting.
        out.append((cell["name"], RewardConfig(**merged).unit_omega()))
    return out


def worst_case_judge_calls(cfg: dict, *, group_size: int | None = None) -> int:
    """Judge calls if every capped step runs, at ``group_size`` (default: the ceiling)."""
    rlsf = cfg["rlsf"]
    caps = rlsf["caps"]
    steps = (caps["max_steps"] or 0) + (caps["max_grid_steps"] or 0)
    if group_size is None:
        group_size = caps["group_size_ceiling"]
    return steps * rlsf["rollout"]["prompts_per_step"] * group_size


# GRPO wiring
# compute_rewards hands GRPOTrainer a combined z-score, so the trainer's own group
# normalization has to be off. See assert_single_normalization.
_REWARD_SCALING = "none"


def rollout_batch(cfg: dict) -> int:
    """Completions generated per rollout step, and so judge calls per rollout step."""
    rollout = cfg["rlsf"]["rollout"]
    return rollout["prompts_per_step"] * rollout["group_size"]


def optimizer_steps(cfg: dict, rollout_steps: int) -> int:
    """Optimizer steps for a number of rollouts: GRPO reuses each rollout mu times."""
    return rollout_steps * cfg["rlsf"]["train"]["num_iterations"]


def assert_single_normalization(args) -> None:
    """Refuse a GRPOConfig that would z-score an already z-scored reward."""
    if args.scale_rewards != _REWARD_SCALING:
        raise ValueError(
            f"scale_rewards is {args.scale_rewards!r}. compute_rewards already z-scores each "
            f"component within its group before the omega weights are applied, so the trainer "
            f"would divide that combined z-score by a second standard deviation. The run would "
            f"not crash; it would train on a distorted advantage and the omega grid would stop "
            f"meaning what docs/budget.md says it means. Set train.scale_rewards to "
            f"{_REWARD_SCALING!r}, or hand the trainer the raw omega-weighted sum instead."
        )


def grpo_args(
    cfg: dict,
    *,
    output_dir: str | Path,
    rollout_steps: int | None = None,
    **overrides,
):
    """Build the ``GRPOConfig`` for a run of ``rollout_steps`` rollouts (default: the cap)."""
    from trl import GRPOConfig

    rlsf = cfg["rlsf"]
    rollout, train = rlsf["rollout"], rlsf["train"]
    generation_batch_size = rollout_batch(cfg)
    micro = train["per_device_train_batch_size"]
    if generation_batch_size % micro:
        raise ValueError(
            f"a rollout of {generation_batch_size} completions does not divide into micro-batches "
            f"of {micro}; TRL slices the generation batch by per_device_train_batch_size"
        )
    steps = rlsf["caps"]["max_steps"] if rollout_steps is None else rollout_steps

    fields = {
        "output_dir": str(output_dir),
        "seed": rlsf["seed"],
        "learning_rate": train["learning_rate"],
        "num_iterations": train["num_iterations"],
        "epsilon": train["epsilon"],
        "beta": rlsf["reference"]["beta"],
        "loss_type": train["loss_type"],
        "scale_rewards": train["scale_rewards"],
        "num_generations": rollout["group_size"],
        "temperature": rollout["temperature"],
        "top_p": rollout["top_p"],
        "max_completion_length": cfg["generator"]["max_tokens"],
        "generation_batch_size": generation_batch_size,
        "per_device_train_batch_size": micro,
        # Held equal to TRL's derived steps_per_generation so one rollout is exactly
        # num_iterations optimizer steps, which is what optimizer_steps assumes.
        "gradient_accumulation_steps": generation_batch_size // micro,
        "max_steps": optimizer_steps(cfg, steps),
        "logging_steps": train["logging_steps"],
        "save_strategy": "no",
        "report_to": [],
    }
    args = GRPOConfig(**{**fields, **overrides})
    assert_single_normalization(args)
    if args.steps_per_generation != args.gradient_accumulation_steps:
        raise ValueError(
            f"TRL derived steps_per_generation={args.steps_per_generation} against "
            f"gradient_accumulation_steps={args.gradient_accumulation_steps}, so a rollout is no "
            f"longer num_iterations optimizer steps and caps.max_steps stops counting rollouts. "
            f"This happens under multi-process training; grpo_args assumes one process."
        )
    return args


def priced_worst_case(
    calls: int,
    pricing: tuple[float, float],
    *,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Dollar worst case for ``calls`` at USD per million tokens; an estimate, not a
    measurement -- the pilot reports provider counts."""
    per_call = prompt_tokens * pricing[0] / 1e6 + completion_tokens * pricing[1] / 1e6
    return calls * per_call
