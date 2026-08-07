
from __future__ import annotations

from pathlib import Path

import yaml

from src.rlsf.reward import RewardConfig

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


def reward_config(cfg: dict) -> RewardConfig:
    """Build the reward from the config's `reward:` block, ignoring its sub-blocks."""
    block = cfg["rlsf"]["reward"]
    fields = RewardConfig.__dataclass_fields__
    return RewardConfig(**{k: v for k, v in block.items() if k in fields})


def grid_reward_configs(cfg: dict) -> list[tuple[str, RewardConfig]]:
    """The RQ3 weight grid as (cell name, reward config) pairs."""
    base = cfg["rlsf"]["reward"]
    fields = RewardConfig.__dataclass_fields__
    out = []
    for cell in cfg["rlsf"]["weight_grid"]["cells"]:
        merged = {k: v for k, v in {**base, **cell}.items() if k in fields}
        out.append((cell["name"], RewardConfig(**merged)))
    return out


def worst_case_judge_calls(cfg: dict, *, group_size: int | None = None) -> int:
    """Judge calls if every capped step runs, at ``group_size`` (default: the ceiling)."""
    rlsf = cfg["rlsf"]
    caps = rlsf["caps"]
    steps = (caps["max_steps"] or 0) + (caps["max_grid_steps"] or 0)
    if group_size is None:
        group_size = caps["group_size_ceiling"]
    return steps * rlsf["rollout"]["prompts_per_step"] * group_size


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
