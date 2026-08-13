"""
RLSF training: GRPO over the frozen PEFT checkpoint, rewarded by src.rlsf.reward.

Usage:
    python manage.py rlsf_train --cell w3_0.0 --skip_judge   # RL-Metric, free
    python manage.py rlsf_train --cell w3_2.0 --yes          # RLSF-Judge
    python manage.py rlsf_train --dry_run           # CPU, 0.5B, 2 rollouts, no paid calls
"""

from __future__ import annotations

import argparse
import json
import math
import os
import torch

from dataclasses import dataclass, field
from datasets import Dataset
from importlib.metadata import PackageNotFoundError, version as pkg_version
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel
from trl import GRPOTrainer
from pathlib import Path

from src.infer.run import build_zeroshot_user
from src.rlsf.config import (
    drift_rule,
    grid_reward_configs,
    grpo_args,
    judge_concurrency,
    load_config,
    make_judge_client,
    optimizer_steps,
    planned_rollouts,
    reward_config,
    rollout_batch,
)
from src.rlsf.kiwi import KiwiScorer
from src.rlsf.reward import (
    JudgeTiming,
    compute_rewards,
    judge_scores,
    load_centroid,
    load_train_template,
    overlap_scores,
    stylo_scores,
)
from src.rlsf.stop import DriftMonitor
from src.eval.stylometrics import REWARD_FEATURES, subcentroid

# Small enough to run on CPU, same tokenizer family as the locked base.
_DRY_RUN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# What a component contributes when it is held flat: one constant, so group_normalize
# returns zeros and the term drops out rather than inventing a signal.
_HELD_FLAT = 1.0

# Libraries whose version changes the update, recorded per run for reproducibility.
_PINNED = ("transformers", "trl", "peft", "accelerate", "bitsandbytes")

# Trainer log keys carried into the step log. A flat style score is not evidence about the
# reward unless these show the optimizer was actually moving the adapter.
_OPTIM_KEYS = (
    "loss", "grad_norm", "learning_rate", "kl", "entropy",
    "clip_ratio/region_mean", "completions/mean_length",
)


class BudgetExceeded(RuntimeError):
    """A declared spend cap would be crossed by the next judge block."""


@dataclass
class JudgeBudget:
    """The call and dollar caps, checked before each block."""

    max_calls: int
    max_spend_usd: float
    calls: int = 0
    spend_usd: float = 0.0

    def reserve(self, n: int) -> None:
        if self.calls + n > self.max_calls:
            raise BudgetExceeded(
                f"the next block of {n} judge calls would take the run to {self.calls + n}, "
                f"past caps.max_judge_calls of {self.max_calls}. Re-price the arm in "
                f"docs/budget.md and raise the cap deliberately (budget rule 3)."
            )
        # spend_usd trails by one block -- a block's cost is known only after its calls
        # return, so the block that crosses the dollar cap runs to completion before this trips.
        if self.spend_usd >= self.max_spend_usd:
            raise BudgetExceeded(
                f"judge spend has reached ${self.spend_usd:.4f} against "
                f"caps.max_judge_spend_usd of ${self.max_spend_usd:.2f}"
            )
        self.calls += n


@dataclass
class LoopState:
    """What the reward function tells the callback between steps."""

    rollout: int = 0
    stop_reason: str | None = None
    timing: JudgeTiming = field(default_factory=JudgeTiming)
    optim_block: list[dict] = field(default_factory=list)

    def take_optim(self) -> dict:
        """Mean of the optimizer logs collected since the last rollout, and whose they are."""
        block, self.optim_block = self.optim_block, []
        if not block:
            return {}
        out = {}
        for key in sorted(set().union(*block)):
            values = [b[key] for b in block if key in b]
            out[key] = round(sum(values) / len(values), 8)
        # The mu passes run after this rollout's reward call, so they belong to the rollout
        # named here, not to the step-log line they are written on.
        return {"rollout": self.rollout - 1, "passes": len(block), **out}


def completion_text(completion) -> str:
    """The assistant text, whether TRL yields a string or a conversational message list."""
    if isinstance(completion, str):
        return completion
    return "".join(message.get("content") or "" for message in completion)


def build_dataset(cfg: dict, split_file: str | Path, limit: int | None = None):
    """One row per source segment: the chat prompt plus the source and reference the
    reward needs, which TRL forwards to the reward function as extra columns."""

    style = Path(cfg["prompt"]["style_instruction_file"]).read_text(encoding="utf-8")
    rows = []
    with Path(split_file).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": style},
                        {"role": "user", "content": build_zeroshot_user(row["input"])},
                    ],
                    "source": row["input"],
                    "reference": row["output"],
                }
            )
            if limit and len(rows) == limit:
                break
    if not rows:
        raise ValueError(f"{split_file} yielded no rows")
    return Dataset.from_list(rows)


def reserve_vram(fraction: float | None) -> None:
    """Cap this process's share of the card so a co-resident COMET worker can allocate."""
    if not fraction:
        return

    if not torch.cuda.is_available():
        return
    torch.cuda.set_per_process_memory_fraction(float(fraction))
    print(f"trainer capped at {fraction:.0%} of the card; the rest is the COMET worker's")


def library_versions() -> dict:
    """The training stack as installed, so a run's numbers stay attributable to a build."""

    out = {"torch": torch.__version__, "cuda": torch.version.cuda}
    for name in _PINNED:
        try:
            out[name] = pkg_version(name)
        except PackageNotFoundError:
            out[name] = None
    out["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return out


def trainable_snapshot(model) -> list[tuple]:
    """CPU float32 copy of the trainable adapter: the baseline `adapter_delta` measures from."""

    return [
        (name, p.detach().to("cpu", torch.float32).clone())
        for name, p in model.named_parameters()
        if p.requires_grad
    ]


def adapter_delta(model, snapshot: list[tuple]) -> dict:
    """How far the trainable adapter has travelled from its initialization."""

    live = dict(model.named_parameters())
    moved = origin = 0.0
    for name, before in snapshot:
        now = live[name].detach().to("cpu", torch.float32)
        moved += float(((now - before) ** 2).sum())
        origin += float((before**2).sum())
    return {
        "l2": round(math.sqrt(moved), 8),
        "rel": round(math.sqrt(moved / origin), 8) if origin else float("nan"),
    }


def load_policy(*, model_id: str, adapter_path: str | None, ref_adapter_path: str | None,
                dtype: str, device_map, require_ref: bool = False):
    """The policy as a trainable PeftModel, with the frozen KL reference loaded as a second
    "ref" adapter so TRL anchors to the init, not the bare base."""

    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_id)
    load_kwargs = {"dtype": getattr(torch, dtype)}
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    if adapter_path:

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        if ref_adapter_path:
            # Without a "ref" adapter TRL's disable_adapter reverts to the bare base, so KL
            # would anchor to base rather than the frozen init (grpo_trainer use_adapter).
            model.load_adapter(ref_adapter_path, adapter_name="ref", is_trainable=False)
            model.set_adapter("default")
            assert "ref" in model.peft_config, "ref adapter did not register; KL would anchor to base"
            live = [n for n, p in model.named_parameters() if "ref" in n and p.requires_grad]
            assert not live, f"ref adapter is trainable ({len(live)} params); the KL anchor would drift"
    else:

        model = get_peft_model(
            model,
            LoraConfig(
                r=32,
                lora_alpha=64,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    if require_ref and "ref" not in model.peft_config:
        raise SystemExit(
            f"beta is non-zero but no 'ref' adapter registered, so TRL's disable_adapter would "
            f"anchor the KL at the bare {model_id} instead of the frozen initialization. The "
            f"arm would penalize distance from a policy it never started at. Set "
            f"generator.adapter_path and rlsf.reference.adapter_path, or set beta to 0 "
            f"deliberately."
        )
    active = getattr(model, "active_adapters", None) or model.active_adapter
    print(f"adapters {sorted(model.peft_config)}, active {active}")
    # use_cache is left on: TRL disables it per forward during training and rollout
    # generation needs the KV cache, which turning it off here makes quadratic.
    return model, tokenizer


def arm_reward_config(cfg: dict, cell: str | None):
    if cell is None:
        return reward_config(cfg)
    cells = dict(grid_reward_configs(cfg))
    if cell not in cells:
        raise SystemExit(
            f"--cell {cell!r} is not a pre-registered grid cell. Available: {sorted(cells)}. "
            f"Exploratory cells are re-ranked on the cached pool, never trained."
        )
    return cells[cell]


def arm_path(path: str | Path, cell: str | None) -> Path:
    """A default output path tagged with the arm, so two arms cannot overwrite each other."""
    path = Path(path)
    return path if cell is None else path.with_name(f"{path.stem}_{cell}{path.suffix}")


def sidecar(step_log: Path, suffix: str) -> Path:
    """A file named after the step log it belongs to: steps.jsonl -> steps_<suffix>."""
    return step_log.with_name(f"{step_log.stem}_{suffix}")


def run_manifest(cfg: dict, *, cell: str | None, rc, steps: int, held_flat: list[str]) -> dict:
    """What this arm is, written before the first rollout so a crashed run still says."""
    gen, rlsf = cfg["generator"], cfg["rlsf"]
    beta = rlsf["reference"]["beta"]
    if not beta:
        kl_anchor = None                                      # KL off; no reference used
    elif gen.get("adapter_path"):
        kl_anchor = rlsf["reference"].get("adapter_path") or gen["adapter_path"]
    else:
        kl_anchor = f"base:{gen['model']}"                    # no init adapter: disable reverts to base
    return {
        "cell": cell or "config reward: block",
        "omega": {name: round(w, 6) for name, w in rc.weights.items()},
        "omega_norm": round(math.hypot(*rc.weights.values()), 6),
        "held_flat": held_flat,
        "steps_requested": steps,
        "generator": {
            "model": gen["model"],
            "adapter_path": gen.get("adapter_path"),
            "load_in_4bit": gen.get("load_in_4bit", False),
            "max_tokens": gen["max_tokens"],
        },
        "rollout": dict(rlsf["rollout"]),
        "train": dict(rlsf["train"]),
        "beta": beta,
        "kl_anchor": kl_anchor,
        "seed": rlsf["seed"],
        "normalization": rc.normalization,
        "on_violation": rc.on_violation,
        "length_band": [rc.len_min_ratio, rc.len_max_ratio],
        "drift_rule": vars(drift_rule(cfg)),
        "train_file": cfg["data"]["train_file"],
        "versions": library_versions(),
    }


def _stylo_centroid(cfg: dict, rc) -> dict | None:
    """The reward-side centroid, loaded only for a cell that weights the stylometric term."""
    if not rc.w_stylo:
        return None

    return subcentroid(load_centroid(cfg["data"]["split_centroid_file"]), REWARD_FEATURES)


def make_reward_fn(
    *,
    rc,
    cell: str | None = None,
    group_size: int,
    kiwi,
    judge,
    template: str | None,
    centroid: dict,
    stylo_centroid: dict | None = None,
    monitor: DriftMonitor,
    step_log: Path,
    judge_workers: int,
    budget: JudgeBudget,
    state: LoopState,
    delta_fn=None,
):
    """The single reward function TRL calls once per rollout."""

    def style_reward(prompts, completions, completion_ids=None, **kwargs) -> list[float | None]:
        del prompts, completion_ids
        hyps = [completion_text(c) for c in completions]
        sources, refs = kwargs["source"], kwargs["reference"]

        raw = {rc.overlap_metric: overlap_scores(hyps, refs, rc.overlap_metric)}
        raw["kiwi"] = kiwi.score(sources, hyps) if kiwi is not None else [_HELD_FLAT] * len(hyps)
        if rc.w_stylo:
            # Counted off the text, so it adds no call and no cap pressure.
            raw["stylo"] = stylo_scores(hyps, stylo_centroid)
        if judge is None:
            raw["judge"] = [_HELD_FLAT] * len(hyps)
        else:
            budget.reserve(len(hyps))
            # Timed per block and summed: judge_scores assigns wall_s rather than adding to
            # it, so a shared timer would report the run's call time over one block's clock.
            block = JudgeTiming()
            raw["judge"] = judge_scores(
                judge, template, sources, refs, hyps,
                max_workers=judge_workers, timing=block,
            )
            state.timing.wall_s += block.wall_s
            state.timing.call_s += block.call_s
            state.timing.calls += block.calls
            budget.spend_usd = judge.usage.summary()["cost_usd"]

        rewards, _, log = compute_rewards(
            sources, hyps, refs,
            cfg=rc,
            group_size=group_size,
            component_scores=raw,
            centroid=centroid,
            step=state.rollout,
        )
        # Read before the line is written, so the log carries the verdict the run acted on
        # rather than one replayed later against a rule that may have been edited since.
        verdict = monitor.update(log)
        delta = delta_fn() if delta_fn is not None else {}
        line = {
            # Named per line, so lines pooled across arms stay attributable without the sidecar.
            "cell": cell,
            **log.as_dict(),
            "drift": verdict.as_dict(),
            "optim": state.take_optim(),
            "adapter_delta": delta,
        }
        with step_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")

        if verdict.tripped:
            state.stop_reason = verdict.reason
        state.rollout += 1
        print(
            f"  rollout {log.step}: reward {log.reward_mean:+.3f} sd {log.reward_sd:.3f}, "
            f"{log.n_feasible}/{log.n_samples} feasible, "
            f"{log.degenerate_groups}/{log.n_groups} degenerate, "
            f"adapter {delta.get('rel', float('nan')):.2e} from init, "
            f"{budget.calls} judge calls (${budget.spend_usd:.4f})"
        )
        return [float(r) if math.isfinite(r) else None for r in rewards]

    return style_reward


def make_optim_callback(state: LoopState):
    """Collects the trainer's own optimizer metrics for the next step-log line."""

    class OptimLog(TrainerCallback):
        def on_log(self, args, trainer_state, control, logs=None, **kwargs):
            kept = {
                k: float(v)
                for k, v in (logs or {}).items()
                if k in _OPTIM_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if kept:
                state.optim_block.append(kept)
            return control

    return OptimLog()


def make_stop_callback(state: LoopState):
    """Ends the run when the register-drift rule fires."""

    class DriftStop(TrainerCallback):
        def on_step_end(self, args, trainer_state, control, **kwargs):
            if state.stop_reason:
                control.should_training_stop = True
            return control

    return DriftStop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RLSF policy with GRPO.")
    parser.add_argument("--config", default="configs/rlsf.yaml")
    parser.add_argument(
        "--cell",
        default=None,
        help="pre-registered weight_grid cell to train, e.g. w3_0.0; default: the reward: block",
    )
    parser.add_argument("--steps", type=int, default=None, help="rollouts; default: plan.rollouts")
    parser.add_argument("--limit", type=int, default=None, help="training rows to read")
    parser.add_argument("--skip_judge", action="store_true", help="no paid calls; judge held flat")
    parser.add_argument("--skip_kiwi", action="store_true", help="no COMET worker; kiwi held flat")
    parser.add_argument("--max_completion_length", type=int, default=None)
    parser.add_argument("--out", default=None, help="default: output.step_log")
    parser.add_argument("--adapter_out", default=None, help="default: output.adapter_dir")
    parser.add_argument("--yes", action="store_true", help="confirm the spend and proceed")
    parser.add_argument(
        "--overwrite", action="store_true", help="discard an existing step log for this arm"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=f"CPU wiring check: {_DRY_RUN_MODEL}, no adapter, 2 rollouts, no paid calls",
    )
    return parser.parse_args()


def main() -> None:
    # Read once, when the CUDA allocator initializes: a 7B policy plus a frozen reference
    # adapter fragments the arena badly enough that a free-but-scattered gigabyte fails.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = _parse_args()
    cfg = load_config(args.config, require_caps=not args.dry_run)
    rlsf = cfg["rlsf"]
    group_size = rlsf["rollout"]["group_size"]
    steps = args.steps or planned_rollouts(cfg)

    skip_judge, skip_kiwi = args.skip_judge, args.skip_kiwi
    overrides: dict = {}
    if args.dry_run:
        skip_judge = skip_kiwi = True
        steps = args.steps or 2
        # Two prompts a rollout and a short completion budget, not sixteen and 1024: this
        # checks signatures on a CPU, not throughput. mu, beta and the reward path are left
        # at their configured values. The ref-adapter branch is not exercised: dry_run has no
        # init adapter, so TRL disables adapters to the bare 0.5B base for the KL reference.
        rlsf["rollout"]["prompts_per_step"] = 2
        # One sequence at a time: a 151,936-token vocabulary makes the fp32 logits of a
        # four-sequence micro-batch large enough to swap on an ordinary development box.
        rlsf["train"]["per_device_train_batch_size"] = 1
        overrides["use_cpu"] = True
        cfg["generator"].update(
            model=_DRY_RUN_MODEL, adapter_path=None, dtype="float32",
            device_map=None, max_tokens=128,
        )
    if args.max_completion_length:
        cfg["generator"]["max_tokens"] = args.max_completion_length

    cap = rlsf["caps"]["max_steps"]
    if cap and steps > cap:
        raise SystemExit(
            f"--steps {steps} exceeds caps.max_steps of {cap}. Running longer than the "
            f"authorized envelope is a widening under budget rule 3: re-price it in "
            f"docs/budget.md and raise the cap deliberately."
        )

    rc = arm_reward_config(cfg, args.cell)
    per_rollout = rollout_batch(cfg)
    judge_calls = 0 if skip_judge else steps * per_rollout
    print(
        f"plan: {steps} rollouts x {rlsf['rollout']['prompts_per_step']} prompts x G="
        f"{group_size} = {per_rollout} completions/rollout, "
        f"{optimizer_steps(cfg, steps)} optimizer steps at mu={rlsf['train']['num_iterations']}"
    )
    print(
        f"      omega {args.cell or 'reward: block'} = "
        + ", ".join(f"{name} {w:.4f}" for name, w in rc.weights.items())
    )
    held_flat = []
    if skip_judge:
        held_flat.append("judge")
        print("      judge skipped: 0 paid calls, judge component held flat")
    else:
        print(f"      {judge_calls} judge calls at concurrency {judge_concurrency(cfg)}")
        if not args.yes:
            raise SystemExit("refusing to spend without --yes (budget rule 1)")
    if skip_kiwi:
        held_flat.append("kiwi")
        print("      kiwi skipped: adequacy component held flat")
    # At omega_3 = 0 the judge score cannot enter the reward, so skipping it is the arm as
    # declared rather than a component silently missing from a reward that weights it.
    unweighted = {"judge": rc.w_judge == 0 and not rc.gated, "kiwi": rc.w_kiwi == 0}
    if any(not unweighted[name] for name in held_flat):
        print("      this is a wiring check, not a training result")

    step_log = arm_path(args.out or cfg["output"]["step_log"], None if args.out else args.cell)
    step_log.parent.mkdir(parents=True, exist_ok=True)
    if step_log.exists() and step_log.stat().st_size and not args.overwrite:
        raise SystemExit(
            f"{step_log} already holds a run of this arm. A halt is a result and is reported "
            f"at the step it halted, not restarted over its own log: move it aside, or pass "
            f"--overwrite if this rerun is a deliberate replacement."
        )
    step_log.write_text("", encoding="utf-8")
    adapter_out = arm_path(
        args.adapter_out or cfg["output"]["adapter_dir"], None if args.adapter_out else args.cell
    )
    manifest = run_manifest(cfg, cell=args.cell, rc=rc, steps=steps, held_flat=held_flat)
    sidecar(step_log, "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    training_args = grpo_args(cfg, output_dir=adapter_out, rollout_steps=steps, **overrides)
    kiwi_cfg = rlsf["reward"]["kiwi"]
    if not skip_kiwi and kiwi_cfg.get("gpus"):
        reserve_vram(rlsf["train"].get("gpu_memory_fraction"))
    ref_adapter_path = None
    if rlsf["reference"]["beta"] and cfg["generator"].get("adapter_path"):
        ref_adapter_path = rlsf["reference"].get("adapter_path") or cfg["generator"]["adapter_path"]
    model, tokenizer = load_policy(
        model_id=cfg["generator"]["model"],
        adapter_path=cfg["generator"].get("adapter_path"),
        ref_adapter_path=ref_adapter_path,
        dtype=cfg["generator"]["dtype"],
        device_map=cfg["generator"].get("device_map"),
        # The dry run has no init adapter and anchors to the bare base by design.
        require_ref=bool(rlsf["reference"]["beta"]) and not args.dry_run,
    )
    snapshot = trainable_snapshot(model)
    dataset = build_dataset(cfg, cfg["data"]["train_file"], args.limit or cfg["data"]["limit"])
    print(f"policy {cfg['generator']['model']} on {model.device}, {len(dataset)} training rows")

    state = LoopState()
    caps = rlsf["caps"]
    budget = JudgeBudget(caps["max_judge_calls"] or 0, caps["max_judge_spend_usd"] or 0.0)
    judge = None if skip_judge else make_judge_client(cfg)


    with _kiwi_or_none(kiwi_cfg, skip_kiwi) as kiwi:
        reward_fn = make_reward_fn(
            rc=rc,
            cell=args.cell,
            group_size=group_size,
            kiwi=kiwi,
            judge=judge,
            template=None if skip_judge else load_train_template(),
            centroid=load_centroid(cfg["data"]["centroid_file"]),
            stylo_centroid=_stylo_centroid(cfg, rc),
            monitor=DriftMonitor(drift_rule(cfg)),
            step_log=step_log,
            judge_workers=judge_concurrency(cfg),
            budget=budget,
            state=state,
            delta_fn=lambda: adapter_delta(model, snapshot),
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[make_stop_callback(state), make_optim_callback(state)],
        )
        trainer.train()

    trainer.save_model(str(adapter_out))
    print(f"\n{state.rollout} rollouts, adapter written to {adapter_out}, log {step_log}")
    if state.stop_reason:
        print(f"stopped early on the register-drift rule: {state.stop_reason}")
    usage = None
    if judge is not None:
        usage = judge.usage.summary()
        usage["model"] = cfg["judge"]["model"]
        usage.update(state.timing.summary())
        sidecar(step_log, "usage.json").write_text(
            json.dumps(usage, indent=2) + "\n", encoding="utf-8"
        )
        print(f"judge usage: {usage}")

    manifest["outcome"] = {
        "rollouts": state.rollout,
        # The step it halted at, which is what the arm is reported at.
        "halted_at_step": state.rollout - 1 if state.stop_reason else None,
        "stop_reason": state.stop_reason,
        "adapter_dir": str(adapter_out),
        "adapter_delta": adapter_delta(model, snapshot),
        "judge": usage,
    }
    sidecar(step_log, "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _kiwi_or_none(kiwi_cfg: dict, skip: bool):
    """The COMET worker as a context manager, or a no-op one when it is held flat."""
    if skip:
        from contextlib import nullcontext

        return nullcontext(None)
    return KiwiScorer(
        model=kiwi_cfg["model"],
        batch_size=kiwi_cfg["batch_size"],
        python=kiwi_cfg["python"],
        gpus=kiwi_cfg.get("gpus"),
    )


if __name__ == "__main__":
    main()
