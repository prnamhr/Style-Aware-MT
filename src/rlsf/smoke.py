"""
Reward-path smoke test: a few groups end to end, no PPO update.

Verifies the wiring only -- Kiwi handshake, within-group reward variance, StepLog write,
and the length band's rejection rate. Produces no reportable figure.

Usage:
    python manage.py rlsf_smoke --segments 20 --group_size 4 --yes
    python manage.py rlsf_smoke --segments 4 --skip_judge      # free: no paid calls
    python manage.py rlsf_smoke --hyps_file outputs/rlsf/smoke_hyps.jsonl --yes  # no sampling
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.infer.run import build_zeroshot_user, make_client
from src.rlsf.config import (
    drift_rule,
    judge_concurrency,
    load_config,
    make_judge_client,
    reward_config,
)
from src.rlsf.kiwi import KiwiScorer
from src.rlsf.reward import (
    JudgeTiming,
    compute_rewards,
    group_normalize,
    judge_scores,
    load_centroid,
    load_train_template,
    overlap_scores,
)

_PRICING = (0.15, 0.60)  # gpt-4o-mini, matching src/infer/openai_client.py
_EST_TOKENS = (700, 40)  # prompt, completion; midpoint of the docs/budget.md range

# A group whose combined rewards are identical carries no advantage and contributes no
# gradient. Some is tolerable, a quarter is not.
_MAX_DEGENERATE = 0.25
_MIN_SD = 1e-9
# "Not flooring most of the batch": more than half the samples must clear the band.
_MIN_FEASIBLE = 0.5


def plan(segments: int, group_size: int) -> dict:
    """Call volume and priced estimate, stated before spending (budget rule 1)."""
    calls = segments * group_size
    per_call = _EST_TOKENS[0] * _PRICING[0] / 1e6 + _EST_TOKENS[1] * _PRICING[1] / 1e6
    return {"samples": calls, "judge_calls": calls, "est_usd": round(calls * per_call, 4)}


def group_variance_report(
    raw: dict[str, list[float]], feasible: np.ndarray, group_size: int
) -> dict[str, dict]:
    """Per component, how many groups normalize to all zeros."""
    out: dict[str, dict] = {}
    for name, values in raw.items():
        arr = np.asarray(values, dtype=float)
        degenerate, sds = 0, []
        for start in range(0, len(arr), group_size):
            sl = slice(start, start + group_size)
            z = group_normalize(arr[sl], feasible[sl])
            sds.append(float(z.std(ddof=0)))
            if not np.any(z):
                degenerate += 1
        n_groups = len(sds)
        out[name] = {
            "groups": n_groups,
            "degenerate": degenerate,
            "degenerate_frac": round(degenerate / n_groups, 3) if n_groups else 0.0,
            "min_group_sd": round(min(sds), 4) if sds else 0.0,
        }
    return out


def reward_degeneracy(
    rewards: np.ndarray, feasible: np.ndarray, group_size: int
) -> dict:
    """Fraction of groups whose combined reward has no spread across feasible samples."""
    rewards = np.asarray(rewards, dtype=float)
    sds = []
    for start in range(0, len(rewards), group_size):
        sl = slice(start, start + group_size)
        # Infeasible entries are excluded, not floored in: `worst - 1.0` would manufacture
        # spread in a flat group. Below two feasible samples there is no spread to measure.
        values = rewards[sl][feasible[sl]]
        sds.append(float(values.std(ddof=0)) if values.size >= 2 else 0.0)
    degenerate = sum(sd < _MIN_SD for sd in sds)
    return {
        "groups": len(sds),
        "degenerate": degenerate,
        "degenerate_frac": round(degenerate / len(sds), 3) if sds else 1.0,
        "min_group_sd": round(min(sds), 6) if sds else 0.0,
    }


def verdicts(
    kiwi_ready: bool, degenerate_frac: float, log_written: bool, log: dict
) -> list[tuple[str, bool, str]]:
    """The four checks, as (name, passed, detail)."""
    feasible_frac = log["n_feasible"] / log["n_samples"] if log["n_samples"] else 0.0
    return [
        (
            "kiwi handshake",
            kiwi_ready,
            "worker ready" if kiwi_ready else "worker did not report ready",
        ),
        (
            "reward variance",
            degenerate_frac <= _MAX_DEGENERATE,
            f"{degenerate_frac:.0%} of groups have no reward spread across their "
            f"feasible samples",
        ),
        ("steplog written", log_written, f"n_samples={log.get('n_samples')}"),
        (
            "length band",
            feasible_frac > _MIN_FEASIBLE,
            f"{log['n_feasible']}/{log['n_samples']} feasible "
            f"({feasible_frac:.0%}), ratio mean {log['length_ratio_mean']:.2f}, "
            f"{log.get('n_unmeasured', 0)} unmeasured",
        ),
    ]


def _load_rows(path: Path, n: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == n:
                break
    if len(rows) < n:
        raise ValueError(f"{path} has {len(rows)} rows, asked for {n}")
    return rows


def _dump_hyps(
    path: Path, sources: list[str], refs: list[str], hyps: list[str], group_size: int
) -> None:
    """One row per segment, carrying its source, reference and completions."""
    with path.open("w", encoding="utf-8") as fh:
        for i, (src, ref) in enumerate(zip(sources, refs)):
            row = {
                "source": src,
                "reference": ref,
                "hyps": hyps[i * group_size : (i + 1) * group_size],
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_hyps(path: Path, n: int, group_size: int) -> tuple[list[str], list[str], list[str]]:
    """Sources, references and flattened completions from a dump written by an earlier run."""
    rows = _load_rows(path, n)
    sizes = {len(r["hyps"]) for r in rows}
    if sizes != {group_size}:
        raise ValueError(
            f"{path} carries {sorted(sizes)} completions per segment, not {group_size}; "
            f"pass --group_size to match the dump"
        )
    return (
        [r["source"] for r in rows],
        [r["reference"] for r in rows],
        [h for r in rows for h in r["hyps"]],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the RLSF reward path.")
    parser.add_argument("--config", default="configs/rlsf.yaml")
    parser.add_argument("--segments", type=int, default=20)
    parser.add_argument("--group_size", type=int, default=None, help="default: config rollout")
    parser.add_argument("--max_judge_calls", type=int, default=None, help="default: pilot block")
    parser.add_argument("--skip_judge", action="store_true", help="no paid calls; judge held flat")
    parser.add_argument(
        "--judge_concurrency", type=int, default=None, help="default: config judge.concurrency"
    )
    parser.add_argument("--yes", action="store_true", help="confirm the spend and proceed")
    parser.add_argument("--out", default="outputs/rlsf/smoke_steps.jsonl")
    parser.add_argument(
        "--hyps_file",
        default=None,
        help="score completions from an earlier dump instead of sampling; no generator is loaded",
    )
    args = parser.parse_args()

    # Caps gate PPO training; this pilot is what measures the rate needed to declare them,
    # so it loads without them and is bounded by its own judge-call ceiling instead.
    cfg = load_config(args.config, require_caps=False)
    group_size = args.group_size or cfg["rlsf"]["rollout"]["group_size"]
    ceiling = args.max_judge_calls or cfg["rlsf"]["pilot"]["judge_calls"]
    if args.judge_concurrency is not None:
        cfg["judge"]["concurrency"] = args.judge_concurrency  # validated by the reader below
    workers = judge_concurrency(cfg)

    p = plan(args.segments, group_size)
    print(f"plan: {args.segments} segments x G={group_size} = {p['samples']} samples")
    if args.skip_judge:
        print("      judge skipped: 0 paid calls, judge component held flat")
    else:
        print(
            f"      {p['judge_calls']} judge calls at concurrency {workers}, "
            f"~${p['est_usd']} (ceiling {ceiling})"
        )
        if p["judge_calls"] > ceiling:
            raise SystemExit(
                f"{p['judge_calls']} judge calls exceeds the pilot ceiling of {ceiling}. "
                f"Lower --segments, or raise --max_judge_calls deliberately."
            )
        if not args.yes:
            raise SystemExit("refusing to spend without --yes (budget rule 1)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.hyps_file:
        sources, refs, hyps = _load_hyps(Path(args.hyps_file), args.segments, group_size)
        print(f"loaded {len(hyps)} completions from {args.hyps_file}, no generator started")
    else:
        rows = _load_rows(Path(cfg["data"]["dev_file"]), args.segments)
        sources = [r["input"] for r in rows]
        refs = [r["output"] for r in rows]

        rollout = cfg["rlsf"]["rollout"]
        style = Path(cfg["prompt"]["style_instruction_file"]).read_text(encoding="utf-8")
        policy = make_client(cfg["generator"])
        print(f"sampling {group_size} completions per segment at T={rollout['temperature']} ...")
        hyps = []
        for src in sources:
            hyps += policy.complete_many(
                style,
                build_zeroshot_user(src),
                n=group_size,
                temperature=rollout["temperature"],
                top_p=rollout["top_p"],
            )

        # Sampling is the expensive half and runs elsewhere (Colab); dump it so the reward
        # path can be re-run locally with --hyps_file as many times as it takes.
        dump_path = out_path.parent / "smoke_hyps.jsonl"
        _dump_hyps(dump_path, sources, refs, hyps, group_size)
        print(f"wrote {dump_path}")

    # Every sample in a group shares its group's source and reference.
    src_rep = [s for s in sources for _ in range(group_size)]
    ref_rep = [r for r in refs for _ in range(group_size)]

    rc = reward_config(cfg)
    raw = {rc.overlap_metric: overlap_scores(hyps, ref_rep, rc.overlap_metric)}

    kiwi_cfg = cfg["rlsf"]["reward"]["kiwi"]
    kiwi_ready = False
    with KiwiScorer(
        model=kiwi_cfg["model"], batch_size=kiwi_cfg["batch_size"], python=kiwi_cfg["python"]
    ) as kiwi:
        kiwi_ready = True
        print(f"kiwi worker ready (reference_free={kiwi.reference_free})")
        raw["kiwi"] = kiwi.score(src_rep, hyps)

    judge = None
    timing = JudgeTiming()
    if args.skip_judge:
        raw["judge"] = [1.0] * len(hyps)
    else:
        judge = make_judge_client(cfg)
        raw["judge"] = judge_scores(
            judge,
            load_train_template(),
            src_rep,
            ref_rep,
            hyps,
            max_workers=workers,
            timing=timing,
        )

    rewards, feasible, log = compute_rewards(
        src_rep,
        hyps,
        ref_rep,
        cfg=rc,
        group_size=group_size,
        component_scores=raw,
        centroid=load_centroid(cfg["data"]["centroid_file"]),
        step=0,
    )

    out_path.write_text(json.dumps(log.as_dict()) + "\n", encoding="utf-8")
    written = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])

    # Per component: diagnostics for reading which term went flat, never the verdict.
    # A component can normalize to zeros while the combined reward still spreads.
    variance = group_variance_report(raw, feasible, group_size)
    degeneracy = reward_degeneracy(rewards, feasible, group_size)
    print(f"\nreward mean {log.reward_mean:.3f} sd {log.reward_sd:.3f}, wrote {out_path}")
    for name, stats in variance.items():
        print(f"  {name:6s} degenerate groups {stats['degenerate']}/{stats['groups']}")
    print(f"  {'reward':6s} degenerate groups {degeneracy['degenerate']}/{degeneracy['groups']}"
          f"  (min group sd {degeneracy['min_group_sd']})")

    # One draw of the drift baseline, not the baseline: the rule averages the run's first
    # steps, and a single step's error is wide enough that one draw cannot stand for it.
    rule = drift_rule(cfg)
    print(f"\ndrift stop: {rule.feature} z {log.z[rule.feature]:+.3f} +/- "
          f"{log.z_se[rule.feature]:.3f} over {args.segments} prompts. The rule opens its "
          f"baseline on the first {rule.baseline_steps} training steps and fires only after "
          f"{rule.window} consecutive steps past it.")

    print()
    failed = 0
    for name, ok, detail in verdicts(
        kiwi_ready, degeneracy["degenerate_frac"], out_path.exists(), written
    ):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        failed += not ok
    if judge is not None:
        # The measured per-call rate; docs/budget.md carries only an estimate until this.
        usage = judge.usage.summary()
        usage["per_call_usd"] = usage["cost_usd"] / usage["calls"] if usage["calls"] else 0.0
        usage["model"] = cfg["judge"]["model"]
        usage["concurrency"] = workers
        usage.update(timing.summary())
        (out_path.parent / "smoke_usage.json").write_text(
            json.dumps(usage, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\njudge usage: {usage}")
        print(
            f"judge latency: {timing.wall_s:.1f}s for {usage['calls']} calls at concurrency "
            f"{workers}, {timing.achieved_parallelism:.1f}x achieved. GPU idle per step is "
            f"the wall clock, not the call time."
        )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
