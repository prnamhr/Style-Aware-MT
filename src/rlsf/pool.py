"""
Dev-slice best-of-N pool: sample, score with the full reward path, dump. No update.

Usage:
    python manage.py rlsf_pool --yes
    python manage.py rlsf_pool --segments 4 --skip_judge     # free: no paid calls
    python manage.py rlsf_pool --resume --yes                # continue after a disconnect
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from src.data.rlsf_dev import sha256_file
from src.infer.run import build_zeroshot_user, make_client
from src.infer.usage import Usage
from src.rlsf.config import (
    judge_concurrency,
    load_config,
    make_judge_client,
    reward_config,
)
from src.rlsf.kiwi import KiwiScorer
from src.rlsf.reward import JudgeTiming, judge_scores, load_train_template, overlap_scores
from src.rlsf.train import JudgeBudget

# Usage artefacts the per-call rate is measured from, newest first.
_USAGE_RECORDS = (Path("outputs/rlsf/pool_usage.json"), Path("outputs/rlsf/smoke_usage.json"))

_HELD_FLAT = 1.0

_DEFAULT_CHUNK = 25


def measured_per_call_usd(client, model: str, records=_USAGE_RECORDS) -> float:
    for path in records:
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("model") != model or not record.get("calls"):
            continue
        # Cost is linear in tokens, so pricing the totals once and dividing is exact.
        usage = Usage(pricing=client.usage.pricing)
        usage.add(model, record["prompt_tokens"], record["completion_tokens"])
        return usage.cost_usd / record["calls"]
    raise FileNotFoundError(
        f"no usage artefact among {[str(p) for p in records]} measures {model}, so this run "
        f"cannot state a priced worst case before spending (budget rule 1). Run the pilot "
        f"first: python manage.py rlsf_smoke --yes (budget rule 2)."
    )


def sidecar(pool_path: str | Path, suffix: str) -> Path:
    """A file named after the pool it belongs to: pool.jsonl -> pool_<suffix>."""
    pool_path = Path(pool_path)
    return pool_path.with_name(f"{pool_path.stem}_{suffix}")


def plan(segments: int, n: int, per_call_usd: float) -> dict:
    """Call volume and priced cost, stated before spending (rule 1)."""
    calls = segments * n
    return {"samples": calls, "judge_calls": calls, "est_usd": round(calls * per_call_usd, 4)}


def pool_settings(cfg: dict, *, n: int | None = None, chunk: int | None = None) -> dict:
    """The `pool:` block, with the sample count checked against the authorized ceiling."""
    block = cfg["rlsf"].get("pool") or {}
    out = {
        "n": n or block.get("n") or cfg["rlsf"]["rollout"]["group_size"],
        "chunk": chunk or block.get("chunk") or _DEFAULT_CHUNK,
        "judge_calls": block.get("judge_calls"),
    }
    ceiling = cfg["rlsf"]["caps"]["group_size_ceiling"]
    if out["n"] > ceiling:
        raise ValueError(
            f"pool.n {out['n']} exceeds the authorized group-size ceiling {ceiling}. Judge "
            f"calls scale one-for-one with it, so this is a widening of the authorised run "
            f"under budget rule 3: re-price the pool in docs/budget.md and raise the ceiling."
        )
    if out["judge_calls"] is None:
        raise ValueError(
            "rlsf.pool.judge_calls is undeclared. The pool spends one judge call per sampled "
            "completion, so an undeclared pool has no upper bound: price it in docs/budget.md "
            "and record the ceiling here first (budget rule 1)."
        )
    if out["chunk"] < 1:
        raise ValueError(f"pool.chunk must be >= 1, got {out['chunk']}")
    return out


def read_dev(path: str | Path, limit: int | None = None) -> tuple[list[str], list[str]]:
    """Sources and references from the dev slice, in file order."""
    sources, refs = [], []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            sources.append(row["input"])
            refs.append(row["output"])
            if limit and len(sources) == limit:
                break
    if not sources:
        raise ValueError(f"{path} yielded no rows")
    return sources, refs


def _jsonable(value: float) -> float | None:
    """nan -> null: an artefact meant to be read back should not carry a bare NaN literal."""
    return float(value) if math.isfinite(value) else None


def pack_rows(
    sources: list[str],
    refs: list[str],
    hyps: list[str],
    raw: dict[str, list[float]],
    n: int,
    *,
    start: int = 0,
) -> list[dict]:
    """One row per segment: its completions and every per-sample score, in sampling order."""
    rows = []
    for i, (src, ref) in enumerate(zip(sources, refs)):
        sl = slice(i * n, (i + 1) * n)
        rows.append(
            {
                "idx": start + i,
                "source": src,
                "reference": ref,
                "hyps": hyps[sl],
                "scores": {
                    name: [_jsonable(v) for v in values[sl]] for name, values in raw.items()
                },
            }
        )
    return rows


def append_rows(path: str | Path, rows: list[dict]) -> None:
    with Path(path).open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_pool(path: str | Path) -> list[dict]:
    """The pool as written, nulls restored to nan so the reward path sees them unmeasured."""
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            row["scores"] = {
                name: [float("nan") if v is None else float(v) for v in values]
                for name, values in row["scores"].items()
            }
            rows.append(row)
    return rows


def resume_point(rows: list[dict], sources: list[str], n: int) -> int:
    """Segments a partial pool already covers; raises if it was built against other data."""
    if len(rows) > len(sources):
        raise ValueError(
            f"the pool holds {len(rows)} segments but the slice has {len(sources)}: this pool "
            f"was built against a different slice or a different --segments"
        )
    for i, row in enumerate(rows):
        if row["idx"] != i or row["source"] != sources[i]:
            raise ValueError(
                f"pool row {i} is not segment {i} of the dev slice, so scores in it cannot be "
                f"attributed to these sources; rebuild rather than resume"
            )
        if len(row["hyps"]) != n:
            raise ValueError(f"pool row {i} carries {len(row['hyps'])} completions, not {n}")
    return len(rows)


def score_chunk(
    sources: list[str],
    refs: list[str],
    hyps: list[str],
    *,
    n: int,
    metric: str,
    kiwi,
    judge,
    template: str | None,
    workers: int,
    timing: JudgeTiming | None = None,
) -> dict[str, list[float]]:
    """Every per-sample component score for one chunk of segments."""
    src_rep = [s for s in sources for _ in range(n)]
    ref_rep = [r for r in refs for _ in range(n)]
    raw = {metric: overlap_scores(hyps, ref_rep, metric)}
    raw["kiwi"] = kiwi.score(src_rep, hyps) if kiwi is not None else [_HELD_FLAT] * len(hyps)
    if judge is None:
        raw["judge"] = [_HELD_FLAT] * len(hyps)
    else:
        raw["judge"] = judge_scores(
            judge, template, src_rep, ref_rep, hyps, max_workers=workers, timing=timing
        )
    return raw


def manifest(
    cfg: dict,
    *,
    n: int,
    segments: int,
    pool_path: Path,
    resumed_from: int,
    held_flat: list[str],
    usage: dict | None,
) -> dict:
    """Provenance for a pool that later runs re-rank without re-sampling."""
    gen, rollout = cfg["generator"], cfg["rlsf"]["rollout"]
    dev_file = Path(cfg["data"]["dev_file"])
    return {
        "purpose": "omega selection by best-of-N reranking over the RLSF dev slice",
        "generator": {
            "model": gen["model"],
            "adapter_path": gen.get("adapter_path"),
            "load_in_4bit": gen.get("load_in_4bit", False),
            "max_tokens": gen["max_tokens"],
            "temperature": rollout["temperature"],
            "top_p": rollout["top_p"],
            "seed": cfg["rlsf"]["seed"],
        },
        "pool": {
            "n": n,
            "segments": segments,
            "samples": segments * n,
            "overlap_metric": reward_config(cfg).overlap_metric,
            "kiwi_model": cfg["rlsf"]["reward"]["kiwi"]["model"],
            "held_flat": held_flat,
            "resumed_from_segment": resumed_from,
        },
        "judge": {
            "model": cfg["judge"]["model"],
            "temperature": cfg["judge"]["temperature"],
            "seed": cfg["judge"]["seed"],
            "template_file": cfg["template_file"],
            "usage": usage,
            "usage_covers": "this invocation only; a resumed pool paid for the rest earlier",
        },
        "hashes": {
            dev_file.name: sha256_file(dev_file),
            pool_path.name: sha256_file(pool_path),
        },
        "caveats": [
            "The dev slice is held out from PPO updates only. The PEFT checkpoint sampled "
            "here was trained on all of train.jsonl, this slice included.",
            "Dev-slice figures select the reward weights and are never reported as a result; "
            "val remains the reported split.",
            "Rewards are group-normalized within a segment, so pool scores are comparable "
            "across omega cells only at a fixed n.",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the dev-slice best-of-N reward pool.")
    parser.add_argument("--config", default="configs/rlsf.yaml")
    parser.add_argument("--segments", type=int, default=None, help="default: the whole dev slice")
    parser.add_argument("--n", type=int, default=None, help="samples per segment; default: pool.n")
    parser.add_argument("--chunk", type=int, default=None, help="segments scored per append")
    parser.add_argument("--out", default=None, help="default: output.pool")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted pool")
    parser.add_argument("--skip_judge", action="store_true", help="no paid calls; judge held flat")
    parser.add_argument("--skip_kiwi", action="store_true", help="no COMET worker; kiwi held flat")
    parser.add_argument("--judge_concurrency", type=int, default=None)
    parser.add_argument("--yes", action="store_true", help="confirm the spend and proceed")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, require_caps=True)
    if args.judge_concurrency is not None:
        cfg["judge"]["concurrency"] = args.judge_concurrency  # validated by the reader below
    workers = judge_concurrency(cfg)
    settings = pool_settings(cfg, n=args.n, chunk=args.chunk)
    n, chunk = settings["n"], settings["chunk"]

    sources, refs = read_dev(cfg["data"]["dev_file"], args.segments)
    segments = len(sources)
    out_path = Path(args.out or cfg["output"]["pool"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = 0
    if args.resume and out_path.exists():
        done = resume_point(read_pool(out_path), sources, n)
        print(f"resuming {out_path}: {done}/{segments} segments already scored")
    elif out_path.exists():
        raise SystemExit(
            f"{out_path} already exists. Pass --resume to continue it, or move it aside; "
            f"appending to a pool built under other settings would mix two samplings."
        )

    # Built before the plan is priced, not with the loop: the rate is read off the client's
    # own pricing table, and constructing one spends nothing.
    judge = None if args.skip_judge else make_judge_client(cfg)

    remaining = segments - done
    rate = 0.0 if judge is None else measured_per_call_usd(judge, cfg["judge"]["model"])
    p = plan(remaining, n, rate)
    print(f"plan: {remaining} segments x N={n} = {p['samples']} samples")
    if args.skip_judge:
        print("      judge skipped: 0 paid calls, judge component held flat")
    else:
        print(
            f"      {p['judge_calls']} judge calls at concurrency {workers}, ~${p['est_usd']} "
            f"at a measured ${rate:.3e}/call (pool ceiling {settings['judge_calls']})"
        )
        if done * n + p["judge_calls"] > settings["judge_calls"]:
            raise SystemExit(
                f"{p['judge_calls']} judge calls would take this pool past its declared "
                f"ceiling of {settings['judge_calls']}. Re-price it in docs/budget.md and "
                f"raise rlsf.pool.judge_calls deliberately (budget rule 3)."
            )
        if not args.yes:
            raise SystemExit("refusing to spend without --yes (budget rule 1)")
    if args.skip_kiwi:
        print("      kiwi skipped: adequacy component held flat")
    if remaining == 0:
        print("nothing left to sample")

    rc = reward_config(cfg)
    caps = cfg["rlsf"]["caps"]
    budget = JudgeBudget(caps["max_judge_calls"], caps["max_judge_spend_usd"])
    template = None if args.skip_judge else load_train_template()
    style = Path(cfg["prompt"]["style_instruction_file"]).read_text(encoding="utf-8")
    rollout = cfg["rlsf"]["rollout"]
    total_timing = JudgeTiming()

    policy = None
    if remaining:
        from transformers import set_seed

        policy = make_client(cfg["generator"])
        print(f"policy {cfg['generator']['model']} + {cfg['generator'].get('adapter_path')}")

    started = time.perf_counter()
    with _kiwi_or_none(cfg["rlsf"]["reward"]["kiwi"], args.skip_kiwi) as kiwi:
        if kiwi is not None:
            print(f"kiwi worker ready (reference_free={kiwi.reference_free})")
        for start in range(done, segments, chunk):
            stop = min(start + chunk, segments)
            chunk_src, chunk_ref = sources[start:stop], refs[start:stop]

            hyps: list[str] = []
            for i, src in enumerate(chunk_src):
                # Seeded per segment, not once per process: a pool resumed after a Colab
                # disconnect then holds the same draws an uninterrupted run would have made.
                set_seed(cfg["rlsf"]["seed"] + start + i)
                hyps += policy.complete_many(
                    style,
                    build_zeroshot_user(src),
                    n=n,
                    temperature=rollout["temperature"],
                    top_p=rollout["top_p"],
                )

            block = JudgeTiming()
            if judge is not None:
                budget.reserve(len(hyps))
            raw = score_chunk(
                chunk_src, chunk_ref, hyps,
                n=n, metric=rc.overlap_metric, kiwi=kiwi, judge=judge,
                template=template, workers=workers, timing=block,
            )
            if judge is not None:
                total_timing.wall_s += block.wall_s
                total_timing.call_s += block.call_s
                total_timing.calls += block.calls
                budget.spend_usd = judge.usage.summary()["cost_usd"]

            append_rows(out_path, pack_rows(chunk_src, chunk_ref, hyps, raw, n, start=start))
            elapsed = time.perf_counter() - started
            rate = (stop - done) / elapsed
            print(
                f"  {stop}/{segments} segments, {elapsed / 60:.1f} min elapsed, "
                f"{(segments - stop) / rate / 60:.1f} min left, "
                f"{budget.calls} judge calls (${budget.spend_usd:.4f})"
            )

    held_flat = [name for name, skip in (("kiwi", args.skip_kiwi), ("judge", args.skip_judge))
                 if skip]
    usage = None
    if judge is not None:
        usage = judge.usage.summary()
        usage["per_call_usd"] = usage["cost_usd"] / usage["calls"] if usage["calls"] else 0.0
        usage.update(total_timing.summary())
        # Sidecars named after the pool, so a free pass into the same directory does not
        # overwrite the provenance of the pool that was paid for.
        sidecar(out_path, "usage.json").write_text(
            json.dumps(usage, indent=2) + "\n", encoding="utf-8"
        )

    man = manifest(
        cfg, n=n, segments=segments, pool_path=out_path,
        resumed_from=done, held_flat=held_flat, usage=usage,
    )
    man_path = sidecar(out_path, "manifest.json")
    man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nwrote {out_path} ({segments} segments x N={n}) and {man_path}")
    if held_flat:
        print(f"components held flat: {', '.join(held_flat)} -- this pool cannot select omega")
    if usage is not None:
        print(f"judge usage: {usage}")
        print(
            f"judge latency: {total_timing.wall_s / 60:.1f} min over {usage['calls']} calls at "
            f"concurrency {workers}, {total_timing.achieved_parallelism:.1f}x achieved"
        )
    print("next: python manage.py rlsf_omega  (free: no GPU, no paid calls)")


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
