"""
Evaluation-time LLM-as-Judge over the Batch API.

Usage:
    python manage.py judge_batch --conditions zeroshot --split val \\
        --config configs/judge_eval_gpt.yaml --limit 25
    python manage.py judge_batch --conditions zeroshot random_fewshot knn_fewshot \\
        afsp_margin afsp_full peft commercial_haiku \\
        --split val --config configs/judge_eval_gpt.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from src.eval._io import condition_path, load_condition, merge_results, read_completed_jsonl
from src.eval.judge import (
    _DEFAULT_TEMPLATE,
    _JUDGE_SYSTEM,
    _aggregate,
    assert_cache_identity,
    assert_results_identity,
    bind_output,
    build_prompt,
    judge_results_path,
    judge_segment_dir,
    judge_stem,
    parse_score,
    template_digest,
)
from src.infer.openai_batch import BatchChatClient

_CUSTOM_ID = "seg_{i}"


def state_path(results_dir: Path, split: str, tag: str | None) -> Path:
    return results_dir / f"{judge_stem(split, tag)}_batch_state.json"


def read_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_state(path: Path, state: dict) -> None:
    """Persist synchronously: losing a batch id means paying for the batch twice."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(state, indent=2) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def resume_point(cache_path: Path, sources: list[str]) -> int:
    """Number of leading segments already scored, verifying they still align."""
    done = read_completed_jsonl(cache_path)
    for j, rec in enumerate(done):
        if rec.get("input") != sources[j]:
            raise ValueError(
                f"resume misalignment in {cache_path} at segment {j}: cached source differs "
                f"from the current outputs; delete the cache to re-judge"
            )
    return len(done)


def append_results(
    cache_path: Path,
    sources: list[str],
    start: int,
    collected: dict[str, dict],
) -> tuple[int, int]:
    """Append returned scores in segment order, stopping at the first absent segment.

    A request the judge failed on comes back with ``text=None`` and is written as a
    null score. An id absent from the batch entirely means the batch did not cover
    this range, so the tail is left unwritten rather than cached as nulls that a
    later run would mistake for finished work.
    """
    written = failed = 0
    with cache_path.open("a", encoding="utf-8") as f:
        for i in range(start, len(sources)):
            got = collected.get(_CUSTOM_ID.format(i=i))
            if got is None:
                break
            score = parse_score(got["text"]) if got["text"] is not None else None
            rec = {"input": sources[i], "score": score}
            if got["error"] is not None:
                rec["error"] = got["error"]
            if score is None:
                failed += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            written += 1
    return written, failed


def run_condition(
    client: BatchChatClient,
    template: str,
    condition: str,
    sources: list[str],
    preds: list[str],
    refs: list[str],
    *,
    cache_path: Path,
    work_dir: Path,
    state: dict,
    state_file: Path,
    poll_interval: float,
    timeout: float | None,
) -> list[int | None]:
    """Score one condition through the batch endpoint; returns scores in order."""
    start = resume_point(cache_path, sources)
    if start >= len(sources):
        print(f"  {condition}: already complete ({start}/{len(sources)})")
        return [r.get("score") for r in read_completed_jsonl(cache_path)]
    if start:
        print(f"  resuming: {start}/{len(sources)} already scored")

    entry = state.get(condition)
    # `n` guards against resuming a --limit pilot batch as if it covered the full split.
    if entry and entry.get("start") == start and entry.get("n") == len(sources) - start:
        batch_id = entry["batch_id"]
        print(f"  resuming in-flight batch {batch_id} (submitted for segments {start}+)")
    else:
        requests = [
            client.build_request(
                _CUSTOM_ID.format(i=i),
                _JUDGE_SYSTEM,
                build_prompt(template, sources[i], refs[i], preds[i]),
            )
            for i in range(start, len(sources))
        ]
        print(f"  submitting {len(requests)} requests ...")
        batch_id = client.submit(requests, work_dir, f"{condition}_{start}")
        # Persisted BEFORE polling: an interrupted poll must resume this batch,
        # never submit a second one for the same segments.
        state[condition] = {"batch_id": batch_id, "start": start, "n": len(requests)}
        write_state(state_file, state)
        print(f"  submitted batch {batch_id}")

    batch = client.poll(batch_id, interval=poll_interval, timeout=timeout)
    collected = client.collect(batch)
    written, failed = append_results(cache_path, sources, start, collected)
    complete = batch.status == "completed" and start + written == len(sources)
    print(
        f"  batch {batch.status}: wrote {written} segment(s)"
        f"{f', {failed} unscored' if failed else ''}"
    )
    if batch.status == "completed" and not complete:
        print(
            f"  WARNING: batch {batch_id} completed but covered only {written} of the "
            f"{len(sources) - start} segments requested; it was submitted for a different "
            f"range -- the remainder is left unwritten for a fresh submission"
        )
    if complete:
        state.pop(condition, None)
    else:
        reason = client.failure_reason(batch)
        print(
            f"  batch did not complete ({batch.status}"
            f"{f': {reason}' if reason else ''}); {len(sources) - start - written} "
            f"segment(s) left unwritten -- re-run to submit them"
        )
        state.pop(condition, None)  # the job is terminal; a re-run must submit afresh
    write_state(state_file, state)
    return [r.get("score") for r in read_completed_jsonl(cache_path)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation-time LLM-as-Judge over the OpenAI Batch API."
    )
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--split", default="val", help="output split tag (default: val)")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--config", required=True, help="YAML with a `judge:` generator block")
    parser.add_argument("--tag", default=None, help="judge tag (default: the config's `tag:`)")
    parser.add_argument("--allow_model_overwrite", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=None, help="DIAGNOSTIC pilot: first N segments per condition"
    )
    parser.add_argument("--poll_interval", type=float, default=30.0)
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="seconds to poll before giving up; the batch keeps running server-side "
        "and a re-run resumes polling it (default: wait indefinitely)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if "judge" not in cfg:
        raise ValueError(f"{args.config} has no `judge:` block")
    gen = cfg["judge"]
    if gen.get("provider", "openai") != "openai":
        raise ValueError(
            f"the batch transport is OpenAI-only; `{args.config}` sets provider "
            f"'{gen.get('provider')}'. Use `manage.py judge` for other providers."
        )
    template_path = Path(cfg.get("template_file", _DEFAULT_TEMPLATE))
    template = template_path.read_text(encoding="utf-8")
    digest = template_digest(template)
    tag = args.tag if args.tag is not None else cfg.get("tag")
    judge_model = gen.get("model", "unknown")

    client = BatchChatClient(
        model=judge_model,
        temperature=gen.get("temperature"),
        max_tokens=gen.get("max_tokens", 256),
        seed=gen.get("seed"),
        reasoning_effort=gen.get("reasoning_effort"),
        pricing=gen.get("pricing"),
        discount=gen.get("batch_discount", 0.5),
        completion_window=gen.get("completion_window", "24h"),
    )

    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    cache_dir = judge_segment_dir(results_dir, args.split, tag)
    out_path = judge_results_path(results_dir, args.split, tag)
    state_file = state_path(results_dir, args.split, tag)
    work_dir = results_dir / f"{judge_stem(args.split, tag)}_batch"
    assert_results_identity(out_path, judge_model, allow_overwrite=args.allow_model_overwrite)
    state = read_state(state_file)

    print(
        f"judge {judge_model} [batch]  tag={tag or '(none)'}  template={template_path} [{digest}]"
    )
    results: dict[str, dict] = {}
    for cond in args.conditions:
        if not condition_path(out_dir, cond, args.split).exists():
            print(f"skip {cond}: {condition_path(out_dir, cond, args.split)} not found")
            continue
        sources, preds, refs = load_condition(out_dir, cond, args.split)
        if args.limit is not None:
            sources, preds, refs = sources[: args.limit], preds[: args.limit], refs[: args.limit]
        print(f"Judging {len(preds)} segments for {cond} with {judge_model} ...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        assert_cache_identity(
            cache_dir, {"model": judge_model, "tag": tag, "template_sha256": digest}
        )
        output_sha256 = bind_output(cache_dir, cond, condition_path(out_dir, cond, args.split))
        scores = run_condition(
            client,
            template,
            cond,
            sources,
            preds,
            refs,
            cache_path=cache_dir / f"{cond}.jsonl",
            work_dir=work_dir,
            state=state,
            state_file=state_file,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
        )
        mean, coverage = _aggregate(scores)
        results[cond] = {
            "n": len(scores),
            "model": judge_model,
            "judge_tag": tag,
            "template_sha256": digest,
            "output_sha256": output_sha256,
            "transport": "batch",
            "mean": mean,
            "coverage": round(coverage, 4),
            "sources": sources,
            "segments": scores,
        }
        mean_str = f"{mean:.3f}" if mean is not None else "n/a"
        print(f"  {cond:<16} Φ {mean_str}  (coverage {coverage:.0%})")

    if not results:
        return
    if args.limit is not None:
        print(
            f"\npilot only (--limit {args.limit}): segment cache written to {cache_dir}, "
            f"{out_path} deliberately NOT written. Re-run without --limit for the full split."
        )
    else:
        preserved = merge_results(out_path, results)
        if preserved:
            print(f"preserved {len(preserved)} condition(s) not re-scored: {', '.join(preserved)}")
        print(f"Wrote {out_path}")

    session = client.usage.summary()
    usage_path = results_dir / f"{judge_stem(args.split, tag)}_usage.json"
    prior = {}
    if usage_path.exists():
        prior = json.loads(usage_path.read_text(encoding="utf-8")).get("cumulative", {})
    cumulative = {k: round(prior.get(k, 0) + v, 6) for k, v in session.items()}
    usage_path.write_text(
        json.dumps(
            {
                "model": judge_model,
                "tag": tag,
                "transport": "batch",
                "batch_discount": client.discount,
                "conditions": sorted(results),
                "limit": args.limit,
                "priced": client.priced,
                "session": session,
                "cumulative": cumulative,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Judge usage: {session}")
    if not client.priced:
        print(
            f"  note: no pricing for {judge_model}, so cost_usd is a floor of 0, not the "
            f"actual spend; set `pricing: [in, out]` in the judge block to account it"
        )
    print(f"Wrote {usage_path}  (cumulative ${cumulative['cost_usd']:.2f})")


if __name__ == "__main__":
    main()
