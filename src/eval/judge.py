"""
Evaluation-time LLM-as-Judge for register fidelity (RQ1 primary style axis, Φ).

Usage:
    python manage.py judge --conditions zeroshot --split val \\
        --config configs/judge_eval.yaml
    python manage.py judge --conditions zeroshot --split val \\
        --config configs/judge_eval_gpt.yaml --tag gpt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import yaml

from src.eval._io import condition_path, load_condition, merge_results, read_completed_jsonl
from src.infer.run import make_client

_RESULTS_DIR = Path("results")
_DEFAULT_TEMPLATE = Path("prompts/judge_eval.txt")
_JUDGE_SYSTEM = (
    "You are a careful, consistent evaluator of translation style. Follow the instructions exactly."
)
_META_NAME = "_meta.json"

# Prefer an explicit "Score: N" verdict; N must be a lone integer 1-5 -- reject a
# further digit ("12") or a decimal ("3.5"), but allow a sentence-final "3.".
_LONE_1_5 = r"([1-5])(?!\d)(?!\.\d)"
_SCORE_RE = re.compile(r"score\s*[:=]?\s*" + _LONE_1_5, re.IGNORECASE)


class _LazyClient:
    """Build the judge client on first use, not up front.

    When every segment cache under ``judge_<split>_segments/`` is already
    complete, re-deriving the aggregates costs no API calls -- so it should not
    require provider credentials either.
    """

    def __init__(self, gen: dict) -> None:
        self._gen = gen
        self._client = None

    def complete(self, system: str, user: str) -> str:
        if self._client is None:
            self._client = make_client(self._gen)
        return self._client.complete(system, user)

    @property
    def usage(self):
        """Usage of the underlying client, or None if it was never built."""
        return None if self._client is None else getattr(self._client, "usage", None)


def judge_stem(split: str, tag: str | None = None) -> str:
    """Artefact stem for one judge on one split; ``None`` keeps the original names."""
    return f"judge_{split}" if not tag else f"judge_{tag}_{split}"


def judge_results_path(results_dir: str | Path, split: str, tag: str | None = None) -> Path:
    return Path(results_dir) / f"{judge_stem(split, tag)}.json"


def judge_segment_dir(results_dir: str | Path, split: str, tag: str | None = None) -> Path:
    return Path(results_dir) / f"{judge_stem(split, tag)}_segments"


def template_digest(template: str) -> str:
    """Short sha256 of the frozen rubric, so two judges can be proven to share it."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


def assert_cache_identity(cache_dir: Path, meta: dict) -> None:
    """Bind a segment cache to one (model, tag, template) and refuse a mismatch."""
    path = cache_dir / _META_NAME
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        differing = {k: (prior.get(k), meta[k]) for k in meta if prior.get(k) != meta[k]}
        if differing:
            detail = "; ".join(
                f"{k}: cached {old!r}, now {new!r}" for k, (old, new) in differing.items()
            )
            raise ValueError(
                f"segment cache {cache_dir} was written by a different judge ({detail}). "
                f"Pass a distinct --tag so this judge gets its own cache, or delete the "
                f"cache to re-judge."
            )
        return
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def assert_results_identity(out_path: Path, model: str, *, allow_overwrite: bool) -> None:
    """Refuse to overwrite a results file that another judge model wrote."""
    if allow_overwrite or not out_path.exists():
        return
    try:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return  # merge_results raises a better-targeted error on this path
    others = sorted(
        {rec.get("model") for rec in existing.values() if isinstance(rec, dict)} - {None, model}
    )
    if others:
        raise ValueError(
            f"{out_path} holds scores from {', '.join(others)}, not {model}. Overwriting would "
            f"replace the primary Phi with a different rater's scores. Pass --tag to route this "
            f"judge to its own file, or --allow_model_overwrite if replacement is intended."
        )


def parse_score(text: str) -> int | None:
    """Extract the 1-5 verdict from a judge response."""
    if not text:
        return None
    matches = _SCORE_RE.findall(text)
    if matches:
        return int(matches[-1])
    ints = re.findall(r"(?<![.\d])" + _LONE_1_5, text)
    return int(ints[-1]) if ints else None


def build_prompt(template: str, source: str, reference: str, prediction: str) -> str:
    """Fill the frozen template. Inserted texts are arguments, so stray braces in
    the source/reference/prediction are not re-interpreted as fields."""
    return template.format(source=source, reference=reference, prediction=prediction)


def score_condition(
    client,
    template: str,
    sources: list[str],
    preds: list[str],
    refs: list[str],
    *,
    cache_path: Path | None = None,
) -> list[int | None]:
    """Judge each segment; returns one score per segment, in order."""
    done = read_completed_jsonl(cache_path) if cache_path is not None else []
    for j, rec in enumerate(done):
        if rec.get("input") != sources[j]:
            raise ValueError(
                f"resume misalignment in {cache_path} at segment {j}: cached source differs "
                f"from the current outputs; delete the cache to re-judge"
            )
    scores: list[int | None] = [rec.get("score") for rec in done]
    start = len(scores)
    if start:
        print(f"  resuming judge: {start}/{len(sources)} already scored")

    f = cache_path.open("a", encoding="utf-8") if cache_path is not None else None
    try:
        for i in range(start, len(sources)):
            try:
                prompt = build_prompt(template, sources[i], refs[i], preds[i])
                score = parse_score(client.complete(_JUDGE_SYSTEM, prompt))
                error = None
            except Exception as e:  # a bad call must not lose the segments already judged
                score = None
                error = f"{type(e).__name__}: {e}"
                print(f"  segment {i + 1} judge call failed, recorded null: {error}")
            scores.append(score)
            if f is not None:
                rec = {"input": sources[i], "score": score}
                if error is not None:
                    rec["error"] = error
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
    finally:
        if f is not None:
            f.close()
    return scores


def _aggregate(scores: list[int | None]) -> tuple[float | None, float]:
    """(mean over parsed scores, coverage). Mean is None if nothing parsed."""
    parsed = [s for s in scores if s is not None]
    coverage = len(parsed) / len(scores) if scores else 0.0
    mean = sum(parsed) / len(parsed) if parsed else None
    return mean, coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation-time LLM-as-Judge (register Φ).")
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--split", default="val", help="output split tag (default: val)")
    parser.add_argument("--out_dir", default="outputs", help="inference output directory")
    parser.add_argument("--results_dir", default=str(_RESULTS_DIR))
    parser.add_argument("--config", required=True, help="YAML with a `judge:` generator block")
    parser.add_argument(
        "--tag",
        default=None,
        help="judge tag for a second/cross-family pass; routes artefacts to "
        "judge_<tag>_<split>.json (default: the config's `tag:`, else unsuffixed)",
    )
    parser.add_argument(
        "--allow_model_overwrite",
        action="store_true",
        help="permit writing over a results file scored by a different judge model",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="score only the first N segments per condition -- a DIAGNOSTIC pilot for "
        "measuring real per-call cost before committing to the full split. The cache "
        "resumes, so the full run continues from where the pilot stopped.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if "judge" not in cfg:
        raise ValueError(f"{args.config} has no `judge:` block")
    template_path = Path(cfg.get("template_file", _DEFAULT_TEMPLATE))
    template = template_path.read_text(encoding="utf-8")
    digest = template_digest(template)

    client = _LazyClient(cfg["judge"])
    judge_model = cfg["judge"].get("model", "unknown")
    tag = args.tag if args.tag is not None else cfg.get("tag")

    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    cache_dir = judge_segment_dir(results_dir, args.split, tag)
    out_path = judge_results_path(results_dir, args.split, tag)
    assert_results_identity(out_path, judge_model, allow_overwrite=args.allow_model_overwrite)
    print(f"judge {judge_model}  tag={tag or '(none)'}  template={template_path} [{digest}]")
    results: dict[str, dict] = {}
    for cond in args.conditions:
        path = condition_path(out_dir, cond, args.split)
        if not path.exists():
            print(f"skip {cond}: {path} not found")
            continue
        sources, preds, refs = load_condition(out_dir, cond, args.split)
        if args.limit is not None:
            sources, preds, refs = sources[: args.limit], preds[: args.limit], refs[: args.limit]
        print(f"Judging {len(preds)} segments for {cond} with {judge_model} ...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        assert_cache_identity(
            cache_dir, {"model": judge_model, "tag": tag, "template_sha256": digest}
        )
        scores = score_condition(
            client, template, sources, preds, refs, cache_path=cache_dir / f"{cond}.jsonl"
        )
        mean, coverage = _aggregate(scores)
        results[cond] = {
            "n": len(scores),
            "model": judge_model,
            "judge_tag": tag,
            "template_sha256": digest,
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
    usage = client.usage
    if usage is not None:
        print(f"Judge usage: {usage.summary()}")
        priced = judge_model in usage.pricing
        if not priced:
            print(
                f"  note: no pricing for {judge_model}, so cost_usd is a floor of 0, not the "
                f"actual spend; set `pricing: [in, out]` in the judge block to account it"
            )

        usage_path = results_dir / f"{judge_stem(args.split, tag)}_usage.json"
        session = usage.summary()
        prior = {}
        if usage_path.exists():
            prior = json.loads(usage_path.read_text(encoding="utf-8")).get("cumulative", {})
        cumulative = {k: round(prior.get(k, 0) + v, 6) for k, v in session.items()}
        usage_path.write_text(
            json.dumps(
                {
                    "model": judge_model,
                    "tag": tag,
                    "conditions": sorted(results),
                    "limit": args.limit,
                    "priced": priced,
                    "session": session,
                    "cumulative": cumulative,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {usage_path}  (cumulative ${cumulative['cost_usd']:.2f})")


if __name__ == "__main__":
    main()
