"""
Evaluation-time LLM-as-Judge for register fidelity (RQ1 primary style axis, Φ).
"""

from __future__ import annotations

import argparse
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


def parse_score(text: str) -> int | None:
    """Extract the 1-5 verdict from a judge response.

    Prefers the last ``Score: N`` match; falls back to the last standalone 1-5
    integer. Returns ``None`` when nothing is parseable -- e.g. an empty string
    from a safety refusal -- so the segment is dropped rather than mis-scored.
    """
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
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if "judge" not in cfg:
        raise ValueError(f"{args.config} has no `judge:` block")
    template_path = Path(cfg.get("template_file", _DEFAULT_TEMPLATE))
    template = template_path.read_text(encoding="utf-8")

    client = _LazyClient(cfg["judge"])
    judge_model = cfg["judge"].get("model", "unknown")

    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    cache_dir = results_dir / f"judge_{args.split}_segments"
    results: dict[str, dict] = {}
    for cond in args.conditions:
        path = condition_path(out_dir, cond, args.split)
        if not path.exists():
            print(f"skip {cond}: {path} not found")
            continue
        sources, preds, refs = load_condition(out_dir, cond, args.split)
        print(f"Judging {len(preds)} segments for {cond} with {judge_model} ...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        scores = score_condition(
            client, template, sources, preds, refs, cache_path=cache_dir / f"{cond}.jsonl"
        )
        mean, coverage = _aggregate(scores)
        results[cond] = {
            "n": len(scores),
            "model": judge_model,
            "mean": mean,
            "coverage": round(coverage, 4),
            "sources": sources,
            "segments": scores,
        }
        mean_str = f"{mean:.3f}" if mean is not None else "n/a"
        print(f"  {cond:<16} Φ {mean_str}  (coverage {coverage:.0%})")

    if not results:
        return
    out_path = results_dir / f"judge_{args.split}.json"
    preserved = merge_results(out_path, results)
    if preserved:
        print(f"preserved {len(preserved)} condition(s) not scored here: {', '.join(preserved)}")
    usage = client.usage
    if usage is not None:
        print(f"Judge usage: {usage.summary()}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
