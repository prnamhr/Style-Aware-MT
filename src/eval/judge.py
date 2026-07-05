"""
Evaluation-time LLM-as-Judge for register fidelity (RQ1 primary style axis, Φ).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from src.eval._io import condition_path, load_condition

_RESULTS_DIR = Path("results")
_DEFAULT_TEMPLATE = Path("prompts/judge_eval.txt")
# The rubric lives in the (user) template; keep the system role minimal and fixed.
_JUDGE_SYSTEM = (
    "You are a careful, consistent evaluator of translation style. Follow the instructions exactly."
)

# Prefer an explicit "Score: N" verdict; N must be a lone integer 1-5 -- reject a
# further digit ("12") or a decimal ("3.5"), but allow a sentence-final "3.".
_LONE_1_5 = r"([1-5])(?!\d)(?!\.\d)"
_SCORE_RE = re.compile(r"score\s*[:=]?\s*" + _LONE_1_5, re.IGNORECASE)


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
    client, template: str, sources: list[str], preds: list[str], refs: list[str]
) -> list[int | None]:
    """Judge each segment; returns one score (or ``None``) per segment, in order."""
    scores: list[int | None] = []
    for s, p, r in zip(sources, preds, refs):
        resp = client.complete(_JUDGE_SYSTEM, build_prompt(template, s, r, p))
        scores.append(parse_score(resp))
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

    from src.infer.run import make_client

    client = make_client(cfg["judge"])
    judge_model = cfg["judge"].get("model", "unknown")

    out_dir = Path(args.out_dir)
    results: dict[str, dict] = {}
    for cond in args.conditions:
        path = condition_path(out_dir, cond, args.split)
        if not path.exists():
            print(f"skip {cond}: {path} not found")
            continue
        sources, preds, refs = load_condition(out_dir, cond, args.split)
        print(f"Judging {len(preds)} segments for {cond} with {judge_model} ...")
        scores = score_condition(client, template, sources, preds, refs)
        mean, coverage = _aggregate(scores)
        results[cond] = {
            "n": len(scores),
            "model": judge_model,
            "mean": mean,
            "coverage": round(coverage, 4),
            "segments": scores,
        }
        mean_str = f"{mean:.3f}" if mean is not None else "n/a"
        print(f"  {cond:<16} Φ {mean_str}  (coverage {coverage:.0%})")

    if not results:
        return
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"judge_{args.split}.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    usage = getattr(client, "usage", None)
    if usage is not None:
        print(f"Judge usage: {usage.summary()}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
