"""
Quick comparison of inference outputs: BLEU, chrF, and an archaic-register proxy.

Usage:
    python -m src.eval.quick --conditions zeroshot knn_fewshot afsp_full --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

from src.eval._io import load_condition
from src.eval.stylometrics import _MARKERS


def _marker_rate(texts: list[str]) -> float:
    """Mean archaic-marker hits per segment."""
    if not texts:
        return 0.0
    return sum(len(_MARKERS.findall(t)) for t in texts) / len(texts)


def segment_scores(preds: list[str], refs: list[str], metric: str = "chrf") -> list[float]:
    """Per-segment sentence-level chrF (default) or BLEU, for the paired bootstrap.

    Sentence-level BLEU uses ``effective_order`` so short segments are not driven
    to zero by missing higher-order n-grams. This is the reusable surface-metric
    entry point for any ``<condition>_<split>.jsonl`` file; the corpus-level
    ``score`` below uses the same sacrebleu metrics.
    """
    metric = metric.lower()
    if metric == "chrf":
        m = CHRF()
        return [m.sentence_score(p, [r]).score for p, r in zip(preds, refs)]
    if metric == "bleu":
        m = BLEU(effective_order=True)
        return [m.sentence_score(p, [r]).score for p, r in zip(preds, refs)]
    raise ValueError(f"unknown metric '{metric}' (expected chrf|bleu)")


def score(condition: str, out_dir: Path, split: str) -> dict:
    _, preds, refs = load_condition(out_dir, condition, split)

    bleu = BLEU().corpus_score(preds, [refs]).score
    chrf = CHRF().corpus_score(preds, [refs]).score
    return {
        "condition": condition,
        "n": len(preds),
        "BLEU": round(bleu, 2),
        "chrF": round(chrf, 2),
        "marker_rate": round(_marker_rate(preds), 2),
        "ref_marker_rate": round(_marker_rate(refs), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick BLEU/chrF/register comparison.")
    parser.add_argument("--conditions", nargs="+", default=["zeroshot", "knn_fewshot"])
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--split", default="val", help="output split tag to score (default: val)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results = []
    for cond in args.conditions:
        path = out_dir / f"{cond}_{args.split}.jsonl"
        if not path.exists():
            print(f"skip {cond}: {path} not found")
            continue
        results.append(score(cond, out_dir, args.split))

    if not results:
        return
    cols = ["condition", "n", "BLEU", "chrF", "marker_rate", "ref_marker_rate"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in results)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in results:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()
