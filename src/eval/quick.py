"""Quick comparison of inference outputs: BLEU, chrF, and an archaic-register proxy.

This is the smoke-test scorer, not the final evaluation harness (COMET / paired
bootstrap / LLM-as-Judge come later). Its purpose is to confirm the loop runs and
to give an early read on whether the kNN-few-shot baseline shifts the register
relative to the reference condition, per RQ2.

The register proxy counts second-person sacred forms and vocatives ("thou",
"thee", "thy", "art", "hast", "doth", "O ...") per segment -- a crude but
direction-correct stand-in for the stylometric features.

Usage:
    python -m src.eval.quick --conditions reference knn_fewshot --split val
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

_MARKERS = re.compile(
    r"\b(thou|thee|thy|thine|art|hast|hath|dost|doth|shalt|wilt|unto|ye)\b"
    r"|\bO\b",
)


def _marker_rate(texts: list[str]) -> float:
    """Mean archaic-marker hits per segment."""
    if not texts:
        return 0.0
    return sum(len(_MARKERS.findall(t)) for t in texts) / len(texts)


def score(condition: str, out_dir: Path, split: str) -> dict:
    path = out_dir / f"{condition}_{split}.jsonl"
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    preds = [r["prediction"] for r in rows]
    refs = [r["output"] for r in rows]

    bleu = BLEU().corpus_score(preds, [refs]).score
    chrf = CHRF().corpus_score(preds, [refs]).score
    return {
        "condition": condition,
        "n": len(rows),
        "BLEU": round(bleu, 2),
        "chrF": round(chrf, 2),
        "marker_rate": round(_marker_rate(preds), 2),
        "ref_marker_rate": round(_marker_rate(refs), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick BLEU/chrF/register comparison.")
    parser.add_argument("--conditions", nargs="+", default=["reference", "knn_fewshot"])
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
