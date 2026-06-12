"""
Quick comparison of inference outputs: BLEU, chrF, and an archaic-register proxy.

Usage:
    python -m src.eval.quick --conditions reference knn_fewshot --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

from src.eval.stylometrics import _MARKERS


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
