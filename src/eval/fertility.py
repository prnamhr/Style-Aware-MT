"""
Tokenizer fertility on the Persian/Arabic source side.

Usage:
    python manage.py fertility --model Qwen/Qwen2.5-7B-Instruct
    python manage.py fertility --model Qwen/Qwen2.5-7B-Instruct --split all
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

_SPLIT_DIR = Path("data/splits")


def _load_sources(split: str) -> list[str]:
    """
    Return the source-side strings for the requested split.
    """
    if split == "all":
        files = sorted(_SPLIT_DIR.glob("*.jsonl"))
    else:
        files = [_SPLIT_DIR / f"{split}.jsonl"]

    sources: list[str] = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"no such split file: {path}")
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                src = json.loads(line).get("input", "").strip()
                if src:
                    sources.append(src)
    return sources


def measure(model: str, sources: list[str]) -> dict:
    """
    Compute corpus-level and per-sentence fertility for model.
    """
    tok = AutoTokenizer.from_pretrained(model)

    total_words = 0
    total_tokens = 0
    per_sentence: list[float] = []

    for src in sources:
        n_words = len(src.split())
        if n_words == 0:
            continue
        # No special tokens: we count only the content the model encodes.
        n_tokens = len(tok.encode(src, add_special_tokens=False))
        total_words += n_words
        total_tokens += n_tokens
        per_sentence.append(n_tokens / n_words)

    per_sentence.sort()
    return {
        "model": model,
        "sentences": len(per_sentence),
        "total_words": total_words,
        "total_tokens": total_tokens,
        "fertility_corpus": total_tokens / total_words if total_words else 0.0,
        "fertility_mean_per_sentence": statistics.fmean(per_sentence) if per_sentence else 0.0,
        "fertility_median": statistics.median(per_sentence) if per_sentence else 0.0,
        "fertility_p90": per_sentence[int(0.90 * (len(per_sentence) - 1))] if per_sentence else 0.0,
        "fertility_max": per_sentence[-1] if per_sentence else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenizer fertility on the source side.")
    parser.add_argument(
        "--model", required=True, help="HF tokenizer id, e.g. Qwen/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test", "all"],
        help="which split's source side to measure (default: train)",
    )
    args = parser.parse_args()

    sources = _load_sources(args.split)
    stats = measure(args.model, sources)

    width = max(len(k) for k in stats)
    print(f"\nFertility report  (split={args.split})")
    print("-" * (width + 24))
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<{width}}  {v:.4f}")
        else:
            print(f"  {k:<{width}}  {v}")

    fert = stats["fertility_corpus"]
    if fert < 2.5:
        verdict = "LOCK IT — fertility under 2.5 tok/word, tokenizer handles the source well."
    elif fert < 4.0:
        verdict = "BORDERLINE — between 2.5 and 4; weigh against alternatives."
    else:
        verdict = "CATASTROPHIC — 4+ tok/word; do not switch to this base."
    print(f"\n  -> {verdict}\n")


if __name__ == "__main__":
    main()
