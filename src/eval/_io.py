"""
Shared IO for the evaluation backbone.
"""

from __future__ import annotations

import json
from pathlib import Path


def condition_path(out_dir: str | Path, condition: str, split: str) -> Path:
    return Path(out_dir) / f"{condition}_{split}.jsonl"


def load_condition(
    out_dir: str | Path, condition: str, split: str
) -> tuple[list[str], list[str], list[str]]:
    """Return ``(sources, predictions, references)`` for one condition, in file order."""
    path = condition_path(out_dir, condition, split)
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    sources = [r["input"] for r in rows]
    preds = [r.get("prediction", "") for r in rows]
    refs = [r["output"] for r in rows]
    return sources, preds, refs
