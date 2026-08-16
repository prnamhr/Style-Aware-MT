"""
Shared IO for the evaluation backbone.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def condition_path(out_dir: str | Path, condition: str, split: str) -> Path:
    return Path(out_dir) / f"{condition}_{split}.jsonl"


def merge_results(path: str | Path, new: dict[str, dict]) -> list[str]:
    path = Path(path)
    existing: dict = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{path} is not valid JSON, so prior results cannot be preserved; "
                f"move it aside to start fresh: {e}"
            ) from e
        if not isinstance(loaded, dict):
            raise ValueError(
                f"{path} holds {type(loaded).__name__}, expected a condition -> result map"
            )
        existing = loaded

    preserved = sorted(set(existing) - set(new))
    merged = {**existing, **new}

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(merged, indent=2) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    return preserved


def read_completed_jsonl(path: str | Path) -> list[dict]:
    """Return the complete JSONL records already written to path."""
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict] = []
    good_bytes = 0
    with path.open("rb") as f:
        for raw in f:
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                break  # torn tail from an interrupted write; nothing past here is trusted
            good_bytes += len(raw)
    if good_bytes != path.stat().st_size:
        with path.open("rb+") as f:
            f.truncate(good_bytes)
    return records


def load_condition(
    out_dir: str | Path, condition: str, split: str
) -> tuple[list[str], list[str], list[str]]:
    """Return (sources, predictions, references) for one condition, in file order."""
    path = condition_path(out_dir, condition, split)
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    sources = [r["input"] for r in rows]
    preds = [r.get("prediction", "") for r in rows]
    refs = [r["output"] for r in rows]
    return sources, preds, refs
