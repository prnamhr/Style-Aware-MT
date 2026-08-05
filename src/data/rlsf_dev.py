"""
Carve the RLSF dev slice out of train.jsonl at work level.

Usage:

    python manage.py rlsf_dev
    python manage.py rlsf_dev --target 500 --seed 42
    python manage.py rlsf_dev --exclude Ishraqat --dry_run
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

_SPLITS_DIR = Path("data/splits")
_UNKNOWN = ""


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def work_sizes(records: list[dict], group_key: str = "source") -> dict[str, int]:
    """Segment count per work, in the order works first appear."""
    counts: Counter[str] = Counter()
    for rec in records:
        counts[rec.get("metadata", {}).get(group_key) or _UNKNOWN] += 1
    return dict(counts)


_EXHAUSTIVE_LIMIT = 20


def select_works(
    sizes: dict[str, int],
    target: int,
    *,
    exclude: set[str] | None = None,
) -> list[str]:
    """Choose whole works whose segment count is closest to ``target``.
    """
    if target < 1:
        raise ValueError(f"target must be >= 1, got {target}")
    blocked = {_UNKNOWN} | (exclude or set())
    available = sorted(
        ((w, n) for w, n in sizes.items() if w not in blocked),
        key=lambda item: (-item[1], item[0]),
    )
    if len(available) < 2:
        raise ValueError(
            f"need at least 2 labelled works to carve a dev slice and leave a remainder, "
            f"found {len(available)}"
        )

    names = [w for w, _ in available]
    counts = [n for _, n in available]

    if len(available) <= _EXHAUSTIVE_LIMIT:
        best: tuple[int, int, list[str]] | None = None
        # Exclude the empty set and the full set: both leave one side unusable.
        for mask in range(1, (1 << len(names)) - 1):
            total = sum(counts[i] for i in range(len(names)) if mask >> i & 1)
            picked = sorted(names[i] for i in range(len(names)) if mask >> i & 1)
            key = (abs(total - target), len(picked), picked)
            if best is None or key < best:
                best = key
        assert best is not None
        return best[2]

    chosen: list[str] = []
    total = 0
    for work, n in available:
        if total + n <= target:
            chosen.append(work)
            total += n
    if not chosen:
        chosen = [min(available, key=lambda item: (item[1], item[0]))[0]]
    if len(chosen) == len(available):
        chosen = chosen[:-1]
    return sorted(chosen)


def partition(
    records: list[dict],
    dev_works: set[str],
    group_key: str = "source",
) -> tuple[list[dict], list[dict]]:
    """Split records into (dev, train_remainder), preserving input order."""
    dev, rest = [], []
    for rec in records:
        work = rec.get("metadata", {}).get(group_key) or _UNKNOWN
        (dev if work in dev_works else rest).append(rec)
    return dev, rest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Carve a work-level RLSF dev slice out of train.jsonl."
    )
    parser.add_argument("--splits_dir", default=str(_SPLITS_DIR))
    parser.add_argument("--target", type=int, default=500, help="approximate dev segments")
    parser.add_argument("--group_key", default="source", help="metadata field naming the work")
    parser.add_argument(
        "--exclude", nargs="*", default=[], help="works that must stay in training"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="recorded for provenance; selection is deterministic"
    )
    parser.add_argument("--dry_run", action="store_true", help="report the plan, write nothing")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    train_path = splits_dir / "train.jsonl"
    if not train_path.exists():
        raise SystemExit(f"missing {train_path}; run `python manage.py split` first")

    records = load_jsonl(train_path)
    sizes = work_sizes(records, args.group_key)
    dev_works = select_works(sizes, args.target, exclude=set(args.exclude))
    dev, rest = partition(records, set(dev_works), args.group_key)

    print(f"train.jsonl: {len(records)} segments across {len(sizes)} works")
    print(f"dev slice:   {len(dev)} segments, target {args.target}, {len(dev_works)} works")
    for work in dev_works:
        print(f"  {sizes[work]:5d}  {work}")
    print(f"remainder:   {len(rest)} segments")
    if _UNKNOWN in sizes:
        print(f"  ({sizes[_UNKNOWN]} unlabelled segments forced to the remainder)")

    if args.dry_run:
        print("dry run: nothing written")
        return

    dev_path = splits_dir / "rlsf_dev.jsonl"
    rest_path = splits_dir / "rlsf_train.jsonl"
    write_jsonl(dev_path, dev)
    write_jsonl(rest_path, rest)

    manifest = {
        "config": {
            "source_file": str(train_path),
            "group_key": args.group_key,
            "target": args.target,
            "seed": args.seed,
            "exclude": sorted(args.exclude),
            "selection": (
                "whole works, subset minimizing |total - target|; ties toward fewer works "
                "then lexicographic; unlabelled forced to remainder"
            ),
        },
        "counts": {
            "train_input": len(records),
            "rlsf_dev": len(dev),
            "rlsf_train": len(rest),
            "dev_works": len(dev_works),
        },
        "dev_works": {work: sizes[work] for work in dev_works},
        "work_sizes": sizes,
        "hashes": {
            "train.jsonl": sha256_file(train_path),
            "rlsf_dev.jsonl": sha256_file(dev_path),
            "rlsf_train.jsonl": sha256_file(rest_path),
        },
        "caveats": [
            "Held out from PPO updates only. The PEFT checkpoint RLSF initializes from "
            "was trained on all of train.jsonl, this slice included, so the slice is not "
            "unseen by the model.",
            "results/stylometrics_centroid.json was built over all 10,860 train targets, "
            "including these works.",
            "Dev-slice figures select the reward weights and are never reported as a "
            "result; val remains the reported split.",
        ],
    }
    manifest_path = splits_dir / "rlsf_dev_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dev_path}, {rest_path}, {manifest_path}")


if __name__ == "__main__":
    main()
