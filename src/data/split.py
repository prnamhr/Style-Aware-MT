import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

ARABIC_DIACRITICS_RE = re.compile(r"[ً-ٰٟۖ-ۭ]")
PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
WS_RE = re.compile(r"\s+")

UNKNOWN_SOURCE_KEYS = {"", None}


def norm_key(text: str, is_source: bool) -> str:
    """
    Aggressive normalization used ONLY for cross-boundary duplicate matching.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    if is_source:
        text = ARABIC_DIACRITICS_RE.sub("", text)
    text = text.casefold()
    text = PUNCT_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def assign_works(work_sizes: dict, fracs: dict, forced_train: set) -> dict:
    """
    Greedy multiway partition: keep whole works intact, hit `fracs` closely.
    """
    total = sum(work_sizes.values())
    targets = {k: total * f for k, f in fracs.items()}
    assigned = {k: [] for k in fracs}
    filled = {k: 0 for k in fracs}

    # Forced works go to train up front and count against its target.
    for work in sorted(forced_train, key=lambda w: (-work_sizes[w], w)):
        assigned["train"].append(work)
        filled["train"] += work_sizes[work]

    rest = sorted(
        (w for w in work_sizes if w not in forced_train),
        key=lambda w: (-work_sizes[w], w),
    )
    for work in rest:
        # Split with the largest remaining deficit wins; name breaks ties.
        split = min(fracs, key=lambda k: (filled[k] - targets[k], k))
        assigned[split].append(work)
        filled[split] += work_sizes[work]

    return assigned


def dedup_against_seen(records: list, seen_inputs: set, seen_outputs: set) -> tuple:
    """
    Drop records whose normalized input OR output is already in `seen`.
    """
    kept, dropped = [], []
    for rec in records:
        ik = norm_key(rec["input"], is_source=True)
        ok = norm_key(rec["output"], is_source=False)
        if ik in seen_inputs or ok in seen_outputs:
            dropped.append(rec)
            continue
        seen_inputs.add(ik)
        seen_outputs.add(ok)
        kept.append(rec)
    return kept, dropped


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, records: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split cleaned data into train/val/test at the WORK level (whole "
            "books never span a boundary), then de-duplicate across the "
            "train/val/test boundary to prevent retrieval leakage."
        )
    )
    parser.add_argument("--input_file", type=str, default="data/processed/sentences_cleaned.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/splits")
    parser.add_argument(
        "--group_key", type=str, default="source", help="metadata field to group on"
    )
    parser.add_argument("--train_frac", type=float, default=0.80)
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--test_frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fracs = {"train": args.train_frac, "val": args.val_frac, "test": args.test_frac}
    if abs(sum(fracs.values()) - 1.0) > 1e-6:
        parser.error(f"fractions must sum to 1.0, got {sum(fracs.values())}")

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_file}...")
    df = pd.read_json(input_file, lines=True)
    records = df.to_dict(orient="records")
    print(f"  {len(records)} records loaded.")

    # --- 1. Group by work and bin-pack whole works into splits ---
    work_sizes: dict = {}
    work_records: dict = {}
    forced_train = set()
    for rec in records:
        work = rec.get("metadata", {}).get(args.group_key)
        key = work if work not in UNKNOWN_SOURCE_KEYS else "<unknown>"
        if work in UNKNOWN_SOURCE_KEYS:
            forced_train.add(key)
        work_sizes[key] = work_sizes.get(key, 0) + 1
        work_records.setdefault(key, []).append(rec)

    print(f"  {len(work_sizes)} distinct works.")
    assignment = assign_works(work_sizes, fracs, forced_train)

    split_records = {k: [] for k in fracs}
    work_to_split = {}
    for split, works in assignment.items():
        for work in works:
            split_records[split].extend(work_records[work])
            work_to_split[work] = split

    pre_counts = {k: len(v) for k, v in split_records.items()}

    # --- 2. Cross-boundary de-duplication (train kept intact, priority order) ---
    seen_inputs, seen_outputs = set(), set()
    # Seed the seen sets with the ENTIRE train partition; train is never trimmed.
    for rec in split_records["train"]:
        seen_inputs.add(norm_key(rec["input"], is_source=True))
        seen_outputs.add(norm_key(rec["output"], is_source=False))

    split_records["val"], dropped_val = dedup_against_seen(
        split_records["val"], seen_inputs, seen_outputs
    )
    split_records["test"], dropped_test = dedup_against_seen(
        split_records["test"], seen_inputs, seen_outputs
    )

    # --- 3. Write splits ---
    print(f"Saving splits to {output_dir}...")
    paths = {}
    for split in ("train", "val", "test"):
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, split_records[split])
        paths[split] = path

    # --- 4. Leakage audit: assert zero cross-boundary key overlap remains ---
    def keyset(recs, is_source):
        field = "input" if is_source else "output"
        return {norm_key(r[field], is_source) for r in recs}

    train_in = keyset(split_records["train"], True)
    train_out = keyset(split_records["train"], False)
    leaks = 0
    for split in ("val", "test"):
        leaks += len(keyset(split_records[split], True) & train_in)
        leaks += len(keyset(split_records[split], False) & train_out)
    assert leaks == 0, f"Leakage audit FAILED: {leaks} overlapping keys with train"

    # --- 5. Manifest: hashes + full split provenance for reproducibility ---
    final_counts = {k: len(v) for k, v in split_records.items()}
    total_final = sum(final_counts.values())
    manifest = {
        "config": {
            "input_file": str(input_file),
            "group_key": args.group_key,
            "fracs": fracs,
            "seed": args.seed,
            "split_method": "whole-work bin-pack + cross-boundary normalized-exact dedup",
            "dedup_rule": "drop val/test record if normalized input OR output appears in train",
        },
        "counts": {
            "loaded": len(records),
            "pre_dedup": pre_counts,
            "final": final_counts,
            "dropped_dedup": {"val": len(dropped_val), "test": len(dropped_test)},
            "final_ratios": {k: round(v / total_final, 4) for k, v in final_counts.items()},
        },
        "work_to_split": work_to_split,
        "hashes": {f"{s}.jsonl": sha256_file(p) for s, p in paths.items()},
    }
    (output_dir / "hashes.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Done!")
    for split in ("train", "val", "test"):
        r = final_counts[split] / total_final
        print(
            f"  {split:5s}: {final_counts[split]:6d}  ({r:.1%})  [{len(assignment[split])} works]"
        )
    print(f"  dropped by cross-boundary dedup: val={len(dropped_val)} test={len(dropped_test)}")
    print("  leakage audit: PASS (0 overlapping normalized keys with train)")


if __name__ == "__main__":
    main()
