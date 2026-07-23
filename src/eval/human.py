"""
Human-judgment scaffold for the evaluation backbone.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.eval._io import load_condition

SHEET_HEADER = [
    "segment_id",
    "system",
    "source",
    "reference",
    "prediction",
    "adequacy_1to5",
    "style_1to5",
    "notes",
]


def _flat(text: str) -> str:
    """Collapse whitespace so a field is safe inside one TSV cell."""
    return " ".join(text.split())


def write_scoring_sheet(
    records: list[dict], results_dir: str | Path, split: str, *, filename_tag: str = ""
) -> tuple[Path, Path]:
    """Write a blind-scoring TSV and a segment-grouped Markdown digest."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{filename_tag}" if filename_tag else ""
    tsv_path = results_dir / f"human_eval{tag}_{split}.tsv"
    md_path = results_dir / f"human_eval{tag}_{split}.md"

    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(SHEET_HEADER) + "\n")
        for r in records:
            f.write(
                "\t".join(
                    [
                        str(r["segment_id"]),
                        str(r["system"]),
                        _flat(r["source"]),
                        _flat(r["reference"]),
                        _flat(r["prediction"]),
                        "",  # adequacy_1to5
                        "",  # style_1to5
                        "",  # notes
                    ]
                )
                + "\n"
            )

    # Markdown grouped by segment, preserving first-seen segment order.
    seen: dict = {}
    for r in records:
        seen.setdefault(r["segment_id"], r)
    n_segments = len(seen)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Human evaluation — {split} ({n_segments} segments)\n\n")
        f.write(
            f"Score each prediction in `{tsv_path.name}`: "
            "**adequacy** (1–5, meaning preserved) and **style** "
            "(1–5, Shoghi Effendi scriptural register). Leave `notes` for anything notable.\n\n"
        )
        for sid, head in seen.items():
            f.write(f"## Segment {sid}\n\n")
            f.write(f"- **Source:** {head['source']}\n")
            f.write(f"- **Reference:** {head['reference']}\n\n")
            for r in records:
                if r["segment_id"] == sid:
                    f.write(f"- `{r['system']}` — {r['prediction']}\n")
            f.write("\n")
    return tsv_path, md_path


def build_records(
    systems: list[str],
    data: dict[str, tuple[list[str], list[str], list[str]]],
    sample_ids: list[int],
    *,
    blind: bool,
    seed: int,
) -> tuple[list[dict], dict[str, str]]:
    """Assemble sampled (segment, system) records; optionally blind the labels."""
    ref_sources, _, ref_refs = data[systems[0]]

    key: dict[str, str] = {}
    label_of = {s: s for s in systems}
    if blind:
        letters = [f"System {chr(ord('A') + i)}" for i in range(len(systems))]
        shuffled = list(systems)
        random.Random(seed).shuffle(shuffled)  # randomize which system gets which letter
        label_of = {sys: letters[i] for i, sys in enumerate(shuffled)}
        key = {letters[i]: sys for i, sys in enumerate(shuffled)}

    records: list[dict] = []
    for sid in sample_ids:
        order = list(systems)
        if blind:
            # Shuffle presentation order per segment so position never encodes system.
            random.Random(seed + sid).shuffle(order)
        for sys in order:
            _, preds, _ = data[sys]
            records.append(
                {
                    "segment_id": sid,
                    "system": label_of[sys],
                    "source": ref_sources[sid],
                    "reference": ref_refs[sid],
                    "prediction": preds[sid],
                }
            )
    return records, key


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a human-judgment scoring sheet.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        required=True,
        help="condition tags to score, e.g. zeroshot knn_fewshot afsp_full",
    )
    parser.add_argument("--out_dir", default="outputs", help="where the prediction JSONLs live")
    parser.add_argument("--split", default="val", help="split tag to score (default: val)")
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="segments to sample for hand-scoring (0 = all; shared across systems)",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed for sampling and blinding")
    parser.add_argument(
        "--blind",
        action="store_true",
        help="hide system names behind random letters and shuffle order per segment",
    )
    parser.add_argument("--results_dir", default="results", help="where the sheet is written")
    parser.add_argument(
        "--tag", default="", help="optional filename tag, e.g. 'ladder' -> human_eval_ladder_val.*"
    )
    args = parser.parse_args()

    data: dict[str, tuple[list[str], list[str], list[str]]] = {}
    for cond in args.conditions:
        path = Path(args.out_dir) / f"{cond}_{args.split}.jsonl"
        if not path.exists():
            print(f"skip {cond}: {path} not found")
            continue
        data[cond] = load_condition(args.out_dir, cond, args.split)
    if not data:
        print("no conditions found; generate predictions first")
        return

    systems = list(data)
    lengths = {c: len(v[1]) for c, v in data.items()}
    n = min(lengths.values())
    if len(set(lengths.values())) > 1:
        print(f"warning: conditions differ in length {lengths}; using the first {n} segments")

    if args.sample and args.sample < n:
        sample_ids = sorted(random.Random(args.seed).sample(range(n), args.sample))
    else:
        sample_ids = list(range(n))

    records, blind_key = build_records(systems, data, sample_ids, blind=args.blind, seed=args.seed)
    tsv_path, md_path = write_scoring_sheet(
        records, args.results_dir, args.split, filename_tag=args.tag
    )
    print(f"Wrote human-eval sheet: {tsv_path}  and  {md_path}")
    print(f"  systems={systems}  segments={len(sample_ids)}  blind={args.blind}")

    if blind_key:
        tag = f"_{args.tag}" if args.tag else ""
        key_path = Path(args.results_dir) / f"human_eval{tag}_{args.split}_key.json"
        key_path.write_text(json.dumps(blind_key, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  blind key (do not open before scoring): {key_path}")


if __name__ == "__main__":
    main()
