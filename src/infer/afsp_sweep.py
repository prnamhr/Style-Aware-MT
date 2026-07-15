from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from src.eval._io import load_condition
from src.eval.quick import score as quick_score
from src.eval.stylometrics import aggregate, distance_to_centroid

DEFAULT_KS = (1, 2, 3, 4)
DEFAULT_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def cell_tag(k: int, lam: float) -> str:
    """Filesystem-safe, collision-free tag for one (k, lambda) cell."""
    return f"afsp_k{k}_l{lam:g}"


def _sweep_dir(cfg: dict) -> Path:
    return Path(cfg["output"]["dir"]) / "sweep"


def generate_grid(
    cfg: dict, ks: list[int], lambdas: list[float], *, overwrite: bool = False
) -> str:
    """Generate predictions for every (k, lambda) cell with the local Qwen base.
    """
    import random as _random

    from src.infer.run import (
        _load_configured_glossary,
        build_fewshot_user,
        make_client,
        order_exemplars,
    )
    from src.retrieval.afsp import AFSPRetriever, load_centroid
    from src.retrieval.retrieve import RetrievalIndex

    gen = cfg["generator"]
    prompt_cfg = cfg.get("prompt", {})
    style_instruction = Path(prompt_cfg["style_instruction_file"]).read_text(encoding="utf-8")
    ordering = prompt_cfg.get("ordering", "most_similar_last")
    rng = _random.Random(prompt_cfg.get("ordering_seed", 42))
    glossary = _load_configured_glossary(cfg)

    eval_file = Path(cfg["data"]["eval_file"])
    split = eval_file.stem
    if split == "test":
        raise ValueError("refusing to sweep on the sealed test split; sweep on val")
    with eval_file.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    limit = cfg["data"].get("limit")
    if limit:
        rows = rows[:limit]
    sources = [r["input"] for r in rows]

    retr = cfg["retrieval"]
    af = cfg.get("afsp", {})
    index = RetrievalIndex(retr["index_dir"], embed_model=retr["embed_model"])
    retriever = AFSPRetriever(
        index,
        load_centroid(af["centroid_file"]),  # loaded so lambda>0 cells can rerank
        index_dir=retr["index_dir"],
        beta=af.get("beta", 0.3),
        knn_hubness=af.get("knn_hubness", 5),
        pool_mult=af.get("pool_mult", 4),
        lambda_style=0.0,
    )

    sweep_dir = _sweep_dir(cfg)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    client = make_client(gen)

    for k in ks:
        for lam in lambdas:
            tag = cell_tag(k, lam)
            out_path = sweep_dir / f"{tag}_{split}.jsonl"
            if out_path.exists() and not overwrite:
                print(f"skip {tag}: {out_path} exists (use --overwrite to regenerate)")
                continue
            retriever.lambda_style = float(lam)  # 0 -> margin only; >0 -> full rerank
            print(f"[{tag}] selecting k={k} (lambda={lam}) for {len(sources)} sources ...")
            selected = retriever.select(sources, k=k)
            ordered = [order_exemplars(ex, ordering, rng) for ex in selected]
            user_msgs = [build_fewshot_user(s, ex, glossary) for s, ex in zip(sources, ordered)]

            print(f"[{tag}] generating {len(rows)} translations with {gen['model']} ...")
            # Write to a temp file and atomically rename on completion so an
            # interruption mid-cell (e.g. a dead Colab session) never leaves a
            # partial file that resume would skip as "done".
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                for row, user in zip(rows, user_msgs):
                    prediction = client.complete(style_instruction, user)
                    f.write(
                        json.dumps(
                            {
                                "input": row["input"],
                                "output": row["output"],
                                "prediction": prediction,
                                "condition": tag,
                                "k": k,
                                "lambda_style": float(lam),
                                "model": gen["model"],
                                "metadata": row.get("metadata", {}),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, out_path)
    return split


def score_grid(cfg: dict, ks: list[int], lambdas: list[float], split: str) -> list[dict]:
    """Score every present cell with free/local metrics only.
    """
    sweep_dir = _sweep_dir(cfg)
    centroid_path = Path(cfg.get("afsp", {}).get("centroid_file", ""))
    centroid = (
        json.loads(centroid_path.read_text(encoding="utf-8")) if centroid_path.exists() else None
    )

    rows: list[dict] = []
    for k in ks:
        for lam in lambdas:
            tag = cell_tag(k, lam)
            path = sweep_dir / f"{tag}_{split}.jsonl"
            if not path.exists():
                print(f"skip {tag}: {path} not found")
                continue
            s = quick_score(tag, sweep_dir, split)
            _, preds, _ = load_condition(sweep_dir, tag, split)
            agg = aggregate(preds)
            row = {
                "tag": tag,
                "k": k,
                "lambda": lam,
                "n": s["n"],
                "chrF": s["chrF"],
                "BLEU": s["BLEU"],
                "marker_rate": s["marker_rate"],
            }
            if centroid is not None:
                row["stylo_dist"] = round(distance_to_centroid(agg["mean"], centroid), 4)
            rows.append(row)
    return rows


def recommend(rows: list[dict], adequacy_margin: float) -> dict | None:
    """Recommend a (k, lambda) cell: best register fidelity within an adequacy band.
    """
    if not rows:
        return None
    if not all("stylo_dist" in r for r in rows):
        return max(rows, key=lambda r: (r["chrF"], -r["k"]))
    best_chrf = max(r["chrF"] for r in rows)
    band = [r for r in rows if r["chrF"] >= best_chrf - adequacy_margin]
    return min(band, key=lambda r: (r["stylo_dist"], -r["chrF"], r["k"]))


def _print_table(rows: list[dict], pick: dict | None) -> None:
    cols = ["tag", "k", "lambda", "n", "chrF", "BLEU", "marker_rate"]
    if rows and "stylo_dist" in rows[0]:
        cols.append("stylo_dist")
    # Best register fidelity first (lower stylo_dist), else best chrF first.
    key = (lambda r: r["stylo_dist"]) if "stylo_dist" in cols else (lambda r: -r["chrF"])
    ordered = sorted(rows, key=key)
    widths = {c: max(len(c), *(len(str(r[c])) for r in ordered)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols) + "  <-"
    print(header)
    print("-" * len(header))
    for r in ordered:
        mark = "  <== recommended" if pick and r["tag"] == pick["tag"] else ""
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols) + mark)


def main() -> None:
    parser = argparse.ArgumentParser(description="AFSP k x lambda_style sweep on val (Qwen).")
    parser.add_argument("--config", default="configs/afsp_sweep.yaml")
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--lambdas", nargs="+", type=float, default=list(DEFAULT_LAMBDAS))
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="skip generation; only (re)score and select over existing cells",
    )
    parser.add_argument("--overwrite", action="store_true", help="regenerate existing cells")
    parser.add_argument(
        "--adequacy-margin",
        type=float,
        default=1.0,
        help="chrF band below the best within which register fidelity decides (default 1.0)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    if args.score_only:
        split = Path(cfg["data"]["eval_file"]).stem
    else:
        split = generate_grid(cfg, args.ks, args.lambdas, overwrite=args.overwrite)

    rows = score_grid(cfg, args.ks, args.lambdas, split)
    if not rows:
        print("no scored cells; generate the grid first (drop --score-only)")
        return

    pick = recommend(rows, args.adequacy_margin)
    print(f"\nAFSP sweep ({split}, {len(rows)} cells)  -- register fidelity first")
    _print_table(rows, pick)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"afsp_sweep_{split}.json"
    out_path.write_text(
        json.dumps(
            {
                "split": split,
                "adequacy_margin": args.adequacy_margin,
                "cells": rows,
                "recommended": pick,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")
    if pick:
        print(
            f"\nRecommended: k={pick['k']}, lambda_style={pick['lambda']}  "
            f"(chrF {pick['chrF']}"
            + (f", stylo_dist {pick['stylo_dist']}" if "stylo_dist" in pick else "")
            + ")"
        )
        print("Freeze before touching test.jsonl -- set in configs/base_qwen.yaml:")
        print(f"    retrieval.k       : {pick['k']}")
        print(f"    afsp.lambda_style : {pick['lambda']}")


if __name__ == "__main__":
    main()
