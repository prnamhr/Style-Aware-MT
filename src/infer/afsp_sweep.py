from __future__ import annotations

import argparse
import json
import os
import random as _random
import yaml

from src.eval._io import load_condition
from src.eval.quick import score as quick_score
from pathlib import Path
from src.eval.stylometrics import aggregate, distance_to_centroid, register_band_distance
from src.infer.run import (
    _load_configured_glossary,
    build_fewshot_user,
    make_client,
    order_exemplars,
)
from src.retrieval.afsp import AFSPRetriever, load_centroid
from src.retrieval.retrieve import RetrievalIndex
from src.infer.run import build_zeroshot_user, make_client
from src.retrieval.afsp import _resolve_direction


# Previous full grid (5 x 8 = 40 cells). Kept, labelled, for reproducing the
# earlier sweep with `--ks {PREVIOUS_KS} --lambdas {PREVIOUS_LAMBDAS}`.
PREVIOUS_KS = (1, 2, 4, 8, 16)
PREVIOUS_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)

# New usual default. lambda only reranks inside the top pool_mult*k margin pool,
DEFAULT_KS = (4, 8, 16)
DEFAULT_LAMBDAS = (0.0, 0.1, 0.25, 0.75, 1.0)

ZEROSHOT_TAG = "afsp_zeroshot"


def cell_tag(k: int, lam: float) -> str:
    """Filesystem-safe, collision-free tag for one (k, lambda) cell."""
    return f"afsp_k{k}_l{lam:g}"


def _sweep_dir(cfg: dict) -> Path:
    return Path(cfg["output"]["dir"]) / "sweep"


def generate_cells(cfg: dict, cells: list[tuple[int, float]], *, overwrite: bool = False) -> str:
    """Generate predictions for the given (k, lambda) cells with the local Qwen base.
    """
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
        style_objective=af.get("style_objective", "bandpass"),
        style_target_sigma=af.get("style_target_sigma", 1.0),
        style_register_direction=af.get("style_register_direction"),
    )

    sweep_dir = _sweep_dir(cfg)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    client = make_client(gen)

    for k, lam in cells:
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
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        failures = 0
        with tmp_path.open("w", encoding="utf-8") as f:
            for row, user in zip(rows, user_msgs):
                try:
                    prediction = client.complete(style_instruction, user)
                    error = None
                except Exception as e:  # one bad segment must not discard the cell
                    prediction = ""
                    error = f"{type(e).__name__}: {e}"
                    failures += 1
                record = {
                    "input": row["input"],
                    "output": row["output"],
                    "prediction": prediction,
                    "condition": tag,
                    "k": k,
                    "lambda_style": float(lam),
                    "model": gen["model"],
                    "metadata": row.get("metadata", {}),
                }
                if error is not None:
                    record["error"] = error
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)
        if failures:
            print(
                f"[{tag}] WARNING: {failures}/{len(rows)} segments failed and were recorded "
                f"with an empty prediction (see the `error` field); --overwrite to retry."
            )
    return split


def generate_zeroshot(cfg: dict, *, overwrite: bool = False) -> str:
    """Generate the zero-shot reference
    """

    gen = cfg["generator"]
    style_instruction = Path(cfg["prompt"]["style_instruction_file"]).read_text(encoding="utf-8")

    eval_file = Path(cfg["data"]["eval_file"])
    split = eval_file.stem
    if split == "test":
        raise ValueError("refusing to sweep on the sealed test split; sweep on val")
    with eval_file.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    limit = cfg["data"].get("limit")
    if limit:
        rows = rows[:limit]

    sweep_dir = _sweep_dir(cfg)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    out_path = sweep_dir / f"{ZEROSHOT_TAG}_{split}.jsonl"
    if out_path.exists() and not overwrite:
        print(f"skip {ZEROSHOT_TAG}: {out_path} exists (use --overwrite to regenerate)")
        return split

    client = make_client(gen)
    print(f"[{ZEROSHOT_TAG}] generating {len(rows)} zero-shot translations with {gen['model']} ...")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    failures = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            try:
                prediction = client.complete(style_instruction, build_zeroshot_user(row["input"]))
                error = None
            except Exception as e:  # one bad segment must not discard the cell
                prediction = ""
                error = f"{type(e).__name__}: {e}"
                failures += 1
            record = {
                "input": row["input"],
                "output": row["output"],
                "prediction": prediction,
                "condition": ZEROSHOT_TAG,
                "k": 0,
                "lambda_style": None,
                "model": gen["model"],
                "metadata": row.get("metadata", {}),
            }
            if error is not None:
                record["error"] = error
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)
    if failures:
        print(
            f"[{ZEROSHOT_TAG}] WARNING: {failures}/{len(rows)} segments failed and were recorded "
            f"with an empty prediction (see the `error` field); --overwrite to retry."
        )
    return split


def score_zeroshot(cfg: dict, split: str) -> dict | None:
    """Score the zero-shot anchor, if present. Returns an anchor-flagged row."""
    sweep_dir = _sweep_dir(cfg)
    path = sweep_dir / f"{ZEROSHOT_TAG}_{split}.jsonl"
    if not path.exists():
        print(f"skip {ZEROSHOT_TAG}: {path} not found")
        return None
    centroid_path = Path(cfg.get("afsp", {}).get("centroid_file", ""))
    centroid = (
        json.loads(centroid_path.read_text(encoding="utf-8")) if centroid_path.exists() else None
    )
    s = quick_score(ZEROSHOT_TAG, sweep_dir, split)
    _, preds, _ = load_condition(sweep_dir, ZEROSHOT_TAG, split)
    row = {
        "tag": ZEROSHOT_TAG,
        "k": 0,
        "lambda": None,
        "anchor": True,
        "n": s["n"],
        "chrF": s["chrF"],
        "BLEU": s["BLEU"],
        "marker_rate": s["marker_rate"],
    }
    if centroid is not None:
        agg_mean = aggregate(preds)["mean"]
        row["stylo_dist"] = round(distance_to_centroid(agg_mean, centroid), 4)
        register_fit = _register_fit_fn(cfg, centroid)
        row["register_fit"] = register_fit(agg_mean)
    return row


def generate_grid(
    cfg: dict, ks: list[int], lambdas: list[float], *, overwrite: bool = False
) -> str:
    """Generate predictions for every (k, lambda) cell of the full grid."""
    cells = [(k, lam) for k in ks for lam in lambdas]
    return generate_cells(cfg, cells, overwrite=overwrite)


def _register_fit_fn(cfg: dict, centroid: dict | None):
    """Build the selector's register-fidelity metric: direction-weighted band
    """
    if centroid is None:
        return None

    af = cfg.get("afsp", {})
    direction = _resolve_direction(af.get("style_register_direction"), centroid)
    target_sigma = float(af.get("select_target_sigma", 0.5))
    return lambda agg_mean: round(
        register_band_distance(agg_mean, centroid, target_sigma, direction), 4
    )


def score_grid(cfg: dict, ks: list[int], lambdas: list[float], split: str) -> list[dict]:
    """Score every present cell with free/local metrics only."""
    sweep_dir = _sweep_dir(cfg)
    centroid_path = Path(cfg.get("afsp", {}).get("centroid_file", ""))
    centroid = (
        json.loads(centroid_path.read_text(encoding="utf-8")) if centroid_path.exists() else None
    )
    register_fit = _register_fit_fn(cfg, centroid)

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
                # stylo_dist retained undirected for reporting; register_fit decides.
                row["stylo_dist"] = round(distance_to_centroid(agg["mean"], centroid), 4)
                row["register_fit"] = register_fit(agg["mean"])
            rows.append(row)
    return rows


def recommend(rows: list[dict], adequacy_margin: float) -> dict | None:
    """Recommend a (k, lambda) cell: best register fidelity within an adequacy band."""
    rows = [r for r in rows if not r.get("anchor")]
    if not rows:
        return None
    if not all("register_fit" in r for r in rows):
        return max(rows, key=lambda r: (r["chrF"], -r["k"]))
    best_chrf = max(r["chrF"] for r in rows)
    band = [r for r in rows if r["chrF"] >= best_chrf - adequacy_margin]
    return min(band, key=lambda r: (r["register_fit"], -r["chrF"], r["k"]))


def ranked_cells(rows: list[dict], adequacy_margin: float) -> list[dict]:
    """Cells ordered best-first by the same rule ``recommend`` uses to pick one.
    """
    rows = [r for r in rows if not r.get("anchor")]
    if not rows:
        return []
    if not all("register_fit" in r for r in rows):
        return sorted(rows, key=lambda r: (-r["chrF"], r["k"]))
    best_chrf = max(r["chrF"] for r in rows)

    def fidelity(r: dict) -> tuple:
        return (r["register_fit"], -r["chrF"], r["k"])

    band = sorted((r for r in rows if r["chrF"] >= best_chrf - adequacy_margin), key=fidelity)
    out_band = sorted((r for r in rows if r["chrF"] < best_chrf - adequacy_margin), key=fidelity)
    return band + out_band


def _print_table(rows: list[dict], pick: dict | None) -> None:
    cols = ["tag", "k", "lambda", "n", "chrF", "BLEU", "marker_rate"]
    have_fit = bool(rows) and "register_fit" in rows[0]
    if rows and "stylo_dist" in rows[0]:
        cols.append("stylo_dist")
    if have_fit:
        cols.append("register_fit")
    # Best register fidelity first (lower register_fit), else best chrF first.
    key = (lambda r: r["register_fit"]) if have_fit else (lambda r: -r["chrF"])
    ordered = sorted(rows, key=key)
    widths = {c: max(len(c), *(len(str(r[c])) for r in ordered)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols) + "  <-"
    print(header)
    print("-" * len(header))
    for r in ordered:
        if r.get("anchor"):
            mark = "  (zero-shot anchor)"
        elif pick and r["tag"] == pick["tag"]:
            mark = "  <== recommended"
        else:
            mark = ""
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
        "--no-zeroshot",
        action="store_true",
        help="skip the zero-shot reference anchor (x=0)",
    )
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
        if not args.no_zeroshot:
            generate_zeroshot(cfg, overwrite=args.overwrite)

    rows = score_grid(cfg, args.ks, args.lambdas, split)
    if not rows:
        print("no scored cells; generate the grid first (drop --score-only)")
        return
    if not args.no_zeroshot:
        anchor = score_zeroshot(cfg, split)
        if anchor is not None:
            rows.append(anchor)

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
                "select_target_sigma": float(cfg.get("afsp", {}).get("select_target_sigma", 0.5)),
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
            + (f", register_fit {pick['register_fit']}" if "register_fit" in pick else "")
            + (f", stylo_dist {pick['stylo_dist']}" if "stylo_dist" in pick else "")
            + ")"
        )
        print("Freeze before touching test.jsonl -- set in configs/base_qwen.yaml:")
        print(f"    retrieval.k       : {pick['k']}")
        print(f"    afsp.lambda_style : {pick['lambda']}")


if __name__ == "__main__":
    main()
