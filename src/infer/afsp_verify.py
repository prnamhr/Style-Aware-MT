"""
Confirm the AFSP sweep's proxy-picked cells on the full val split with the real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.eval._io import load_condition
from src.infer.afsp_sweep import _sweep_dir, generate_cells, ranked_cells


def _run_judge(judge_config: str, top: list[dict], sweep_dir: Path, split: str) -> dict[str, dict]:
    """Judge each top cell on ``split``; resumable per-cell segment cache."""
    from src.eval.judge import (
        _DEFAULT_TEMPLATE,
        _RESULTS_DIR,
        _aggregate,
        score_condition,
    )
    from src.infer.run import make_client

    cfg = yaml.safe_load(Path(judge_config).read_text(encoding="utf-8"))
    if "judge" not in cfg:
        raise SystemExit(f"{judge_config} has no `judge:` block")
    template = Path(cfg.get("template_file", _DEFAULT_TEMPLATE)).read_text(encoding="utf-8")
    client = make_client(cfg["judge"])
    judge_model = cfg["judge"].get("model", "unknown")
    cache_dir = _RESULTS_DIR / f"judge_{split}_segments"
    cache_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, dict] = {}
    for r in top:
        tag = r["tag"]
        sources, preds, refs = load_condition(sweep_dir, tag, split)
        print(f"Judging {len(preds)} segments for {tag} with {judge_model} ...")
        scores = score_condition(
            client, template, sources, preds, refs, cache_path=cache_dir / f"{tag}.jsonl"
        )
        mean, coverage = _aggregate(scores)
        out[tag] = {"mean": mean, "coverage": round(coverage, 4), "model": judge_model}
        mean_str = f"{mean:.3f}" if mean is not None else "n/a"
        print(f"  {tag:<16} Φ {mean_str}  (coverage {coverage:.0%})")

    usage = getattr(client, "usage", None)
    if usage is not None:
        print(f"Judge usage: {usage.summary()}")
    return out


def _freeze_pick(
    top: list[dict],
    comet_by_tag: dict[str, float],
    judge_by_tag: dict[str, dict],
    comet_adequacy: float,
) -> dict:
    """Real-metric analogue of the sweep's selection rule.

    When judge scores are present for every cell, keep the cells within
    ``comet_adequacy`` of the best COMET and pick the highest register Φ among
    them (register fidelity within an adequacy band, now on the reported metrics).
    Otherwise fall back to the best COMET cell.
    """
    have_judge = judge_by_tag and all(
        judge_by_tag.get(r["tag"], {}).get("mean") is not None for r in top
    )
    if have_judge:
        best_comet = max(comet_by_tag[r["tag"]] for r in top)
        band = [r for r in top if comet_by_tag[r["tag"]] >= best_comet - comet_adequacy]
        return max(band, key=lambda r: judge_by_tag[r["tag"]]["mean"])
    return max(top, key=lambda r: comet_by_tag[r["tag"]])


def _print_table(
    top: list[dict],
    comet_by_tag: dict[str, float],
    judge_by_tag: dict[str, dict],
    proxy_tag: str,
    freeze_tag: str,
) -> None:
    have_judge = bool(judge_by_tag)
    header = f"{'tag':<16}  {'k':>2}  {'lambda':>6}  {'chrF':>6}  {'stylo':>7}  {'COMET':>7}"
    if have_judge:
        header += f"  {'Φ':>5}"
    print(header)
    print("-" * len(header))
    for r in sorted(top, key=lambda r: comet_by_tag[r["tag"]], reverse=True):
        tag = r["tag"]
        stylo = r.get("stylo_dist", "-")
        line = (
            f"{tag:<16}  {r['k']:>2}  {r['lambda']:>6g}  {r['chrF']:>6}  "
            f"{stylo:>7}  {comet_by_tag[tag]:>7.4f}"
        )
        if have_judge:
            mean = judge_by_tag.get(tag, {}).get("mean")
            line += f"  {mean:>5.3f}" if mean is not None else f"  {'n/a':>5}"
        marks = []
        if tag == proxy_tag:
            marks.append("proxy pick")
        if tag == freeze_tag:
            marks.append("<== freeze")
        if marks:
            line += "   " + ", ".join(marks)
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm the AFSP proxy-picked cells on full val with COMET (+ optional judge)."
    )
    parser.add_argument("--config", default="configs/afsp_sweep.yaml")
    parser.add_argument(
        "--sweep-result",
        default=None,
        help="results/afsp_sweep_<split>.json (default: derived from the config's eval_file)",
    )
    parser.add_argument(
        "--val-file",
        default="data/splits/val.jsonl",
        help="full split to confirm on; must not be the sealed test split",
    )
    parser.add_argument(
        "--top", type=int, default=3, help="how many top proxy cells to verify (default 3)"
    )
    parser.add_argument(
        "--judge-config",
        default=None,
        help="YAML with a `judge:` block; omit to skip the paid judge pass (COMET only)",
    )
    parser.add_argument(
        "--adequacy-margin",
        type=float,
        default=None,
        help="chrF band for ranking the proxy cells; default reuses the sweep result's margin",
    )
    parser.add_argument(
        "--comet-adequacy",
        type=float,
        default=0.01,
        help="COMET band within which judge Φ decides the freeze (default 0.01)",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="COMET batch size")
    parser.add_argument(
        "--gpus", type=int, default=None, help="COMET GPU count; default auto (1 if CUDA else 0)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="regenerate the top cells on full val"
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    if args.sweep_result:
        result_path = Path(args.sweep_result)
    else:
        sweep_split = Path(cfg["data"]["eval_file"]).stem
        result_path = Path("results") / f"afsp_sweep_{sweep_split}.json"
    if not result_path.exists():
        raise SystemExit(
            f"sweep result not found: {result_path} -- run `manage.py afsp_sweep` first"
        )
    sweep = json.loads(result_path.read_text(encoding="utf-8"))
    rows = sweep.get("cells", [])
    if not rows:
        raise SystemExit(f"{result_path} has no scored cells")
    margin = (
        args.adequacy_margin
        if args.adequacy_margin is not None
        else sweep.get("adequacy_margin", 1.0)
    )

    ordered = ranked_cells(rows, margin)
    top = ordered[: max(1, args.top)]
    proxy_pick = sweep.get("recommended") or top[0]
    proxy_tag = proxy_pick["tag"]

    val_file = Path(args.val_file)
    val_split = val_file.stem
    if val_split == "test":
        raise SystemExit("refusing to verify on the sealed test split; --val-file must be val")

    print(
        f"Confirming top {len(top)} proxy cells from {result_path.name} "
        f"({[r['tag'] for r in top]}) on {val_split}"
    )

    # Regenerate only the top cells on the full val split (skip-if-exists / resumable).
    gen_cfg = {**cfg, "data": {**cfg.get("data", {}), "eval_file": str(val_file), "limit": None}}
    cells = [(r["k"], r["lambda"]) for r in top]
    generate_cells(gen_cfg, cells, overwrite=args.overwrite)

    # COMET (free/local), model loaded once and reused across cells.
    from src.eval import comet as comet_mod

    sweep_dir = _sweep_dir(cfg)
    comet_model = comet_mod.load_model()
    comet_by_tag: dict[str, float] = {}
    for r in top:
        tag = r["tag"]
        sources, preds, refs = load_condition(sweep_dir, tag, val_split)
        res = comet_mod.score(
            sources, preds, refs, model=comet_model, batch_size=args.batch_size, gpus=args.gpus
        )
        comet_by_tag[tag] = res["system"]
        print(f"  {tag:<16} COMET {res['system']:.4f}  (n={len(preds)})")

    # Judge (paid) only when a config is supplied.
    judge_by_tag: dict[str, dict] = {}
    if args.judge_config:
        judge_by_tag = _run_judge(args.judge_config, top, sweep_dir, val_split)

    freeze = _freeze_pick(top, comet_by_tag, judge_by_tag, args.comet_adequacy)
    freeze_tag = freeze["tag"]
    held = freeze_tag == proxy_tag

    print(f"\nAFSP verification ({val_split}, {len(top)} cells)  -- reported metrics")
    _print_table(top, comet_by_tag, judge_by_tag, proxy_tag, freeze_tag)

    if held:
        print(f"\nProxy pick {proxy_tag} holds up on the reported metrics -- freeze it.")
    else:
        print(
            f"\nRunner-up {freeze_tag} overtakes the proxy pick {proxy_tag} on the reported "
            f"metrics -- freeze {freeze_tag} instead (this is exactly the catch the confirmation "
            f"pass exists for)."
        )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"afsp_verify_{val_split}.json"
    out_path.write_text(
        json.dumps(
            {
                "split": val_split,
                "sweep_result": str(result_path),
                "adequacy_margin": margin,
                "comet_adequacy": args.comet_adequacy,
                "judge": bool(judge_by_tag),
                "proxy_pick": proxy_tag,
                "freeze": freeze_tag,
                "proxy_pick_held": held,
                "cells": [
                    {
                        "tag": r["tag"],
                        "k": r["k"],
                        "lambda": r["lambda"],
                        "chrF": r.get("chrF"),
                        "stylo_dist": r.get("stylo_dist"),
                        "comet_system": comet_by_tag[r["tag"]],
                        "judge_mean": judge_by_tag.get(r["tag"], {}).get("mean"),
                        "judge_coverage": judge_by_tag.get(r["tag"], {}).get("coverage"),
                    }
                    for r in top
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")
    print("Freeze before touching test.jsonl -- set in configs/base_qwen.yaml:")
    print(f"    retrieval.k       : {freeze['k']}")
    print(f"    afsp.lambda_style : {freeze['lambda']}")


if __name__ == "__main__":
    main()
