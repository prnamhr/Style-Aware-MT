"""
Confirm the PEFT sweep's proxy-picked candidates on the reported metrics before freezing.

The real-metric analogue of src.peft.sweep, matched to src.infer.afsp_verify: take the
top proxy candidates (each an (r, lr, epoch) checkpoint, ranked on chrF + register_fit),
score them on COMET (adequacy) and, when a --judge-config is given, the LLM-judge
register score Phi, then freeze on COMET adequacy band + judge Phi -- the exact rule
AFSP freezes on. The freeze/judge logic is reused from src.infer.afsp_verify so PEFT
and AFSP are frozen on one identical procedure.

Usage:
    python -m src.peft.verify                                        # COMET only
    python -m src.peft.verify --judge-config configs/judge_eval.yaml # COMET + judge Phi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.peft.sweep import _sweep_dir, generate_cell, ranked_cells


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm PEFT proxy candidates on full val with COMET (+ optional judge Phi)."
    )
    parser.add_argument("--config", default="configs/peft_sweep.yaml")
    parser.add_argument(
        "--sweep-result",
        default=None,
        help="results/peft_sweep_<split>.json (default: derived from the config's eval_file)",
    )
    parser.add_argument(
        "--val-file",
        default="data/splits/val.jsonl",
        help="full split to confirm on; must not be the sealed test split",
    )
    parser.add_argument(
        "--top", type=int, default=3, help="how many top proxy candidates to verify (default 3)"
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
        help="chrF band for ranking the proxy candidates; default reuses the sweep result's margin",
    )
    parser.add_argument(
        "--comet-adequacy",
        type=float,
        default=0.01,
        help="COMET band within which judge Phi decides the freeze (default 0.01)",
    )
    parser.add_argument(
        "--min-judge-coverage",
        type=float,
        default=0.9,
        help="when --judge-config is set, minimum per-candidate judge coverage required to freeze "
        "on both axes; below this the run refuses rather than falling back to COMET (default 0.9)",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="COMET batch size")
    parser.add_argument(
        "--gpus", type=int, default=None, help="COMET GPU count; default auto (1 if CUDA else 0)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="regenerate the top candidates on full val"
    )
    args = parser.parse_args()

    # Heavy imports (COMET/torch) kept lazy so --help needs no GPU stack.
    from src.eval import comet as comet_mod
    from src.eval._io import load_condition
    from src.infer.afsp_verify import _freeze_pick, _run_judge

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    if args.sweep_result:
        result_path = Path(args.sweep_result)
    else:
        sweep_split = Path(cfg["data"]["eval_file"]).stem
        result_path = Path("results") / f"peft_sweep_{sweep_split}.json"
    if not result_path.exists():
        raise SystemExit(
            f"sweep result not found: {result_path} -- run `manage.py peft_sweep` first"
        )
    sweep = json.loads(result_path.read_text(encoding="utf-8"))
    rows = sweep.get("cells", [])
    if not rows:
        raise SystemExit(f"{result_path} has no scored candidates")
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
        f"Confirming top {len(top)} proxy candidates from {result_path.name} "
        f"({[r['tag'] for r in top]}) on {val_split}"
    )

    # Regenerate only the top candidates on the full val split (skip-if-exists / resumable).
    sweep_dir = _sweep_dir(cfg)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with val_file.open(encoding="utf-8") as f:
        val_rows = [json.loads(line) for line in f if line.strip()]
    style = Path(cfg["prompt"]["style_instruction_file"]).read_text(encoding="utf-8")
    for r in top:
        generate_cell(
            cfg,
            Path(r["checkpoint"]),
            r["tag"],
            sweep_dir,
            val_split,
            val_rows,
            style,
            overwrite=args.overwrite,
        )

    comet_model = comet_mod.load_model()
    comet_by_tag: dict[str, float] = {}
    comet_segments_by_tag: dict[str, list[float]] = {}
    shared_sources: list[str] | None = None
    for r in top:
        tag = r["tag"]
        sources, preds, refs = load_condition(sweep_dir, tag, val_split)
        # Every top candidate is generated from the same val rows in the same order, so
        # segment i lines up across candidates -- the precondition a paired bootstrap needs.
        if shared_sources is None:
            shared_sources = sources
        elif sources != shared_sources:
            raise SystemExit(
                f"segment misalignment: {tag} sources differ from the earlier candidates on "
                f"{val_split}; regenerate the top candidates (--overwrite) before verifying"
            )
        res = comet_mod.score(
            sources, preds, refs, model=comet_model, batch_size=args.batch_size, gpus=args.gpus
        )
        comet_by_tag[tag] = res["system"]
        comet_segments_by_tag[tag] = res["segments"]
        print(f"  {tag:<24} COMET {res['system']:.4f}  (n={len(preds)})")

    judge_by_tag: dict[str, dict] = {}
    if args.judge_config:
        judge_by_tag = _run_judge(args.judge_config, top, sweep_dir, val_split)
        thin = [
            tag
            for tag in (r["tag"] for r in top)
            if judge_by_tag.get(tag, {}).get("mean") is None
            or judge_by_tag.get(tag, {}).get("coverage", 0.0) < args.min_judge_coverage
        ]
        if thin:
            raise SystemExit(
                "refusing to freeze: a --judge-config was given (confirm on both axes) but the "
                f"judge pass is unusable for {thin} (mean missing, or coverage below "
                f"--min-judge-coverage {args.min_judge_coverage:.0%}). Fix the judge run (the "
                f"per-segment cache under results/judge_{val_split}_segments/ is resumable) or "
                "rerun without --judge-config to freeze on COMET alone deliberately."
            )

    freeze = _freeze_pick(top, comet_by_tag, judge_by_tag, args.comet_adequacy)
    freeze_tag = freeze["tag"]
    held = freeze_tag == proxy_tag

    print(f"\nPEFT verification ({val_split}, {len(top)} candidates) -- reported metrics")
    have_judge = bool(judge_by_tag)
    header = f"{'tag':<24}  {'r':>3}  {'lr':>7}  {'ep':>2}  {'chrF':>6}  {'stylo':>7}  {'COMET':>7}"
    if have_judge:
        header += f"  {'Phi':>5}"
    print(header)
    print("-" * len(header))
    for r in sorted(top, key=lambda r: comet_by_tag[r["tag"]], reverse=True):
        tag = r["tag"]
        stylo = r.get("stylo_dist", "-")
        line = (
            f"{tag:<24}  {r['r']:>3}  {r['lr']:>7g}  {str(r.get('epoch', '-')):>2}  "
            f"{r.get('chrF', '-'):>6}  {stylo:>7}  {comet_by_tag[tag]:>7.4f}"
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

    if held:
        print(f"\nProxy pick {proxy_tag} holds up on the reported metrics -- freeze it.")
    else:
        print(
            f"\nRunner-up {freeze_tag} overtakes the proxy pick {proxy_tag} on the reported "
            f"metrics -- freeze {freeze_tag} instead (the catch the confirmation pass exists for)."
        )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"peft_verify_{val_split}.json"
    out_path.write_text(
        json.dumps(
            {
                "split": val_split,
                "sweep_result": str(result_path),
                "adequacy_margin": margin,
                "comet_adequacy": args.comet_adequacy,
                "judge": have_judge,
                "proxy_pick": proxy_tag,
                "freeze": freeze_tag,
                "freeze_checkpoint": freeze["checkpoint"],
                "proxy_pick_held": held,
                "sources": shared_sources,
                "cells": [
                    {
                        "tag": r["tag"],
                        "r": r["r"],
                        "lr": r["lr"],
                        "epoch": r.get("epoch"),
                        "checkpoint": r["checkpoint"],
                        "chrF": r.get("chrF"),
                        "stylo_dist": r.get("stylo_dist"),
                        "comet_system": comet_by_tag[r["tag"]],
                        "comet_segments": comet_segments_by_tag[r["tag"]],
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
    print("Freeze before touching test.jsonl -- set in configs/peft_qwen.yaml:")
    print(f"    generator.adapter_path : {freeze['checkpoint']}")



if __name__ == "__main__":
    main()
