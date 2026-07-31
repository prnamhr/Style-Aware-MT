"""
PEFT (LoRA) hyperparameter sweep over the Phase 2 grid, with MATCHED selection.

Selection is deliberately identical in shape to the AFSP sweep (src.infer.afsp_sweep),
but epoch is a tuned axis: a candidate here is one trained (r, lr, epoch) checkpoint.
Each cell trains for the full epoch budget with load_best_model_at_end:false and
save_total_limit:null, so every epoch checkpoint is kept. eval_loss (val token-level
cross-entropy) is used ONLY as a free pre-filter to prune which checkpoints get
generated -- never as the selector. The reported adapter is picked on the same axis
AFSP freezes on: chrF adequacy band + register fidelity (register_fit / Phi) here,
confirmed on COMET (+ judge Phi) in src.peft.verify.

Pipeline: train (save every epoch) -> eval_loss prunes candidates -> generate val ->
score chrF + register_fit -> rank. Then `python -m src.peft.verify` for COMET/Phi.

Usage:
    python -m src.peft.sweep --config configs/peft_sweep.yaml
    python -m src.peft.sweep --dry-run                 # print the grid plan, train nothing
    python -m src.peft.sweep --score-only              # re-score/rank existing candidates
    python -m src.peft.sweep --epochs-keep -1          # keep every epoch (no eval_loss prune)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import yaml

from src.peft.train import train


def _lr_tag(lr: float) -> str:
    """Compact learning-rate tag for a filesystem path: 2e-4, 1e-4, ..."""
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _cell_dir(output_base: str, r: int, lr: float) -> Path:
    return Path(output_base) / f"peft_lora_r{r}_lr{_lr_tag(lr)}"


def cell_tag(r: int, lr: float) -> str:
    """Filesystem-safe tag for one (r, lr) cell (without the epoch axis)."""
    return f"peft_r{r}_lr{_lr_tag(lr)}"


def candidate_tag(r: int, lr: float, epoch: int | None) -> str:
    """Filesystem-safe, collision-free tag for one (r, lr, epoch) candidate."""
    return cell_tag(r, lr) if epoch is None else f"{cell_tag(r, lr)}_e{epoch}"


def _sweep_dir(cfg: dict) -> Path:
    return Path(cfg["output"]["dir"]) / "peft_sweep"


def _cell_alpha(cell: dict, r: int) -> int:
    """lora_alpha for a cell. Default alpha = 2r holds the effective LoRA scale.\
    """
    return int(cell.get("alpha", 2 * r))


def build_cell_config(base_cfg: dict, cell: dict, output_base: str) -> tuple[dict, Path]:
    """Copy base_cfg and apply this cell's r / alpha / lr / output_dir overrides."""
    cfg = copy.deepcopy(base_cfg)
    r = int(cell["r"])
    lr = float(cell["lr"])
    out_dir = _cell_dir(output_base, r, lr)
    cfg["peft"]["lora"]["r"] = r
    cfg["peft"]["lora"]["alpha"] = _cell_alpha(cell, r)
    cfg["peft"]["train"]["learning_rate"] = lr
    cfg["peft"]["output_dir"] = str(out_dir)
    return cfg, out_dir


def _read_val_rows(cfg: dict) -> list[dict]:
    eval_file = Path(cfg["data"]["eval_file"])
    if eval_file.stem == "test":
        raise ValueError("refusing to sweep on the sealed test split; sweep on val")
    with eval_file.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    limit = cfg["data"].get("limit")
    return rows[:limit] if limit else rows


def enumerate_candidates(cfg: dict, output_base: str) -> list[dict]:
    """Every (r, lr, epoch) checkpoint across the grid, from each cell's manifest.

    Falls back to the cell's root adapter as a single epoch-less candidate when no
    epoch manifest is present (e.g. a cell trained with load_best_model_at_end:true).
    """
    candidates: list[dict] = []
    for cell in cfg["sweep"]["grid"]:
        r, lr = int(cell["r"]), float(cell["lr"])
        alpha = _cell_alpha(cell, r)
        cell_dir = _cell_dir(output_base, r, lr)
        anchor = bool(cell.get("anchor", False))
        manifest_path = cell_dir / "epoch_checkpoints.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for m in manifest:
                if not m.get("checkpoint"):
                    continue
                candidates.append(
                    {
                        "tag": candidate_tag(r, lr, m["epoch"]),
                        "r": r,
                        "alpha": alpha,
                        "lr": lr,
                        "epoch": m["epoch"],
                        "checkpoint": m["checkpoint"],
                        "eval_loss": m.get("eval_loss"),
                        "anchor": anchor,
                        "cell_dir": str(cell_dir),
                    }
                )
        elif (cell_dir / "adapter_config.json").exists():
            candidates.append(
                {
                    "tag": candidate_tag(r, lr, None),
                    "r": r,
                    "alpha": alpha,
                    "lr": lr,
                    "epoch": None,
                    "checkpoint": str(cell_dir),
                    "eval_loss": _read_eval_loss(cell_dir),
                    "anchor": anchor,
                    "cell_dir": str(cell_dir),
                }
            )
    return candidates


def prune_by_eval_loss(candidates: list[dict], keep: int | None) -> list[dict]:
    """Per-cell pre-filter: keep the `keep` lowest-eval_loss epochs of each (r, lr).

    eval_loss decides only which checkpoints are worth generating; it never picks
    the reported adapter. keep <= 0 (or None) keeps every epoch. Pruned candidates
    are logged so the cap is never silent.
    """
    if keep is None or keep <= 0:
        return candidates
    by_cell: dict[tuple, list[dict]] = {}
    for c in candidates:
        by_cell.setdefault((c["r"], c["lr"]), []).append(c)
    kept: list[dict] = []
    for cs in by_cell.values():
        # Missing eval_loss sorts last so a checkpoint we can't rank is dropped first.
        ranked = sorted(cs, key=lambda c: (c["eval_loss"] is None, c["eval_loss"] or 0.0))
        kept.extend(ranked[:keep])
        for c in ranked[keep:]:
            log_el = f"{c['eval_loss']:.4f}" if c["eval_loss"] is not None else "n/a"
            print(f"prune {c['tag']}: eval_loss {log_el} (keeping best {keep}/cell for generation)")
    return kept


def generate_cell(
    cfg: dict,
    adapter_path: Path,
    tag: str,
    sweep_dir: Path,
    split: str,
    rows: list[dict],
    style: str,
    *,
    overwrite: bool = False,
) -> None:
    """Generate zero-shot val predictions with this candidate's adapter checkpoint.

    Byte-identical prompt to the `peft` inference condition (zero-shot user directive;
    the register is carried by the adapter), so the sweep scores what the frozen
    system would emit.
    """
    from src.infer.run import build_zeroshot_user, make_client

    out_path = sweep_dir / f"{tag}_{split}.jsonl"
    if out_path.exists() and not overwrite:
        print(f"skip {tag}: {out_path} exists (use --overwrite to regenerate)")
        return

    gen = {**cfg["generator"], "adapter_path": str(adapter_path)}
    client = make_client(gen)
    print(f"[{tag}] generating {len(rows)} val translations with {adapter_path} ...")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    failures = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            try:
                prediction = client.complete(style, build_zeroshot_user(row["input"]))
                error = None
            except Exception as e:  # one bad segment must not discard the candidate
                prediction = ""
                error = f"{type(e).__name__}: {e}"
                failures += 1
            record = {
                "input": row["input"],
                "output": row["output"],
                "prediction": prediction,
                "condition": tag,
                "model": gen["model"],
                "adapter_path": str(adapter_path),
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


def _free_gpu() -> None:
    """Drop the trainer's model before loading the inference model (OOM guard)."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _read_eval_loss(out_dir: Path) -> float | None:
    """Reference only -- eval_loss never drives selection."""
    p = out_dir / "train_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("eval_loss")


def score_candidates(cfg: dict, candidates: list[dict], split: str) -> list[dict]:
    """Score each present candidate on the matched proxy axis (chrF + register_fit)."""
    # afsp_sweep pulls src.infer.run -> torch at import; keep it lazy so --dry-run
    # and --help need no GPU stack.
    from src.eval._io import load_condition
    from src.eval.quick import score as quick_score
    from src.eval.stylometrics import aggregate, distance_to_centroid
    from src.infer.afsp_sweep import _register_fit_fn

    sweep_dir = _sweep_dir(cfg)
    centroid_path = Path(cfg.get("register", {}).get("centroid_file", ""))
    centroid = (
        json.loads(centroid_path.read_text(encoding="utf-8")) if centroid_path.exists() else None
    )
    # Reuse AFSP's register-fit builder verbatim so Phi is the SAME axis; it reads the
    # direction/target-sigma from cfg["afsp"], so shim the `register:` block into place.
    register_fit = _register_fit_fn({"afsp": cfg.get("register", {})}, centroid)

    rows: list[dict] = []
    for c in candidates:
        tag = c["tag"]
        path = sweep_dir / f"{tag}_{split}.jsonl"
        if not path.exists():
            print(f"skip {tag}: {path} not found")
            continue
        s = quick_score(tag, sweep_dir, split)
        _, preds, _ = load_condition(sweep_dir, tag, split)
        agg = aggregate(preds)
        keys = ("tag", "r", "alpha", "lr", "epoch", "checkpoint", "eval_loss", "anchor")
        row = {
            **{k: c[k] for k in keys},
            "n": s["n"],
            "chrF": s["chrF"],
            "BLEU": s["BLEU"],
            "marker_rate": s["marker_rate"],
        }
        if centroid is not None:
            row["stylo_dist"] = round(distance_to_centroid(agg["mean"], centroid), 4)
            row["register_fit"] = register_fit(agg["mean"])
        rows.append(row)
    return rows


def ranked_cells(rows: list[dict], adequacy_margin: float) -> list[dict]:
    """Candidates best-first by the rule AFSP uses: chrF adequacy band, then Phi.

    Unlike src.infer.afsp_sweep, the `anchor` flag here is a COSMETIC label for the
    documented-default cell (r=16, lr=2e-4), not an exclusion: it is a first-class
    (r, lr, epoch) candidate and can be selected. (AFSP's anchor is the zero-shot
    reference, a different condition, so it is filtered there; PEFT has no such
    non-candidate reference in the grid.)
    """
    if not rows:
        return []

    def _epoch(r: dict) -> int:
        return r["epoch"] if r.get("epoch") is not None else 0

    if not all("register_fit" in r for r in rows):
        return sorted(rows, key=lambda r: (-r["chrF"], r["r"], _epoch(r), -r["lr"]))
    best_chrf = max(r["chrF"] for r in rows)

    def fidelity(r: dict) -> tuple:
        return (r["register_fit"], -r["chrF"], r["r"], _epoch(r), -r["lr"])

    band = sorted((r for r in rows if r["chrF"] >= best_chrf - adequacy_margin), key=fidelity)
    out_band = sorted((r for r in rows if r["chrF"] < best_chrf - adequacy_margin), key=fidelity)
    return band + out_band


def _fmt(v: object) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def _print_table(rows: list[dict], pick: dict | None) -> None:
    cols = ["tag", "r", "alpha", "lr", "epoch", "n", "chrF", "BLEU", "marker_rate"]
    have_fit = bool(rows) and "register_fit" in rows[0]
    if rows and "stylo_dist" in rows[0]:
        cols.append("stylo_dist")
    if have_fit:
        cols.append("register_fit")
    cols.append("eval_loss")
    key = (lambda r: r["register_fit"]) if have_fit else (lambda r: -r["chrF"])
    ordered = sorted(rows, key=key)  # anchor included: it is a selectable candidate
    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in ordered)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols) + "  <-"
    print(header)
    print("-" * len(header))
    for r in ordered:
        marks = []
        if r.get("anchor"):
            marks.append("(anchor)")
        if pick and r["tag"] == pick["tag"]:
            marks.append("<== proxy pick")
        line = "  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols)
        print(line + ("  " + " ".join(marks) if marks else ""))


def _print_plan(cells: list[dict], output_base: str, epochs: int) -> None:
    print(f"\nPEFT sweep: {len(cells)} cell(s) x {epochs} epoch(s) kept = candidate grid\n")
    header = (
        f"{'cell':<6}{'r':>4}{'alpha':>7}{'lr':>10}   output_dir (checkpoints: epoch 1..{epochs})"
    )
    print(header)
    print("-" * (len(header) + 6))
    for i, cell in enumerate(cells, 1):
        r, lr = int(cell["r"]), float(cell["lr"])
        tag = " (anchor)" if cell.get("anchor") else ""
        alpha = _cell_alpha(cell, r)
        print(f"{i:<6}{r:>4}{alpha:>7}{_lr_tag(lr):>10}   {_cell_dir(output_base, r, lr)}{tag}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="PEFT (LoRA) hyperparameter sweep (matched).")
    parser.add_argument("--config", default="configs/peft_sweep.yaml")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the grid plan and exit (no training)"
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="skip train+generate; only (re)score and rank existing candidates",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="retrain/regenerate even if present"
    )
    parser.add_argument(
        "--epochs-keep",
        type=int,
        default=2,
        help="per cell, generate only the N lowest-eval_loss epochs (pre-filter, not the "
        "selector); <=0 keeps every epoch (default 2)",
    )
    parser.add_argument(
        "--adequacy-margin",
        type=float,
        default=1.0,
        help="chrF band below the best within which register fidelity decides (default 1.0)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sweep_cfg = cfg.get("sweep", {})
    cells = sweep_cfg.get("grid", [])
    output_base = sweep_cfg.get("output_base", "models")
    if not cells:
        raise ValueError(f"no sweep.grid cells in {args.config}")
    epochs = int(cfg["peft"]["train"].get("num_train_epochs", 3))

    _print_plan(cells, output_base, epochs)
    if args.dry_run:
        print("--dry-run: no training performed.")
        return

    split = Path(cfg["data"]["eval_file"]).stem
    sweep_dir = _sweep_dir(cfg)
    keep = None if args.epochs_keep <= 0 else args.epochs_keep

    if not args.score_only:
        sweep_dir.mkdir(parents=True, exist_ok=True)
        rows_val = _read_val_rows(cfg)
        style = Path(cfg["prompt"]["style_instruction_file"]).read_text(encoding="utf-8")
        # 1) Train each cell for the full epoch budget, keeping every epoch checkpoint.
        for i, cell in enumerate(cells, 1):
            cell_cfg, out_dir = build_cell_config(cfg, cell, output_base)
            r = cell_cfg["peft"]["lora"]["r"]
            lr = cell_cfg["peft"]["train"]["learning_rate"]
            bar = "=" * 70
            print(f"\n{bar}\n[cell {i}/{len(cells)}] r={r} lr={_lr_tag(lr)} -> {out_dir}\n{bar}")
            if (out_dir / "epoch_checkpoints.json").exists() and not args.overwrite:
                print(f"skip training {out_dir}: epoch manifest present (--overwrite to retrain)")
            else:
                train(cell_cfg)
                _free_gpu()
        # 2) eval_loss pre-filter, then 3) generate val for the surviving candidates.
        candidates = prune_by_eval_loss(enumerate_candidates(cfg, output_base), keep)
        for c in candidates:
            generate_cell(
                cfg,
                Path(c["checkpoint"]),
                c["tag"],
                sweep_dir,
                split,
                rows_val,
                style,
                overwrite=args.overwrite,
            )
            _free_gpu()
    else:
        candidates = prune_by_eval_loss(enumerate_candidates(cfg, output_base), keep)

    # 4) Score the generated candidates on the matched proxy axis and rank.
    rows = score_candidates(cfg, candidates, split)
    if not rows:
        print("no scored candidates; run without --score-only to train+generate first")
        return
    ranked = ranked_cells(rows, args.adequacy_margin)
    pick = ranked[0] if ranked else None
    print(f"\nPEFT sweep ({split}, {len(rows)} candidates) -- register fidelity first")
    _print_table(rows, pick)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"peft_sweep_{split}.json"
    out_path.write_text(
        json.dumps(
            {
                "split": split,
                "adequacy_margin": args.adequacy_margin,
                "epochs_keep": keep,
                "select_target_sigma": float(
                    cfg.get("register", {}).get("select_target_sigma", 0.5)
                ),
                "selection_note": "candidates are (r, lr, epoch); eval_loss is a generation "
                "pre-filter only. Ranked on chrF adequacy band + register_fit (Phi). Confirm "
                "on COMET/judge via src.peft.verify before freezing.",
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
            f"\nProxy pick: r={pick['r']}, alpha={pick['alpha']}, lr={_lr_tag(pick['lr'])}, "
            f"epoch={pick['epoch']}  (chrF {pick['chrF']}"
            + (f", register_fit {pick['register_fit']}" if "register_fit" in pick else "")
            + f")  -> {pick['checkpoint']}"
        )
        print("Confirm on the reported metrics before freezing:")
        print("    python -m src.peft.verify --judge-config configs/judge_eval.yaml")


if __name__ == "__main__":
    main()
