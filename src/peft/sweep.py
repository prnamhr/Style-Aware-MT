"""
PEFT (LoRA) hyperparameter sweep over the Phase 2 grid.

Usage:
    python -m src.peft.sweep --config configs/peft_sweep.yaml
    python -m src.peft.sweep --dry-run          # print the plan, train nothing
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from src.peft.train import train


def _lr_tag(lr: float) -> str:
    """Compact learning-rate tag for a filesystem path: 2e-4, 1e-4, ..."""
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _cell_dir(output_base: str, r: int, lr: float) -> Path:
    return Path(output_base) / f"peft_lora_r{r}_lr{_lr_tag(lr)}"


def build_cell_config(base_cfg: dict, cell: dict, output_base: str) -> tuple[dict, Path]:
    """Copy base_cfg and apply this cell's r / lr / output_dir overrides."""
    cfg = copy.deepcopy(base_cfg)
    r = int(cell["r"])
    lr = float(cell["lr"])
    out_dir = _cell_dir(output_base, r, lr)
    cfg["peft"]["lora"]["r"] = r
    cfg["peft"]["train"]["learning_rate"] = lr
    cfg["peft"]["output_dir"] = str(out_dir)
    return cfg, out_dir


def _print_plan(cells: list[dict], output_base: str) -> None:
    print(f"\nPEFT sweep: {len(cells)} cell(s), one adapter each (sequential)\n")
    header = f"{'cell':<6}{'r':>4}{'lr':>10}   {'output_dir'}"
    print(header)
    print("-" * (len(header) + 20))
    for i, cell in enumerate(cells, 1):
        r, lr = int(cell["r"]), float(cell["lr"])
        tag = " (anchor)" if cell.get("anchor") else ""
        print(f"{i:<6}{r:>4}{_lr_tag(lr):>10}   {_cell_dir(output_base, r, lr)}{tag}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="PEFT (LoRA) hyperparameter sweep.")
    parser.add_argument("--config", default="configs/peft_sweep.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the grid plan and exit (no training)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sweep_cfg = cfg.get("sweep", {})
    cells = sweep_cfg.get("grid", [])
    output_base = sweep_cfg.get("output_base", "models")
    if not cells:
        raise ValueError(f"no sweep.grid cells in {args.config}")

    _print_plan(cells, output_base)
    if args.dry_run:
        print("--dry-run: no training performed.")
        return

    results: list[dict] = []
    for i, cell in enumerate(cells, 1):
        cell_cfg, out_dir = build_cell_config(cfg, cell, output_base)
        r = cell_cfg["peft"]["lora"]["r"]
        lr = cell_cfg["peft"]["train"]["learning_rate"]
        bar = "=" * 70
        print(f"\n{bar}\n[cell {i}/{len(cells)}] r={r} lr={_lr_tag(lr)} -> {out_dir}\n{bar}")
        train(cell_cfg)

        # train() writes final dev metrics here; pull eval_loss for the leaderboard.
        metrics_path = out_dir / "train_metrics.json"
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        )
        results.append(
            {
                "r": r,
                "lr": lr,
                "anchor": bool(cell.get("anchor", False)),
                "output_dir": str(out_dir),
                "eval_loss": metrics.get("eval_loss"),
            }
        )

    # Rank by dev eval_loss (lower is better); cells that failed to report sink last.
    ranked = sorted(results, key=lambda r: (r["eval_loss"] is None, r["eval_loss"] or 0.0))
    print("\nSweep leaderboard (by dev eval_loss):")
    print(f"{'r':>4}{'lr':>10}{'eval_loss':>14}   output_dir")
    for row in ranked:
        el = f"{row['eval_loss']:.4f}" if row["eval_loss"] is not None else "n/a"
        tag = " (anchor)" if row["anchor"] else ""
        print(f"{row['r']:>4}{_lr_tag(row['lr']):>10}{el:>14}   {row['output_dir']}{tag}")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    lb_path = results_dir / "peft_sweep_leaderboard.json"
    lb_path.write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    print(f"\nWrote leaderboard: {lb_path}")


if __name__ == "__main__":
    main()
