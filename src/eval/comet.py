"""
Reference-based COMET scoring of inference outputs against the gold targets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from comet import download_model, load_from_checkpoint

from src.eval._io import condition_path, file_digest, load_condition, merge_results

DEFAULT_MODEL = "Unbabel/wmt22-comet-da"
_RESULTS_DIR = Path("results")


def load_model(model_name: str = DEFAULT_MODEL):
    """Download (if needed) and load a COMET checkpoint."""
    return load_from_checkpoint(download_model(model_name))


def _resolve_gpus(gpus: int | None) -> int:
    if gpus is not None:
        return gpus
    try:
        import torch

        return 1 if torch.cuda.is_available() else 0
    except Exception:
        return 0


def score(
    sources: list[str],
    hyps: list[str],
    refs: list[str],
    *,
    model=None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 8,
    gpus: int | None = None,
) -> dict:
    """Segment- and system-level COMET for one condition."""
    if model is None:
        model = load_model(model_name)
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, refs)]
    out = model.predict(data, batch_size=batch_size, gpus=_resolve_gpus(gpus), progress_bar=False)
    return {"system": float(out.system_score), "segments": [float(x) for x in out.scores]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference-based COMET scoring.")
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--split", default="val", help="output split tag (default: val)")
    parser.add_argument("--out_dir", default="outputs", help="inference output directory")
    parser.add_argument("--results_dir", default=str(_RESULTS_DIR))
    parser.add_argument(
        "--results_path",
        default=None,
        help="scores file to write; default results_dir/comet_<split>.json",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--gpus", type=int, default=None, help="GPU count; default auto (1 if CUDA else 0)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    present = [c for c in args.conditions if condition_path(out_dir, c, args.split).exists()]
    for c in args.conditions:
        if c not in present:
            print(f"skip {c}: {condition_path(out_dir, c, args.split)} not found")
    if not present:
        return

    model = load_model(args.model)  # loaded once, reused across conditions
    results: dict[str, dict] = {}
    for cond in present:
        sources, preds, refs = load_condition(out_dir, cond, args.split)
        res = score(sources, preds, refs, model=model, batch_size=args.batch_size, gpus=args.gpus)
        results[cond] = {
            "n": len(preds),
            "model": args.model,
            "output_sha256": file_digest(condition_path(out_dir, cond, args.split)),
            "sources": sources,
            **res,
        }
        print(f"{cond:<16} COMET {res['system']:.4f}  (n={len(preds)})")

    out_path = Path(args.results_path or Path(args.results_dir) / f"comet_{args.split}.json")
    preserved = merge_results(out_path, results)
    if preserved:
        print(f"preserved {len(preserved)} condition(s) not scored here: {', '.join(preserved)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
