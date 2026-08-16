"""
Checkpoint selection for a trained RLSF arm, on the dev slice.

Usage:
    python manage.py rlsf_select --cell w3_0.0
    python manage.py rlsf_select --cell w3_2.0 --skip_kiwi        # chrF and register only
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

from src.eval._io import load_condition
from src.eval.quick import _marker_rate
from src.eval.stylometrics import (
    HELDOUT_FEATURES,
    REWARD_FEATURES,
    aggregate,
    distance_to_centroid,
    signed_z,
    subcentroid,
)
from src.infer.run import build_zeroshot_user, make_client
from src.rlsf.config import load_config
from src.rlsf.reward import load_centroid
from src.rlsf.train import _kiwi_or_none, arm_path

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")

# chrF points a checkpoint may give up against the best in the arm before it leaves the
# adequacy band. Matches src.peft.sweep's default, so the two selections read the same way.
_ADEQUACY_MARGIN = 1.0


def checkpoints(adapter_dir: str | Path) -> list[tuple[str, Path]]:
    """(tag, path) per saved checkpoint, in optimizer-step order, final adapter last."""
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise SystemExit(f"{adapter_dir} does not exist; train the arm before selecting from it")
    out = []
    for path in adapter_dir.iterdir():
        m = _CHECKPOINT_RE.match(path.name)
        if m and (path / "adapter_config.json").exists():
            out.append((int(m.group(1)), f"step{m.group(1)}", path))
    out.sort()
    tagged = [(tag, path) for _, tag, path in out]
    if (adapter_dir / "adapter_config.json").exists():
        tagged.append(("final", adapter_dir))
    if not tagged:
        raise SystemExit(
            f"{adapter_dir} holds no adapter_config.json, at its root or in a checkpoint-N/ "
            f"subdirectory. Nothing was saved: check train.save_every_rollouts."
        )
    return tagged


def generate(gen: dict, rows: list[dict], style: str, out_path: Path, *, tag: str) -> None:
    """Greedy dev generations for one checkpoint, at the locked inference decoding."""

    client = make_client(gen)
    print(f"[{tag}] generating {len(rows)} dev translations with {gen['adapter_path']} ...")
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    failures = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            try:
                prediction = client.complete(style, build_zeroshot_user(row["input"]))
                error = None
            except Exception as e:  # one bad segment must not discard the checkpoint
                prediction, error = "", f"{type(e).__name__}: {e}"
                failures += 1
            record = {
                "input": row["input"],
                "output": row["output"],
                "prediction": prediction,
                "condition": tag,
                "model": gen["model"],
                "adapter_path": str(gen["adapter_path"]),
            }
            if error is not None:
                record["error"] = error
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)
    if failures:
        print(f"[{tag}] WARNING: {failures}/{len(rows)} segments recorded with an empty prediction")


def free_gpu() -> None:
    """Drop one checkpoint's model before the next is loaded."""

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def score_checkpoint(tag: str, select_dir: Path, centroid: dict, kiwi) -> dict:
    """Adequacy and register for one checkpoint's dev generations."""

    sources, preds, refs = load_condition(select_dir, tag, "dev")
    agg = aggregate(preds)
    row = {
        "tag": tag,
        "n": len(preds),
        "chrF": round(CHRF().corpus_score(preds, [refs]).score, 2),
        "BLEU": round(BLEU().corpus_score(preds, [refs]).score, 2),
        "marker_rate": round(_marker_rate(preds), 2),
        # Split so the register the reward could see and the register it could not are
        # reported apart: a gain confined to REWARD_FEATURES is the proxy, not the target.
        "dist_reward": round(
            distance_to_centroid(agg["mean"], subcentroid(centroid, REWARD_FEATURES)), 4
        ),
        "dist_heldout": round(
            distance_to_centroid(agg["mean"], subcentroid(centroid, HELDOUT_FEATURES)), 4
        ),
        "z": {k: round(v, 4) for k, v in signed_z(agg["mean"], centroid).items()},
    }
    if kiwi is not None:
        scores = kiwi.score(sources, preds)
        row["kiwi"] = round(sum(scores) / len(scores), 4)
    return row


def ranked(rows: list[dict], margin: float) -> list[dict]:
    """Best first: inside the arm's own chrF adequacy band, then nearest on held-out register."""
    if not rows:
        return []
    best_chrf = max(r["chrF"] for r in rows)

    def fidelity(r: dict) -> tuple:
        return (r["dist_heldout"], -r["chrF"], r["tag"])

    band = sorted((r for r in rows if r["chrF"] >= best_chrf - margin), key=fidelity)
    out = sorted((r for r in rows if r["chrF"] < best_chrf - margin), key=fidelity)
    return band + out


def _print_table(rows: list[dict], pick: dict) -> None:
    cols = ["tag", "n", "chrF", "BLEU", "marker_rate", "dist_reward", "dist_heldout"]
    if "kiwi" in rows[0]:
        cols.append("kiwi")
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("\n" + "  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        marker = " <-" if r["tag"] == pick["tag"] else ""
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols) + marker)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select an RLSF checkpoint on the dev slice.")
    parser.add_argument("--config", default="configs/rlsf.yaml")
    parser.add_argument("--cell", required=True, help="the trained arm, e.g. w3_2.0")
    parser.add_argument("--adapter_dir", default=None, help="default: output.adapter_dir per arm")
    parser.add_argument(
        "--dev-limit",
        type=int,
        default=0,
        help="dev segments per checkpoint; 0 = full dev slice (499)",
    )
    parser.add_argument("--adequacy-margin", type=float, default=_ADEQUACY_MARGIN)
    parser.add_argument("--skip_kiwi", action="store_true", help="no COMET worker")
    parser.add_argument("--overwrite", action="store_true", help="regenerate existing dev files")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, require_caps=False)
    adapter_dir = args.adapter_dir or arm_path(cfg["output"]["adapter_dir"], args.cell)
    found = checkpoints(adapter_dir)

    style = Path(cfg["prompt"]["style_instruction_file"]).read_text(encoding="utf-8")
    with Path(cfg["data"]["dev_file"]).open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rows = rows[: args.dev_limit] if args.dev_limit else rows

    select_dir = Path(cfg["output"]["dir"]) / "rlsf" / f"select_{args.cell}"
    select_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(found)} checkpoints in {adapter_dir}, {len(rows)} dev segments each")

    # The locked inference decoding, not the rollout's: a checkpoint is selected as the
    # system it would be deployed as. max_tokens comes from the arm so nothing is truncated
    # here that was feasible during training.
    base_gen = {
        **cfg["generator"],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": cfg["rlsf"]["seed"],
    }
    for tag, path in found:
        out_path = select_dir / f"{tag}_dev.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"skip {tag}: {out_path} exists (--overwrite to regenerate)")
            continue
        generate({**base_gen, "adapter_path": str(path)}, rows, style, out_path, tag=tag)
        free_gpu()

    centroid = load_centroid(cfg["data"]["split_centroid_file"])
    with _kiwi_or_none(cfg["rlsf"]["reward"]["kiwi"], args.skip_kiwi) as kiwi:
        scored = [score_checkpoint(tag, select_dir, centroid, kiwi) for tag, _ in found]

    order = ranked(scored, args.adequacy_margin)
    pick = order[0]
    _print_table(scored, pick)
    print(
        f"\nselected {pick['tag']}: chrF {pick['chrF']}, held-out register distance "
        f"{pick['dist_heldout']} (band = best chrF - {args.adequacy_margin})"
    )

    out = {
        "cell": args.cell,
        "adapter_dir": str(adapter_dir),
        "dev_file": cfg["data"]["dev_file"],
        "n_segments": len(rows),
        "adequacy_margin": args.adequacy_margin,
        "rule": "chrF adequacy band, then held-out register distance; no judge",
        "selected": pick["tag"],
        "selected_path": str(dict(found)[pick["tag"]]),
        "ranked": [r["tag"] for r in order],
        "rows": scored,
    }
    result_path = Path("results") / f"rlsf_select_{args.cell}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
