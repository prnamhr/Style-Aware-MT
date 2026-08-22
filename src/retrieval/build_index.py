"""
Build the retrieval index over the Persian/Arabic source side of the training split.

Usage:
    python -m src.retrieval.build_index --config configs/base_qwen.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.data.split import sha256_file
from src.retrieval.embed import embed_passages, load_model
from src.retrieval.leakage import load_quarantine


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _check_not_overwriting(index_dir: Path) -> None:
    """Refuse to turn an existing unquarantined index into a quarantined one."""
    meta_path = index_dir / "meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("quarantine_file"):
        raise ValueError(
            f"{index_dir} holds an unquarantined index; build the quarantined one into a "
            "separate directory, e.g. --index_dir data/knn_index_clean"
        )


def build_index(
    train_file: Path,
    index_dir: Path,
    embed_model: str,
    batch_size: int = 32,
    quarantine: Path | None = None,
) -> None:
    rows = _read_jsonl(train_file)
    dropped: list[int] = []
    if quarantine is not None:
        _check_not_overwriting(index_dir)
        dropped = load_quarantine(quarantine, train_file)
        keep = set(range(len(rows))) - set(dropped)
        rows = [r for i, r in enumerate(rows) if i in keep]
        print(f"Quarantine {quarantine}: dropped {len(dropped)} of {len(dropped) + len(rows)} rows")

    pairs = [{"input": r["input"], "output": r["output"]} for r in rows]
    passages = [p["input"] for p in pairs]

    print(f"Embedding {len(passages)} Persian/Arabic training sources with {embed_model} ...")
    model = load_model(embed_model)
    embeddings = embed_passages(model, passages, batch_size=batch_size).astype(np.float32)

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", embeddings)
    with (index_dir / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta = {
        "embed_model": embed_model,
        "indexed_side": "source",
        "n_passages": len(passages),
        "dim": int(embeddings.shape[1]),
        "source_file": str(train_file),
    }
    if quarantine is not None:
        meta["quarantine_file"] = str(quarantine)
        meta["quarantine_sha256"] = sha256_file(quarantine)
        meta["n_quarantined"] = len(dropped)
    with (index_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Wrote index to {index_dir}/ : embeddings {embeddings.shape}, {len(pairs)} pairs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AFSP retrieval index.")
    parser.add_argument("--config", default="configs/base_qwen.yaml")
    parser.add_argument("--train_file", default=None, help="override config train_file")
    parser.add_argument("--index_dir", default=None, help="override config index_dir")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--quarantine", default=None, help="pool rows to drop, from `manage.py leakage`"
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    retr = cfg["retrieval"]
    train_file = Path(args.train_file or retr["train_file"])
    index_dir = Path(args.index_dir or retr["index_dir"])

    build_index(
        train_file,
        index_dir,
        retr["embed_model"],
        batch_size=args.batch_size,
        quarantine=Path(args.quarantine) if args.quarantine else None,
    )


if __name__ == "__main__":
    main()
