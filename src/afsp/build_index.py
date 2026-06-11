"""Build the AFSP retrieval index over the English target side of the training split.

The index is a dense embedding matrix plus the aligned (source, target) exemplar
pairs. Retrieval is cross-lingual: a Persian/Arabic source query is embedded and
matched by cosine similarity against the embedded English targets. The matched
row maps back to its full (source -> target) pair, which becomes a few-shot
exemplar at inference time.

At this corpus size (~10.8k rows, 1024-dim) a brute-force matmul over L2-normalized
vectors is instant, so no FAISS dependency is needed for the smoke test.

Artifacts written to ``index_dir``:
  * ``embeddings.npy``  -- float32 [N, D], L2-normalized passage embeddings
  * ``pairs.jsonl``     -- N lines of {"input": ..., "output": ...}, row-aligned
  * ``meta.json``       -- embed model name, counts, dim, source file

Usage:
    python -m src.afsp.build_index --config configs/openai_smoke.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.afsp.embed import embed_passages, load_model


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_index(train_file: Path, index_dir: Path, embed_model: str, batch_size: int = 32) -> None:
    rows = _read_jsonl(train_file)
    pairs = [{"input": r["input"], "output": r["output"]} for r in rows]
    passages = [p["output"] for p in pairs]

    print(f"Embedding {len(passages)} English training targets with {embed_model} ...")
    model = load_model(embed_model)
    embeddings = embed_passages(model, passages, batch_size=batch_size).astype(np.float32)

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", embeddings)
    with (index_dir / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta = {
        "embed_model": embed_model,
        "n_passages": len(passages),
        "dim": int(embeddings.shape[1]),
        "source_file": str(train_file),
    }
    with (index_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Wrote index to {index_dir}/ : embeddings {embeddings.shape}, {len(pairs)} pairs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AFSP retrieval index.")
    parser.add_argument("--config", default="configs/openai_smoke.yaml")
    parser.add_argument("--train_file", default=None, help="override config train_file")
    parser.add_argument("--index_dir", default=None, help="override config index_dir")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    afsp = cfg["afsp"]
    train_file = Path(args.train_file or afsp["train_file"])
    index_dir = Path(args.index_dir or afsp["index_dir"])

    build_index(train_file, index_dir, afsp["embed_model"], batch_size=args.batch_size)


if __name__ == "__main__":
    main()
