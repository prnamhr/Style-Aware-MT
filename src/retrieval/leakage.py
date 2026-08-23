"""
Near-duplicate leakage audit: eval sentences against the retrieval pool.
Usage:
    python -m src.retrieval.leakage --config configs/sparse_retrieval.yaml --split val test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.data.split import normalize_key, sha256_file
from src.retrieval.embed import embed_queries
from src.retrieval.retrieve import RetrievalIndex


def char_ngrams(text: str, n: int = 4) -> set[str]:
    """Character n-grams of the match-normalized text."""
    key = normalize_key(text)
    if len(key) < n:
        return {key} if key else set()
    return {key[i : i + n] for i in range(len(key) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def audit(
    index: RetrievalIndex,
    rows: list[dict],
    *,
    cos_thr: float = 0.95,
    jac_thr: float = 0.7,
    char_n: int = 4,
    top_m: int = 10,
    batch: int = 256,
) -> tuple[list[dict], np.ndarray]:
    """Flag eval rows with a near-duplicate in the pool."""
    pool_src = [char_ngrams(p["input"], char_n) for p in index.pairs]
    pool_tgt = [char_ngrams(p["output"], char_n) for p in index.pairs]

    flags: list[dict] = []
    max_cos = np.zeros(len(rows))
    model = index._model_lazy()
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        q = embed_queries(model, [r["input"] for r in chunk]).astype(np.float32)
        sims = q @ index.embeddings.T  # [B, N] cosine
        top = np.argsort(-sims, axis=1)[:, :top_m]
        for j, row in enumerate(chunk):
            i = start + j
            max_cos[i] = float(sims[j].max())
            src_ng = char_ngrams(row["input"], char_n)
            tgt_ng = char_ngrams(row["output"], char_n)
            for p in top[j]:
                cos = float(sims[j, p])
                js = jaccard(src_ng, pool_src[p])
                jt = jaccard(tgt_ng, pool_tgt[p])
                if cos <= cos_thr and js <= jac_thr and jt <= jac_thr:
                    continue
                flags.append(
                    {
                        "eval_row": i,
                        "pool_row": int(p),
                        "cos": round(cos, 4),
                        "jaccard_source": round(js, 4),
                        "jaccard_target": round(jt, 4),
                        "eval_source": row["input"],
                        "pool_source": index.pairs[p]["input"],
                        "eval_target": row["output"],
                        "pool_target": index.pairs[p]["output"],
                    }
                )
    return flags, max_cos


def load_quarantine(
    path: str | Path, train_file: str | Path, *, allow_test: bool = False
) -> list[int]:
    """Read a quarantine list, refusing one written against a different pool."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "test" in payload.get("splits", []) and not allow_test:
        raise ValueError(
            f"quarantine list {path} was audited against the sealed test split "
            f"({payload['splits']}); a val-time pool must not be pruned with test "
            "near-duplicates. Re-run: python manage.py leakage --split val --write-quarantine"
        )
    actual = sha256_file(Path(train_file))
    if payload.get("train_sha256") != actual:
        raise ValueError(
            f"quarantine list {path} was written against a different {train_file} "
            f"({payload.get('train_sha256', '?')[:12]} != {actual[:12]}). "
            "Re-run: python manage.py leakage --write-quarantine"
        )
    return sorted(payload["pool_rows"])


def _read_jsonl(path: Path, limit: int | None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def _histogram(values: np.ndarray) -> dict[str, int]:
    edges = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1.01]
    counts, _ = np.histogram(values, bins=edges)
    return {f"{edges[i]:.2f}-{edges[i + 1]:.2f}": int(c) for i, c in enumerate(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit eval splits for pool near-duplicates.")
    parser.add_argument("--config", default="configs/sparse_retrieval.yaml")
    parser.add_argument("--split", nargs="+", default=["val"])
    parser.add_argument("--index_dir", default=None)
    parser.add_argument("--cos", type=float, default=None)
    parser.add_argument("--jaccard", type=float, default=None)
    parser.add_argument("--char_n", type=int, default=None)
    parser.add_argument("--top_m", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--write-quarantine", action="store_true")
    parser.add_argument(
        "--unseal-test",
        action="store_true",
        help="permit auditing data/splits/test.jsonl; only at the final test pass",
    )
    args = parser.parse_args()

    if "test" in args.split and not args.unseal_test:
        raise SystemExit(
            "refusing to audit the sealed test split; drop it from --split, or pass "
            "--unseal-test at the final pass"
        )

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    retr, leak = cfg["retrieval"], cfg.get("leakage", {})
    index_dir = args.index_dir or retr["index_dir"]
    cos_thr = args.cos if args.cos is not None else leak.get("cos", 0.95)
    jac_thr = args.jaccard if args.jaccard is not None else leak.get("jaccard", 0.7)
    char_n = args.char_n or leak.get("char_n", 4)
    top_m = args.top_m or leak.get("top_m", 10)

    index = RetrievalIndex(index_dir, embed_model=retr["embed_model"])
    out_dir = Path(cfg.get("output", {}).get("dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    quarantined: set[int] = set()
    for split in args.split:
        rows = _read_jsonl(Path(f"data/splits/{split}.jsonl"), args.limit)
        print(f"Auditing {len(rows)} {split} rows against {len(index.pairs)} pool rows ...")
        flags, max_cos = audit(
            index, rows, cos_thr=cos_thr, jac_thr=jac_thr, char_n=char_n, top_m=top_m
        )
        pool_rows = sorted({f["pool_row"] for f in flags})
        quarantined |= set(pool_rows)

        payload = {
            "config": {
                "split": split,
                "index_dir": index_dir,
                "cos": cos_thr,
                "jaccard": jac_thr,
                "char_n": char_n,
                "top_m": top_m,
            },
            "n_eval_rows": len(rows),
            "n_pool_rows": len(index.pairs),
            "n_flags": len(flags),
            "n_eval_rows_flagged": len({f["eval_row"] for f in flags}),
            "n_pool_rows_flagged": len(pool_rows),
            "max_cos_histogram": _histogram(max_cos),
            "flags": sorted(flags, key=lambda f: -f["cos"]),
        }
        path = out_dir / f"leakage_{split}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"  {payload['n_eval_rows_flagged']}/{len(rows)} eval rows flagged, "
            f"{len(pool_rows)} pool rows implicated -> {path}"
        )

    if args.write_quarantine:
        train_file = retr["train_file"]
        q_path = Path(leak.get("quarantine_file", "data/splits/pool_quarantine.json"))
        q_path.write_text(
            json.dumps(
                {
                    "train_file": str(train_file),
                    "train_sha256": sha256_file(Path(train_file)),
                    "splits": args.split,
                    "thresholds": {"cos": cos_thr, "jaccard": jac_thr, "char_n": char_n},
                    "pool_rows": sorted(quarantined),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        share = len(quarantined) / len(index.pairs)
        print(f"Quarantined {len(quarantined)} pool rows ({share:.2%}) -> {q_path}")


if __name__ == "__main__":
    main()
