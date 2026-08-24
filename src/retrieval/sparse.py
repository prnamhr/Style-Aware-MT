"""
Sparse retrieval channel: one nearest exemplar per rare query term, cosine kNN for the rest.

Usage:
    python -m src.retrieval.sparse --config configs/sparse_retrieval.yaml --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import yaml

from src.retrieval.rarity import load_irregular, tokenize
from src.retrieval.retrieve import RetrievalIndex

# Rarity slots filled, against the m the query was eligible for.
ROUTES = ("full", "partial", "dense")


def _pair_key(pair: dict) -> tuple[str, str]:
    return (pair["input"], pair["output"])


def _fallback_select(fallback, queries: list[str], k: int) -> list[list[dict]]:
    """Call whichever selection method the dense retriever exposes."""
    if hasattr(fallback, "select"):
        return fallback.select(queries, k=k)
    return fallback.retrieve(queries, k=k)


class SparseRetriever:
    """Up to ``m`` exemplars, one per rare query term, with the rest filled by cosine kNN."""

    def __init__(
        self,
        index: RetrievalIndex,
        irregular: dict[str, int],
        fallback,
        *,
        zwnj: str = "keep",
        m: int = 4,
    ):
        self.index = index
        self.fallback = fallback
        self.zwnj = zwnj
        self.m = int(m)

        # Column order is rarity rank, so a sorted column list is sorted rarest first.
        self.terms: list[str] = sorted(irregular, key=irregular.__getitem__)
        self._col = {t: i for i, t in enumerate(self.terms)}

        rows, cols = [], []
        self._row_of: dict[tuple[str, str], int] = {}
        for r, pair in enumerate(index.pairs):
            self._row_of.setdefault(_pair_key(pair), r)
            hits = {self._col[t] for t in tokenize(pair["input"], self.zwnj) if t in self._col}
            rows.extend([r] * len(hits))
            cols.extend(hits)
        shape = (len(index.pairs), len(self.terms))
        self.postings = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=shape
        )
        self._by_term = self.postings.tocsc()

    def query_terms(self, query: str) -> list[int]:
        """Column indices of the frozen terms the query carries, deduplicated, rarest first."""
        seen = {self._col[t] for t in tokenize(query, self.zwnj) if t in self._col}
        return sorted(seen)

    def targeted_terms(self, cols: list[int], slots: int | None = None) -> list[int]:
        """The rarest of them, at most one per rare slot: the terms an exemplar is fetched for."""
        return cols[: self.m if slots is None else slots]

    def _rows_with(self, col: int) -> np.ndarray:
        m = self._by_term
        return m.indices[m.indptr[col] : m.indptr[col + 1]]

    def _nearest_per_term(self, cols: list[int], qemb: np.ndarray) -> tuple[list[int], list[int]]:
        """For each targeted term, the most similar pool row carrying it that is still free."""
        picked, served, taken = [], [], set()
        for c in cols:
            rows = self._rows_with(c)
            if rows.size == 0:
                continue
            sims = self.index.embeddings[rows] @ qemb
            for j in np.argsort(-sims, kind="stable"):
                row = int(rows[j])
                key = _pair_key(self.index.pairs[row])
                # One training example fills one slot, however many rare terms it carries.
                if key not in taken:
                    taken.add(key)
                    picked.append(row)
                    served.append(c)
                    break
        return picked, served

    def select_with_trace(self, queries: list[str], k: int) -> tuple[list[list[dict]], list[dict]]:
        """Exemplars per query, with the routing record behind each."""
        slots = min(self.m, k)
        cols_per_query = [self.query_terms(q) for q in queries]
        targeted = [self.targeted_terms(c, slots) for c in cols_per_query]
        qemb = self.index.encode(queries)
        rare = [self._nearest_per_term(t, qemb[i]) for i, t in enumerate(targeted)]

        # One batched fallback call covers both the no-match and short-fill cases.
        needs_fallback = [i for i, (rows, _) in enumerate(rare) if len(rows) < k]
        filler: dict[int, list[dict]] = {}
        if needs_fallback:
            got = _fallback_select(self.fallback, [queries[i] for i in needs_fallback], k + slots)
            filler = dict(zip(needs_fallback, got))

        selected, traces = [], []
        for i, (rows, served) in enumerate(rare):
            picked = list(rows)
            seen = {_pair_key(self.index.pairs[r]) for r in rows}
            n_sparse = len(picked)
            for cand in filler.get(i, []):
                if len(picked) >= k:
                    break
                key = _pair_key(cand)
                if key not in seen:
                    seen.add(key)
                    picked.append(self._row_of[key])
            # Both arms hand order_exemplars a cosine-ranked list, so only membership differs.
            sims = self.index.embeddings[picked] @ qemb[i]
            final = [picked[j] for j in np.argsort(-sims, kind="stable")]
            if n_sparse == 0:
                route = "dense"
            elif n_sparse >= slots:
                route = "full"
            else:
                route = "partial"
            selected.append([self.index.pairs[r] for r in final])
            traces.append(
                {
                    "route": route,
                    "n_query_terms": len(cols_per_query[i]),
                    "n_targeted": len(targeted[i]),
                    "n_sparse": n_sparse,
                    "n_knn": len(picked) - n_sparse,
                    "query_terms": [self.terms[c] for c in cols_per_query[i]],
                    "targeted_terms": [self.terms[c] for c in targeted[i]],
                    "served_terms": [self.terms[c] for c in served],
                    "sparse_rows": rows,
                    "final_rows": final,
                }
            )
        return selected, traces

    def select(self, queries: list[str], k: int) -> list[list[dict]]:
        """Drop-in for :meth:`AFSPRetriever.select`, discarding the trace."""
        return self.select_with_trace(queries, k)[0]


def mean_intra_similarity(index: RetrievalIndex, selections: list[list[dict]]) -> float:
    """Mean pairwise cosine within each exemplar set: the redundancy readout."""
    row_of = {}
    for i, pair in enumerate(index.pairs):
        row_of.setdefault(_pair_key(pair), i)
    scores = []
    for pairs in selections:
        rows = [row_of[_pair_key(p)] for p in pairs if _pair_key(p) in row_of]
        if len(rows) < 2:
            continue
        sims = index.embeddings[rows] @ index.embeddings[rows].T
        iu = np.triu_indices(len(rows), k=1)
        scores.append(float(sims[iu].mean()))
    return float(np.mean(scores)) if scores else float("nan")


def _read_jsonl(path: Path, limit: int | None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run the sparse retrieval channel.")
    parser.add_argument("--config", default="configs/sparse_retrieval.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--index_dir", default=None)
    parser.add_argument("--rarity", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--m", type=int, default=None, help="rare slots; the rest fill by cosine")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--out", default=None, help="override the report path, e.g. for a sweep")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    retr, spa, rar = cfg["retrieval"], cfg["sparse"], cfg["rarity"]
    index_dir = args.index_dir or retr["index_dir"]
    k = args.k if args.k is not None else retr["k"]
    m = args.m if args.m is not None else spa.get("m", 4)

    eval_file = Path(f"data/splits/{args.split}.jsonl")
    rows = _read_jsonl(eval_file, args.limit or cfg["data"].get("limit"))
    sources = [r["input"] for r in rows]

    index = RetrievalIndex(index_dir, embed_model=retr["embed_model"])
    irregular = load_irregular(args.rarity or rar.get("out", "results/rarity_train.json"))
    retriever = SparseRetriever(index, irregular, index, zwnj=rar.get("zwnj", "keep"), m=m)

    print(f"Selecting k={k} (up to m={m} rare) for {len(sources)} {args.split} sources ...")
    selected, traces = retriever.select_with_trace(sources, k)
    print("Dense-only baseline for the redundancy comparison ...")
    baseline = index.retrieve(sources, k=k)

    routes = {r: sum(t["route"] == r for t in traces) for r in ROUTES}
    n_terms = np.array([t["n_query_terms"] for t in traces])
    filled = np.array([t["n_sparse"] for t in traces])
    targeted = np.array([t["n_targeted"] for t in traces])
    eligible = {str(thr): round(float((n_terms >= thr).mean()), 4) for thr in range(1, 7)}
    payload = {
        "config": {
            "split": args.split,
            "index_dir": index_dir,
            "k": k,
            "m": m,
            "n_frozen": len(irregular),
        },
        "n_queries": len(sources),
        "routes": routes,
        "route_fractions": {r: round(n / len(sources), 4) for r, n in routes.items()},
        "query_terms": {
            "mean": round(float(n_terms.mean()), 3),
            "deciles": [int(q) for q in np.quantile(n_terms, np.arange(0, 1.1, 0.1))],
            "share_at_or_above": eligible,
        },
        "n_sparse": {
            "slots": min(m, k),
            "mean": round(float(filled.mean()), 3),
            "histogram": {str(v): int((filled == v).sum()) for v in range(min(m, k) + 1)},
        },
        # A targeted term goes unserved only when every pool row carrying it is already taken.
        "terms_served": round(float(filled.sum() / targeted.sum()), 4) if targeted.sum() else None,
        "intra_set_similarity": {
            "sparse": round(mean_intra_similarity(index, selected), 4),
            "dense_baseline": round(mean_intra_similarity(index, baseline), 4),
        },
        "examples": [
            {
                "source": sources[i],
                "trace": traces[i],
                "exemplars": [p["input"] for p in selected[i]],
            }
            for i in [j for j, t in enumerate(traces) if t["route"] != "dense"][: args.examples]
        ],
    }

    out_dir = Path(cfg.get("output", {}).get("dir", "results"))
    out_path = Path(args.out) if args.out else out_dir / f"sparse_selection_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  routes: {payload['route_fractions']}")
    print(
        f"  rare slots filled: mean {payload['n_sparse']['mean']}, "
        f"{payload['n_sparse']['histogram']}"
    )
    print(f"  queries with >= n rare terms: {eligible}")
    print(f"  targeted terms served: {payload['terms_served']}")
    print(f"  intra-set cosine: {payload['intra_set_similarity']}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
