"""
Adaptive Few-Shot Prompting (AFSP): exemplar selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.stylometrics import features, register_salience
from src.retrieval.embed import embed_queries
from src.retrieval.retrieve import RetrievalIndex


def load_centroid(path: str | Path) -> dict:
    """Load the target-register centroid, or raise if it has not been built."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"target-register centroid not found at {p}. Build it first with:\n"
            "    python manage.py stylometrics --build-centroid"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def _topk_mean(matrix: np.ndarray, m: int) -> np.ndarray:
    """Row-wise mean of the ``m`` largest values in a 2-D array."""
    m = max(1, min(m, matrix.shape[1]))
    top = np.partition(matrix, -m, axis=1)[:, -m:]
    return top.mean(axis=1)


def _minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalisation to [0, 1]. A constant vector maps to zeros."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class AFSPRetriever:
    """Adaptive exemplar selection over a shared :class:`RetrievalIndex`."""

    def __init__(
        self,
        index: RetrievalIndex,
        centroid: dict | None,
        *,
        index_dir: str | Path | None = None,
        beta: float = 0.3,
        knn_hubness: int = 5,
        pool_mult: int = 4,
        lambda_style: float = 0.3,
    ):
        self.index = index
        self.centroid = centroid
        self.index_dir = Path(index_dir) if index_dir is not None else None
        self.beta = float(beta)
        self.knn_hubness = int(knn_hubness)
        self.pool_mult = max(1, int(pool_mult))
        self.lambda_style = float(lambda_style)

        n = self.index.embeddings.shape[0]
        self._chub: np.ndarray | None = None
        # Lazily filled per-candidate target-register distances (NaN = not computed).
        self._style_cache = np.full(n, np.nan, dtype=np.float64)

    # -- Margin-based scoring (mechanism 1) --------------------------------

    def _candidate_hubness(self, batch: int = 512) -> np.ndarray:
        """Mean cosine similarity of each index row to its ``knn_hubness`` nearest
        neighbours.

        Larger values indicate hub rows situated in dense regions of the
        embedding space. The computation proceeds in row batches to bound memory
        and excludes self-similarity. The result depends only on the index and
        ``knn_hubness``, and is cached under ``index_dir``.
        """
        if self._chub is not None:
            return self._chub

        e = self.index.embeddings
        n = e.shape[0]
        cache_path = (
            self.index_dir / f"hubness_top{self.knn_hubness}.npy"
            if self.index_dir is not None
            else None
        )
        if cache_path is not None and cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape == (n,):
                self._chub = cached
                return cached

        hub = np.empty(n, dtype=np.float64)
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            block = e[start:stop] @ e.T  # [b, N] cosine (rows are L2-normalized)
            # Exclude self-similarity before taking the neighbourhood mean.
            rows = np.arange(start, stop)
            block[np.arange(stop - start), rows] = -np.inf
            hub[start:stop] = _topk_mean(block, self.knn_hubness)

        self._chub = hub
        if cache_path is not None:
            np.save(cache_path, hub)
        return hub

    # -- Target-distribution-priority selection (mechanism 2) --------------

    def _style_fit(self, indices: np.ndarray) -> np.ndarray:
        """Register salience of the pooled candidates: the mean standardised deviation"""
        for i in indices:
            if np.isnan(self._style_cache[i]):
                target = self.index.pairs[i]["output"]
                self._style_cache[i] = register_salience(features(target), self.centroid)
        return self._style_cache[indices]

    # -- Selection ---------------------------------------------------------

    def select(self, queries: list[str], k: int) -> list[list[dict]]:
        e = self.index.embeddings
        q = embed_queries(self.index._model_lazy(), queries).astype(np.float32)
        sims = (q @ e.T).astype(np.float64)  # [Q, N] cosine

        if self.beta > 0:
            qhub = _topk_mean(sims, self.knn_hubness)  # [Q]
            chub = self._candidate_hubness()  # [N]
            margin = sims - self.beta * 0.5 * (qhub[:, None] + chub[None, :])
        else:
            margin = sims

        use_style = self.lambda_style > 0 and self.centroid is not None
        pool = min(self.pool_mult * k, e.shape[0])

        selected: list[list[dict]] = []
        for qi in range(sims.shape[0]):
            row = margin[qi]
            # The `pool` highest-margin candidates (unordered slice, ranked below).
            pool_idx = np.argpartition(-row, pool - 1)[:pool]
            sim_component = _minmax(row[pool_idx])
            if use_style:
                style_component = _minmax(self._style_fit(pool_idx))
            else:
                style_component = np.zeros(pool)
            combined = (1 - self.lambda_style) * sim_component + self.lambda_style * style_component
            # Descending order yields most-relevant-first (highest combined score first).
            order = np.argsort(-combined, kind="stable")
            chosen = pool_idx[order][:k]
            selected.append([self.index.pairs[i] for i in chosen])
        return selected
