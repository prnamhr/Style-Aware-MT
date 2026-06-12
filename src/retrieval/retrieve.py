"""
Load the retrieval index and fetch top-k exemplars for a source query.

Retrieval is brute-force cosine similarity over L2-normalized embeddings, which
is exact and instant at this corpus size. The query (Persian/Arabic source) is
embedded with the same model used to build the index; the instruct query prefix
is applied on the query side only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.retrieval.embed import embed_queries, load_model


class RetrievalIndex:
    def __init__(self, index_dir: str | Path, embed_model: str | None = None):
        index_dir = Path(index_dir)
        self.embeddings: np.ndarray = np.load(index_dir / "embeddings.npy")
        with (index_dir / "pairs.jsonl").open(encoding="utf-8") as f:
            self.pairs: list[dict] = [json.loads(line) for line in f if line.strip()]
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        self.embed_model_name = embed_model or meta["embed_model"]
        self._model = None

    def _model_lazy(self):
        if self._model is None:
            self._model = load_model(self.embed_model_name)
        return self._model

    def retrieve(self, queries: list[str], k: int) -> list[list[dict]]:
        """Return, per query, the top-k exemplar pairs ordered most-similar first."""
        q = embed_queries(self._model_lazy(), queries).astype(np.float32)
        # Embeddings are L2-normalized, so the dot product is cosine similarity.
        sims = q @ self.embeddings.T  # [Q, N]
        top = np.argsort(-sims, axis=1)[:, :k]
        return [[self.pairs[i] for i in row] for row in top]
