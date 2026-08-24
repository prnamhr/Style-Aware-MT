"""
Tests for the sparse channel: the rare-term cap, one exemplar per term, and the kNN fill.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.sparse import SparseRetriever

# term -> rarity rank, rarest first.
IRREGULAR = {t: r for r, t in enumerate(("الف", "ب", "ج", "د", "ه"))}


class _Index:
    def __init__(self, sources: list[str], embeddings, query_embedding=None):
        self.pairs = [{"input": s, "output": f"en:{s}"} for s in sources]
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        # Zeros leave every cosine tied, so a test opts in to a similarity ordering.
        self.query = (
            np.zeros(self.embeddings.shape[1])
            if query_embedding is None
            else np.asarray(query_embedding, dtype=np.float32)
        )

    def encode(self, queries):
        return np.tile(self.query, (len(queries), 1))


class _Fallback:
    """Stands in for the dense retriever: returns the same ranked pairs for every query."""

    def __init__(self, pairs: list[dict]):
        self.pairs = pairs
        self.calls: list[int] = []

    def select(self, queries, k):
        self.calls.append(k)
        return [self.pairs[:k] for _ in queries]


def _build(sources, embeddings, fallback_rows=(), query_embedding=None, **kw):
    index = _Index(sources, embeddings, query_embedding)
    fallback = _Fallback([index.pairs[r] for r in fallback_rows])
    return SparseRetriever(index, IRREGULAR, fallback, **kw), index, fallback


def test_query_with_no_rare_term_gets_k_cosine_exemplars():
    retriever, _, fallback = _build(["الف", "ب", "ج"], np.eye(3), fallback_rows=(0, 1, 2))
    selected, traces = retriever.select_with_trace(["واژه دیگر"], k=3)

    assert traces[0]["n_query_terms"] == 0
    assert traces[0]["route"] == "dense"
    assert len(selected[0]) == 3
    assert fallback.calls == [6]  # k + slots, so the de-duplicating fill has slack


def test_each_rare_term_takes_its_most_similar_training_example():
    # "الف" sits in rows 0 and 2; row 2 is the one aligned with the query.
    sources = ["الف", "ب", "الف"]
    embeddings = [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    retriever, _, _ = _build(sources, embeddings, query_embedding=[1.0, 0.0], m=1)

    _, traces = retriever.select_with_trace(["الف"], k=1)

    assert traces[0]["sparse_rows"] == [2]


def test_a_training_example_is_not_used_twice():
    # Row 0 carries both terms. It answers "الف", so "ب" has to fall to its next best row.
    sources = ["الف ب", "ب", "ج"]
    retriever, _, _ = _build(sources, np.eye(3), m=2)

    _, traces = retriever.select_with_trace(["الف ب"], k=2)

    assert traces[0]["sparse_rows"] == [0, 1]
    assert traces[0]["served_terms"] == ["الف", "ب"]


def test_a_term_with_no_pool_example_leaves_its_slot_to_the_fill():
    # "الف" is on the list but absent from the pool, so only "ج" is served.
    sources = ["ج", "واژه"]
    retriever, _, _ = _build(sources, np.eye(2), fallback_rows=(1, 0), m=2)

    selected, traces = retriever.select_with_trace(["الف ج"], k=2)

    assert traces[0]["n_targeted"] == 2
    assert traces[0]["n_sparse"] == 1
    assert traces[0]["served_terms"] == ["ج"]
    assert len(selected[0]) == 2


def test_the_targeted_terms_are_the_m_rarest_the_query_carries():
    index = _Index(["الف", "ب", "ج", "د", "ه"], np.eye(5))
    retriever = SparseRetriever(index, IRREGULAR, _Fallback([]), m=4)

    cols = retriever.query_terms("ه د ج ب الف")
    targeted = retriever.targeted_terms(cols)

    assert len(cols) == 5
    # "ه" is the least rare of the five, so it is the one dropped.
    assert [retriever.terms[c] for c in targeted] == ["الف", "ب", "ج", "د"]


def test_the_rarest_term_is_served_first():
    """Rows 0 and 1 both carry the rarer term and the less rare one, so whichever term
    is served first takes the better row. Rank order, not query order, decides."""
    sources = ["ب الف", "الف ب"]
    retriever, _, _ = _build(sources, np.eye(2), m=2)

    _, traces = retriever.select_with_trace(["ب الف"], k=2)

    assert traces[0]["served_terms"] == ["الف", "ب"]
    assert traces[0]["sparse_rows"] == [0, 1]


@pytest.mark.parametrize(("n_matches", "expected_rare"), [(0, 0), (1, 1), (2, 2), (3, 3), (5, 4)])
def test_the_prompt_holds_k_exemplars_however_many_rare_terms_matched(n_matches, expected_rare):
    rare_rows = ["الف", "ب", "ج", "د", "ه"]
    sources = rare_rows + [f"واژه{i}" for i in range(7)]
    retriever, _, _ = _build(sources, np.eye(12), fallback_rows=tuple(range(5, 12)) + (0, 1, 2, 3))

    query = " ".join(rare_rows[:n_matches])
    selected, traces = retriever.select_with_trace([query], k=8)

    inputs = [p["input"] for p in selected[0]]
    assert traces[0]["n_sparse"] == expected_rare
    assert traces[0]["n_knn"] == 8 - expected_rare
    assert len(inputs) == len(set(inputs)) == 8


def test_the_fill_does_not_repeat_a_rare_pick():
    sources = ["الف ب", "ج د", "ه"]
    retriever, _, _ = _build(sources, np.eye(3), fallback_rows=(0, 2), m=2)

    selected, traces = retriever.select_with_trace(["الف ج"], k=3)

    inputs = [p["input"] for p in selected[0]]
    assert traces[0]["route"] == "full"
    assert traces[0]["n_sparse"] == 2
    assert len(inputs) == len(set(inputs)) == 3
    assert "ه" in inputs  # row 0 was already a rare pick, so the fill skips it


def test_route_labels_follow_the_filled_slots():
    sources = ["الف", "ب", "ج", "د", "ه"]
    retriever, _, _ = _build(sources, np.eye(5), fallback_rows=(0, 1, 2, 3, 4), m=2)

    _, traces = retriever.select_with_trace(["الف ب ج", "الف", "واژه"], k=4)

    assert [t["route"] for t in traces] == ["full", "partial", "dense"]
    assert [t["n_sparse"] for t in traces] == [2, 1, 0]


def test_final_order_is_cosine_ranked_across_both_channels():
    # Row 2 carries no rare term but is nearest the query, so the fill outranks the rare pick.
    sources = ["الف", "ب", "ج د"]
    embeddings = [[0.6, 0.8], [0.0, 1.0], [1.0, 0.0]]
    retriever, _, _ = _build(
        sources, embeddings, fallback_rows=(2, 0, 1), query_embedding=[1.0, 0.0], m=2
    )
    selected, traces = retriever.select_with_trace(["الف ب"], k=3)

    assert traces[0]["n_sparse"] == 2
    assert traces[0]["final_rows"] == [2, 0, 1]
    assert [p["input"] for p in selected[0]] == ["ج د", "الف", "ب"]
