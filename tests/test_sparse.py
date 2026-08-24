"""
Tests for the sparse channel: routing, the query-term cap, greedy coverage, and redundancy.
"""

from __future__ import annotations

import numpy as np

from src.retrieval.sparse import SparseRetriever

IRREGULAR = {t: 3.0 for t in ("الف", "ب", "ج", "د", "ه")}


class _Index:
    def __init__(self, sources: list[str], embeddings, query_embedding=None):
        self.pairs = [{"input": s, "output": f"en:{s}"} for s in sources]
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        # Zeros leave the final cosine sort a stable no-op, so a test opts in to reordering.
        self.query = (
            np.zeros(self.embeddings.shape[1])
            if query_embedding is None
            else np.asarray(query_embedding, dtype=np.float32)
        )

    def encode(self, queries):
        return np.tile(self.query, (len(queries), 1))


class _Fallback:
    """Stands in for AFSPRetriever: returns the same ranked pairs for every query."""

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


def test_short_query_routes_to_the_dense_fallback():
    retriever, index, fallback = _build(
        ["الف ب", "ج د"], np.eye(2), fallback_rows=(1, 0), min_query_terms=4
    )
    selected, traces = retriever.select_with_trace(["الف ب"], k=2)

    assert traces[0]["route"] == "dense"
    assert traces[0]["n_sparse"] == 0
    assert [p["input"] for p in selected[0]] == ["ج د", "الف ب"]
    assert fallback.calls == [4]  # asks for 2k so the de-duplicating fill has slack


def test_greedy_prefers_the_candidate_covering_most_rarity():
    sources = ["الف ب", "الف", "ج د"]
    retriever, _, _ = _build(sources, np.eye(3), min_query_terms=4, redundancy=0.0)
    selected, traces = retriever.select_with_trace(["الف ب ج د"], k=2)

    assert [p["input"] for p in selected[0]] == ["الف ب", "ج د"]
    assert traces[0]["route"] == "full"
    assert traces[0]["coverage"] == 1.0


def test_redundancy_penalty_breaks_a_coverage_tie_toward_the_dissimilar_candidate():
    # rows 1 and 2 cover the same terms; row 1 is a near-copy of the already-chosen row 0.
    sources = ["الف ب", "ج د", "ج د"]
    embeddings = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    query = ["الف ب ج د"]

    flat, _, _ = _build(sources, embeddings, min_query_terms=4, redundancy=0.0)
    penalised, _, _ = _build(sources, embeddings, min_query_terms=4, redundancy=0.3)

    assert flat.select_with_trace(query, k=2)[1][0]["sparse_rows"] == [0, 1]
    assert penalised.select_with_trace(query, k=2)[1][0]["sparse_rows"] == [0, 2]


def test_short_coverage_fills_from_the_fallback_without_repeating():
    sources = ["الف ب", "ج د", "ه"]
    retriever, index, fallback = _build(
        sources, np.eye(3), fallback_rows=(0, 2), min_query_terms=4, redundancy=0.0
    )
    selected, traces = retriever.select_with_trace(["الف ب ج د"], k=3)

    inputs = [p["input"] for p in selected[0]]
    assert traces[0]["route"] == "partial"
    assert traces[0]["n_sparse"] == 2
    assert len(inputs) == len(set(inputs)) == 3
    assert inputs[2] == "ه"  # row 0 was already selected, so the fill skips it


def test_query_with_no_pool_match_routes_dense():
    retriever, _, _ = _build(["الف ب"], np.eye(1), fallback_rows=(0,), min_query_terms=1)
    _, traces = retriever.select_with_trace(["ج د ه"], k=2)

    assert traces[0]["route"] == "dense"
    assert traces[0]["n_query_terms"] == 3


def test_sparse_slots_are_capped_at_m():
    sources = ["الف", "ب", "ج", "د", "ه"]
    retriever, _, _ = _build(
        sources, np.eye(5), fallback_rows=(0, 1, 2, 3, 4), m=2, min_query_terms=1, redundancy=0.0
    )
    selected, traces = retriever.select_with_trace(["الف ب ج د ه"], k=4)

    assert traces[0]["n_sparse"] == 2
    assert traces[0]["route"] == "full"
    assert len(selected[0]) == 4  # the remaining k-m slots come from cosine top-k


def test_query_with_no_irregular_terms_gets_k_cosine_exemplars():
    retriever, _, _ = _build(
        ["الف", "ب", "ج"], np.eye(3), fallback_rows=(0, 1, 2), m=4, min_query_terms=1
    )
    selected, traces = retriever.select_with_trace(["واژه دیگر"], k=3)

    assert traces[0]["n_query_terms"] == 0
    assert traces[0]["route"] == "dense"
    assert len(selected[0]) == 3


def test_query_terms_are_capped_at_m_and_keep_the_rarest():
    # Five rare terms, four slots: the greedy is asked to cover only the four rarest.
    graded = {"الف": 9.0, "ب": 8.0, "ج": 7.0, "د": 6.0, "ه": 5.0}
    index = _Index(["الف", "ب", "ج", "د", "ه"], np.eye(5))
    retriever = SparseRetriever(
        index, graded, _Fallback([]), m=4, min_query_terms=1, redundancy=0.0
    )
    cols = retriever.query_terms("الف ب ج د ه")
    targeted = retriever.targeted_terms(cols)

    assert len(cols) == 5
    assert [retriever.terms[c] for c in targeted] == sorted(["الف", "ب", "ج", "د"])

    _, traces = retriever.select_with_trace(["الف ب ج د ه"], k=4)
    # "ه" is the least rare, so it is dropped and its exemplar is never a rarity pick.
    assert traces[0]["n_query_terms"] == 5 and traces[0]["n_targeted"] == 4
    assert traces[0]["targeted_terms"] == sorted(["الف", "ب", "ج", "د"])
    assert traces[0]["coverage"] == 1.0
    assert [index.pairs[r]["input"] for r in sorted(traces[0]["sparse_rows"])] == [
        "الف",
        "ب",
        "ج",
        "د",
    ]


def test_the_cap_ties_break_deterministically_on_the_term():
    # All five tie in IDF, so only the alphabetical secondary key decides the four kept.
    index = _Index(["الف"], np.eye(1))
    retriever = SparseRetriever(index, IRREGULAR, _Fallback([]), m=4, min_query_terms=1)
    targeted = retriever.targeted_terms(retriever.query_terms("الف ب ج د ه"))

    assert [retriever.terms[c] for c in targeted] == sorted(["الف", "ب", "ج", "د"])


def test_final_order_is_cosine_ranked_across_both_channels():
    # Row 2 carries no rare term but is nearest the query, so the fill outranks the rarity pick.
    sources = ["الف", "ب", "ج د"]
    embeddings = [[0.6, 0.8], [0.0, 1.0], [1.0, 0.0]]
    retriever, _, _ = _build(
        sources,
        embeddings,
        fallback_rows=(2, 0, 1),
        query_embedding=[1.0, 0.0],
        m=2,
        min_query_terms=1,
        redundancy=0.0,
    )
    selected, traces = retriever.select_with_trace(["الف ب"], k=3)

    assert traces[0]["n_sparse"] == 2
    assert traces[0]["final_rows"] == [2, 0, 1]
    assert [p["input"] for p in selected[0]] == ["ج د", "الف", "ب"]
