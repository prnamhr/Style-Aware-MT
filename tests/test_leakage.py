"""
Tests for the near-duplicate audit and the quarantine hash guard.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.retrieval.build_index import _check_not_overwriting
from src.retrieval.leakage import audit, char_ngrams, jaccard, load_quarantine


class _Index:
    """Stands in for RetrievalIndex with hand-written embeddings, so no model loads."""

    def __init__(self, pairs, embeddings):
        self.pairs = pairs
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def _model_lazy(self):
        return None


def _patch_embed(monkeypatch, vectors):
    monkeypatch.setattr(
        "src.retrieval.leakage.embed_queries",
        lambda model, texts: np.asarray(vectors, dtype=np.float32),
    )


def test_jaccard_of_identical_text_is_one():
    assert jaccard(char_ngrams("در نام خداوند"), char_ngrams("در نام خداوند")) == 1.0


def test_normalization_is_shared_with_the_split_dedup():
    # normalize_key strips punctuation and case, so these must produce the same n-grams.
    assert char_ngrams("در نام خداوند!") == char_ngrams("در نام خداوند")


def test_identical_source_is_flagged(monkeypatch):
    index = _Index(
        [{"input": "در نام خداوند", "output": "In the name of God"}],
        [[1.0, 0.0]],
    )
    _patch_embed(monkeypatch, [[1.0, 0.0]])
    rows = [{"input": "در نام خداوند", "output": "In the name of God"}]

    flags, max_cos = audit(index, rows, cos_thr=0.95, jac_thr=0.7, top_m=1)

    assert len(flags) == 1
    assert flags[0]["jaccard_source"] == 1.0
    assert flags[0]["jaccard_target"] == 1.0
    assert max_cos[0] == pytest.approx(1.0)


def test_unrelated_row_is_not_flagged(monkeypatch):
    index = _Index([{"input": "کتاب اقدس", "output": "The Most Holy Book"}], [[1.0, 0.0]])
    _patch_embed(monkeypatch, [[0.0, 1.0]])
    rows = [{"input": "بامداد روشن", "output": "A bright morning"}]

    flags, max_cos = audit(index, rows, cos_thr=0.95, jac_thr=0.7, top_m=1)

    assert flags == []
    assert max_cos[0] == pytest.approx(0.0)


def test_target_side_leak_is_flagged_when_the_source_reads_as_distinct(monkeypatch):
    """The reference can leak even when the source paraphrase does not."""
    index = _Index([{"input": "کتاب اقدس", "output": "In the name of God"}], [[1.0, 0.0]])
    _patch_embed(monkeypatch, [[0.0, 1.0]])
    rows = [{"input": "بامداد روشن", "output": "In the name of God"}]

    flags, _ = audit(index, rows, cos_thr=0.95, jac_thr=0.7, top_m=1)

    assert len(flags) == 1
    assert flags[0]["jaccard_source"] < 0.7
    assert flags[0]["jaccard_target"] == 1.0


def test_quarantine_written_against_a_different_pool_is_rejected(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text('{"input": "a", "output": "b"}\n', encoding="utf-8")
    q = tmp_path / "pool_quarantine.json"
    q.write_text(json.dumps({"train_sha256": "0" * 64, "pool_rows": [3]}), encoding="utf-8")

    with pytest.raises(ValueError, match="different"):
        load_quarantine(q, train)


def test_quarantined_build_refuses_to_overwrite_an_unquarantined_index(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"n_passages": 10}), encoding="utf-8")

    with pytest.raises(ValueError, match="separate directory"):
        _check_not_overwriting(tmp_path)


def test_rebuilding_a_quarantined_index_in_place_is_allowed(tmp_path):
    (tmp_path / "meta.json").write_text(
        json.dumps({"quarantine_file": "data/splits/pool_quarantine.json"}), encoding="utf-8"
    )

    _check_not_overwriting(tmp_path)
