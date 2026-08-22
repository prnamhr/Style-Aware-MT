"""
Tests for the irregular-term list: tokenization and the document-frequency band.
"""

from __future__ import annotations

import random

import pytest

from src.retrieval.rarity import (
    df_histogram,
    irregular_terms,
    review_sample,
    term_stats,
    tokenize,
    zwnj_collisions,
)

ZWNJ = "‌"


def test_zwnj_modes_split_the_same_word_differently():
    text = f"می{ZWNJ}رود"
    assert tokenize(text, "keep") == [f"می{ZWNJ}رود"]
    assert tokenize(text, "strip") == ["میرود"]
    assert tokenize(text, "split") == ["می", "رود"]


def test_diacritics_and_digits_are_dropped():
    # Persian digits live in the Arabic block, so the script test alone lets them through.
    assert tokenize("بِسْمِ ۱۲۳ abc کتاب") == ["بسم", "کتاب"]


def test_unknown_zwnj_mode_rejected():
    with pytest.raises(ValueError, match="zwnj must be one of"):
        tokenize("کتاب", "fold")


def _banded_docs() -> list[str]:
    """25 documents placing one term at each of df 1, 2, 20, 21, and one at df 25."""
    docs = []
    for i in range(25):
        toks = ["کتاب"]
        toks += ["یک"] * (i < 1)
        toks += ["دو"] * (i < 2)
        toks += ["بیست"] * (i < 20)
        toks += ["بیستویک"] * (i < 21)
        docs.append(" ".join(toks))
    return docs


def test_band_excludes_both_edges():
    picked = irregular_terms(term_stats(_banded_docs()), df_min=2, df_max=20)

    assert set(picked) == {"دو", "بیست"}


def test_band_edges_are_inclusive():
    stats = term_stats(_banded_docs())

    assert set(irregular_terms(stats, df_min=1, df_max=1)) == {"یک"}
    assert "بیستویک" in irregular_terms(stats, df_min=2, df_max=21)


def test_inverted_band_rejected():
    with pytest.raises(ValueError, match="df_min <= df_max"):
        irregular_terms(term_stats(["کتاب الف"]), df_min=5, df_max=2)


def test_selection_is_deterministic_under_document_shuffling():
    """Terms inside the band tie in IDF at each df, so the written order must not
    depend on the order sklearn happened to see the documents in."""
    docs = [f"کتاب واژه{i} نام{i % 4}" for i in range(40)]
    shuffled = list(docs)
    random.Random(0).shuffle(shuffled)

    a = irregular_terms(term_stats(docs), df_min=2, df_max=20)
    b = irregular_terms(term_stats(shuffled), df_min=2, df_max=20)
    assert list(a) == list(b) and a


def test_df_histogram_buckets_cover_every_term():
    stats = term_stats([f"کتاب واژه{i}" for i in range(5)])
    assert sum(df_histogram(stats.df).values()) == len(stats.terms)


def test_zwnj_collisions_pair_the_split_and_joined_spellings():
    terms = [f"بی{ZWNJ}خبر", "بیخبر", f"بی{ZWNJ}مثال", "کتاب"]

    # "بی‌مثال" has no joined counterpart in this vocabulary, so it is not a collision.
    assert zwnj_collisions(terms) == [(f"بی{ZWNJ}خبر", "بیخبر")]


def test_review_sample_is_seeded_and_leads_with_the_least_rare():
    irregular = {f"واژه{i}": 9.0 for i in range(20)}
    df_of = {t: 1 for t in irregular}
    df_of["واژه7"] = 3

    assert review_sample(irregular, df_of, 5) == review_sample(irregular, df_of, 5)
    assert len(review_sample(irregular, df_of, 5)) == 5
    # The sample is drawn first, then ordered by falling df so real vocabulary leads.
    assert review_sample(irregular, df_of, len(irregular))[0] == "واژه7"
