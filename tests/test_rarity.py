"""
Tests for the irregular-term list: tokenization, the rarity cut, and its tie-break.
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


def test_rarity_cut_takes_the_rarest_terms():
    # "کتاب" appears in every document; the others appear once each.
    docs = [f"کتاب واژه{i}" for i in range(10)]
    picked = irregular_terms(term_stats(docs), top_frac=0.5)

    assert "کتاب" not in picked
    assert set(picked) == {f"واژه{i}" for i in range(10)}


def test_cut_extends_to_the_tie_boundary_rather_than_slicing_it():
    """Half this vocabulary is hapax, so a rank cut inside the tie block would
    return an alphabetical prefix rather than a rarity list."""
    docs = [f"کتاب واژه{i}" for i in range(10)]
    picked = irregular_terms(term_stats(docs), top_frac=0.1)

    # 10% of 11 terms is one term, but all ten hapax terms share that IDF.
    assert len(picked) == 10
    assert len(set(picked.values())) == 1


def test_min_df_excludes_the_hapax_tie_block():
    docs = ["کتاب الف", "کتاب الف", "کتاب واژه"]
    stats = term_stats(docs)
    assert "واژه" in irregular_terms(stats, top_frac=1.0, min_df=1)
    assert "واژه" not in irregular_terms(stats, top_frac=1.0, min_df=2)


def test_tie_break_is_deterministic_under_document_shuffling():
    """Most of the rarest 20% sits in one df==1 block, so the cut must not depend
    on the order sklearn happened to see the documents in."""
    docs = [f"کتاب واژه{i} نام{i}" for i in range(40)]
    shuffled = list(docs)
    random.Random(0).shuffle(shuffled)

    a = irregular_terms(term_stats(docs), top_frac=0.2)
    b = irregular_terms(term_stats(shuffled), top_frac=0.2)
    assert list(a) == list(b)


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
