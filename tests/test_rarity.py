"""
Tests for the frozen rarity list: tokenization, the min_df floor, and the fixed size.
"""

from __future__ import annotations

import random

import pytest

from src.retrieval.rarity import (
    df_histogram,
    frozen_terms,
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


def test_terms_below_min_df_are_excluded():
    picked = frozen_terms(term_stats(_banded_docs()), min_df=2, freeze_n=500)

    assert "یک" not in picked
    assert set(picked) == {"دو", "بیست", "بیستویک", "کتاب"}


def test_the_list_is_truncated_to_freeze_n_rarest_first():
    picked = frozen_terms(term_stats(_banded_docs()), min_df=2, freeze_n=2)

    # df 2 and df 20 are the two rarest at or above the floor; df 21 and 25 are cut.
    assert list(picked) == ["دو", "بیست"]


def test_the_list_is_exactly_freeze_n_when_the_vocabulary_allows():
    docs = [f"کتاب واژه{i % 60} نام{i % 7}" for i in range(120)]
    freeze_n = 40
    stats = term_stats(docs)

    picked = frozen_terms(stats, min_df=2, freeze_n=freeze_n)
    assert len(picked) == freeze_n
    assert int((stats.df >= 2).sum()) > freeze_n


def test_a_short_vocabulary_yields_fewer_than_freeze_n():
    picked = frozen_terms(term_stats(["کتاب الف", "کتاب ب"]), min_df=2, freeze_n=500)

    assert set(picked) == {"کتاب"}


@pytest.mark.parametrize(("min_df", "freeze_n"), [(0, 500), (2, 0)])
def test_out_of_range_arguments_rejected(min_df, freeze_n):
    with pytest.raises(ValueError, match="need (min_df|freeze_n) >="):
        frozen_terms(term_stats(["کتاب الف"]), min_df=min_df, freeze_n=freeze_n)


def test_selection_is_deterministic_under_document_shuffling():
    """Terms tie in IDF at each df, so which of them the truncation keeps must not
    depend on the order sklearn happened to see the documents in."""
    docs = [f"کتاب واژه{i} نام{i % 4}" for i in range(40)]
    shuffled = list(docs)
    random.Random(0).shuffle(shuffled)

    # The four نام terms tie at df 10, so the truncation to 3 is decided by the tie-break alone.
    a = frozen_terms(term_stats(docs), min_df=2, freeze_n=3)
    b = frozen_terms(term_stats(shuffled), min_df=2, freeze_n=3)
    assert list(a) == list(b) == ["نام0", "نام1", "نام2"]


def test_df_histogram_buckets_cover_every_term():
    stats = term_stats([f"کتاب واژه{i}" for i in range(5)])
    assert sum(df_histogram(stats.df).values()) == len(stats.terms)


def test_zwnj_collisions_pair_the_split_and_joined_spellings():
    terms = [f"بی{ZWNJ}خبر", "بیخبر", f"بی{ZWNJ}مثال", "کتاب"]

    # "بی‌مثال" has no joined counterpart in this vocabulary, so it is not a collision.
    assert zwnj_collisions(terms) == [(f"بی{ZWNJ}خبر", "بیخبر")]


def test_review_sample_is_seeded_and_leads_with_the_least_rare():
    frozen = {f"واژه{i}": 9.0 for i in range(20)}
    df_of = {t: 1 for t in frozen}
    df_of["واژه7"] = 3

    assert review_sample(frozen, df_of, 5) == review_sample(frozen, df_of, 5)
    assert len(review_sample(frozen, df_of, 5)) == 5
    # The sample is drawn first, then ordered by falling df so real vocabulary leads.
    assert review_sample(frozen, df_of, len(frozen))[0] == "واژه7"
