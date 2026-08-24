"""
Tests for the frozen rarity list: tokenization, the df ranking, and its tie-breaks.
"""

from __future__ import annotations

import random

import pytest

from src.retrieval.rarity import (
    df_histogram,
    frozen_terms,
    load_irregular,
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


def test_the_list_runs_from_rarest_to_less_rare():
    picked = frozen_terms(term_stats(_banded_docs()), min_df=1, freeze_n=500)

    assert list(picked) == ["یک", "دو", "بیست", "بیستویک", "کتاب"]
    assert list(picked.values()) == [0, 1, 2, 3, 4]


def test_the_list_is_truncated_to_the_freeze_n_rarest():
    picked = frozen_terms(term_stats(_banded_docs()), min_df=1, freeze_n=2)

    assert list(picked) == ["یک", "دو"]


def test_terms_below_min_df_are_excluded():
    picked = frozen_terms(term_stats(_banded_docs()), min_df=2, freeze_n=500)

    assert "یک" not in picked
    assert list(picked)[0] == "دو"


def test_document_frequency_outranks_total_frequency():
    """One term occurs 10 times in a single sentence, the other once in each of two.
    Rarity is how many sentences carry a term, so the first is the rarer of the two."""
    docs = ["الف " * 10, "ب", "ب"]

    picked = frozen_terms(term_stats(docs), min_df=1, freeze_n=500)

    assert list(picked) == ["الف", "ب"]


def test_total_frequency_breaks_a_document_frequency_tie():
    # Both sit at df 2; "ج" occurs three times to "د" once, so "د" is the rarer.
    docs = ["ج ج د", "ج د"]

    picked = frozen_terms(term_stats(docs), min_df=1, freeze_n=500)

    assert list(picked) == ["د", "ج"]


def test_the_token_string_breaks_what_frequency_cannot():
    docs = ["ب الف", "ب الف"]

    picked = frozen_terms(term_stats(docs), min_df=1, freeze_n=500)

    # Identical df and tf, so only the deterministic token order is left to decide.
    assert list(picked) == sorted(["الف", "ب"])


def test_the_list_is_exactly_freeze_n_when_the_vocabulary_allows():
    docs = [f"کتاب واژه{i % 60} نام{i % 7}" for i in range(120)]
    freeze_n = 40
    stats = term_stats(docs)

    picked = frozen_terms(stats, min_df=1, freeze_n=freeze_n)
    assert len(picked) == freeze_n
    assert len(stats.terms) > freeze_n


def test_a_short_vocabulary_yields_fewer_than_freeze_n():
    picked = frozen_terms(term_stats(["کتاب الف", "کتاب ب"]), min_df=2, freeze_n=500)

    assert set(picked) == {"کتاب"}


@pytest.mark.parametrize(("min_df", "freeze_n"), [(0, 500), (1, 0)])
def test_out_of_range_arguments_rejected(min_df, freeze_n):
    with pytest.raises(ValueError, match="need (min_df|freeze_n) >="):
        frozen_terms(term_stats(["کتاب الف"]), min_df=min_df, freeze_n=freeze_n)


def test_selection_is_deterministic_under_document_shuffling():
    """Terms tie in df and tf in bulk, so which of them the truncation keeps must not
    depend on the order sklearn happened to see the documents in."""
    docs = [f"کتاب واژه{i} نام{i % 4}" for i in range(40)]
    shuffled = list(docs)
    random.Random(0).shuffle(shuffled)

    a = frozen_terms(term_stats(docs), min_df=1, freeze_n=10)
    b = frozen_terms(term_stats(shuffled), min_df=1, freeze_n=10)
    assert list(a) == list(b)


def test_the_written_list_reloads_as_the_same_ranking(tmp_path):
    import json

    frozen = frozen_terms(term_stats(_banded_docs()), min_df=1, freeze_n=3)
    path = tmp_path / "rarity.json"
    path.write_text(
        json.dumps({"terms": [[t, 1, 1] for t in frozen]}, ensure_ascii=False), encoding="utf-8"
    )

    assert load_irregular(path) == frozen


def test_df_histogram_buckets_cover_every_term():
    stats = term_stats([f"کتاب واژه{i}" for i in range(5)])
    assert sum(df_histogram(stats.df).values()) == len(stats.terms)


def test_zwnj_collisions_pair_the_split_and_joined_spellings():
    terms = [f"بی{ZWNJ}خبر", "بیخبر", f"بی{ZWNJ}مثال", "کتاب"]

    # "بی‌مثال" has no joined counterpart in this vocabulary, so it is not a collision.
    assert zwnj_collisions(terms) == [(f"بی{ZWNJ}خبر", "بیخبر")]
