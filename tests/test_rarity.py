"""
Tests for the frozen rarity list: tokenization, character surprisal, and the df band.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.retrieval.rarity import (
    char_surprisal,
    df_histogram,
    frozen_terms,
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


_SHAPED = ["abab", "abac", "abad", "xyzw"]


def test_surprisal_is_higher_for_the_term_off_the_dominant_pattern():
    scores = char_surprisal(_SHAPED, [100, 100, 100, 1])

    assert scores[3] > scores[0]


def test_surprisal_follows_the_token_frequency_weights():
    """The same four types, reweighted: whichever pattern carries the mass is the
    unremarkable one, so the ranking must invert with tf rather than with type counts."""
    common = char_surprisal(_SHAPED, [100, 100, 100, 1])
    rare = char_surprisal(_SHAPED, [1, 1, 1, 100])

    assert np.argmax(common) == 3
    assert rare[3] < rare[0]


def test_surprisal_is_a_per_character_mean():
    """A repeated pattern is scored per position, so lengthening a term with more of
    the same characters cannot make it arbitrarily more unusual."""
    scores = char_surprisal(["ab", "ababababab"], [1, 1], n=2)

    assert scores[1] < scores[0] * 2


@pytest.mark.parametrize(("terms", "tf", "n"), [(["ab"], [1, 2], 4), (["ab"], [1], 1)])
def test_surprisal_rejects_bad_arguments(terms, tf, n):
    with pytest.raises(ValueError, match="need n >= 2|tf must have one weight"):
        char_surprisal(terms, tf, n=n)


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


def test_only_terms_inside_the_df_band_are_kept():
    picked = frozen_terms(term_stats(_banded_docs()), min_df=2, max_df=21, freeze_n=500)

    # df 1 is one-off noise below the floor; df 25 is ordinary vocabulary above the ceiling.
    assert set(picked) == {"دو", "بیست", "بیستویک"}


def test_an_absent_ceiling_admits_the_whole_tail():
    picked = frozen_terms(term_stats(_banded_docs()), min_df=2, max_df=None, freeze_n=500)

    assert set(picked) == {"دو", "بیست", "بیستویک", "کتاب"}


def _spread_docs() -> list[str]:
    """120 documents over a vocabulary of shared stems, so no two terms tie in surprisal."""
    words = ["دانش", "دانشمند", "دانا", "دانایی", "خرد"]
    words += ["خردمند", "خردورزی", "مندی", "ورزش", "دانشور"]
    return [
        f"{words[i % len(words)]} {words[(i * 3 + 1) % len(words)]} از این که را با"
        for i in range(120)
    ]


def test_the_list_is_exactly_freeze_n_and_ordered_by_falling_surprisal():
    stats = term_stats(_spread_docs())
    freeze_n = 6

    picked = frozen_terms(stats, min_df=2, max_df=60, freeze_n=freeze_n)
    index_of = {t: i for i, t in enumerate(stats.terms)}
    scores = [stats.surprisal[index_of[t]] for t in picked]

    assert len(picked) == freeze_n
    assert all(a > b for a, b in zip(scores, scores[1:]))
    # Nothing eligible was passed over: the cut is a threshold on the score, not a filter.
    band = [i for i in range(len(stats.terms)) if 2 <= stats.df[i] <= 60]
    dropped = [stats.surprisal[i] for i in band if str(stats.terms[i]) not in picked]
    assert max(dropped) <= min(scores)


def test_a_short_vocabulary_yields_fewer_than_freeze_n():
    picked = frozen_terms(term_stats(["کتاب الف", "کتاب ب"]), min_df=2, freeze_n=500)

    assert set(picked) == {"کتاب"}


@pytest.mark.parametrize(
    ("min_df", "max_df", "freeze_n", "rank", "match"),
    [
        (0, None, 500, "surprisal", "need min_df >="),
        (2, 1, 500, "surprisal", "need max_df >="),
        (2, None, 0, "surprisal", "need freeze_n >="),
        (2, None, 500, "df", "rank must be one of"),
    ],
)
def test_out_of_range_arguments_rejected(min_df, max_df, freeze_n, rank, match):
    with pytest.raises(ValueError, match=match):
        frozen_terms(
            term_stats(["کتاب الف"]),
            min_df=min_df,
            max_df=max_df,
            freeze_n=freeze_n,
            rank=rank,
        )


def test_selection_is_deterministic_under_document_shuffling():
    """Surprisal is a property of the vocabulary and its token frequencies, so the list
    must not depend on the order sklearn happened to see the documents in."""
    docs = _spread_docs()
    shuffled = list(docs)
    random.Random(0).shuffle(shuffled)

    a = frozen_terms(term_stats(docs), min_df=2, max_df=60, freeze_n=6)
    b = frozen_terms(term_stats(shuffled), min_df=2, max_df=60, freeze_n=6)
    assert list(a) == list(b)


def test_df_histogram_buckets_cover_every_term():
    stats = term_stats([f"کتاب واژه{i}" for i in range(5)])
    assert sum(df_histogram(stats.df).values()) == len(stats.terms)


def test_zwnj_collisions_pair_the_split_and_joined_spellings():
    terms = [f"بی{ZWNJ}خبر", "بیخبر", f"بی{ZWNJ}مثال", "کتاب"]

    # "بی‌مثال" has no joined counterpart in this vocabulary, so it is not a collision.
    assert zwnj_collisions(terms) == [(f"بی{ZWNJ}خبر", "بیخبر")]
