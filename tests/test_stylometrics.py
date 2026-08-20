"""Sanity checks for src/eval/stylometrics.py -- no pytest dependency.

Run directly:  python tests/test_stylometrics.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.stylometrics import FUNCTION_WORDS, _words, features  # noqa: E402

# A real train target (data/splits/train.jsonl, record 0).
REF = (
    "Verily, the birds abiding within the domains of My Kingdom and the doves "
    "dwelling in the rose-garden of My wisdom utter such melodies and warblings as "
    "are inscrutable to all but God, the Lord of the kingdoms of earth and heaven."
)

EXPECTED_TOKENS = 42
EXPECTED_CONTENT = 18
EXPECTED_TYPES = 31  # the x6, of x4, and x3, my x2 -> 11 duplicates removed

HAND_COUNTED = [
    (REF, 0, EXPECTED_TOKENS),
    (
        "This is the Day wherein the earth hath told out her tidings and hath laid bare "
        "her treasures.",
        2,
        18,
    ),
    ("He hath no associate in His judgment nor any helper in His sovereignty.", 1, 13),
    ("Unto Thee be praise, O Lord my God!", 3, 8),
    ("Thou art the God of power, of glory and bounty.", 2, 10),
    (
        "Blessing be unto Thee, O Lord of Names, and glory be unto Thee, O Creator of the heavens,",
        6,
        18,
    ),
]

# Mean marker_rate the case-sensitive regex produced over the 10,860 train targets.
CASE_SENSITIVE_MARKER_MEAN = 0.0327
MARKER_BAND = (0.050, 0.065)


def test_tokenizing_strips_punctuation() -> None:
    words = _words(REF)
    assert len(words) == EXPECTED_TOKENS, len(words)
    # "Verily," and "God," lose their trailing comma; "rose-garden" splits in two.
    assert "verily" in words and "god" in words
    assert "rose" in words and "garden" in words
    assert "verily," not in words


def test_lexical_density_matches_hand_count() -> None:
    f = features(REF)
    content = sum(1 for w in _words(REF) if w not in FUNCTION_WORDS)
    assert content == EXPECTED_CONTENT, content
    assert math.isclose(f["lex_density"], EXPECTED_CONTENT / EXPECTED_TOKENS, rel_tol=1e-9)


def test_ttr_and_root_ttr() -> None:
    f = features(REF)
    assert math.isclose(f["ttr"], EXPECTED_TYPES / EXPECTED_TOKENS, rel_tol=1e-9)
    assert math.isclose(f["root_ttr"], EXPECTED_TYPES / math.sqrt(EXPECTED_TOKENS), rel_tol=1e-9)
    # root_ttr is the length-robust variant; on a 42-token segment it exceeds raw ttr.
    assert f["root_ttr"] > f["ttr"]


def test_single_sentence_has_zero_variance() -> None:
    f = features(REF)  # one sentence
    assert f["sent_len_var"] == 0.0
    assert math.isclose(f["sent_len_mean"], EXPECTED_TOKENS, rel_tol=1e-9)


def test_multi_sentence_variance_positive() -> None:
    text = "He came. Then the long and winding river carried them onward to the sea."
    f = features(text)
    assert f["sent_len_var"] > 0.0  # 2 words vs 13 words


def test_empty_text_is_all_zeros() -> None:
    assert features("   ") == {k: 0.0 for k in features(REF)}


def test_marker_rate_normalized_per_word() -> None:
    # "verily" is a function word for lex density but NOT an archaic marker; "thou"
    # and "thee" are. Two markers over four words = 0.5.
    f = features("thou and thee speak")
    assert math.isclose(f["marker_rate"], 2 / 4, rel_tol=1e-9)


def test_capitalized_markers_are_counted() -> None:
    # Sentence-initial and vocative capitals are the same markers; before the pronoun
    # class was made case-insensitive only the lowercase "art" here scored.
    assert math.isclose(features("Thou art")["marker_rate"], 1.0, rel_tol=1e-9)
    assert math.isclose(features("Thy grace")["marker_rate"], 1 / 2, rel_tol=1e-9)


def test_vocative_o_stays_case_sensitive() -> None:
    # The scoped (?i:...) must not leak onto the O branch, or every ordinary "o" counts.
    assert math.isclose(features("O Lord")["marker_rate"], 1 / 2, rel_tol=1e-9)
    assert features("o lord")["marker_rate"] == 0.0


def test_marker_rate_matches_hand_counts_on_train_targets() -> None:
    for text, markers, tokens in HAND_COUNTED:
        f = features(text)
        assert len(_words(text)) == tokens, (text, len(_words(text)))
        assert math.isclose(f["marker_rate"], markers / tokens, rel_tol=1e-9), text


def test_committed_centroid_marker_rate_in_hand_checked_band() -> None:
    """Guard against a centroid file left over from the case-sensitive regex."""
    path = Path(__file__).resolve().parent.parent / "results" / "stylometrics_centroid.json"
    if not path.exists():
        print("skip centroid guard: build it with python manage.py stylometrics --build-centroid")
        return
    centroid = json.loads(path.read_text(encoding="utf-8"))
    mean = centroid["mean"][centroid["features"].index("marker_rate")]
    assert MARKER_BAND[0] <= mean <= MARKER_BAND[1], mean
    assert mean > CASE_SENSITIVE_MARKER_MEAN * 1.5, mean

    hand_mean = statistics.fmean(m / t for _, m, t in HAND_COUNTED)
    assert mean < hand_mean, (mean, hand_mean)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")
