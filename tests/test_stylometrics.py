"""Sanity checks for src/eval/stylometrics.py -- no pytest dependency.

Run directly:  python tests/test_stylometrics.py

The lexical-density check is hand-labeled: the reference excerpt below is a real
training target, and its 42 tokens were split into content vs. function words by
hand (see the comment block) so the code's counts can be confirmed against a human
reading -- exactly the sanity check the metric's approximate nature calls for.
"""

from __future__ import annotations

import math
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

# Hand count (rose-garden splits on the hyphen into two tokens -> 42 tokens):
#   FUNCTION (24): verily the within the of my and the in the of my such and as
#                  are to all but the of the of and
#   CONTENT  (18): birds abiding domains Kingdom doves dwelling rose garden wisdom
#                  utter melodies warblings inscrutable God Lord kingdoms earth heaven
EXPECTED_TOKENS = 42
EXPECTED_CONTENT = 18
EXPECTED_TYPES = 31  # the x6, of x4, and x3, my x2 -> 11 duplicates removed


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
    # and "thee" are. Two markers over four words = 0.5. (The shared marker regex is
    # case-sensitive, matching quick.py, so the words are given lowercase here.)
    f = features("thou and thee speak")
    assert math.isclose(f["marker_rate"], 2 / 4, rel_tol=1e-9)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")
