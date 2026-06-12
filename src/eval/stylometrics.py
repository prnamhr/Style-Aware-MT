"""
Per-segment stylometric features and a target-register centroid.

Usage:
    python manage.py stylometrics --build-centroid
    python manage.py stylometrics --targets-split train
    python manage.py stylometrics --conditions reference knn_fewshot --split val
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np

_SPLIT_DIR = Path("data/splits")
_CENTROID_PATH = Path("results/stylometrics_centroid.json")

_MARKERS = re.compile(
    r"\b(thou|thee|thy|thine|art|hast|hath|dost|doth|shalt|wilt|unto|ye)\b"
    r"|\bO\b",
)

# Standard English function words plus the archaic grammatical forms (thee/thou/hath
# family, archaic adverbs and relative forms) that recur in the authorized register.
# Counted as *function* words, so a heavily archaic passage scores LOWER lexical
# density -- lex_density and marker_rate are therefore complementary, not redundant.
# fmt: off
# Hand-grouped by category for readability; ruff would explode these to one
# token per line, so formatting is disabled across both word-list sets.
_STANDARD_FUNCTION_WORDS = {
    # articles / determiners
    "a", "an", "the", "this", "that", "these", "those", "such", "no", "every",
    "each", "any", "all", "some", "both", "either", "neither", "much", "many",
    "more", "most", "few", "less", "least", "own", "same", "other", "another",
    # pronouns
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "who", "whom", "whose", "which", "what", "whatever",
    "whoever", "whomever", "whichever",
    # prepositions
    "of", "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "over", "under", "again", "then", "once", "here", "there", "when",
    "where", "why", "how", "as", "until", "while", "than", "upon", "within",
    "without", "amid", "amidst", "among", "amongst", "beneath", "beside", "beyond",
    "toward", "towards", "throughout",
    # conjunctions
    "and", "but", "or", "nor", "so", "yet", "if", "because", "although", "though",
    "whereas", "whether", "unless", "since", "lest",
    # auxiliaries / common verbs of being
    "be", "am", "is", "are", "was", "were", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "will", "would", "shall", "should",
    "may", "might", "must", "can", "could", "ought",
    # negation / particles
    "not", "only", "even", "just", "also", "too", "very", "ever", "never",
}

# Archaic / register-aware extension. The user asked that the thee/thou/hath family
# and archaic relatives be treated as function words.
_ARCHAIC_FUNCTION_WORDS = {
    "thou", "thee", "thy", "thine", "ye", "hath", "doth", "dost", "hast", "art",
    "wert", "wast", "wilt", "shalt", "canst", "couldst", "wouldst", "shouldst",
    "didst", "hadst", "mayest", "mayst", "unto", "verily", "whilst", "wherefore", "whence",
    "whither", "hither", "thither", "thence", "hence", "thereof", "therein",
    "thereto", "thereon", "thereby", "therewith", "thereunto", "wherein",
    "whereof", "whereby", "whereupon", "wherewith", "herein", "hereof", "hereby",
    "hereunto", "withal", "nay", "yea",
}
# fmt: on

FUNCTION_WORDS = frozenset(_STANDARD_FUNCTION_WORDS | _ARCHAIC_FUNCTION_WORDS)

# Feature vector order. feature_vector() and the centroid follow this exactly.
FEATURE_NAMES = [
    "lex_density",
    "ttr",
    "root_ttr",
    "sent_len_mean",
    "sent_len_var",
    "marker_rate",
]

_WORD_RE = re.compile(r"[a-z']+")
_SENT_SPLIT_RE = re.compile(r"[.!?]+\s+")


def _words(text: str) -> list[str]:
    """Lowercased word tokens with surrounding punctuation stripped.

    Using a letter/apostrophe class means ``"word."`` and ``"word"`` collapse to the
    same token, so the type and total counts are not polluted by punctuation.
    """
    return _WORD_RE.findall(text.lower())


def _sentences(text: str) -> list[str]:
    """Split a segment into sentences on terminal punctuation."""
    return [s for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]


def features(text: str) -> dict[str, float]:
    """Per-segment stylometric features. Empty/wordless text yields all zeros."""
    words = _words(text)
    n = len(words)
    if n == 0:
        return dict.fromkeys(FEATURE_NAMES, 0.0)

    types = len(set(words))
    content = sum(1 for w in words if w not in FUNCTION_WORDS)

    sent_lengths = [len(_words(s)) for s in _sentences(text)]
    sent_lengths = [length for length in sent_lengths if length > 0] or [n]

    return {
        "lex_density": content / n,
        "ttr": types / n,
        "root_ttr": types / math.sqrt(n),
        "sent_len_mean": statistics.fmean(sent_lengths),
        # Population variance: a single-sentence segment is genuinely 0 spread.
        "sent_len_var": statistics.pvariance(sent_lengths) if len(sent_lengths) > 1 else 0.0,
        "marker_rate": len(_MARKERS.findall(text)) / n,
    }


def feature_vector(text: str) -> list[float]:
    """Features ordered by ``FEATURE_NAMES`` -- the form the style rerank consumes."""
    f = features(text)
    return [f[name] for name in FEATURE_NAMES]


def build_centroid(targets: list[str]) -> dict:
    """Per-feature mean and (sample) std over the training English targets.

    ``std`` is floored at a small epsilon so a degenerate zero-variance feature
    cannot blow up the z-scoring the rerank does against this centroid.
    """
    vectors = [feature_vector(t) for t in targets if t.strip()]
    matrix = np.asarray(vectors, dtype=float)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    std = np.where(std < 1e-9, 1e-9, std)
    return {
        "n_segments": int(matrix.shape[0]),
        "features": FEATURE_NAMES,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def aggregate(texts: list[str]) -> dict:
    """Per-feature mean and std *across segments* -- the std is the H2 signal."""
    vectors = [feature_vector(t) for t in texts if t.strip()]
    matrix = np.asarray(vectors, dtype=float)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros(len(FEATURE_NAMES))
    return {
        "n": int(matrix.shape[0]),
        "mean": dict(zip(FEATURE_NAMES, mean.tolist())),
        "std": dict(zip(FEATURE_NAMES, std.tolist())),
    }


def distance_to_centroid(agg_mean: dict[str, float], centroid: dict) -> float:
    """Standardized (z-scored) Euclidean distance of a mean vector to the centroid."""
    mean = np.asarray(centroid["mean"], dtype=float)
    std = np.asarray(centroid["std"], dtype=float)
    vec = np.asarray([agg_mean[name] for name in centroid["features"]], dtype=float)
    return float(np.linalg.norm((vec - mean) / std))


def _load_field(path: Path, field: str) -> list[str]:
    """Return one text field from every JSONL record in ``path``."""
    out: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            val = json.loads(line).get(field, "")
            if val and val.strip():
                out.append(val)
    return out


def _print_table(rows: list[dict], cols: list[str]) -> None:
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def _report_row(label: str, agg: dict, centroid: dict | None) -> dict:
    row = {"label": label, "n": agg["n"]}
    for name in FEATURE_NAMES:
        row[name] = round(agg["mean"][name], 4)
        row[f"{name}_sd"] = round(agg["std"][name], 4)
    if centroid is not None:
        row["stylo_dist"] = round(distance_to_centroid(agg["mean"], centroid), 4)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-segment stylometrics + target centroid.")
    parser.add_argument(
        "--build-centroid",
        action="store_true",
        help="build the target-register centroid over train targets and save it",
    )
    parser.add_argument(
        "--targets-split",
        choices=["train", "val", "test"],
        help="report feature aggregates over a split's gold English targets",
    )
    parser.add_argument("--conditions", nargs="+", help="inference conditions to report on")
    parser.add_argument("--split", default="val", help="output split tag (default: val)")
    parser.add_argument("--out_dir", default="outputs", help="inference output directory")
    args = parser.parse_args()

    if args.build_centroid:
        targets = _load_field(_SPLIT_DIR / "train.jsonl", "output")
        centroid = build_centroid(targets)
        _CENTROID_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CENTROID_PATH.write_text(json.dumps(centroid, indent=2) + "\n", encoding="utf-8")
        print(f"\nTarget-register centroid  (n={centroid['n_segments']})  -> {_CENTROID_PATH}")
        width = max(len(n) for n in FEATURE_NAMES)
        print("-" * (width + 28))
        for name, m, s in zip(FEATURE_NAMES, centroid["mean"], centroid["std"]):
            print(f"  {name:<{width}}  mean {m:.4f}   std {s:.4f}")
        print()
        return

    centroid = None
    if _CENTROID_PATH.exists():
        centroid = json.loads(_CENTROID_PATH.read_text(encoding="utf-8"))

    rows: list[dict] = []
    if args.targets_split:
        targets = _load_field(_SPLIT_DIR / f"{args.targets_split}.jsonl", "output")
        rows.append(_report_row(f"target:{args.targets_split}", aggregate(targets), centroid))

    out_dir = Path(args.out_dir)
    for cond in args.conditions or []:
        path = out_dir / f"{cond}_{args.split}.jsonl"
        if not path.exists():
            print(f"skip {cond}: {path} not found")
            continue
        preds = _load_field(path, "prediction")
        rows.append(_report_row(cond, aggregate(preds), centroid))

    if not rows:
        print("nothing to report: pass --targets-split and/or --conditions")
        return

    # Each feature is shown with its across-segment std (the H2 variance signal).
    cols = ["label", "n"]
    for name in FEATURE_NAMES:
        cols += [name, f"{name}_sd"]
    if centroid is not None:
        cols.append("stylo_dist")
    _print_table(rows, cols)


if __name__ == "__main__":
    main()
