"""
Character-unusualness over the retrieval pool: the "irregular" (rare) source-term list.

Usage:
    python -m src.retrieval.rarity --config configs/sparse_retrieval.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from src.data.preprocess import ARABIC_SCRIPT_RE
from src.data.split import ARABIC_DIACRITICS

ZWNJ = "‌"
ZWNJ_MODES = ("keep", "strip", "split")
RANKS = ("surprisal", "idf")

PAD, END = "^", "$"
SMOOTHING = 0.5

_DIACRITIC_RE = re.compile(f"[{ARABIC_DIACRITICS}]")
# A token is a run of word characters, optionally joined across ZWNJ.
_TOKEN_RE = re.compile(rf"[^\W_]+(?:{ZWNJ}[^\W_]+)*")


def tokenize(text: str, zwnj: str = "keep") -> list[str]:
    """Split a Persian/Arabic source into vocabulary tokens.

    ``zwnj`` selects how U+200C is treated: ``keep`` joins across it, ``strip``
    deletes it, ``split`` breaks on it. NFKC leaves ZWNJ in place, so this is the
    one normalisation choice the cleaned corpus has not already made.
    """
    if zwnj not in ZWNJ_MODES:
        raise ValueError(f"zwnj must be one of {ZWNJ_MODES}, got {zwnj!r}")
    text = _DIACRITIC_RE.sub("", text or "")
    if zwnj == "strip":
        text = text.replace(ZWNJ, "")
    elif zwnj == "split":
        text = text.replace(ZWNJ, " ")
    # Digits (Persian ones share the Arabic block) and transliteration are not vocabulary.
    return [t for t in _TOKEN_RE.findall(text) if ARABIC_SCRIPT_RE.search(t) and not t.isdigit()]


def char_surprisal(terms, tf, n: int = 4) -> np.ndarray:
    """Mean per-character negative log-probability of each term under a character n-gram model."""
    if n < 2:
        raise ValueError(f"need n >= 2, got {n}")
    terms = [str(t) for t in terms]
    weights = np.asarray(tf, dtype=np.float64)
    if weights.shape != (len(terms),):
        raise ValueError(f"tf must have one weight per term, got {weights.shape}")

    pad = PAD * (n - 1)
    context: defaultdict[str, float] = defaultdict(float)
    ngram: defaultdict[tuple[str, str], float] = defaultdict(float)
    alphabet: set[str] = {END}
    for term, w in zip(terms, weights):
        s = pad + term + END
        alphabet.update(term)
        for i in range(n - 1, len(s)):
            ctx = s[i - n + 1 : i]
            context[ctx] += w
            ngram[(ctx, s[i])] += w

    denom = SMOOTHING * len(alphabet)
    out = np.empty(len(terms), dtype=np.float64)
    for j, term in enumerate(terms):
        s = pad + term + END
        total = 0.0
        for i in range(n - 1, len(s)):
            ctx = s[i - n + 1 : i]
            p = (ngram.get((ctx, s[i]), 0.0) + SMOOTHING) / (context.get(ctx, 0.0) + denom)
            total -= math.log(p)
        out[j] = total / (len(s) - n + 1)
    return out


class TermStats(NamedTuple):
    terms: np.ndarray
    idf: np.ndarray
    df: np.ndarray
    tf: np.ndarray
    surprisal: np.ndarray
    postings: sp.csc_matrix
    n_docs: int


def term_stats(sources: list[str], zwnj: str = "keep", char_n: int = 4) -> TermStats:
    """Frequency, smoothed IDF and character surprisal of every term in the pool."""
    vec = CountVectorizer(analyzer=partial(tokenize, zwnj=zwnj))
    counts = vec.fit_transform(sources)
    binary = counts.copy()
    binary.data = np.ones_like(binary.data)
    # TfidfTransformer derives its own document frequency, so the raw counts are fine here.
    idf = TfidfTransformer(use_idf=True, smooth_idf=True).fit(counts).idf_
    terms = vec.get_feature_names_out()
    tf = np.asarray(counts.sum(axis=0)).ravel()
    return TermStats(
        terms=terms,
        idf=idf,
        df=np.asarray(binary.sum(axis=0)).ravel(),
        tf=tf,
        surprisal=char_surprisal(terms, tf, n=char_n),
        postings=binary.tocsc(),
        n_docs=counts.shape[0],
    )


def eligible_terms(stats: TermStats, min_df: int, max_df: int | None) -> np.ndarray:
    """Indices of the terms whose pool document frequency lies in the closed band."""
    if min_df < 1:
        raise ValueError(f"need min_df >= 1, got {min_df}")
    if max_df is not None and max_df < min_df:
        raise ValueError(f"need max_df >= min_df, got max_df={max_df}, min_df={min_df}")
    keep = stats.df >= min_df
    if max_df is not None:
        keep &= stats.df <= max_df
    return np.flatnonzero(keep)


def frozen_terms(
    stats: TermStats,
    min_df: int = 2,
    max_df: int | None = None,
    freeze_n: int = 500,
    rank: str = "surprisal",
) -> dict[str, float]:
    """The freeze_n most unusual terms inside the df band"""
    if freeze_n < 1:
        raise ValueError(f"need freeze_n >= 1, got {freeze_n}")
    if rank not in RANKS:
        raise ValueError(f"rank must be one of {RANKS}, got {rank!r}")
    score = stats.surprisal if rank == "surprisal" else stats.idf
    eligible = eligible_terms(stats, min_df, max_df)
    # The term string is a determinism guard against exact score ties, not a ranking key.
    order = sorted(eligible, key=lambda i: (-score[i], stats.terms[i]))[:freeze_n]
    return {str(stats.terms[i]): float(stats.idf[i]) for i in order}


def df_histogram(df: np.ndarray) -> dict[str, int]:
    """Bucketed document-frequency counts, for reading where the frozen list falls."""
    buckets = {"1": 0, "2": 0, "3": 0, "4-9": 0, "10-99": 0, "100+": 0}
    for v in df:
        if v <= 3:
            buckets[str(int(v))] += 1
        elif v < 10:
            buckets["4-9"] += 1
        elif v < 100:
            buckets["10-99"] += 1
        else:
            buckets["100+"] += 1
    return buckets


def load_irregular(path: str | Path) -> dict[str, float]:
    """Load a written rarity list back as ``{term: idf}``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"rarity list not found at {p}. Build it first with:\n"
            "    python manage.py rarity --config configs/sparse_retrieval.yaml"
        )
    payload = json.loads(p.read_text(encoding="utf-8"))
    return {term: float(idf) for term, idf, *_rest in payload["terms"]}


def _read_sources(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line)["input"] for line in f if line.strip()]


def _example_doc(stats: TermStats, term_idx: int, sources: list[str]) -> str:
    col = stats.postings
    rows = col.indices[col.indptr[term_idx] : col.indptr[term_idx + 1]]
    return sources[int(rows[0])] if len(rows) else ""


def zwnj_collisions(terms) -> list[tuple[str, str]]:
    """Terms whose ZWNJ-joined form is also in the vocabulary, e.g. بی‌خبر / بیخبر.

    Each pair is one word counted as two, which inflates the rarity of both. Only
    meaningful for ``zwnj="keep"``; the other modes cannot produce a collision.
    """
    vocab = set(map(str, terms))
    return sorted(
        (t, t.replace(ZWNJ, "")) for t in vocab if ZWNJ in t and t.replace(ZWNJ, "") in vocab
    )


def _write_top_tsv(path: Path, rows: list[tuple[int, str, float, float, int, str]]) -> None:
    lines = ["rank\tterm\tsurprisal\tidf\tdf\texample"]
    lines += [f"{r}\t{t}\t{s:.4f}\t{idf:.4f}\t{df}\t{ex}" for r, t, s, idf, df, ex in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the most unusual pool terms in a df band.")
    parser.add_argument("--config", default="configs/sparse_retrieval.yaml")
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--min_df", type=int, default=None)
    parser.add_argument("--max_df", type=int, default=None, help="0 lifts the ceiling")
    parser.add_argument("--freeze_n", type=int, default=None, help="size of the frozen list")
    parser.add_argument("--rank", choices=RANKS, default=None)
    parser.add_argument("--zwnj", choices=ZWNJ_MODES, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--top", type=int, default=50, help="rows written to the review TSV")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rar = cfg.get("rarity", {})
    train_file = Path(args.train_file or cfg["retrieval"]["train_file"])
    min_df = args.min_df if args.min_df is not None else rar.get("min_df", 2)
    max_df = args.max_df if args.max_df is not None else rar.get("max_df")
    max_df = None if max_df in (0, None) else int(max_df)
    freeze_n = args.freeze_n if args.freeze_n is not None else rar.get("freeze_n", 500)
    rank = args.rank or rar.get("rank", "surprisal")
    zwnj = args.zwnj or rar.get("zwnj", "keep")
    out_path = Path(args.out or rar.get("out", "results/rarity_train.json"))

    sources = _read_sources(train_file)
    print(f"Scoring {len(sources)} pool sources (zwnj={zwnj}, rank={rank}) ...")
    stats = term_stats(sources, zwnj=zwnj)
    frozen = frozen_terms(stats, min_df=min_df, max_df=max_df, freeze_n=freeze_n, rank=rank)

    index_of = {t: i for i, t in enumerate(stats.terms)}
    df_of = {t: int(stats.df[index_of[t]]) for t in frozen}
    sur_of = {t: float(stats.surprisal[index_of[t]]) for t in frozen}
    sel_df = np.array(list(df_of.values()))
    sel_sur = np.array(list(sur_of.values()))
    payload = {
        "config": {
            "train_file": str(train_file),
            "min_df": min_df,
            "max_df": max_df,
            "freeze_n": freeze_n,
            "rank": rank,
            "zwnj": zwnj,
        },
        "n_docs": stats.n_docs,
        "n_terms": int(len(stats.terms)),
        "n_eligible": int(len(eligible_terms(stats, min_df, max_df))),
        "n_frozen": len(frozen),
        "selected_frac": round(len(frozen) / len(stats.terms), 4),
        "df_observed": [int(sel_df.min()), int(sel_df.max())] if sel_df.size else None,
        "idf_range": [min(frozen.values()), max(frozen.values())] if frozen else None,
        "surprisal_range": [float(sel_sur.min()), float(sel_sur.max())] if sel_sur.size else None,
        "df_histogram": {"pool": df_histogram(stats.df), "frozen": df_histogram(sel_df)},
        "zwnj_collisions": zwnj_collisions(stats.terms),
        "terms": [[t, idf, df_of[t], sur_of[t]] for t, idf in frozen.items()],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    tsv_path = out_path.with_name(f"{out_path.stem}_top{args.top}.tsv")
    _write_top_tsv(
        tsv_path,
        [
            (r, t, sur_of[t], frozen[t], df_of[t], _example_doc(stats, index_of[t], sources))
            for r, t in enumerate(list(frozen)[: args.top], start=1)
        ],
    )

    ceiling = max_df if max_df is not None else "inf"
    print(
        f"  {len(stats.terms)} pool terms, {payload['n_eligible']} in df [{min_df}, {ceiling}] "
        f"-> {len(frozen)} frozen (requested {freeze_n})"
    )
    frac, observed = payload["selected_frac"], payload["df_observed"]
    print(f"  {frac:.1%} of the vocabulary, realized df {observed}")
    print(f"  idf {payload['idf_range']}, surprisal {payload['surprisal_range']}")
    print(f"  df histogram (frozen): {payload['df_histogram']['frozen']}")
    collisions = payload["zwnj_collisions"]
    print(f"  ZWNJ variant collisions: {len(collisions)}", collisions[:3] or "")
    print(f"Wrote {out_path} and {tsv_path}")


if __name__ == "__main__":
    main()
