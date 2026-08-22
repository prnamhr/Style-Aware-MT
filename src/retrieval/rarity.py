"""
IDF over the retrieval pool: the "irregular" (rare) source-term list.

Usage:
    python -m src.retrieval.rarity --config configs/sparse_retrieval.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import re
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from src.data.preprocess import ARABIC_SCRIPT_RE
from src.data.split import ARABIC_DIACRITICS

ZWNJ = "\u200c"
ZWNJ_MODES = ("keep", "strip", "split")

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


class TermStats(NamedTuple):
    terms: np.ndarray
    idf: np.ndarray
    df: np.ndarray
    postings: sp.csc_matrix
    n_docs: int


def term_stats(sources: list[str], zwnj: str = "keep") -> TermStats:
    """Document frequency and smoothed IDF of every term in the pool."""
    vec = CountVectorizer(analyzer=partial(tokenize, zwnj=zwnj), binary=True)
    counts = vec.fit_transform(sources)
    idf = TfidfTransformer(use_idf=True, smooth_idf=True).fit(counts).idf_
    return TermStats(
        terms=vec.get_feature_names_out(),
        idf=idf,
        df=np.asarray(counts.sum(axis=0)).ravel(),
        postings=counts.tocsc(),
        n_docs=counts.shape[0],
    )


def irregular_terms(stats: TermStats, df_min: int = 2, df_max: int = 20) -> dict[str, float]:
    """Terms whose pool document frequency lies in the closed band.
    """
    if df_min < 1 or df_max < df_min:
        raise ValueError(f"need 1 <= df_min <= df_max, got df_min={df_min}, df_max={df_max}")
    keep = np.flatnonzero((stats.df >= df_min) & (stats.df <= df_max))
    order = sorted(keep, key=lambda i: (-stats.idf[i], stats.terms[i]))
    return {str(stats.terms[i]): float(stats.idf[i]) for i in order}


def df_histogram(df: np.ndarray) -> dict[str, int]:
    """Bucketed document-frequency counts, for reading where the selected band falls."""
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
    return {term: float(idf) for term, idf, _df in payload["terms"]}


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


def review_sample(irregular: dict[str, float], df_of: dict[str, int], n: int) -> list[str]:
    """A seeded sample of the list, least-rare first, for the manual eyeball.

    The selected terms are near-uniformly tied in IDF, so the head of any IDF sort
    is an alphabetical accident. A sample is the only honest view of the set.
    """
    terms = sorted(irregular)
    picked = random.Random(0).sample(terms, min(n, len(terms)))
    return sorted(picked, key=lambda t: (-df_of[t], t))


def _write_top_tsv(path: Path, rows: list[tuple[str, float, int, str]]) -> None:
    lines = ["term\tidf\tdf\texample"]
    lines += [f"{t}\t{idf:.4f}\t{df}\t{ex}" for t, idf, df, ex in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the irregular-term list by pool IDF.")
    parser.add_argument("--config", default="configs/sparse_retrieval.yaml")
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--df_min", type=int, default=None)
    parser.add_argument("--df_max", type=int, default=None)
    parser.add_argument("--zwnj", choices=ZWNJ_MODES, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--top", type=int, default=50, help="rows written to the review TSV")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rar = cfg.get("rarity", {})
    train_file = Path(args.train_file or cfg["retrieval"]["train_file"])
    df_min = args.df_min if args.df_min is not None else rar.get("df_min", 2)
    df_max = args.df_max if args.df_max is not None else rar.get("df_max", 20)
    zwnj = args.zwnj or rar.get("zwnj", "keep")
    out_path = Path(args.out or rar.get("out", "results/rarity_train.json"))

    sources = _read_sources(train_file)
    print(f"Computing IDF over {len(sources)} pool sources (zwnj={zwnj}) ...")
    stats = term_stats(sources, zwnj=zwnj)
    irregular = irregular_terms(stats, df_min=df_min, df_max=df_max)

    index_of = {t: i for i, t in enumerate(stats.terms)}
    df_of = {t: int(stats.df[index_of[t]]) for t in irregular}
    sel_df = np.array(list(df_of.values()))
    payload = {
        "config": {
            "train_file": str(train_file),
            "df_min": df_min,
            "df_max": df_max,
            "zwnj": zwnj,
        },
        "n_docs": stats.n_docs,
        "n_terms": int(len(stats.terms)),
        "n_irregular": len(irregular),
        "selected_frac": round(len(irregular) / len(stats.terms), 4),
        "df_observed": [int(sel_df.min()), int(sel_df.max())] if sel_df.size else None,
        "idf_range": [min(irregular.values()), max(irregular.values())] if irregular else None,
        "df_histogram": {"pool": df_histogram(stats.df), "irregular": df_histogram(sel_df)},
        "zwnj_collisions": zwnj_collisions(stats.terms),
        "terms": [[t, idf, df_of[t]] for t, idf in irregular.items()],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    tsv_path = out_path.with_name(f"{out_path.stem}_sample{args.top}.tsv")
    _write_top_tsv(
        tsv_path,
        [
            (t, irregular[t], df_of[t], _example_doc(stats, index_of[t], sources))
            for t in review_sample(irregular, df_of, args.top)
        ],
    )

    frac, observed = payload["selected_frac"], payload["df_observed"]
    print(f"  {len(stats.terms)} pool terms -> {len(irregular)} in df band [{df_min}, {df_max}]")
    print(f"  {frac:.1%} of the vocabulary, observed df {observed}")
    print(f"  df histogram (irregular): {payload['df_histogram']['irregular']}")
    collisions = payload["zwnj_collisions"]
    print(f"  ZWNJ variant collisions: {len(collisions)}", collisions[:3] or "")
    print(f"Wrote {out_path} and {tsv_path}")


if __name__ == "__main__":
    main()
