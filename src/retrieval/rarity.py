"""
Document frequency over the training pool: the frozen rare source-term list.

Usage:
    python -m src.retrieval.rarity --config configs/sparse_retrieval.yaml
"""

from __future__ import annotations

import argparse
import json
import re
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import yaml
from sklearn.feature_extraction.text import CountVectorizer

from src.data.preprocess import ARABIC_SCRIPT_RE
from src.data.split import ARABIC_DIACRITICS

ZWNJ = "‌"
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
    df: np.ndarray
    tf: np.ndarray
    postings: sp.csc_matrix
    n_docs: int


def term_stats(sources: list[str], zwnj: str = "keep") -> TermStats:
    """Document and total frequency of every term in the training pool."""
    vec = CountVectorizer(analyzer=partial(tokenize, zwnj=zwnj))
    counts = vec.fit_transform(sources)
    binary = counts.copy()
    binary.data = np.ones_like(binary.data)
    return TermStats(
        terms=vec.get_feature_names_out(),
        df=np.asarray(binary.sum(axis=0)).ravel(),
        tf=np.asarray(counts.sum(axis=0)).ravel(),
        postings=binary.tocsc(),
        n_docs=counts.shape[0],
    )


def rarity_key(stats: TermStats, i: int) -> tuple[int, int, str]:
    """Sort key ranking term ``i`` from rarest to least rare.

    Document frequency is the score. Total frequency separates terms that occur in
    the same number of sentences, and the term string makes the rest deterministic:
    df alone leaves thousands of terms tied.
    """
    return (int(stats.df[i]), int(stats.tf[i]), str(stats.terms[i]))


def frozen_terms(stats: TermStats, min_df: int = 1, freeze_n: int = 500) -> dict[str, int]:
    """The ``freeze_n`` rarest pool terms, as ``{term: rank}`` with rank 0 the rarest."""
    if min_df < 1:
        raise ValueError(f"need min_df >= 1, got {min_df}")
    if freeze_n < 1:
        raise ValueError(f"need freeze_n >= 1, got {freeze_n}")
    eligible = np.flatnonzero(stats.df >= min_df)
    order = sorted(eligible, key=partial(rarity_key, stats))[:freeze_n]
    return {str(stats.terms[i]): rank for rank, i in enumerate(order)}


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


def load_irregular(path: str | Path) -> dict[str, int]:
    """Load a written rarity list back as ``{term: rank}``, rarest first."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"rarity list not found at {p}. Build it first with:\n"
            "    python manage.py rarity --config configs/sparse_retrieval.yaml"
        )
    payload = json.loads(p.read_text(encoding="utf-8"))
    return {str(term): rank for rank, (term, *_rest) in enumerate(payload["terms"])}


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


def _write_top_tsv(path: Path, rows: list[tuple[int, str, int, int, str]]) -> None:
    lines = ["rank\tterm\tdf\ttf\texample"]
    lines += [f"{r}\t{t}\t{df}\t{tf}\t{ex}" for r, t, df, tf, ex in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the rarest training-pool terms.")
    parser.add_argument("--config", default="configs/sparse_retrieval.yaml")
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--min_df", type=int, default=None, help="floor on document frequency")
    parser.add_argument("--freeze_n", type=int, default=None, help="size of the frozen list")
    parser.add_argument("--zwnj", choices=ZWNJ_MODES, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--top", type=int, default=50, help="rows written to the review TSV")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rar = cfg.get("rarity", {})
    train_file = Path(args.train_file or cfg["retrieval"]["train_file"])
    min_df = args.min_df if args.min_df is not None else rar.get("min_df", 1)
    freeze_n = args.freeze_n if args.freeze_n is not None else rar.get("freeze_n", 500)
    zwnj = args.zwnj or rar.get("zwnj", "keep")
    out_path = Path(args.out or rar.get("out", "results/rarity_train.json"))

    sources = _read_sources(train_file)
    print(f"Counting term frequencies over {len(sources)} training sources (zwnj={zwnj}) ...")
    stats = term_stats(sources, zwnj=zwnj)
    frozen = frozen_terms(stats, min_df=min_df, freeze_n=freeze_n)

    index_of = {t: i for i, t in enumerate(stats.terms)}
    df_of = {t: int(stats.df[index_of[t]]) for t in frozen}
    tf_of = {t: int(stats.tf[index_of[t]]) for t in frozen}
    sel_df = np.array(list(df_of.values()))
    payload = {
        "config": {
            "train_file": str(train_file),
            "min_df": min_df,
            "freeze_n": freeze_n,
            "zwnj": zwnj,
        },
        "n_docs": stats.n_docs,
        "n_terms": int(len(stats.terms)),
        "n_eligible": int((stats.df >= min_df).sum()),
        "n_frozen": len(frozen),
        "selected_frac": round(len(frozen) / len(stats.terms), 4),
        "df_observed": [int(sel_df.min()), int(sel_df.max())] if sel_df.size else None,
        "df_histogram": {"pool": df_histogram(stats.df), "frozen": df_histogram(sel_df)},
        "zwnj_collisions": zwnj_collisions(stats.terms),
        "terms": [[t, df_of[t], tf_of[t]] for t in frozen],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    tsv_path = out_path.with_name(f"{out_path.stem}_top{args.top}.tsv")
    _write_top_tsv(
        tsv_path,
        [
            (rank, t, df_of[t], tf_of[t], _example_doc(stats, index_of[t], sources))
            for t, rank in list(frozen.items())[: args.top]
        ],
    )

    frac, observed = payload["selected_frac"], payload["df_observed"]
    print(
        f"  {len(stats.terms)} pool terms, {payload['n_eligible']} at df >= {min_df} "
        f"-> {len(frozen)} frozen (requested {freeze_n})"
    )
    print(f"  {frac:.1%} of the vocabulary, realized df {observed}")
    print(f"  df histogram (frozen): {payload['df_histogram']['frozen']}")
    collisions = payload["zwnj_collisions"]
    print(f"  ZWNJ variant collisions: {len(collisions)}", collisions[:3] or "")
    print(f"Wrote {out_path} and {tsv_path}")


if __name__ == "__main__":
    main()
