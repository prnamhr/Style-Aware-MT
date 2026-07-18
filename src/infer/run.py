"""
Test-set inference, provider-agnostic across the configured generator.

The condition is one rung of an ablation ladder, each adding one component over
the one before it so any register shift is attributable to that component:

    zeroshot        instruction only, no exemplars
    random_fewshot  k random exemplars (isolates "having examples at all")
    knn_fewshot     k cosine top-k exemplars (isolates relevance-based retrieval)
    afsp_margin     + margin/hub-penalised selection, no register rerank (lambda=0)
    afsp_full       + target-register rerank (lambda>0) -- the full AFSP method

The register glossary is a controlled prompt augmentation (see the `prompt:`
config block), applied uniformly to every few-shot rung, not an AFSP-only rider.

Usage:
    python -m src.infer.run --condition zeroshot    --config configs/openai_smoke.yaml
    python -m src.infer.run --condition knn_fewshot  --config configs/openai_smoke.yaml
    python -m src.infer.run --condition afsp_full    --config configs/base_qwen.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import yaml

from src.eval._io import read_completed_jsonl

# Demonstration ordering is a controlled experimental flag. Exemplars reach
ORDERINGS = ("most_similar_last", "most_similar_first", "random")

# Ablation ladder: each rung adds one component over the previous one.
CONDITIONS = ("zeroshot", "random_fewshot", "knn_fewshot", "afsp_margin", "afsp_full")


def order_exemplars(
    exemplars: list[dict], ordering: str = "most_similar_last", rng: random.Random | None = None
) -> list[dict]:
    """Arrange canonically most-relevant-first exemplars into final prompt order."""
    if ordering == "most_similar_last":
        return list(reversed(exemplars))
    if ordering == "most_similar_first":
        return list(exemplars)
    if ordering == "random":
        out = list(exemplars)
        (rng or random).shuffle(out)
        return out
    raise ValueError(f"unknown ordering '{ordering}' (expected {'|'.join(ORDERINGS)})")


def make_client(gen: dict):
    """
    Build the generator client for the configured provider.
    """
    provider = gen.get("provider", "openai")
    if provider == "openai":
        from src.infer.openai_client import ChatClient

        return ChatClient(
            model=gen["model"],
            temperature=gen.get("temperature"),  # None -> omitted (reasoning models reject it)
            max_tokens=gen.get("max_tokens", 1024),
            seed=gen.get("seed"),
            reasoning_effort=gen.get("reasoning_effort"),
        )
    if provider == "anthropic":
        from src.infer.anthropic_client import AnthropicChatClient

        return AnthropicChatClient(
            model=gen["model"],
            max_tokens=gen.get("max_tokens", 1024),
            thinking=gen.get("thinking", False),
        )
    if provider == "local":
        from src.infer.local_client import LocalChatClient

        return LocalChatClient(
            model=gen["model"],
            max_tokens=gen.get("max_tokens", 1024),
            temperature=gen.get("temperature", 0.0),
            top_p=gen.get("top_p", 1.0),
            seed=gen.get("seed", 42),
            dtype=gen.get("dtype", "bfloat16"),
            device_map=gen.get("device_map"),
            load_in_4bit=gen.get("load_in_4bit", False),
        )
    raise ValueError(f"unknown provider '{provider}' (expected openai|anthropic|local)")


def _read_jsonl(path: Path, limit: int | None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def build_zeroshot_user(source: str) -> str:
    return f"Translate the following text into English:\n\n{source}"


def load_glossary(path: str | Path | None) -> list[tuple[str, str]]:
    """Read tab-separated ``source_term<TAB>target_term`` register pairs.

    Blank lines and lines beginning with ``#`` are ignored. A missing or unset
    path returns an empty glossary, which disables word-level weighting.
    """
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    pairs: list[tuple[str, str]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cells = line.split("\t")
        if len(cells) >= 2 and cells[0].strip() and cells[1].strip():
            pairs.append((cells[0].strip(), cells[1].strip()))
    return pairs


def _term_line(source: str, glossary: list[tuple[str, str]], max_pairs: int = 6) -> str:
    hits = [(s, t) for s, t in glossary if s in source][:max_pairs]
    if not hits:
        return ""
    return "[Terms] " + " | ".join(f"{s} → {t}" for s, t in hits)


def build_fewshot_user(
    source: str, exemplars: list[dict], glossary: list[tuple[str, str]] | None = None
) -> str:
    """Assemble the few-shot user prompt shared by every retrieval condition
    (random_fewshot, knn_fewshot, afsp_margin, afsp_full).

    When ``glossary`` is non-empty, a ``[Terms]`` register line is injected
    before each exemplar and before the query. An empty or ``None`` glossary
    disables it, in which case the output is byte-identical to the plain
    baseline prompt -- so the glossary is a controlled augmentation applied
    uniformly across conditions, not an AFSP-only confound.

    Exemplars arrive already in final prompt order.
    """
    glossary = glossary or []
    blocks = ["Here are example translations in the required style:\n"]
    for e in exemplars:
        term_line = _term_line(e["input"], glossary)
        if term_line:
            blocks.append(term_line)
        blocks.append(f"Source: {e['input']}\nEnglish: {e['output']}\n")
    blocks.append("Now translate the following text into English in the same style:\n")
    query_terms = _term_line(source, glossary)
    if query_terms:
        blocks.append(query_terms)
    blocks.append(f"Source: {source}\nEnglish:")
    return "\n".join(blocks)


def _select_random(
    sources: list[str], pool: list[dict], k: int, rng: random.Random
) -> list[list[dict]]:
    """Draw ``k`` random exemplars per source from the training pool (seeded).

    This is the ``random_fewshot`` rung: it holds shot count fixed but removes
    relevance, isolating the effect of retrieval over merely having examples.
    """
    n = len(pool)
    k = min(k, n)
    return [[pool[i] for i in rng.sample(range(n), k)] for _ in sources]


def _select_afsp(sources, cfg, retr, index, k, *, rerank):
    """AFSP exemplar selection.

    ``rerank=False`` is the ``afsp_margin`` rung: it forces ``lambda_style = 0``
    (margin/hub-penalised selection only) and needs no register centroid.
    ``rerank=True`` is the full method, using the configured ``lambda_style`` and
    the target-register centroid.
    """
    from src.retrieval.afsp import AFSPRetriever, load_centroid

    af = cfg.get("afsp", {})
    retriever = AFSPRetriever(
        index,
        load_centroid(af["centroid_file"]) if rerank else None,
        index_dir=retr["index_dir"],
        beta=af.get("beta", 0.3),
        knn_hubness=af.get("knn_hubness", 5),
        pool_mult=af.get("pool_mult", 4),
        lambda_style=af.get("lambda_style", 0.3) if rerank else 0.0,
        style_objective=af.get("style_objective", "bandpass"),
        style_target_sigma=af.get("style_target_sigma", 1.0),
        style_register_direction=af.get("style_register_direction"),
    )
    return retriever.select(sources, k=k)


def _load_configured_glossary(cfg: dict) -> list[tuple[str, str]]:
    """Load the register glossary as a controlled prompt augmentation.

    It is read from the ``prompt:`` block so it applies uniformly to every
    few-shot condition rather than riding along with AFSP alone. The legacy
    ``afsp:`` location is honoured as a fallback for older configs. Disabled
    (empty) unless ``word_pairs`` is truthy.
    """
    prompt_cfg = cfg.get("prompt", {})
    af = cfg.get("afsp", {})
    if not prompt_cfg.get("word_pairs", af.get("word_pairs", False)):
        return []
    return load_glossary(prompt_cfg.get("glossary_file", af.get("glossary_file")))


def run(condition: str, cfg: dict) -> None:
    gen = cfg["generator"]
    prompt_cfg = cfg.get("prompt", {})
    style_instruction = Path(prompt_cfg["style_instruction_file"]).read_text(encoding="utf-8")
    ordering = prompt_cfg.get("ordering", "most_similar_last")
    # Seeded so `random` ordering and random_fewshot sampling are reproducible.
    rng = random.Random(prompt_cfg.get("ordering_seed", 42))
    eval_file = Path(cfg["data"]["eval_file"])
    split = eval_file.stem  # e.g. "val" -- tags outputs so val results never look like test
    test_rows = _read_jsonl(eval_file, cfg["data"].get("limit"))
    sources = [r["input"] for r in test_rows]

    # Register glossary is applied uniformly to every few-shot rung (a controlled
    # augmentation), so it never confounds the selection-mechanism comparison.
    glossary = _load_configured_glossary(cfg)

    # Build the per-segment user messages for the chosen ablation rung.
    if condition == "zeroshot":
        user_msgs = [build_zeroshot_user(s) for s in sources]
    elif condition in ("random_fewshot", "knn_fewshot", "afsp_margin", "afsp_full"):
        from src.retrieval.retrieve import RetrievalIndex

        retr = cfg["retrieval"]
        k = retr["k"]
        index = RetrievalIndex(retr["index_dir"], embed_model=retr["embed_model"])
        if condition == "random_fewshot":
            print(f"Sampling k={k} random exemplars for {len(sources)} sources ...")
            selected = _select_random(sources, index.pairs, k, rng)
        elif condition == "knn_fewshot":
            print(f"Retrieving k={k} exemplars for {len(sources)} sources ({ordering}) ...")
            selected = index.retrieve(sources, k=k)
        else:  # afsp_margin | afsp_full
            rerank = condition == "afsp_full"
            print(f"{condition}: selecting k={k} for {len(sources)} sources ({ordering}) ...")
            selected = _select_afsp(sources, cfg, retr, index, k, rerank=rerank)
        ordered = [order_exemplars(ex, ordering, rng) for ex in selected]
        user_msgs = [build_fewshot_user(s, ex, glossary) for s, ex in zip(sources, ordered)]
    else:
        raise ValueError(f"unknown condition '{condition}' (expected {'|'.join(CONDITIONS)})")

    client = make_client(gen)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{condition}_{split}.jsonl"

    # Resume: skip segments already written by an earlier (interrupted) pass.
    # read_completed_jsonl repairs a torn final line so we never append onto it.
    done = read_completed_jsonl(out_path)
    if len(done) > len(test_rows):
        raise ValueError(
            f"{out_path} has {len(done)} records but only {len(test_rows)} eval rows; "
            f"delete it to regenerate (did `data.limit` shrink?)"
        )
    for j, rec in enumerate(done):
        if rec.get("input") != test_rows[j]["input"]:
            raise ValueError(
                f"resume misalignment in {out_path} at segment {j}: recorded source differs "
                f"from the eval file; delete the file to regenerate"
            )
    start = len(done)
    if start:
        print(f"resuming {condition}: {start}/{len(test_rows)} already done")

    total = len(test_rows)
    print(f"Generating {total} translations with {gen['model']} ({condition}) ...")
    failures = 0
    # Append (not truncate) so resumed and prior-pass segments accumulate; each
    # line is flushed+fsync'd so an interruption loses at most the in-flight call.
    with out_path.open("a", encoding="utf-8") as f:
        for i in range(start, total):
            row, user = test_rows[i], user_msgs[i]
            try:
                prediction = client.complete(style_instruction, user)
                error = None
            except Exception as e:  # one bad segment must not discard the pass
                prediction = ""
                error = f"{type(e).__name__}: {e}"
                failures += 1
                print(f"  segment {i + 1} failed, recorded empty: {error}")
            record = {
                "input": row["input"],
                "output": row["output"],  # reference target, for scoring
                "prediction": prediction,
                "condition": condition,
                "model": gen["model"],
                "metadata": row.get("metadata", {}),
            }
            if error is not None:
                record["error"] = error
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            if (i + 1) % 5 == 0 or i + 1 == total:
                print(f"  {i + 1}/{total}")

    if failures:
        print(
            f"WARNING: {failures}/{total} segments failed and were recorded with an empty "
            f"prediction (see the `error` field). Delete those lines and re-run to retry them."
        )

    usage = client.usage.summary()
    (out_dir / f"{condition}_{split}_usage.json").write_text(
        json.dumps({"condition": condition, "model": gen["model"], **usage}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")
    print(f"Usage: {usage}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Provider-agnostic eval-set inference.")
    parser.add_argument("--condition", required=True, choices=list(CONDITIONS))
    parser.add_argument("--config", default="configs/openai_smoke.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run(args.condition, cfg)


if __name__ == "__main__":
    main()
