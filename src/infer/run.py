"""
Test-set inference, provider-agnostic across the configured generator.

Usage:
    python -m src.infer.run --condition reference   --config configs/openai_smoke.yaml
    python -m src.infer.run --condition knn_fewshot --config configs/openai_smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

# Demonstration ordering is a controlled experimental flag. Exemplars reach
ORDERINGS = ("most_similar_last", "most_similar_first", "random")


def order_exemplars(
    exemplars: list[dict], ordering: str = "most_similar_last", rng: random.Random | None = None
) -> list[dict]:
    """Arrange canonically most-relevant-first exemplars into final prompt order.
    """
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


def build_reference_user(source: str) -> str:
    return f"Translate the following text into English:\n\n{source}"


def build_knn_fewshot_user(source: str, exemplars: list[dict]) -> str:
    # Exemplars arrive already in final prompt order
    blocks = [
        "Here are example translations in the required style:\n",
        *(f"Source: {e['input']}\nEnglish: {e['output']}\n" for e in exemplars),
        "Now translate the following text into English in the same style:\n",
        f"Source: {source}\nEnglish:",
    ]
    return "\n".join(blocks)


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


def build_afsp_user(
    source: str, exemplars: list[dict], glossary: list[tuple[str, str]] | None = None
) -> str:
    """Assemble the AFSP user prompt. Register word pairs are injected before each
    exemplar and before the query.

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


def run(condition: str, cfg: dict) -> None:
    gen = cfg["generator"]
    prompt_cfg = cfg.get("prompt", {})
    style_instruction = Path(prompt_cfg["style_instruction_file"]).read_text(encoding="utf-8")
    ordering = prompt_cfg.get("ordering", "most_similar_last")
    # Seeded so `random` ordering is reproducible across reruns and conditions.
    rng = random.Random(prompt_cfg.get("ordering_seed", 42))
    eval_file = Path(cfg["data"]["eval_file"])
    split = eval_file.stem  # e.g. "val" -- tags outputs so val results never look like test
    test_rows = _read_jsonl(eval_file, cfg["data"].get("limit"))
    sources = [r["input"] for r in test_rows]

    # Build the per-segment user messages for the chosen condition.
    if condition == "reference":
        user_msgs = [build_reference_user(s) for s in sources]
    elif condition == "knn_fewshot":
        from src.retrieval.retrieve import RetrievalIndex

        retr = cfg["retrieval"]
        index = RetrievalIndex(retr["index_dir"], embed_model=retr["embed_model"])
        print(f"Retrieving k={retr['k']} exemplars for {len(sources)} sources ({ordering}) ...")
        retrieved = index.retrieve(sources, k=retr["k"])
        ordered = [order_exemplars(ex, ordering, rng) for ex in retrieved]
        user_msgs = [build_knn_fewshot_user(s, ex) for s, ex in zip(sources, ordered)]
    elif condition == "afsp":
        from src.retrieval.afsp import AFSPRetriever, load_centroid
        from src.retrieval.retrieve import RetrievalIndex

        retr = cfg["retrieval"]
        af = cfg["afsp"]
        index = RetrievalIndex(retr["index_dir"], embed_model=retr["embed_model"])
        retriever = AFSPRetriever(
            index,
            load_centroid(af["centroid_file"]),
            index_dir=retr["index_dir"],
            beta=af.get("beta", 0.3),
            knn_hubness=af.get("knn_hubness", 5),
            pool_mult=af.get("pool_mult", 4),
            lambda_style=af.get("lambda_style", 0.3),
        )
        glossary = load_glossary(af.get("glossary_file")) if af.get("word_pairs", True) else []
        print(f"AFSP: selecting k={retr['k']} for {len(sources)} sources ({ordering}) ...")
        selected = retriever.select(sources, k=retr["k"])
        ordered = [order_exemplars(ex, ordering, rng) for ex in selected]
        user_msgs = [build_afsp_user(s, ex, glossary) for s, ex in zip(sources, ordered)]
    else:
        raise ValueError(f"unknown condition '{condition}' (expected reference|knn_fewshot|afsp)")

    client = make_client(gen)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{condition}_{split}.jsonl"

    print(f"Generating {len(test_rows)} translations with {gen['model']} ({condition}) ...")
    with out_path.open("w", encoding="utf-8") as f:
        for i, (row, user) in enumerate(zip(test_rows, user_msgs), 1):
            prediction = client.complete(style_instruction, user)
            f.write(
                json.dumps(
                    {
                        "input": row["input"],
                        "output": row["output"],  # reference target, for scoring
                        "prediction": prediction,
                        "condition": condition,
                        "model": gen["model"],
                        "metadata": row.get("metadata", {}),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if i % 5 == 0 or i == len(test_rows):
                print(f"  {i}/{len(test_rows)}")

    usage = client.usage.summary()
    (out_dir / f"{condition}_{split}_usage.json").write_text(
        json.dumps({"condition": condition, "model": gen["model"], **usage}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")
    print(f"Usage: {usage}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Provider-agnostic eval-set inference.")
    parser.add_argument("--condition", required=True, choices=["reference", "knn_fewshot", "afsp"])
    parser.add_argument("--config", default="configs/openai_smoke.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run(args.condition, cfg)


if __name__ == "__main__":
    main()
