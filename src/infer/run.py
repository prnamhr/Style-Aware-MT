"""
Test-set inference, provider-agnostic across the configured generator.

Usage:
    python -m src.infer.run --condition reference   --config configs/openai_smoke.yaml
    python -m src.infer.run --condition knn_fewshot --config configs/openai_smoke.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


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
    raise ValueError(f"unknown provider '{provider}' (expected openai|anthropic)")


def _read_jsonl(path: Path, limit: int | None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def build_reference_user(source: str) -> str:
    return f"Translate the following text into English:\n\n{source}"


def build_knn_fewshot_user(source: str, exemplars: list[dict]) -> str:
    # Most-similar last: closest to the query, where positional influence is strongest.
    ordered = list(reversed(exemplars))
    blocks = [
        "Here are example translations in the required style:\n",
        *(f"Source: {e['input']}\nEnglish: {e['output']}\n" for e in ordered),
        "Now translate the following text into English in the same style:\n",
        f"Source: {source}\nEnglish:",
    ]
    return "\n".join(blocks)


def run(condition: str, cfg: dict) -> None:
    gen = cfg["generator"]
    style_instruction = Path(cfg["prompt"]["style_instruction_file"]).read_text(encoding="utf-8")
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
        print(f"Retrieving k={retr['k']} exemplars for {len(sources)} sources ...")
        retrieved = index.retrieve(sources, k=retr["k"])
        user_msgs = [build_knn_fewshot_user(s, ex) for s, ex in zip(sources, retrieved)]
    else:
        raise ValueError(f"unknown condition '{condition}' (expected reference|knn_fewshot)")

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
    parser.add_argument("--condition", required=True, choices=["reference", "knn_fewshot"])
    parser.add_argument("--config", default="configs/openai_smoke.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run(args.condition, cfg)


if __name__ == "__main__":
    main()
