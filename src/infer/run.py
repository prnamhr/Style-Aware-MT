"""
Test-set inference, provider-agnostic across the configured generator.

Usage:
    python -m src.infer.run --condition zeroshot    --config configs/openai_smoke.yaml
    python -m src.infer.run --condition knn_fewshot  --config configs/openai_smoke.yaml
    python -m src.infer.run --condition afsp_full    --config configs/base_qwen.yaml
    python -m src.infer.run --condition peft_afsp    --config configs/peft_afsp.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import yaml

from src.eval._io import read_completed_jsonl
from src.eval.stylometrics import fingerprint
from src.infer.anthropic_client import AnthropicChatClient
from src.infer.gemini_client import GeminiChatClient
from src.infer.local_client import LocalChatClient
from src.infer.openai_client import ChatClient
from src.retrieval.afsp import AFSPRetriever, load_centroid
from src.retrieval.rarity import load_irregular
from src.retrieval.retrieve import RetrievalIndex
from src.retrieval.sparse import ROUTES, SparseRetriever

# Demonstration ordering is a controlled experimental flag. Exemplars reach
ORDERINGS = ("most_similar_last", "most_similar_first", "random")

# Ablation ladder: each rung adds one component over the previous one.
CONDITIONS = (
    "zeroshot",
    "random_fewshot",
    "knn_fewshot",
    "afsp_margin",
    "afsp_full",
    "peft",
    "peft_knn",
    "peft_afsp",
    "sparse_knn",
)

# Generated with the LoRA adapter loaded; every other rung runs the frozen base.
ADAPTER_CONDITIONS = ("peft", "peft_knn", "peft_afsp")
# Rungs whose prompt carries exemplars, and the subset that applies the register rerank.
RETRIEVAL_CONDITIONS = (
    "random_fewshot",
    "knn_fewshot",
    "afsp_margin",
    "afsp_full",
    "peft_knn",
    "peft_afsp",
    "sparse_knn",
)
RERANK_CONDITIONS = ("afsp_full", "peft_afsp")


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
        return ChatClient(
            model=gen["model"],
            temperature=gen.get("temperature"),  # None -> omitted (reasoning models reject it)
            max_tokens=gen.get("max_tokens", 1024),
            seed=gen.get("seed"),
            reasoning_effort=gen.get("reasoning_effort"),
            pricing=gen.get("pricing"),  # rates for a model newer than the built-in table
        )
    if provider == "anthropic":
        return AnthropicChatClient(
            model=gen["model"],
            max_tokens=gen.get("max_tokens", 1024),
            thinking=gen.get("thinking", False),
            temperature=gen.get("temperature"),
        )
    if provider == "gemini":
        return GeminiChatClient(
            model=gen["model"],
            max_tokens=gen.get("max_tokens", 1024),
            temperature=gen.get("temperature"),  # None -> API default
            thinking_budget=gen.get("thinking_budget", 0),  # 0 disables thinking (2.5 models)
        )
    if provider == "local":
        return LocalChatClient(
            model=gen["model"],
            max_tokens=gen.get("max_tokens", 1024),
            temperature=gen.get("temperature", 0.0),
            top_p=gen.get("top_p", 1.0),
            seed=gen.get("seed", 42),
            dtype=gen.get("dtype", "bfloat16"),
            device_map=gen.get("device_map"),
            load_in_4bit=gen.get("load_in_4bit", False),
            adapter_path=gen.get("adapter_path"),  # LoRA adapter for the peft condition
            attn_implementation=gen.get("attn_implementation"),
        )
    raise ValueError(f"unknown provider '{provider}' (expected openai|anthropic|gemini|local)")


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
    """Assemble the few-shot user prompt shared by every retrieval condition."""
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
    """AFSP exemplar selection."""

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


def _select_sparse_knn(sources, cfg, index, k):
    """One nearest exemplar per rare query term for up to ``m`` slots, cosine top-k for the rest."""
    spa, rar = cfg.get("sparse", {}), cfg.get("rarity", {})
    retriever = SparseRetriever(
        index,
        load_irregular(rar.get("out", "results/rarity_train.json")),
        index,
        zwnj=rar.get("zwnj", "keep"),
        m=spa.get("m", 4),
    )
    selected, traces = retriever.select_with_trace(sources, k=k)
    routes = {r: sum(t["route"] == r for t in traces) for r in ROUTES}
    n_sparse = [t["n_sparse"] for t in traces]
    print(f"  routes: {routes}, mean rare slots filled: {sum(n_sparse) / len(n_sparse):.2f}")
    return selected


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


def _sha256(path: str | Path) -> str:
    """Digest of a generation-time input file, so the sidecar pins the exact list used."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _provenance(condition: str, cfg: dict) -> dict:
    """The settings a run was made at, so the sidecar records them rather than the config."""
    gen, prov = cfg["generator"], {}
    if gen.get("adapter_path"):
        prov["adapter_path"] = gen["adapter_path"]
    if condition in RETRIEVAL_CONDITIONS:
        retr, af = cfg["retrieval"], cfg.get("afsp", {})
        prov["k"] = retr["k"]
        prov["ordering"] = cfg.get("prompt", {}).get("ordering", "most_similar_last")
        prov["index_dir"] = retr["index_dir"]
        if condition in ("afsp_margin", "afsp_full", "peft_afsp"):
            rerank = condition in RERANK_CONDITIONS
            prov["beta"] = af.get("beta", 0.3)
            prov["lambda_style"] = af.get("lambda_style", 0.3) if rerank else 0.0
            prov["style_objective"] = af.get("style_objective", "bandpass")
            prov["style_target_sigma"] = af.get("style_target_sigma", 1.0)
            if rerank:
                # The centroid steers retrieval here, so it is a generation-time input.
                prov["centroid"] = {
                    "path": af["centroid_file"],
                    "fingerprint": fingerprint(load_centroid(af["centroid_file"])),
                }
        if condition == "sparse_knn":
            spa, rar = cfg.get("sparse", {}), cfg.get("rarity", {})
            rarity_file = rar.get("out", "results/rarity_train.json")
            prov["m"] = spa.get("m", 4)
            prov["min_df"] = rar.get("min_df", 1)
            prov["freeze_n"] = rar.get("freeze_n", 500)
            prov["rarity_file"] = rarity_file
            prov["rarity_sha256"] = _sha256(rarity_file)
    return prov


def resolve_out_name(condition: str, cfg: dict, override: str | None = None) -> str:
    name = str(override or cfg.get("output", {}).get("name") or condition).strip()
    if not name:
        raise ValueError("output name is empty; give a non-blank --out-name or output.name")
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(
            f"invalid output name '{name}': it is a filename stem, not a path — "
            f"it must not contain separators or start with '.'"
        )
    return name


def run(condition: str, cfg: dict, out_name: str | None = None) -> None:
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

    # Checked before retrieval so a missing adapter fails without loading an index.
    if condition in ADAPTER_CONDITIONS and not gen.get("adapter_path"):
        raise ValueError(
            f"condition '{condition}' requires generator.adapter_path in the config "
            f"(the trained LoRA adapter); see configs/peft_qwen.yaml"
        )

    # Build the per-segment user messages for the chosen ablation rung.
    # `peft` shares the zero-shot prompt: no exemplars, the register is carried by
    # the LoRA adapter weights (loaded via generator.adapter_path in make_client).
    if condition in ("zeroshot", "peft"):
        user_msgs = [build_zeroshot_user(s) for s in sources]
    elif condition in RETRIEVAL_CONDITIONS:
        retr = cfg["retrieval"]
        k = retr["k"]
        index = RetrievalIndex(retr["index_dir"], embed_model=retr["embed_model"])
        if condition == "random_fewshot":
            print(f"Sampling k={k} random exemplars for {len(sources)} sources ...")
            selected = _select_random(sources, index.pairs, k, rng)
        elif condition in ("knn_fewshot", "peft_knn"):
            print(f"Retrieving k={k} exemplars for {len(sources)} sources ({ordering}) ...")
            selected = index.retrieve(sources, k=k)
        elif condition == "sparse_knn":
            m = cfg.get("sparse", {}).get("m", 4)
            print(f"{condition}: k={k} as up to {m} rarity + cosine for {len(sources)} ...")
            selected = _select_sparse_knn(sources, cfg, index, k)
        else:  # afsp_margin | afsp_full | peft_afsp
            rerank = condition in RERANK_CONDITIONS
            print(f"{condition}: selecting k={k} for {len(sources)} sources ({ordering}) ...")
            selected = _select_afsp(sources, cfg, retr, index, k, rerank=rerank)
        ordered = [order_exemplars(ex, ordering, rng) for ex in selected]
        user_msgs = [build_fewshot_user(s, ex, glossary) for s, ex in zip(sources, ordered)]
    else:
        raise ValueError(f"unknown condition '{condition}' (expected {'|'.join(CONDITIONS)})")

    client = make_client(gen)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    name = resolve_out_name(condition, cfg, out_name)
    out_path = out_dir / f"{name}_{split}.jsonl"
    if name != condition:
        print(f"Output name overridden: condition '{condition}' -> {out_path}")

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
        print(f"resuming {name}: {start}/{len(test_rows)} already done")

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
    (out_dir / f"{name}_{split}_usage.json").write_text(
        json.dumps(
            {
                "condition": condition,
                "output_name": name,
                "model": gen["model"],
                "provenance": _provenance(condition, cfg),
                **usage,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")
    print(f"Usage: {usage}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Provider-agnostic eval-set inference.")
    parser.add_argument("--condition", required=True, choices=list(CONDITIONS))
    parser.add_argument("--config", default="configs/openai_smoke.yaml")
    parser.add_argument(
        "--out-name",
        default=None,
        metavar="STEM",
        help="filename stem for outputs/<STEM>_<split>.jsonl, overriding output.name",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run(args.condition, cfg, out_name=args.out_name)


if __name__ == "__main__":
    main()
