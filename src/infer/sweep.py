"""
Model x shot-count sweep for the commercial-API smoke path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from sacrebleu.metrics import BLEU, CHRF

from src.eval.stylometrics import _MARKERS
from src.infer.run import _read_jsonl, build_knn_fewshot_user, build_reference_user, make_client

# Model registry for the smoke sweep. Keys are the --models names; each maps to a
# generator block for src.infer.run.make_client. Cheap by default; flagships opt-in.
MODEL_REGISTRY: dict[str, dict] = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 1024,
        "seed": 42,
    },
    "claude-haiku-4-5": {
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
        "max_tokens": 1024,
        "thinking": False,
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "thinking": False,
    },
    # Flagships -- pricey, opt in explicitly via --models.
    "gpt-5.5": {
        "provider": "openai",
        "model": "gpt-5.5",
        "max_tokens": 4096,
        "reasoning_effort": "low",
    },
    "claude-opus-4-8": {
        "provider": "anthropic",
        "model": "claude-opus-4-8",
        "max_tokens": 1024,
        "thinking": False,
    },
}

DEFAULT_MODELS = ["gpt-4o-mini", "claude-haiku-4-5", "claude-sonnet-4-6"]
DEFAULT_N = [0, 2, 4, 8]


def _marker_rate(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return sum(len(_MARKERS.findall(t)) for t in texts) / len(texts)


def _tag(model: str) -> str:
    """Filesystem-safe run tag from a model id."""
    return model.replace("/", "-")


def _flat(text: str) -> str:
    """Collapse whitespace so a field is safe inside one TSV cell."""
    return " ".join(text.split())


def _build_user_messages(sources: list[str], n: int, cfg: dict) -> tuple[str, list[str]]:
    """Return (condition, per-source user messages) for shot count n."""
    if n == 0:
        return "reference", [build_reference_user(s) for s in sources]

    from src.retrieval.retrieve import RetrievalIndex

    retr = cfg["retrieval"]
    index = RetrievalIndex(retr["index_dir"], embed_model=retr["embed_model"])
    print(f"  retrieving k={n} exemplars for {len(sources)} sources ...")
    retrieved = index.retrieve(sources, k=n)
    return "knn_fewshot", [build_knn_fewshot_user(s, ex) for s, ex in zip(sources, retrieved)]


def run_one(
    model_name: str,
    gen: dict,
    condition: str,
    user_msgs: list[str],
    style: str,
    rows: list[dict],
    out_dir: Path,
    split: str,
    n: int,
) -> dict:
    """Run a single (model, n) combination; write its JSONL and return its metrics row."""
    client = make_client(gen)  # fresh client -> per-run usage/cost is isolated
    preds: list[str] = []
    print(f"[{model_name} | n={n} | {condition}] generating {len(rows)} ...")
    for i, user in enumerate(user_msgs, 1):
        preds.append(client.complete(style, user))
        if i % 10 == 0 or i == len(user_msgs):
            print(f"    {i}/{len(user_msgs)}")

    out_path = out_dir / f"{_tag(model_name)}__n{n}_{split}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row, pred in zip(rows, preds):
            f.write(
                json.dumps(
                    {
                        "input": row["input"],
                        "output": row["output"],
                        "prediction": pred,
                        "condition": condition,
                        "model": gen["model"],
                        "n": n,
                        "metadata": row.get("metadata", {}),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    refs = [r["output"] for r in rows]
    usage = client.usage.summary()
    return {
        "model": model_name,
        "n": n,
        "condition": condition,
        "BLEU": round(BLEU().corpus_score(preds, [refs]).score, 2),
        "chrF": round(CHRF().corpus_score(preds, [refs]).score, 2),
        "marker_rate": round(_marker_rate(preds), 2),
        "ref_marker_rate": round(_marker_rate(refs), 2),
        "cost_usd": usage["cost_usd"],
        "calls": usage["calls"],
        "_preds": preds,  # kept for human-eval dump; stripped before JSON
    }


def _print_table(results: list[dict]) -> None:
    cols = ["model", "n", "condition", "BLEU", "chrF", "marker_rate", "ref_marker_rate", "cost_usd"]
    ranked = sorted(results, key=lambda r: r["chrF"], reverse=True)
    widths = {c: max(len(c), *(len(str(r[c])) for r in ranked)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in ranked:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def write_human_eval(results: list[dict], rows: list[dict], results_dir: Path, split: str) -> None:
    """Write a scoring TSV (one row per segment x run) and a readable Markdown digest."""
    tsv_path = results_dir / f"human_eval_{split}.tsv"
    header = [
        "segment_id",
        "model",
        "n",
        "condition",
        "source",
        "reference",
        "prediction",
        "adequacy_1to5",
        "style_1to5",
        "notes",
    ]
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in results:
            for sid, (row, pred) in enumerate(zip(rows, r["_preds"])):
                f.write(
                    "\t".join(
                        [
                            str(sid),
                            r["model"],
                            str(r["n"]),
                            r["condition"],
                            _flat(row["input"]),
                            _flat(row["output"]),
                            _flat(pred),
                            "",
                            "",
                            "",
                        ]
                    )
                    + "\n"
                )

    # Markdown: grouped by segment so all systems sit under one source for easy reading.
    md_path = results_dir / f"human_eval_{split}.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Human evaluation — {split} ({len(rows)} segments)\n\n")
        f.write(
            "Score each prediction in `human_eval_" + split + ".tsv` "
            "(adequacy = meaning preserved, style = Shoghi Effendi register).\n\n"
        )
        ordered = sorted(results, key=lambda r: (r["model"], r["n"]))
        for sid, row in enumerate(rows):
            f.write(f"## Segment {sid}\n\n")
            f.write(f"- **Source:** {row['input']}\n")
            f.write(f"- **Reference:** {row['output']}\n\n")
            for r in ordered:
                f.write(f"- `{r['model']} n={r['n']}` — {r['_preds'][sid]}\n")
            f.write("\n")
    print(f"\nWrote human-eval files: {tsv_path}  and  {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Model x shot-count smoke sweep.")
    parser.add_argument(
        "--config",
        default="configs/base_qwen.yaml",
        help="supplies prompt/retrieval/data blocks (generator is overridden per model)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=list(MODEL_REGISTRY),
        help="models to compare",
    )
    parser.add_argument(
        "--n", nargs="+", type=int, default=DEFAULT_N, help="shot counts (0 = reference)"
    )
    parser.add_argument("--limit", type=int, default=25, help="segments from the eval split")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    style = Path(cfg["prompt"]["style_instruction_file"]).read_text(encoding="utf-8")
    eval_file = Path(cfg["data"]["eval_file"])
    split = eval_file.stem
    rows = _read_jsonl(eval_file, args.limit)
    sources = [r["input"] for r in rows]
    print(f"Sweep: models={args.models} n={args.n} on {len(rows)} '{split}' segments")

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Retrieval is model-independent, so build each n's prompts once and reuse.
    prompts_by_n = {n: _build_user_messages(sources, n, cfg) for n in args.n}

    results: list[dict] = []
    for model_name in args.models:
        gen = MODEL_REGISTRY[model_name]
        for n in args.n:
            condition, user_msgs = prompts_by_n[n]
            results.append(
                run_one(model_name, gen, condition, user_msgs, style, rows, out_dir, split, n)
            )

    _print_table(results)
    write_human_eval(results, rows, results_dir, split)

    lb_path = results_dir / f"sweep_leaderboard_{split}.json"
    lb_path.write_text(
        json.dumps([{k: v for k, v in r.items() if k != "_preds"} for r in results], indent=2),
        encoding="utf-8",
    )
    print(f"Wrote leaderboard: {lb_path}")


if __name__ == "__main__":
    main()
