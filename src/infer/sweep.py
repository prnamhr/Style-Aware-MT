"""
Model x shot-count knn_fewshot sweep on the commercial-API path.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml
from sacrebleu.metrics import BLEU, CHRF

from src.eval.human import build_records, write_scoring_sheet
from src.eval.stylometrics import _MARKERS
from src.infer.run import (
    _read_jsonl,
    build_fewshot_user,
    build_zeroshot_user,
    make_client,
    order_exemplars,
)
from src.retrieval.retrieve import RetrievalIndex
from src.infer.anthropic_client import _PRICING as AP
from src.infer.openai_client import _PRICING as OP

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

# --- Cost pre-estimate ---------------------------------------------------------
_EST_CHARS = {"src": 78, "ex_tgt": 147, "style": 889, "out": 140, "scaffold": 115, "labels": 20}


def _rates() -> dict[str, tuple[float, float]]:
    """(input, output) USD-per-1M-token rates, pulled from the client pricing tables."""

    return {**OP, **AP}


def estimate_tokens(n: int) -> tuple[float, float]:
    """Approximate (prompt_tokens, completion_tokens) for one knn_fewshot call at n shots."""
    e = _EST_CHARS

    def en(c: float) -> float:
        return c / 4.0  # Latin script

    def fa(c: float) -> float:
        return c / 2.5  # Persian/Arabic script

    out = en(e["out"])
    if n == 0:  # zeroshot user prompt
        return en(e["style"]) + en(44) + fa(e["src"]), out
    per_ex = fa(e["src"]) + en(e["ex_tgt"]) + en(e["labels"])
    prompt = en(e["style"]) + en(e["scaffold"]) + n * per_ex + fa(e["src"]) + en(16)
    return prompt, out


def estimate_cost(model: str, n: int, n_segments: int) -> float:
    """Estimated USD for one (model, n) cell over n_segments calls (0 if pricing unknown)."""
    in_rate, out_rate = _rates().get(model, (0.0, 0.0))
    pt, ot = estimate_tokens(n)
    return (pt * in_rate + ot * out_rate) / 1e6 * n_segments


def print_cost_estimate(models: list[str], ns: list[int], n_segments: int) -> float:
    """Print a per-cell + total cost estimate; return the grand total."""
    print(f"\nEstimated cost -- heuristic; actual lands in the leaderboard ({n_segments} seg/cell)")
    header = "model".ljust(18) + " ".join(f"n={n}".rjust(9) for n in ns) + "     row"
    print(header)
    print("-" * len(header))
    grand = 0.0
    for m in models:
        row = [estimate_cost(m, n, n_segments) for n in ns]
        grand += sum(row)
        print(m.ljust(18) + " ".join(f"${c:7.2f}" for c in row) + f"  ${sum(row):7.2f}")
    print("-" * len(header))
    print(f"{'TOTAL'.ljust(18)}{' ' * (len(header) - 18 - 10)}${grand:9.2f}")
    print(f"total calls: {len(models) * len(ns) * n_segments}\n")
    return grand


def _marker_rate(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return sum(len(_MARKERS.findall(t)) for t in texts) / len(texts)


def _tag(model: str) -> str:
    """Filesystem-safe run tag from a model id."""
    return model.replace("/", "-")


def _build_user_messages(sources: list[str], n: int, cfg: dict) -> tuple[str, list[str]]:
    """Return (condition, per-source user messages) for shot count n."""
    if n == 0:
        return "zeroshot", [build_zeroshot_user(s) for s in sources]

    retr = cfg["retrieval"]
    prompt_cfg = cfg.get("prompt", {})
    ordering = prompt_cfg.get("ordering", "most_similar_last")
    rng = random.Random(prompt_cfg.get("ordering_seed", 42))
    index = RetrievalIndex(retr["index_dir"], embed_model=retr["embed_model"])
    print(f"  retrieving k={n} exemplars for {len(sources)} sources ({ordering}) ...")
    retrieved = index.retrieve(sources, k=n)
    ordered = [order_exemplars(ex, ordering, rng) for ex in retrieved]
    return "knn_fewshot", [build_fewshot_user(s, ex) for s, ex in zip(sources, ordered)]


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


def write_human_eval(
    results: list[dict],
    rows: list[dict],
    results_dir: Path,
    split: str,
    *,
    sample: int,
    seed: int,
    blind: bool,
) -> None:
    """Emit a sampled, optionally-blind human-judgment sheet for the sweep grid.
    """
    sources = [r["input"] for r in rows]
    refs = [r["output"] for r in rows]
    # Reshape each grid cell into the (sources, preds, refs) triple build_records wants.
    systems = [f"{r['model']} n={r['n']}" for r in results]
    data = {sys: (sources, r["_preds"], refs) for sys, r in zip(systems, results)}

    n = len(rows)
    if sample and sample < n:
        sample_ids = sorted(random.Random(seed).sample(range(n), sample))
    else:
        sample_ids = list(range(n))

    records, blind_key = build_records(systems, data, sample_ids, blind=blind, seed=seed)
    tsv_path, md_path = write_scoring_sheet(records, results_dir, split)
    print(f"\nWrote human-eval files: {tsv_path}  and  {md_path}")
    print(f"  systems={len(systems)}  segments={len(sample_ids)}  blind={blind}")
    if blind_key:
        key_path = results_dir / f"human_eval_{split}_key.json"
        key_path.write_text(json.dumps(blind_key, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  blind key (do not open before scoring): {key_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Model x shot-count knn_fewshot sweep.")
    parser.add_argument(
        "--config",
        default="configs/base_qwen.yaml",
        help="supplies sweep/prompt/retrieval/data blocks (generator is overridden per model)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(MODEL_REGISTRY),
        help="models to compare (default: config sweep.models, else the cheap trio)",
    )
    parser.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=None,
        help="shot counts, 0 = zeroshot reference (default: config sweep.n)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="segments from the eval split (default: config data.limit; null = full split)",
    )
    parser.add_argument(
        "--human-sample",
        type=int,
        default=40,
        help="segments sampled into the human-eval sheet (0 = every segment)",
    )
    parser.add_argument("--human-seed", type=int, default=42, help="seed for the human-eval sample")
    parser.add_argument(
        "--blind",
        action="store_true",
        help="blind the human-eval sheet (hide model/n behind random letters)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the cost estimate for the configured grid and exit (no API calls)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sweep_cfg = cfg.get("sweep", {})
    # Config supplies grid/scope defaults; CLI flags override for one-off runs.
    models = args.models or sweep_cfg.get("models", DEFAULT_MODELS)
    ns = args.n if args.n is not None else sweep_cfg.get("n", DEFAULT_N)
    limit = args.limit if args.limit is not None else cfg["data"].get("limit")

    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"unknown model(s) {unknown}; known: {list(MODEL_REGISTRY)}")

    style = Path(cfg["prompt"]["style_instruction_file"]).read_text(encoding="utf-8")
    eval_file = Path(cfg["data"]["eval_file"])
    split = eval_file.stem
    rows = _read_jsonl(eval_file, limit)
    sources = [r["input"] for r in rows]
    print(f"Sweep: models={models} n={ns} on {len(rows)} '{split}' segments")

    # Document the spend up front. The ACTUAL per-cell cost is recorded alongside
    # each cell's scores in results/sweep_leaderboard_<split>.json after the run.
    print_cost_estimate(models, ns, len(rows))
    if args.dry_run:
        print("--dry-run: no API calls made.")
        return

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Retrieval is model-independent, so build each n's prompts once and reuse.
    prompts_by_n = {n: _build_user_messages(sources, n, cfg) for n in ns}

    results: list[dict] = []
    for model_name in models:
        gen = MODEL_REGISTRY[model_name]
        for n in ns:
            condition, user_msgs = prompts_by_n[n]
            results.append(
                run_one(model_name, gen, condition, user_msgs, style, rows, out_dir, split, n)
            )

    _print_table(results)
    write_human_eval(
        results,
        rows,
        results_dir,
        split,
        sample=args.human_sample,
        seed=args.human_seed,
        blind=args.blind,
    )

    lb_path = results_dir / f"sweep_leaderboard_{split}.json"
    lb_path.write_text(
        json.dumps([{k: v for k, v in r.items() if k != "_preds"} for r in results], indent=2),
        encoding="utf-8",
    )
    print(f"Wrote leaderboard: {lb_path}")


if __name__ == "__main__":
    main()
