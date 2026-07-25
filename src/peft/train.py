"""
PEFT (LoRA) supervised fine-tuning of the locked open-source base.
Usage:
    python -m src.peft.train --config configs/peft_qwen.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

# Zero-shot user directive. Kept byte-identical to
# ``src.infer.run.build_zeroshot_user`` so the PEFT inference condition sees the
# exact prompt the adapter was trained on. Replicated (not imported) to avoid
# pulling the commercial-API client imports of ``run.py`` into training.
_ZEROSHOT_USER = "Translate the following text into English:\n\n{source}"


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def build_example(
    tokenizer, style_instruction: str, source: str, target: str, max_seq_len: int
) -> dict:
    """Tokenize one (source, target) pair into a completion-masked example.

    The full chat sequence (system + user + assistant) is tokenized once; the
    prompt prefix (system + user + generation cue) is tokenized separately and
    its length used to mask the prompt tokens to ``-100``. Only the assistant
    (target) tokens contribute to the loss.
    """
    messages = [
        {"role": "system", "content": style_instruction},
        {"role": "user", "content": _ZEROSHOT_USER.format(source=source)},
        {"role": "assistant", "content": target},
    ]
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )

    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_seq_len]
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    labels = list(full_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100  # mask the prompt; supervise only the target tokens

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


class SFTDataset:
    """Minimal map-style dataset of pre-tokenized, completion-masked examples."""

    def __init__(self, rows: list[dict], tokenizer, style_instruction: str, max_seq_len: int):
        self.examples = [
            build_example(tokenizer, style_instruction, r["input"], r["output"], max_seq_len)
            for r in rows
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def train(cfg: dict) -> None:
    # Heavy deps imported lazily so `import`/`--help` need no GPU stack.
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    gen = cfg["generator"]
    pcfg = cfg["peft"]
    lora_cfg = pcfg["lora"]
    tcfg = pcfg["train"]
    seed = tcfg.get("seed", 42)
    set_seed(seed)

    base_model = gen["model"]  # single source of truth for the locked base
    torch_dtype = getattr(torch, gen.get("dtype", "bfloat16"))
    load_in_4bit = pcfg.get("load_in_4bit", False)  # QLoRA fallback
    max_seq_len = tcfg.get("max_seq_len", 1024)
    output_dir = pcfg["output_dir"]

    style_instruction = Path(cfg["prompt"]["style_instruction_file"]).read_text(encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {"dtype": torch_dtype}
    if load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = gen.get("device_map", "auto")
    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    if load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=tcfg.get("gradient_checkpointing", True)
        )

    lora = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        # README: LoRA on the query and value projections of each attention block.
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    if tcfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.config.use_cache = False  # incompatible with gradient checkpointing

    data = cfg["data"]
    train_rows = _read_jsonl(Path(data["train_file"]), data.get("limit"))
    eval_rows = _read_jsonl(Path(data["eval_file"]), data.get("limit"))
    print(f"Tokenizing {len(train_rows)} train / {len(eval_rows)} dev examples ...")
    train_ds = SFTDataset(train_rows, tokenizer, style_instruction, max_seq_len)
    eval_ds = SFTDataset(eval_rows, tokenizer, style_instruction, max_seq_len)

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding="longest", label_pad_token_id=-100
    )

    # When load_best_model_at_end is false (the sweep's fixed-epoch regime), every
    # epoch checkpoint is kept (set save_total_limit: null) so selection can treat
    # epoch as a tuned axis over (r, lr, epoch); eval_loss then only pre-filters
    # which checkpoints get generated, never picks the reported adapter.
    load_best = tcfg.get("load_best_model_at_end", True)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=tcfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 16),
        learning_rate=float(tcfg.get("learning_rate", 2e-4)),
        num_train_epochs=tcfg.get("num_train_epochs", 3),
        max_steps=tcfg.get("max_steps", -1),
        weight_decay=tcfg.get("weight_decay", 0.0),
        warmup_ratio=tcfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        bf16=(gen.get("dtype", "bfloat16") == "bfloat16"),
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        logging_steps=tcfg.get("logging_steps", 10),
        eval_strategy=tcfg.get("eval_strategy", "epoch"),
        save_strategy=tcfg.get("save_strategy", "epoch"),
        save_total_limit=tcfg.get("save_total_limit", 1),
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss" if load_best else None,
        greater_is_better=False if load_best else None,
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()

    # Save the best adapter + tokenizer to output_dir; this is the artifact the
    # `peft` inference condition loads (generator.adapter_path) and the RLSF init.
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    metrics = trainer.evaluate()
    (out / "train_metrics.json").write_text(
        json.dumps({"base_model": base_model, "seed": seed, **metrics}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved LoRA adapter to {out}/ ; final dev metrics: {metrics}")

    # Fixed-epoch regime: record every kept epoch checkpoint with its dev eval_loss
    # so the sweep can enumerate (r, lr, epoch) candidates. eval_loss here is a
    # generation pre-filter only -- the reported adapter is picked on COMET/Phi.
    if not load_best:
        manifest = []
        for entry in trainer.state.log_history:
            if "eval_loss" not in entry or "epoch" not in entry:
                continue
            step = int(entry.get("step", 0))
            ckpt = out / f"checkpoint-{step}"
            manifest.append(
                {
                    "epoch": int(round(entry["epoch"])),
                    "step": step,
                    "eval_loss": float(entry["eval_loss"]),
                    "checkpoint": str(ckpt) if ckpt.exists() else None,
                }
            )
        (out / "epoch_checkpoints.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        kept = [m for m in manifest if m["checkpoint"]]
        print(f"Recorded {len(kept)} epoch checkpoint(s) -> {out / 'epoch_checkpoints.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA supervised fine-tuning (PEFT condition).")
    parser.add_argument("--config", default="configs/peft_qwen.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train(cfg)


if __name__ == "__main__":
    main()
