"""
Local open-source generator (HuggingFace ``transformers`` generate), interface
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from src.infer.usage import Usage

# Local weights have no per-token price; an empty pricing table keeps cost_usd at 0
# while token counts are still accumulated for throughput/repro bookkeeping.
_PRICING: dict[str, tuple[float, float]] = {}


def _generated_len(tokens, pad_id: int | None, eos_id: int | None) -> int:
    """Generated-token count with right-padding removed."""
    n = int(tokens.shape[0])
    if pad_id is None:
        return n
    keep_eos = pad_id == eos_id
    while n > 0 and int(tokens[n - 1]) == pad_id:
        n -= 1
    return n + 1 if keep_eos and n < int(tokens.shape[0]) else n


@dataclass
class LocalChatClient:
    model: str
    max_tokens: int = 1024  # max_new_tokens
    # temperature == 0 -> greedy (deterministic). > 0 -> sampling, RNG fixed by seed.
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    dtype: str = "bfloat16"
    device_map: str | None = None
    load_in_4bit: bool = False
    adapter_path: str | None = None
    usage: Usage = field(default=None)
    _tokenizer: object = field(default=None, repr=False)
    _model: object = field(default=None, repr=False)
    _device: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        set_seed(self.seed)
        self.usage = Usage(pricing=_PRICING)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model)

        torch_dtype = getattr(torch, self.dtype)
        load_kwargs: dict = {"dtype": torch_dtype}
        if self.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        if self.device_map is not None:
            load_kwargs["device_map"] = self.device_map

        self._model = AutoModelForCausalLM.from_pretrained(self.model, **load_kwargs)

        if self.adapter_path:
            # Lazy import: only the PEFT condition needs the peft dependency.
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

        if self.device_map is None and not self.load_in_4bit:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
        else:
            self._device = self._model.device

    def complete(self, system: str, user: str) -> str:
        """Single completion. Greedy unless ``temperature > 0``; the inference path
        every reported condition uses, so its behaviour is deliberately unchanged."""
        return self.complete_many(system, user, n=1)[0]

    def complete_many(
        self,
        system: str,
        user: str,
        *,
        n: int = 1,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> list[str]:
        
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        temp = self.temperature if temperature is None else temperature
        nucleus = self.top_p if top_p is None else top_p
        sampling = bool(temp and temp > 0)
        if n > 1 and not sampling:
            raise ValueError(
                f"n={n} with temperature={temp} is greedy: all {n} samples would be "
                "identical and every group-normalized advantage would be zero. "
                "Pass temperature > 0 for group sampling."
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict = {
            "max_new_tokens": self.max_tokens,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            "num_return_sequences": n,
        }
        if sampling:
            gen_kwargs.update(do_sample=True, temperature=temp, top_p=nucleus)
        else:
            gen_kwargs["do_sample"] = False  # greedy; deterministic

        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)


        pad_id = gen_kwargs["pad_token_id"]
        texts: list[str] = []
        completion_total = 0
        for row in out[:n]:
            new_tokens = row[prompt_len:]
            completion_total += _generated_len(new_tokens, pad_id, self._tokenizer.eos_token_id)
            texts.append(self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        self.usage.add(self.model, prompt_len, completion_total)
        return texts
