"""
Local open-source generator (HuggingFace ``transformers`` generate), interface
"""

from __future__ import annotations
from dataclasses import dataclass, field
from src.infer.usage import Usage
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch


# Local weights have no per-token price; an empty pricing table keeps cost_usd at 0
# while token counts are still accumulated for throughput/repro bookkeeping.
_PRICING: dict[str, tuple[float, float]] = {}


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
        self._model.eval()

        if self.device_map is None and not self.load_in_4bit:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
        else:
            self._device = self._model.device

    def complete(self, system: str, user: str) -> str:

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
        }
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
        else:
            gen_kwargs["do_sample"] = False  # greedy; deterministic

        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)

        # Slice off the prompt so only newly generated tokens are decoded/counted.
        new_tokens = out[0][prompt_len:]
        completion_len = int(new_tokens.shape[0])
        self.usage.add(self.model, prompt_len, completion_len)
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()
