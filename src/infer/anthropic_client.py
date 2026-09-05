"""
Thin Anthropic (Claude) chat wrapper, interface-compatible with ChatClient.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from anthropic import Anthropic
from dotenv import load_dotenv

from src.infer.usage import Usage

load_dotenv()  # populate os.environ from a .env file if one exists

# USD per 1M tokens, (input, output). Estimate only; unknown models -> (0, 0).
_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-fable-5-1": (10.00, 50.00),
}

# Thinking cannot be switched off on these; an explicit thinking block is a 400.
_ALWAYS_THINKING = ("claude-fable-", "claude-mythos-")
# Sampling parameters were removed with Opus 4.7; sending one is a 400.
_NO_SAMPLING = _ALWAYS_THINKING + (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
)


@dataclass
class AnthropicChatClient:
    model: str
    max_tokens: int = 1024
    thinking: bool = False  # adaptive thinking; off keeps the smoke test fast/cheap
    temperature: float | None = None
    effort: str | None = None  # low|medium|high|xhigh|max; depth control where thinking is fixed
    pricing: tuple[float, float] | None = None  # rates for a model newer than the built-in table
    usage: Usage = field(default=None)
    _client: Anthropic = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Both rejections are per-call, so an unguarded config would empty a whole pass.
        if self.temperature is not None and self.model.startswith(_NO_SAMPLING):
            raise ValueError(f"{self.model} rejects temperature; leave it null in the config")
        if not self.thinking and self.model.startswith(_ALWAYS_THINKING):
            raise ValueError(
                f"{self.model} always thinks; a thinking-free row needs another model, "
                f"and output_config.effort is the only depth control here"
            )
        # max_retries gives exponential backoff on rate-limit / transient 5xx.
        self._client = Anthropic(max_retries=5)
        table = dict(_PRICING)
        if self.pricing is not None:
            table[self.model] = tuple(self.pricing)
        self.usage = Usage(pricing=table)

    @property
    def priced(self) -> bool:
        """Whether reported cost is real; False means cost_usd is a floor of 0."""
        return self.model in self.usage.pricing

    def complete(self, system: str, user: str) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        elif self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}

        resp = self._client.messages.create(**kwargs)
        self.usage.add(self.model, resp.usage.input_tokens, resp.usage.output_tokens)

        # Safety classifiers can decline with a 200 + stop_reason "refusal"; the
        # content block may be empty. Surface that as an empty string rather than
        # crashing the run.
        if resp.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in resp.content if b.type == "text").strip()
