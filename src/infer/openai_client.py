"""Thin OpenAI chat wrapper with retries and token/cost accounting.

Kept deliberately small and provider-isolated: the rest of the pipeline talks to
``ChatClient.complete(system, user)`` and never imports the OpenAI SDK directly,
so swapping in a local model later means replacing only this file.

The API key is read from the ``OPENAI_API_KEY`` environment variable, with a
``.env`` file in the project root loaded automatically if present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # populate os.environ from a .env file if one exists

# USD per 1M tokens, (input, output). Used for a rough spend estimate only.
# Unknown models fall back to (0, 0) -- cost is reported as 0, not an error.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    def add(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.calls += 1
        in_rate, out_rate = _PRICING.get(model, (0.0, 0.0))
        self.cost_usd += (prompt_tokens * in_rate + completion_tokens * out_rate) / 1e6

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 4),
        }


@dataclass
class ChatClient:
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int | None = 42
    usage: Usage = field(default_factory=Usage)
    _client: OpenAI = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # max_retries gives exponential backoff on rate-limit / transient 5xx.
        self._client = OpenAI(max_retries=5)

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        u = resp.usage
        self.usage.add(self.model, u.prompt_tokens, u.completion_tokens)
        return (resp.choices[0].message.content or "").strip()
