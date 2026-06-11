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

from src.infer.usage import Usage

load_dotenv()  # populate os.environ from a .env file if one exists

# USD per 1M tokens, (input, output). Used for a rough spend estimate only.
# Unknown models fall back to (0, 0) -- cost is reported as 0, not an error.
# Reasoning tokens on gpt-5.x are billed as output and are included in
# usage.completion_tokens, so the output-rate estimate already covers them.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-5.5": (5.00, 30.00),
}


@dataclass
class ChatClient:
    model: str
    # temperature / seed are omitted from the request when None. Reasoning models
    # (gpt-5.x) reject a custom temperature, so leave it None for those.
    temperature: float | None = 0.0
    max_tokens: int = 1024
    seed: int | None = 42
    # reasoning_effort: none|low|medium|high|xhigh, only for reasoning models.
    reasoning_effort: str | None = None
    usage: Usage = field(default=None)
    _client: OpenAI = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # max_retries gives exponential backoff on rate-limit / transient 5xx.
        self._client = OpenAI(max_retries=5)
        self.usage = Usage(pricing=_PRICING)

    def complete(self, system: str, user: str) -> str:
        # max_completion_tokens is the current chat-completions field (accepted by
        # both reasoning and non-reasoning models); for reasoning models it must
        # also leave room for the hidden reasoning tokens.
        kwargs: dict = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort

        resp = self._client.chat.completions.create(**kwargs)
        u = resp.usage
        self.usage.add(self.model, u.prompt_tokens, u.completion_tokens)
        return (resp.choices[0].message.content or "").strip()
