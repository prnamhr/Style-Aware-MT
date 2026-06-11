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
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


@dataclass
class ChatClient:
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int | None = 42
    usage: Usage = field(default=None)
    _client: OpenAI = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # max_retries gives exponential backoff on rate-limit / transient 5xx.
        self._client = OpenAI(max_retries=5)
        self.usage = Usage(pricing=_PRICING)

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
