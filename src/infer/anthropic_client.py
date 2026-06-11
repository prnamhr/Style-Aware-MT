"""Thin Anthropic (Claude) chat wrapper, interface-compatible with ``ChatClient``.

Exposes the same ``complete(system, user) -> str`` surface and ``.usage`` as the
OpenAI wrapper, so the pipeline is provider-agnostic. The API key is read from
``ANTHROPIC_API_KEY`` (or a project-root ``.env``).

Note on parameters: on Claude Opus 4.x and Fable 5, ``temperature`` / ``top_p`` /
``seed`` are removed and return a 400 if sent, so this client never passes them.
Determinism is steered through the prompt instead. The style instruction already
asks for output-only (no preamble), which also keeps reasoning from leaking into
the visible response when thinking is off.
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
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
}


@dataclass
class AnthropicChatClient:
    model: str
    max_tokens: int = 1024
    thinking: bool = False  # adaptive thinking; off keeps the smoke test fast/cheap
    usage: Usage = field(default=None)
    _client: Anthropic = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # max_retries gives exponential backoff on rate-limit / transient 5xx.
        self._client = Anthropic(max_retries=5)
        self.usage = Usage(pricing=_PRICING)

    def complete(self, system: str, user: str) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        resp = self._client.messages.create(**kwargs)
        self.usage.add(self.model, resp.usage.input_tokens, resp.usage.output_tokens)

        # Safety classifiers can decline with a 200 + stop_reason "refusal"; the
        # content block may be empty. Surface that as an empty string rather than
        # crashing the run.
        if resp.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in resp.content if b.type == "text").strip()
