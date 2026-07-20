
from __future__ import annotations

import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from src.infer.usage import Usage

load_dotenv()  # populate os.environ from a .env file if one exists

_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.0, 0.0),
    "gemini-2.5-flash-lite": (0.0, 0.0),
    "gemini-2.5-pro": (0.0, 0.0),
    "gemini-2.0-flash": (0.0, 0.0),
}


@dataclass
class GeminiChatClient:
    model: str
    max_tokens: int = 1024
    temperature: float | None = 0.0

    thinking_budget: int = 0
    max_retries: int = 5
    usage: Usage = field(default=None)
    _client: genai.Client = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._client = genai.Client() 
        self.usage = Usage(pricing=_PRICING)

    def _config(self, system: str) -> types.GenerateContentConfig:
        kwargs: dict = {
            "system_instruction": system,
            "max_output_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        # thinking_config is only meaningful on 2.5-series models.
        if self.model.startswith("gemini-2.5"):
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        return types.GenerateContentConfig(**kwargs)

    def complete(self, system: str, user: str) -> str:
        config = self._config(system)
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=user, config=config
                )
                break
            except errors.APIError as e:  # 429 (quota/rate) and 5xx are retryable
                code = getattr(e, "code", None)
                if code != 429 and not (isinstance(code, int) and code >= 500):
                    raise
                if attempt == self.max_retries:
                    raise
                time.sleep(min(2 ** attempt, 32))  # exponential backoff, capped

        u = resp.usage_metadata
        prompt_tokens = getattr(u, "prompt_token_count", 0) or 0
        # candidates + any thinking tokens both count as output on Gemini.
        out_tokens = (getattr(u, "candidates_token_count", 0) or 0) + (
            getattr(u, "thoughts_token_count", 0) or 0
        )
        self.usage.add(self.model, prompt_tokens, out_tokens)

        # resp.text is None when the candidate was blocked or produced no text part;
        # surface that as an empty string so the judge records a null rather than crashing.
        return (resp.text or "").strip()
