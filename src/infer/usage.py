"""
Token and cost accounting shared across generator providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Usage:
    pricing: dict[str, tuple[float, float]] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    def add(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.calls += 1
        in_rate, out_rate = self.pricing.get(model, (0.0, 0.0))
        self.cost_usd += (prompt_tokens * in_rate + completion_tokens * out_rate) / 1e6

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 4),
        }
