"""
OpenAI Batch API transport: submit many chat completions, collect them asynchronously.

"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.infer.usage import Usage

load_dotenv()

# Terminal states of a batch job; anything else is still worth polling.
_DONE = {"completed", "failed", "expired", "cancelled"}
_BATCH_MAX_REQUESTS = 50_000


@dataclass
class BatchChatClient:
    """Build, submit, poll and collect one batch of chat completions."""

    model: str
    temperature: float | None = None
    max_tokens: int = 256
    seed: int | None = 42
    reasoning_effort: str | None = None
    # (usd_per_1M_input, usd_per_1M_output) at LIST price.
    pricing: tuple[float, float] | None = None
    # Batch billing multiplier applied to `pricing`, so reported cost is what the
    # batch endpoint actually charges rather than the synchronous rate.
    discount: float = 0.5
    completion_window: str = "24h"
    usage: Usage = field(default=None)
    _client: OpenAI = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._client = OpenAI(max_retries=5)
        table = {}
        if self.pricing is not None:
            table[self.model] = tuple(r * self.discount for r in self.pricing)
        self.usage = Usage(pricing=table)

    @property
    def priced(self) -> bool:
        """Whether reported cost is real; False means cost_usd is a floor of 0."""
        return self.model in self.usage.pricing

    def build_request(self, custom_id: str, system: str, user: str) -> dict:
        """One JSONL line for the batch input file."""
        body: dict = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.seed is not None:
            body["seed"] = self.seed
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    def submit(self, requests: list[dict], work_dir: Path, label: str) -> str:
        """Upload the requests and create the batch; returns the batch id."""
        if not requests:
            raise ValueError("refusing to submit an empty batch")
        if len(requests) > _BATCH_MAX_REQUESTS:
            raise ValueError(
                f"{len(requests)} requests exceeds the {_BATCH_MAX_REQUESTS} per-batch limit; "
                f"split the work across batches"
            )
        seen = {r["custom_id"] for r in requests}
        if len(seen) != len(requests):
            raise ValueError("custom_id values must be unique within a batch")

        work_dir.mkdir(parents=True, exist_ok=True)
        input_path = work_dir / f"{label}_input.jsonl"
        input_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in requests) + "\n",
            encoding="utf-8",
        )
        uploaded = self._client.files.create(file=input_path.open("rb"), purpose="batch")
        batch = self._client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window=self.completion_window,
            metadata={"label": label},
        )
        return batch.id

    def poll(self, batch_id: str, *, interval: float = 30.0, timeout: float | None = None):
        """Block until the batch reaches a terminal state; returns the batch object."""
        waited = 0.0
        while True:
            batch = self._client.batches.retrieve(batch_id)
            counts = getattr(batch, "request_counts", None)
            done = getattr(counts, "completed", 0) if counts else 0
            total = getattr(counts, "total", 0) if counts else 0
            failed = getattr(counts, "failed", 0) if counts else 0
            print(
                f"  [{batch.status}] {done}/{total} completed"
                f"{f', {failed} failed' if failed else ''}",
                flush=True,
            )
            if batch.status in _DONE:
                return batch
            if timeout is not None and waited >= timeout:
                raise TimeoutError(
                    f"batch {batch_id} still {batch.status} after {waited:.0f}s; it is still "
                    f"running server-side -- resume polling rather than resubmitting"
                )
            time.sleep(interval)
            waited += interval

    def _read_jsonl_file(self, file_id: str) -> list[dict]:
        content = self._client.files.content(file_id)
        text = content.text if hasattr(content, "text") else content.read().decode("utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def collect(self, batch) -> dict[str, dict]:
        """Map ``custom_id`` -> ``{"text": str|None, "error": str|None}``.

        Token usage from every successful response is accumulated into ``self.usage``.
        A request that failed server-side yields ``text=None`` with its error, so the
        caller records a null score rather than losing the segment.
        """
        out: dict[str, dict] = {}
        if getattr(batch, "output_file_id", None):
            for rec in self._read_jsonl_file(batch.output_file_id):
                cid = rec.get("custom_id")
                resp = rec.get("response") or {}
                body = resp.get("body") or {}
                if resp.get("status_code") != 200 or "choices" not in body:
                    out[cid] = {"text": None, "error": f"status {resp.get('status_code')}"}
                    continue
                u = body.get("usage") or {}
                self.usage.add(self.model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
                content = (body["choices"][0].get("message") or {}).get("content") or ""
                out[cid] = {"text": content.strip(), "error": None}

        if getattr(batch, "error_file_id", None):
            for rec in self._read_jsonl_file(batch.error_file_id):
                cid = rec.get("custom_id")
                err = rec.get("error") or rec.get("response")
                out.setdefault(cid, {"text": None, "error": json.dumps(err)[:300]})
        return out
