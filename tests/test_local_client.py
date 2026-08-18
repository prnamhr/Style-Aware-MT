"""
Tests for in-place LoRA adapter swapping in the local generator.
"""

from __future__ import annotations

import pytest

from src.infer.local_client import LocalChatClient


class FakeModel:
    """Records the PEFT calls a swap makes, in order."""

    def __init__(self, fail_load: bool = False):
        self.calls: list[tuple[str, str]] = []
        self.fail_load = fail_load

    def load_adapter(self, model_id, adapter_name=None, **kwargs):
        if self.fail_load:
            raise RuntimeError("adapter_config.json not found")
        self.calls.append(("load", adapter_name))

    def set_adapter(self, adapter_name):
        self.calls.append(("set", adapter_name))

    def delete_adapter(self, adapter_name):
        self.calls.append(("delete", adapter_name))

    def eval(self):
        self.calls.append(("eval", ""))


def _client(model: FakeModel, adapter_name: str | None) -> LocalChatClient:
    """A client with the base already 'loaded', bypassing __post_init__."""
    client = LocalChatClient.__new__(LocalChatClient)
    client.model = "Qwen/Qwen2.5-7B-Instruct"
    client.adapter_path = "models/rlsf_grpo_w3_2.0/checkpoint-100"
    client._model = model
    client._adapter_name = adapter_name
    client._swaps = 0
    return client


def test_swap_loads_before_deleting_the_previous_adapter():
    model = FakeModel()
    client = _client(model, "default")

    client.swap_adapter("models/rlsf_grpo_w3_2.0/checkpoint-200")

    assert model.calls[:3] == [("load", "swap1"), ("set", "swap1"), ("delete", "default")]
    assert client._adapter_name == "swap1"
    assert client.adapter_path == "models/rlsf_grpo_w3_2.0/checkpoint-200"


def test_repeated_swaps_keep_one_adapter_resident():
    model = FakeModel()
    client = _client(model, "default")

    for step in (200, 400, 800):
        client.swap_adapter(f"models/rlsf_grpo_w3_2.0/checkpoint-{step}")

    loaded = [name for kind, name in model.calls if kind == "load"]
    deleted = [name for kind, name in model.calls if kind == "delete"]
    assert loaded == ["swap1", "swap2", "swap3"]
    assert deleted == ["default", "swap1", "swap2"]


def test_failed_load_leaves_the_previous_adapter_active():
    """A checkpoint that will not load must not cost the run the one already loaded."""
    model = FakeModel(fail_load=True)
    client = _client(model, "default")

    with pytest.raises(RuntimeError):
        client.swap_adapter("models/rlsf_grpo_w3_2.0/checkpoint-200")

    assert model.calls == []
    assert client._adapter_name == "default"
    assert client.adapter_path == "models/rlsf_grpo_w3_2.0/checkpoint-100"


def test_swap_without_an_adapter_is_refused():
    client = _client(FakeModel(), None)

    with pytest.raises(ValueError, match="adapter_path"):
        client.swap_adapter("models/rlsf_grpo_w3_2.0/checkpoint-200")
