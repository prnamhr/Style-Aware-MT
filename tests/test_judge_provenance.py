"""
Tests for the generation-to-score provenance binding in the judge segment cache.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval._io import file_digest  # noqa: E402
from src.eval.judge import assert_cache_identity, bind_output  # noqa: E402

_IDENTITY = {"model": "claude-haiku-4-5", "tag": None, "template_sha256": "abcd1234"}


def _output(tmp: Path, text: str) -> Path:
    p = tmp / "peft_test.jsonl"
    p.write_text(json.dumps({"input": "s", "prediction": text, "output": "r"}) + "\n", "utf-8")
    return p


def test_digest_tracks_file_bytes(tmp_path):
    a = _output(tmp_path, "first")
    before = file_digest(a)
    assert before == file_digest(a)
    _output(tmp_path, "second")
    assert file_digest(a) != before


def test_binding_is_recorded_alongside_the_judge_identity(tmp_path):
    cache = tmp_path / "judge_test_segments"
    cache.mkdir()
    out = _output(tmp_path, "first")
    assert_cache_identity(cache, dict(_IDENTITY))

    digest = bind_output(cache, "peft", out)

    meta = json.loads((cache / "_meta.json").read_text(encoding="utf-8"))
    assert meta["outputs"]["peft"] == {"file": str(out), "sha256": digest}
    assert meta["template_sha256"] == "abcd1234"


def test_rebinding_the_same_bytes_is_a_no_op(tmp_path):
    cache = tmp_path / "judge_test_segments"
    cache.mkdir()
    out = _output(tmp_path, "first")
    first = bind_output(cache, "peft", out)

    assert bind_output(cache, "peft", out) == first


def test_regenerated_output_is_refused(tmp_path):
    """The hazard this exists for: cross-session decoding drift silently
    reattaching cached scores to different translations."""
    cache = tmp_path / "judge_test_segments"
    cache.mkdir()
    out = _output(tmp_path, "first")
    bind_output(cache, "peft", out)
    _output(tmp_path, "regenerated")

    with pytest.raises(ValueError, match="generation changed after it was judged"):
        bind_output(cache, "peft", out)


def test_a_second_condition_does_not_disturb_the_first(tmp_path):
    cache = tmp_path / "judge_test_segments"
    cache.mkdir()
    peft = _output(tmp_path, "first")
    bind_output(cache, "peft", peft)
    other = tmp_path / "zeroshot_test.jsonl"
    other.write_text("{}\n", encoding="utf-8")

    bind_output(cache, "zeroshot", other)

    meta = json.loads((cache / "_meta.json").read_text(encoding="utf-8"))
    assert set(meta["outputs"]) == {"peft", "zeroshot"}
    assert bind_output(cache, "peft", peft) == meta["outputs"]["peft"]["sha256"]


def test_identity_check_ignores_the_outputs_block(tmp_path):
    """assert_cache_identity compares only the keys it is given, so binding an
    output must not make the next condition look like a different judge."""
    cache = tmp_path / "judge_test_segments"
    cache.mkdir()
    assert_cache_identity(cache, dict(_IDENTITY))
    bind_output(cache, "peft", _output(tmp_path, "first"))

    assert_cache_identity(cache, dict(_IDENTITY))
