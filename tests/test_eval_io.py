"""
Tests for the shared evaluation IO helpers.
"""

from __future__ import annotations

import json

import pytest

from src.eval._io import merge_results


def _write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def test_merge_preserves_unscored_conditions(tmp_path):
    """The regression that lost the AFSP/PEFT scores: scoring one condition
    must not drop the conditions absent from this run."""
    path = tmp_path / "comet_val.json"
    _write(path, {"zeroshot": {"system": 0.1}, "afsp_full": {"system": 0.2}})

    preserved = merge_results(path, {"commercial_haiku": {"system": 0.3}})

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert set(stored) == {"zeroshot", "afsp_full", "commercial_haiku"}
    assert stored["afsp_full"] == {"system": 0.2}
    assert preserved == ["afsp_full", "zeroshot"]


def test_merge_overwrites_rescored_condition(tmp_path):
    path = tmp_path / "comet_val.json"
    _write(path, {"zeroshot": {"system": 0.1}})

    preserved = merge_results(path, {"zeroshot": {"system": 0.9}})

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == {"zeroshot": {"system": 0.9}}
    assert preserved == []


def test_merge_creates_missing_file(tmp_path):
    path = tmp_path / "nested" / "judge_val.json"

    preserved = merge_results(path, {"zeroshot": {"mean": 3.0}})

    assert json.loads(path.read_text(encoding="utf-8")) == {"zeroshot": {"mean": 3.0}}
    assert preserved == []


def test_merge_refuses_to_clobber_unreadable_file(tmp_path):
    """Better to fail loudly than to silently replace results we can't read."""
    path = tmp_path / "comet_val.json"
    path.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        merge_results(path, {"zeroshot": {"system": 0.1}})

    assert path.read_text(encoding="utf-8") == "{ this is not json"


def test_merge_leaves_no_temp_file(tmp_path):
    path = tmp_path / "comet_val.json"
    _write(path, {"zeroshot": {"system": 0.1}})

    merge_results(path, {"peft": {"system": 0.4}})

    assert [p.name for p in tmp_path.iterdir()] == ["comet_val.json"]
