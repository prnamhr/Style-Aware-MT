import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval import stylometrics  # noqa: E402
from src.eval.register_direction import (  # noqa: E402
    LEGACY_DIRECTION,
    compare_directions,
    derive_direction,
    load_direction_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/register_direction.json"
CENTROID = ROOT / "results/stylometrics_centroid.json"
TRAIN = ROOT / "data/splits/train.jsonl"
CANONICAL_CONFIGS = [
    "base_qwen.yaml",
    "afsp_sweep.yaml",
    "qwen_smoke.yaml",
    "peft_sweep.yaml",
    "peft_anchor_e3.yaml",
    "peft_smoke.yaml",
    "peft_afsp.yaml",
]


def _targets() -> list[str]:
    rows = [json.loads(line) for line in TRAIN.read_text(encoding="utf-8").splitlines() if line]
    return [row["output"] for row in rows if row.get("output", "").strip()]


def test_committed_direction_is_reproducible_from_corrected_centroid():
    centroid = json.loads(CENTROID.read_text(encoding="utf-8"))
    expected = load_direction_artifact(ARTIFACT, centroid=centroid)
    actual = derive_direction(_targets(), centroid)
    assert list(actual) == centroid["features"]
    assert np.allclose(
        [actual[name] for name in centroid["features"]],
        [expected[name] for name in centroid["features"]],
        rtol=0,
        atol=1e-12,
    )


def test_all_canonical_configs_read_the_one_direction_artifact():
    for name in CANONICAL_CONFIGS:
        cfg = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
        block = cfg.get("afsp") or cfg.get("register")
        assert block["style_register_direction_file"] == "results/register_direction.json"
        assert "style_register_direction" not in block



def test_committed_comparison_stays_below_the_afsp_rerun_gate():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    comparison = payload["comparison_to_legacy"]
    impact = payload["afsp_sweep_objective_impact"]
    assert comparison["classification"] == "near_identical"
    assert comparison["rerun_afsp_full"] is False
    assert impact["recommendation_changed"] is False
    assert impact["corrected_recommendation"]["tag"] == "afsp_k8_l0.75"

def test_very_different_direction_triggers_afsp_rerun():
    current = {
        "lex_density": -0.355,
        "ttr": 0.098,
        "root_ttr": -0.296,
        "marker_rate": 0.514,
    }
    comparison = compare_directions(current)
    assert comparison["classification"] == "very_different"
    assert comparison["rerun_afsp_full"] is True


def test_legacy_direction_reproduces_under_the_pre_casefix_marker_regex(monkeypatch):
    # Pins the DEVLOG provenance claim: the legacy vector was this same correlation
    # construction run against the case-sensitive archaic-marker regex.
    monkeypatch.setattr(
        stylometrics,
        "_MARKERS",
        re.compile(r"\b(thou|thee|thy|thine|art|hast|hath|dost|doth|shalt|wilt|unto|ye)\b|\bO\b"),
    )
    targets = _targets()
    old_centroid = stylometrics.build_centroid(targets, stylometrics.CENTROID_FEATURES)
    derived = derive_direction(targets, old_centroid)
    for name, legacy in LEGACY_DIRECTION.items():
        assert round(derived[name], 3) == round(legacy, 3), name
