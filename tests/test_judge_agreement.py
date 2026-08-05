from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.judge import (  # noqa: E402
    assert_cache_identity,
    assert_results_identity,
    judge_results_path,
    judge_segment_dir,
    judge_stem,
    template_digest,
)
from src.eval.judge_agreement import (  # noqa: E402
    _load_pair,
    condition_ordering,
    contrast_replication,
    kappa_ci,
    kappa_draws,
    quadratic_weighted_kappa,
    rater_agreement,
)

BOOT = {"n_resamples": 200, "seed": 42, "alpha": 0.05}


# --- artefact routing: the second judge must not land on the first judge's files ---


def test_tag_routes_artefacts_away_from_the_primary_judge() -> None:
    assert judge_stem("val", None) == "judge_val"
    assert judge_stem("val", "gpt") == "judge_gpt_val"
    assert judge_results_path("results", "val", None) != judge_results_path("results", "val", "gpt")
    assert judge_segment_dir("results", "val", None) != judge_segment_dir("results", "val", "gpt")


def test_cache_identity_writes_meta_then_rejects_a_different_judge() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        meta = {"model": "claude-haiku-4-5", "tag": None, "template_sha256": "abc123"}
        assert_cache_identity(cache, meta)  # first use records identity
        assert json.loads((cache / "_meta.json").read_text())["model"] == "claude-haiku-4-5"
        assert_cache_identity(cache, meta)  # same judge resumes fine

        with pytest.raises(ValueError, match="different judge"):
            assert_cache_identity(cache, {**meta, "model": "gpt-5.6"})
        with pytest.raises(ValueError, match="different judge"):
            assert_cache_identity(cache, {**meta, "template_sha256": "deadbeef"})


def test_results_identity_blocks_overwriting_another_raters_phi() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "judge_val.json"
        path.write_text(json.dumps({"zeroshot": {"model": "claude-haiku-4-5", "mean": 2.5}}))

        assert_results_identity(path, "claude-haiku-4-5", allow_overwrite=False)  # same rater
        assert_results_identity(Path(tmp) / "absent.json", "gpt-5.6", allow_overwrite=False)
        assert_results_identity(path, "gpt-5.6", allow_overwrite=True)  # explicit opt-in
        with pytest.raises(ValueError, match="primary Phi"):
            assert_results_identity(path, "gpt-5.6", allow_overwrite=False)


def test_template_digest_is_stable_and_sensitive() -> None:
    assert template_digest("rubric text") == template_digest("rubric text")
    assert template_digest("rubric text") != template_digest("rubric text ")


# --- the agreement statistic itself ---


def test_qwk_endpoints_and_degenerate_cases() -> None:
    a = np.array([1, 2, 3, 4, 5, 3, 3, 2])
    assert quadratic_weighted_kappa(a, a) == pytest.approx(1.0)
    assert quadratic_weighted_kappa(np.array([]), np.array([])) != quadratic_weighted_kappa(
        np.array([]), np.array([])
    )  # NaN != NaN
    # Both raters constant: chance agreement is already perfect, kappa undefined.
    const = np.array([3, 3, 3, 3])
    assert np.isnan(quadratic_weighted_kappa(const, const))
    # One rater constant, the other not: expected disagreement is still non-zero.
    assert np.isfinite(quadratic_weighted_kappa(const, np.array([1, 2, 3, 4])))


def test_qwk_matches_the_textbook_definition() -> None:
    """Hand-computed against the weighted-kappa formula on a 2-level example."""
    a = np.array([1, 1, 5, 5])
    b = np.array([1, 5, 1, 5])
    # obs disagreement = 2/4 cells at distance 4 -> w = 1 ; marginals are 0.5/0.5
    # each way, so expected disagreement is also 0.5. kappa = 1 - 0.5/0.5 = 0.
    assert quadratic_weighted_kappa(a, b) == pytest.approx(0.0, abs=1e-12)


def test_qwk_rejects_scores_outside_the_frozen_rubric() -> None:
    with pytest.raises(ValueError, match="outside the 1-5"):
        quadratic_weighted_kappa(np.array([1, 2, 7]), np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="outside the 1-5"):
        quadratic_weighted_kappa(np.array([1, 2, 3.5]), np.array([1, 2, 3]))


def test_kappa_draws_reproducible_and_chunk_invariant() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(1, 6, 120)
    b = np.clip(a + rng.integers(-1, 2, 120), 1, 5)
    d1 = kappa_draws(a, b, n_resamples=300, seed=7)
    d2 = kappa_draws(a, b, n_resamples=300, seed=7)
    d3 = kappa_draws(a, b, n_resamples=300, seed=7, chunk=17)
    assert np.array_equal(d1, d2)
    assert np.allclose(d1, d3, equal_nan=True)  # chunking must not change the stream
    assert d1.size == 300


def test_kappa_ci_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(1)
    a = rng.integers(1, 6, 300)
    b = np.clip(a + rng.integers(-1, 2, 300), 1, 5)
    res = kappa_ci(a, b, n_resamples=500, seed=42, alpha=0.05)
    assert res["ci_low"] <= res["kappa"] <= res["ci_high"]
    assert res["n"] == 300


def test_kappa_ci_on_a_single_segment_is_degenerate_but_does_not_crash() -> None:
    """One pair cannot support an agreement estimate; it must still not raise.

    Every resample of a single pair is that same pair, so the interval collapses
    onto the point. The two single-pair cases differ: disagreeing raters give a
    defined 0.0 (expected disagreement is non-zero), agreeing raters give NaN.
    Neither is interpretable -- the collapsed interval is what signals that.
    """
    res = kappa_ci(np.array([3]), np.array([4]), n_resamples=50, seed=42, alpha=0.05)
    assert res["n"] == 1
    assert res["kappa"] == pytest.approx(0.0)
    assert res["ci_low"] == res["ci_high"] == pytest.approx(0.0)

    agreeing = kappa_ci(np.array([3]), np.array([3]), n_resamples=50, seed=42, alpha=0.05)
    assert np.isnan(agreeing["kappa"])
    assert agreeing["n_degenerate_resamples"] == 50


def test_kappa_ci_on_no_segments_returns_empty_rather_than_raising() -> None:
    res = kappa_ci(np.array([]), np.array([]), n_resamples=50, seed=42, alpha=0.05)
    assert res["n"] == 0
    assert np.isnan(res["ci_low"]) and np.isnan(res["ci_high"])


# --- loading, alignment, and the condition-level views ---


def _write_condition(root: Path, cond: str, scores_a: list, scores_b: list, split: str = "val"):
    """Lay down one condition's predictions plus both judges' segment caches."""
    out_dir, results = root / "outputs", root / "results"
    dir_a = judge_segment_dir(results, split, None)
    dir_b = judge_segment_dir(results, split, "gpt")
    for d in (out_dir, dir_a, dir_b):
        d.mkdir(parents=True, exist_ok=True)
    sources = [f"src-{i}" for i in range(len(scores_a))]
    (out_dir / f"{cond}_{split}.jsonl").write_text(
        "\n".join(
            json.dumps({"input": s, "prediction": f"p{i}", "output": f"r{i}"})
            for i, s in enumerate(sources)
        )
        + "\n"
    )
    for d, scores in ((dir_a, scores_a), (dir_b, scores_b)):
        (d / f"{cond}.jsonl").write_text(
            "\n".join(json.dumps({"input": s, "score": sc}) for s, sc in zip(sources, scores))
            + "\n"
        )
    return out_dir, dir_a, dir_b


def test_load_pair_counts_coverage_per_rater_and_intersects() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out_dir, dir_a, dir_b = _write_condition(root, "zeroshot", [3, 4, None, 2], [3, None, 5, 2])
        d = _load_pair(out_dir, "val", "zeroshot", dir_a, dir_b)
        assert (d["n_total"], d["n_a"], d["n_b"], d["n_both"]) == (4, 3, 3, 2)
        assert d["index"].tolist() == [0, 3]  # only where BOTH raters parsed


def test_load_pair_rejects_misaligned_judge_segments() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out_dir, dir_a, dir_b = _write_condition(root, "zeroshot", [3, 4], [3, 4])
        rows = [json.loads(x) for x in (dir_b / "zeroshot.jsonl").read_text().splitlines()]
        rows[1]["input"] = "a different segment"
        (dir_b / "zeroshot.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        with pytest.raises(ValueError, match="aligned index-for-index"):
            _load_pair(out_dir, "val", "zeroshot", dir_a, dir_b)


def test_load_pair_skips_a_condition_missing_from_one_judge() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out_dir, dir_a, dir_b = _write_condition(root, "zeroshot", [3, 4], [3, 4])
        (dir_b / "zeroshot.jsonl").unlink()
        assert _load_pair(out_dir, "val", "zeroshot", dir_a, dir_b) is None


def _loaded(pairs: dict[str, tuple[list, list]]) -> dict[str, dict]:
    """Build the in-memory structure the reporting functions consume."""
    out = {}
    for cond, (sa, sb) in pairs.items():
        a = np.asarray([np.nan if v is None else float(v) for v in sa])
        b = np.asarray([np.nan if v is None else float(v) for v in sb])
        idx = np.where(~np.isnan(a) & ~np.isnan(b))[0]
        out[cond] = {
            "condition": cond,
            "n_total": len(sa),
            "n_a": int((~np.isnan(a)).sum()),
            "n_b": int((~np.isnan(b)).sum()),
            "n_both": int(idx.size),
            "index": idx,
            "a_full": a,
            "b_full": b,
            "sources": [f"src-{i}" for i in range(len(sa))],
        }
    return out


def test_rater_agreement_pools_only_jointly_scored_segments() -> None:
    loaded = _loaded(
        {
            "zeroshot": ([1, 2, 3, None], [2, 2, 3, 4]),
            "peft": ([4, 5, 5, 4], [4, 4, 5, 5]),
        }
    )
    res = rater_agreement(loaded, ["zeroshot", "peft"], **BOOT)
    assert res["per_condition"]["zeroshot"]["n"] == 3
    assert res["pooled"]["n"] == 7  # 3 + 4, not 8
    assert 0.0 <= res["pooled"]["exact_agreement"] <= 1.0
    assert res["pooled"]["adjacent_agreement"] >= res["pooled"]["exact_agreement"]


def test_rater_agreement_skips_a_condition_with_no_shared_segment() -> None:
    loaded = _loaded({"zeroshot": ([1, None], [None, 2]), "peft": ([4, 5], [4, 5])})
    res = rater_agreement(loaded, ["zeroshot", "peft"], **BOOT)
    assert "zeroshot" not in res["per_condition"]
    assert res["pooled"]["n"] == 2


def test_condition_ordering_detects_a_rank_flip() -> None:
    agree = _loaded({"a": ([1, 1], [1, 1]), "b": ([3, 3], [3, 3]), "c": ([5, 5], [5, 5])})
    res = condition_ordering(agree, ["a", "b", "c"])
    assert res["ranking_a"] == res["ranking_b"] == ["c", "b", "a"]
    assert res["identical_ranking"] is True
    assert res["spearman"]["rho"] == pytest.approx(1.0)

    flip = _loaded({"a": ([1, 1], [5, 5]), "b": ([3, 3], [3, 3]), "c": ([5, 5], [1, 1])})
    res = condition_ordering(flip, ["a", "b", "c"])
    assert res["identical_ranking"] is False
    assert res["spearman"]["rho"] == pytest.approx(-1.0)


def test_contrast_replication_holds_both_raters_to_the_same_segments() -> None:
    # Segment 2 is unparsed for rater B on knn_fewshot, so it must drop from BOTH
    # raters' columns -- otherwise the two columns are not comparable.
    loaded = _loaded(
        {
            "zeroshot": ([1, 1, 1, 1], [2, 2, 2, 2]),
            "knn_fewshot": ([3, 3, 3, 3], [4, 4, None, 4]),
            "afsp_full": ([4, 4, 4, 4], [5, 5, 5, 5]),
        }
    )
    res = contrast_replication(
        loaded, [("knn_fewshot", "zeroshot"), ("afsp_full", "knn_fewshot")], **BOOT
    )
    knn = res["contrasts"]["knn_fewshot - zeroshot"]
    assert knn["n"] == 3  # the segment B could not score is excluded for A too
    assert knn["judge_a"]["n"] == knn["judge_b"]["n"] == 3
    assert knn["same_sign"] is True
    assert res["family_size"] == 2


def test_contrast_replication_flags_a_sign_flip_between_raters() -> None:
    loaded = _loaded(
        {
            "zeroshot": ([1, 1, 1, 1], [5, 5, 5, 5]),
            "afsp_full": ([4, 4, 4, 4], [2, 2, 2, 2]),
        }
    )
    res = contrast_replication(loaded, [("afsp_full", "zeroshot")], **BOOT)
    row = res["contrasts"]["afsp_full - zeroshot"]
    assert row["judge_a"]["diff"] > 0 > row["judge_b"]["diff"]
    assert row["same_sign"] is False


def test_contrast_replication_skips_a_contrast_whose_condition_is_absent() -> None:
    loaded = _loaded({"zeroshot": ([1, 2], [1, 2])})
    res = contrast_replication(loaded, [("afsp_full", "zeroshot")], **BOOT)
    assert res["contrasts"] == {}
    assert res["family_size"] == 0


def test_contrast_replication_rejects_conditions_scored_on_different_segments() -> None:
    loaded = _loaded({"zeroshot": ([1, 2], [1, 2]), "peft": ([3, 4], [3, 4])})
    loaded["peft"]["sources"] = ["other-0", "other-1"]
    with pytest.raises(ValueError, match="same segments in the same order"):
        contrast_replication(loaded, [("peft", "zeroshot")], **BOOT)


def test_identical_raters_agree_perfectly_and_report_zero_offset() -> None:
    loaded = _loaded({"zeroshot": ([1, 3, 5, 2, 4], [1, 3, 5, 2, 4])})
    res = rater_agreement(loaded, ["zeroshot"], **BOOT)["per_condition"]["zeroshot"]
    assert res["exact_agreement"] == 1.0
    assert res["qwk"]["kappa"] == pytest.approx(1.0)
    assert res["offset"]["diff"] == pytest.approx(0.0)
    assert res["offset"]["significant"] is False
