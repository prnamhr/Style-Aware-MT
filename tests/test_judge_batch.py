from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.judge_batch import (  # noqa: E402
    _CUSTOM_ID,
    append_results,
    read_state,
    resume_point,
    run_condition,
    state_path,
    write_state,
)
from src.infer.openai_batch import BatchChatClient  # noqa: E402

SOURCES = [f"src-{i}" for i in range(6)]
PREDS = [f"pred-{i}" for i in range(6)]
REFS = [f"ref-{i}" for i in range(6)]
TEMPLATE = "rubric {source} {reference} {prediction}"


def _cache(tmp: Path, rows: list[dict]) -> Path:
    p = tmp / "zeroshot.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def _returned(indices, text="Score: 4"):
    return {_CUSTOM_ID.format(i=i): {"text": text, "error": None} for i in indices}


# --- resume accounting ---


def test_resume_point_counts_the_scored_prefix() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = _cache(Path(tmp), [{"input": s, "score": 3} for s in SOURCES[:4]])
        assert resume_point(p, SOURCES) == 4


def test_resume_point_on_an_absent_cache_is_zero() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        assert resume_point(Path(tmp) / "nothing.jsonl", SOURCES) == 0


def test_resume_point_rejects_a_cache_from_different_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = _cache(Path(tmp), [{"input": "a different segment", "score": 3}])
        with pytest.raises(ValueError, match="resume misalignment"):
            resume_point(p, SOURCES)


def test_state_round_trips_and_is_absent_before_first_write() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = state_path(Path(tmp), "val", "gpt")
        assert read_state(path) == {}
        write_state(path, {"zeroshot": {"batch_id": "batch_1", "start": 0, "n": 6}})
        assert read_state(path)["zeroshot"]["batch_id"] == "batch_1"


def test_state_path_is_tag_scoped() -> None:
    assert state_path(Path("results"), "val", None) != state_path(Path("results"), "val", "gpt")


# --- writing returned scores back into the shared cache format ---


def test_append_writes_every_segment_when_the_batch_completed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = _cache(Path(tmp), [])
        written, failed = append_results(p, SOURCES, 0, _returned(range(6)), complete=True)
        assert (written, failed) == (6, 0)
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        assert [r["input"] for r in rows] == SOURCES
        assert {r["score"] for r in rows} == {4}


def test_append_resumes_from_the_existing_prefix() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = _cache(Path(tmp), [{"input": s, "score": 2} for s in SOURCES[:2]])
        written, _ = append_results(p, SOURCES, 2, _returned(range(2, 6)), complete=True)
        assert written == 4
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        assert [r["input"] for r in rows] == SOURCES  # contiguous and in order
        assert [r["score"] for r in rows] == [2, 2, 4, 4, 4, 4]


def test_incomplete_batch_stops_at_the_first_gap_so_the_tail_is_resubmitted() -> None:
    """An expired batch must not freeze nulls into the cache.

    Writing a null for a segment that was merely never run would mark the cache
    complete and that segment would never be retried -- silently dropping it from n.
    """
    with tempfile.TemporaryDirectory() as tmp:
        p = _cache(Path(tmp), [])
        returned = _returned([0, 1, 4, 5])  # 2 and 3 never came back
        written, _ = append_results(p, SOURCES, 0, returned, complete=False)
        assert written == 2  # stops at the gap, does not skip ahead to 4
        assert resume_point(p, SOURCES) == 2  # next run picks up exactly there


def test_completed_batch_records_a_null_for_a_missing_segment() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = _cache(Path(tmp), [])
        written, failed = append_results(p, SOURCES, 0, _returned([0, 1, 2, 3, 5]), complete=True)
        assert (written, failed) == (6, 1)
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        assert rows[4]["score"] is None
        assert "missing from batch output" in rows[4]["error"]


def test_unparseable_and_errored_responses_become_nulls_not_lost_segments() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = _cache(Path(tmp), [])
        collected = {
            _CUSTOM_ID.format(i=0): {"text": "no verdict here", "error": None},
            _CUSTOM_ID.format(i=1): {"text": None, "error": "status 429"},
            _CUSTOM_ID.format(i=2): {"text": "Score: 5", "error": None},
        }
        written, failed = append_results(p, SOURCES[:3], 0, collected, complete=True)
        assert (written, failed) == (3, 2)
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        assert [r["score"] for r in rows] == [None, None, 5]
        assert rows[1]["error"] == "status 429"


# --- request construction ---


def test_request_omits_temperature_and_seed_when_unset() -> None:
    c = BatchChatClient(model="terra", temperature=None, seed=None, reasoning_effort="none")
    body = c.build_request("seg_0", "sys", "user")["body"]
    assert "temperature" not in body and "seed" not in body
    assert body["reasoning_effort"] == "none"
    assert body["max_completion_tokens"] == 256


def test_request_shape_matches_the_batch_endpoint() -> None:
    c = BatchChatClient(model="terra")
    req = c.build_request("seg_7", "sys", "user")
    assert req["method"] == "POST" and req["url"] == "/v1/chat/completions"
    assert req["custom_id"] == "seg_7"
    assert [m["role"] for m in req["body"]["messages"]] == ["system", "user"]


def test_batch_pricing_applies_the_discount() -> None:
    c = BatchChatClient(model="terra", pricing=(2.0, 8.0), discount=0.5)
    assert c.usage.pricing["terra"] == (1.0, 4.0)
    assert c.priced is True
    c.usage.add("terra", 1_000_000, 1_000_000)
    assert c.usage.cost_usd == pytest.approx(5.0)  # 1.0 + 4.0, not 10.0


def test_unpriced_model_is_flagged_rather_than_reported_as_free() -> None:
    c = BatchChatClient(model="terra", pricing=None)
    assert c.priced is False
    c.usage.add("terra", 1_000_000, 1_000_000)
    assert c.usage.cost_usd == 0.0  # a floor, and `priced` is what says so


def test_submit_refuses_an_empty_or_duplicated_batch() -> None:
    c = BatchChatClient(model="terra")
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="empty batch"):
            c.submit([], Path(tmp), "label")
        dupe = [c.build_request("seg_0", "s", "u"), c.build_request("seg_0", "s", "u")]
        with pytest.raises(ValueError, match="unique"):
            c.submit(dupe, Path(tmp), "label")


# --- the money-critical path: never submit twice for the same segments ---


class _FakeBatch:
    def __init__(self, status="completed"):
        self.id = "batch_fake"
        self.status = status
        self.output_file_id = "out"
        self.error_file_id = None
        self.request_counts = None


class _RecordingClient(BatchChatClient):
    """Stands in for the API, counting submissions."""

    def __init__(self, **kw):
        super().__init__(model="terra", **kw)
        self.submissions = 0

    def __post_init__(self):
        self.usage = __import__("src.infer.usage", fromlist=["Usage"]).Usage(pricing={})
        self._client = None

    def submit(self, requests, work_dir, label):
        self.submissions += 1
        self.last_n = len(requests)
        return "batch_fake"

    def poll(self, batch_id, *, interval=30.0, timeout=None):
        return _FakeBatch(self.status)

    def collect(self, batch):
        return _returned(range(self.start, self.start + self.n_return))


def _run(client, tmp: Path, state: dict, cache_rows: list[dict]):
    cache = _cache(tmp, cache_rows)
    sfile = state_path(tmp, "val", "gpt")
    return (
        run_condition(
            client,
            TEMPLATE,
            "zeroshot",
            SOURCES,
            PREDS,
            REFS,
            cache_path=cache,
            work_dir=tmp / "work",
            state=state,
            state_file=sfile,
            poll_interval=0,
            timeout=None,
        ),
        cache,
        sfile,
    )


def test_run_condition_submits_only_the_unscored_tail() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        c = _RecordingClient()
        c.status, c.start, c.n_return = "completed", 2, 4
        scores, cache, _ = _run(c, Path(tmp), {}, [{"input": s, "score": 1} for s in SOURCES[:2]])
        assert c.submissions == 1
        assert c.last_n == 4  # only segments 2..5, not all six
        assert len(scores) == 6


def test_run_condition_resumes_an_in_flight_batch_instead_of_resubmitting() -> None:
    """The state file records a live batch; a re-run must poll it, not pay again."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _RecordingClient()
        c.status, c.start, c.n_return = "completed", 0, 6
        state = {"zeroshot": {"batch_id": "batch_fake", "start": 0, "n": 6}}
        scores, _, sfile = _run(c, Path(tmp), state, [])
        assert c.submissions == 0  # resumed, never resubmitted
        assert len(scores) == 6
        assert read_state(sfile) == {}  # cleared once complete


def test_run_condition_ignores_stale_state_from_a_different_resume_point() -> None:
    """State recorded for segments 0+ must not be reused when 2 are already scored."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _RecordingClient()
        c.status, c.start, c.n_return = "completed", 2, 4
        state = {"zeroshot": {"batch_id": "stale", "start": 0, "n": 6}}
        _run(c, Path(tmp), state, [{"input": s, "score": 1} for s in SOURCES[:2]])
        assert c.submissions == 1  # stale entry discarded, fresh batch for 2..5


def test_run_condition_skips_a_complete_condition_without_submitting() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        c = _RecordingClient()
        c.status, c.start, c.n_return = "completed", 0, 0
        scores, _, _ = _run(c, Path(tmp), {}, [{"input": s, "score": 3} for s in SOURCES])
        assert c.submissions == 0
        assert scores == [3] * 6
