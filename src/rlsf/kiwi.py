

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL = "Unbabel/wmt22-cometkiwi-da"


DEFAULT_PYTHON = Path(".venv-comet/bin/python")
_PYTHON_ENV_VAR = "RLSF_COMET_PYTHON"

_HANDSHAKE_TIMEOUT_S = 600.0


class KiwiError(RuntimeError):
    """The worker failed to start, died, or refused a request."""


def resolve_python(python: str | Path | None = None) -> Path:
    """Locate the COMET interpreter, explicit argument first, then env, then default."""
    candidate = python or os.environ.get(_PYTHON_ENV_VAR) or DEFAULT_PYTHON
    path = Path(candidate)
    if not path.exists():
        raise KiwiError(
            f"COMET interpreter not found at {path}. COMET runs in its own environment "
            f"because unbabel-comet pins transformers/numpy against this one; create it "
            f"with `python -m venv .venv-comet && .venv-comet/bin/pip install -r "
            f"requirements-comet.txt`, or point {_PYTHON_ENV_VAR} at an existing one."
        )
    return path


class KiwiScorer:
    """One long-lived COMET worker. Use as a context manager.

    >>> with KiwiScorer() as kiwi:                     # doctest: +SKIP
    ...     scores = kiwi.score(sources, hypotheses)
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        python: str | Path | None = None,
        batch_size: int = 16,
        gpus: int | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.gpus = gpus
        self._python = resolve_python(python)
        self._cwd = Path(cwd) if cwd is not None else Path.cwd()
        self._proc: subprocess.Popen | None = None
        self._next_id = 0
        self.reference_free = "kiwi" in model.lower()


    def start(self) -> None:
        if self._proc is not None:
            return
        cmd = [
            str(self._python),
            "-m",
            "src.rlsf._kiwi_worker",
            "--model",
            self.model,
            "--batch_size",
            str(self.batch_size),
        ]
        if self.gpus is not None:
            cmd += ["--gpus", str(self.gpus)]
        env = dict(os.environ)
        # The worker is addressed as a module of this repository, so the repo root has
        # to be importable by an interpreter that was never installed into it.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self._cwd), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(self._cwd),
            env=env,
        )
        hello = self._read_line(_HANDSHAKE_TIMEOUT_S)
        if not hello.get("ready"):
            self.stop()
            raise KiwiError(f"worker failed to start: {hello.get('error', hello)}")
        # Trust the worker's own view: it inspected the loaded checkpoint, this class
        # only guessed from the name.
        self.reference_free = bool(hello.get("reference_free", self.reference_free))

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.write(json.dumps({"shutdown": True}) + "\n")
                proc.stdin.flush()
                proc.stdin.close()
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
            proc.wait(timeout=10)

    def __enter__(self) -> KiwiScorer:
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()


    def score(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[str] | None = None,
    ) -> list[float]:
        """Per-segment scores, one per hypothesis, in input order.

        An empty batch returns ``[]`` without touching the worker -- a training step
        whose whole group was rejected by the length constraint is a real case.
        """
        if len(sources) != len(hypotheses):
            raise ValueError(
                f"sources and hypotheses differ in length: {len(sources)} vs {len(hypotheses)}"
            )
        if not sources:
            return []
        if not self.reference_free and references is None:
            raise ValueError(f"{self.model} is reference-based; references are required")
        if references is not None and len(references) != len(sources):
            raise ValueError(
                f"references differ in length: {len(references)} vs {len(sources)}"
            )

        if self._proc is None:
            self.start()
        req_id = self._next_id
        self._next_id += 1
        payload: dict = {"id": req_id, "src": list(sources), "mt": list(hypotheses)}
        if references is not None and not self.reference_free:
            payload["ref"] = list(references)

        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise KiwiError(f"worker died before request {req_id}: {self._stderr()}") from exc

        resp = self._read_line(None)
        if "error" in resp:
            raise KiwiError(f"request {req_id} failed: {resp['error']}")
        scores = resp.get("scores")
        if scores is None or len(scores) != len(hypotheses):
            raise KiwiError(
                f"request {req_id} returned {0 if scores is None else len(scores)} scores "
                f"for {len(hypotheses)} hypotheses"
            )
        return [float(x) for x in scores]

    # -- plumbing 

    def _read_line(self, timeout: float | None) -> dict:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise KiwiError("worker is not running")
        # readline blocks; the timeout argument is advisory and only documents intent
        # for the handshake, where a gated-model download can legitimately take minutes.
        del timeout
        line = proc.stdout.readline()
        if not line:
            raise KiwiError(f"worker closed stdout: {self._stderr()}")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise KiwiError(f"worker emitted non-JSON: {line[:200]!r}") from exc

    def _stderr(self) -> str:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return "(no stderr)"
        try:
            return (proc.stderr.read() or "(empty)")[-2000:]
        except Exception:
            return "(unreadable)"


def main() -> None:
    """Smoke the worker on a handful of segments: `python -m src.rlsf.kiwi`."""
    import argparse

    parser = argparse.ArgumentParser(description="COMET-Kiwi subprocess smoke check.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--python", default=None, help="COMET interpreter path")
    parser.add_argument("--split", default="train", help="split under data/splits/")
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()

    path = Path("data/splits") / f"{args.split}.jsonl"
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
            if len(rows) >= args.limit:
                break
    srcs = [r["input"] for r in rows]
    refs = [r["output"] for r in rows]

    with KiwiScorer(model=args.model, python=args.python) as kiwi:
        print(f"model={kiwi.model}  reference_free={kiwi.reference_free}")
        # Scoring the references as hypotheses: an upper anchor, not an evaluation.
        scores = kiwi.score(srcs, refs, None if kiwi.reference_free else refs)
    for s, r in zip(scores, refs):
        print(f"  {s:.4f}  {r[:70]}")
    print(f"mean={sum(scores) / len(scores):.4f} over {len(scores)} segments", file=sys.stderr)


if __name__ == "__main__":
    main()
