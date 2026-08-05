

from __future__ import annotations

import argparse
import json
import sys

DEFAULT_MODEL = "Unbabel/wmt22-cometkiwi-da"


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent COMET scoring worker.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=None)
    args = parser.parse_args()


    try:
        from comet import download_model, load_from_checkpoint
    except Exception as exc:  # pragma: no cover - environment-dependent
        _emit({"ready": False, "error": f"cannot import comet: {exc}"})
        raise SystemExit(1) from exc

    try:
        model = load_from_checkpoint(download_model(args.model))
    except Exception as exc:  # pragma: no cover - environment-dependent

        _emit({"ready": False, "error": f"cannot load {args.model}: {exc}"})
        raise SystemExit(1) from exc

    if args.gpus is None:
        try:
            import torch

            gpus = 1 if torch.cuda.is_available() else 0
        except Exception:
            gpus = 0
    else:
        gpus = args.gpus

    reference_free = "kiwi" in args.model.lower()
    _emit({"ready": True, "model": args.model, "reference_free": reference_free})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("shutdown"):
            return
        req_id = req.get("id")
        try:
            srcs, hyps = req["src"], req["mt"]
            refs = req.get("ref")
            if reference_free:
                data = [{"src": s, "mt": h} for s, h in zip(srcs, hyps)]
            else:
                if refs is None:
                    raise ValueError(f"{args.model} is reference-based; 'ref' is required")
                data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
            out = model.predict(
                data, batch_size=args.batch_size, gpus=gpus, progress_bar=False
            )
            _emit(
                {
                    "id": req_id,
                    "scores": [float(x) for x in out.scores],
                    "system": float(out.system_score),
                }
            )
        except Exception as exc:
            # Keep serving: one malformed step must not kill a training run.
            _emit({"id": req_id, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
