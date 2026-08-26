"""Derive and load the signed target-register direction used by AFSP/PEFT selection.

Usage:
    python manage.py register_direction
    python manage.py register_direction --output results/register_direction.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from src.eval.stylometrics import (
    CENTROID_FEATURES,
    aggregate,
    build_centroid,
    features,
    fingerprint,
    register_band_distance,
)

_DEFAULT_TRAIN = Path("data/splits/train.jsonl")
_DEFAULT_CENTROID = Path("results/stylometrics_centroid.json")
_DEFAULT_OUTPUT = Path("results/register_direction.json")

# Historical vector used by the seven configs before this derivation was committed.
# It is retained only as the fixed comparison baseline; configs do not consume it.
LEGACY_DIRECTION = {
    "marker_rate": 0.514,
    "lex_density": 0.355,
    "root_ttr": -0.296,
    "ttr": 0.098,
}

# Predeclared comparison thresholds. register_band_distance() normalizes by the sum
# of |direction|, so relative absolute-weight shares and sign changes are the relevant
# objective geometry; cosine similarity captures rotation of the full signed vector.
NEAR_COSINE_MIN = 0.98
MATERIAL_WEIGHT_SHARE_DELTA = 0.05
VERY_COSINE_MAX = 0.90
VERY_WEIGHT_SHARE_DELTA = 0.15


def _read_targets(path: Path) -> list[str]:
    targets: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            target = json.loads(line).get("output", "")
            if target and target.strip():
                targets.append(target)
    if not targets:
        raise ValueError(f"no non-empty target texts found in {path}")
    return targets


def _ordered(direction: dict[str, float], names: list[str]) -> np.ndarray:
    missing = [name for name in names if name not in direction]
    if missing:
        raise ValueError(f"register direction missing entries for {missing}")
    return np.asarray([float(direction[name]) for name in names], dtype=float)


def derive_direction(targets: list[str], centroid: dict) -> dict[str, float]:
    """Return per-feature correlations with train-target centroid distance.

    The committed centroid is treated as fixed. The function first verifies that the
    current feature extractor reproduces its means/stds on the supplied targets, which
    prevents silently deriving a direction against stale centroid statistics.
    """
    names = list(centroid["features"])
    if set(names) != set(CENTROID_FEATURES):
        raise ValueError(
            f"centroid features {names} do not match register features {CENTROID_FEATURES}"
        )

    rebuilt = build_centroid(targets, names)
    for field in ("mean", "std"):
        if not np.allclose(rebuilt[field], centroid[field], rtol=0, atol=1e-12):
            raise ValueError(
                f"current feature extractor does not reproduce centroid {field}; "
                "rebuild results/stylometrics_centroid.json before deriving the direction"
            )

    matrix = np.asarray([[features(text)[name] for name in names] for text in targets], dtype=float)
    mean = np.asarray(centroid["mean"], dtype=float)
    std = np.asarray(centroid["std"], dtype=float)
    z = (matrix - mean) / std
    distance = np.linalg.norm(z, axis=1)

    out: dict[str, float] = {}
    for idx, name in enumerate(names):
        corr = float(np.corrcoef(distance, z[:, idx])[0, 1])
        if not math.isfinite(corr):
            raise ValueError(f"cannot derive finite correlation for {name}")
        out[name] = corr
    return out


def compare_directions(
    current: dict[str, float],
    legacy: dict[str, float] = LEGACY_DIRECTION,
    names: list[str] | None = None,
) -> dict:
    """Compare two directions under thresholds fixed before inspecting the result."""
    names = names or list(CENTROID_FEATURES)
    old = _ordered(legacy, names)
    new = _ordered(current, names)

    old_norm = float(np.linalg.norm(old))
    new_norm = float(np.linalg.norm(new))
    if old_norm <= 0 or new_norm <= 0:
        raise ValueError("register directions must not be all-zero")

    cosine = float(np.dot(old, new) / (old_norm * new_norm))
    cosine = max(-1.0, min(1.0, cosine))
    angle = float(math.degrees(math.acos(cosine)))
    raw_delta = new - old

    old_share = np.abs(old) / np.abs(old).sum()
    new_share = np.abs(new) / np.abs(new).sum()
    share_delta = new_share - old_share
    sign_flips = [
        name
        for name, a, b in zip(names, old, new)
        if np.sign(a) != np.sign(b) and not (a == 0 and b == 0)
    ]
    max_share_delta = float(np.max(np.abs(share_delta)))

    if sign_flips or cosine < VERY_COSINE_MAX or max_share_delta >= VERY_WEIGHT_SHARE_DELTA:
        classification = "very_different"
    elif cosine >= NEAR_COSINE_MIN and max_share_delta < MATERIAL_WEIGHT_SHARE_DELTA:
        classification = "near_identical"
    else:
        classification = "materially_different"

    return {
        "thresholds_predeclared": {
            "near_identical": {
                "cosine_similarity_min": NEAR_COSINE_MIN,
                "max_abs_weight_share_delta_lt": MATERIAL_WEIGHT_SHARE_DELTA,
                "requires_no_sign_flips": True,
            },
            "very_different": {
                "cosine_similarity_lt": VERY_COSINE_MAX,
                "max_abs_weight_share_delta_gte": VERY_WEIGHT_SHARE_DELTA,
                "or_any_sign_flip": True,
            },
            "materially_different": "all remaining cases",
            "rerun_afsp_full_if": "very_different",
        },
        "cosine_similarity": cosine,
        "angle_degrees": angle,
        "l2_delta": float(np.linalg.norm(raw_delta)),
        "max_abs_coefficient_delta": float(np.max(np.abs(raw_delta))),
        "max_abs_weight_share_delta": max_share_delta,
        "sign_flips": sign_flips,
        "coefficient_delta": {name: float(delta) for name, delta in zip(names, raw_delta)},
        "legacy_abs_weight_share": {name: float(value) for name, value in zip(names, old_share)},
        "corrected_abs_weight_share": {name: float(value) for name, value in zip(names, new_share)},
        "weight_share_delta": {name: float(value) for name, value in zip(names, share_delta)},
        "classification": classification,
        "rerun_afsp_full": classification == "very_different",
    }


def compare_afsp_sweep_objective(
    sweep_results: Path,
    sweep_output_dir: Path,
    centroid: dict,
    current: dict[str, float],
    legacy: dict[str, float] = LEGACY_DIRECTION,
) -> dict | None:
    """Re-score existing AFSP sweep outputs under old/new directions, without generation."""
    if not sweep_results.exists():
        return None
    payload = json.loads(sweep_results.read_text(encoding="utf-8"))
    split = payload.get("split", "val")
    sigma = float(payload.get("select_target_sigma", 0.5))
    margin = float(payload.get("adequacy_margin", 1.0))
    names = list(centroid["features"])
    old = _ordered(legacy, names)
    new = _ordered(current, names)

    rows: list[dict] = []
    for cell in payload.get("cells", []):
        tag = cell["tag"]
        path = sweep_output_dir / f"{tag}_{split}.jsonl"
        if not path.exists():
            continue
        preds: list[str] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                pred = json.loads(line).get("prediction", "")
                if pred and pred.strip():
                    preds.append(pred)
        if not preds:
            continue
        mean = aggregate(preds)["mean"]
        old_fit = register_band_distance(mean, centroid, sigma, old)
        new_fit = register_band_distance(mean, centroid, sigma, new)
        rows.append(
            {
                "tag": tag,
                "k": cell.get("k"),
                "lambda": cell.get("lambda"),
                "anchor": bool(cell.get("anchor")),
                "chrF": float(cell["chrF"]),
                "register_fit_legacy": old_fit,
                "register_fit_corrected": new_fit,
                "delta": new_fit - old_fit,
            }
        )

    selectable = [row for row in rows if not row["anchor"]]
    if not selectable:
        return None
    best_chrf = max(row["chrF"] for row in selectable)
    band = [row for row in selectable if row["chrF"] >= best_chrf - margin]

    def pick(key: str) -> dict:
        row = min(band, key=lambda r: (r[key], -r["chrF"], r["k"]))
        return {
            "tag": row["tag"],
            "k": row["k"],
            "lambda": row["lambda"],
            "chrF": row["chrF"],
            "register_fit": row[key],
        }

    old_pick = pick("register_fit_legacy")
    new_pick = pick("register_fit_corrected")
    return {
        "scope": (
            "offline re-score of already-generated AFSP sweep translations; this does not "
            "re-run exemplar selection or translation generation"
        ),
        "source_results": str(sweep_results),
        "source_outputs": str(sweep_output_dir),
        "n_cells_rescored": len(rows),
        "select_target_sigma": sigma,
        "adequacy_margin": margin,
        "legacy_recommendation": old_pick,
        "corrected_recommendation": new_pick,
        "recommendation_changed": old_pick["tag"] != new_pick["tag"],
        "max_abs_register_fit_delta": max(abs(row["delta"]) for row in rows),
    }


def load_direction_artifact(
    path: str | Path,
    *,
    expected_features: list[str] | None = None,
    centroid: dict | None = None,
) -> dict[str, float]:
    """Load the active direction from the generated JSON artifact."""
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    direction = payload.get("direction")
    if not isinstance(direction, dict) or not direction:
        raise ValueError(f"{p} has no non-empty 'direction' mapping")

    names = expected_features or (list(centroid["features"]) if centroid is not None else None)
    if names is not None:
        missing = [name for name in names if name not in direction]
        extra = [name for name in direction if name not in names]
        if missing or extra:
            raise ValueError(f"{p} direction feature mismatch: missing={missing}, extra={extra}")

    if centroid is not None:
        artifact_fp = payload.get("centroid", {}).get("fingerprint")
        active_fp = fingerprint(centroid)
        if artifact_fp and artifact_fp != active_fp:
            raise ValueError(
                f"{p} was derived from centroid {artifact_fp}, but active centroid is {active_fp}; "
                "rerun: python manage.py register_direction"
            )

    return {name: float(value) for name, value in direction.items()}


def configured_direction(block: dict, centroid: dict | None = None) -> dict[str, float] | None:
    """Resolve a config block to one direction, preferring the generated artifact.

    Inline ``style_register_direction`` remains accepted for historical/test configs, but the
    canonical configs all use ``style_register_direction_file``.
    """
    path = block.get("style_register_direction_file")
    if path:
        return load_direction_artifact(path, centroid=centroid)
    direction = block.get("style_register_direction")
    if direction is None:
        return None
    if not isinstance(direction, dict):
        raise ValueError("style_register_direction must be a feature-to-coefficient mapping")
    return {name: float(value) for name, value in direction.items()}


def build_artifact(
    train_file: Path,
    centroid_file: Path,
    sweep_results: Path | None = None,
    sweep_output_dir: Path | None = None,
) -> dict:
    centroid = json.loads(centroid_file.read_text(encoding="utf-8"))
    targets = _read_targets(train_file)
    direction = derive_direction(targets, centroid)
    comparison = compare_directions(direction, names=list(centroid["features"]))
    artifact = {
        "schema_version": 1,
        "method": "pearson_feature_correlation_with_standardized_centroid_distance",
        "derivation": {
            "train_file": str(train_file),
            "target_field": "output",
            "n_segments": len(targets),
            "features": list(centroid["features"]),
            "distance": "euclidean_norm_of_per_segment_z_scores_against_train_target_centroid",
            "coefficient": "pearson_r(distance, feature_z_score)",
        },
        "centroid": {
            "path": str(centroid_file),
            "fingerprint": fingerprint(centroid),
        },
        "direction": direction,
        "legacy_hard_coded_direction": LEGACY_DIRECTION,
        "comparison_to_legacy": comparison,
    }
    if sweep_results is not None and sweep_output_dir is not None:
        impact = compare_afsp_sweep_objective(sweep_results, sweep_output_dir, centroid, direction)
        if impact is not None:
            artifact["afsp_sweep_objective_impact"] = impact
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive the target-register direction from the corrected train centroid."
    )
    parser.add_argument("--train-file", type=Path, default=_DEFAULT_TRAIN)
    parser.add_argument("--centroid", type=Path, default=_DEFAULT_CENTROID)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--afsp-sweep-results", type=Path, default=Path("results/afsp_sweep_val.json")
    )
    parser.add_argument("--afsp-sweep-output-dir", type=Path, default=Path("outputs/sweep"))
    parser.add_argument(
        "--no-sweep-impact", action="store_true", help="skip the offline AFSP sweep re-score"
    )
    args = parser.parse_args()

    artifact = build_artifact(
        args.train_file,
        args.centroid,
        None if args.no_sweep_impact else args.afsp_sweep_results,
        None if args.no_sweep_impact else args.afsp_sweep_output_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    direction = artifact["direction"]
    comp = artifact["comparison_to_legacy"]
    print(f"Wrote {args.output}")
    print("Corrected register direction:")
    for name in artifact["derivation"]["features"]:
        print(f"  {name}: {direction[name]:+.6f}")
    print(
        "Comparison to legacy: "
        f"{comp['classification']} | cosine={comp['cosine_similarity']:.6f} | "
        f"angle={comp['angle_degrees']:.3f} deg | "
        f"max weight-share delta={comp['max_abs_weight_share_delta']:.6f}"
    )
    print(f"Re-run afsp_full: {'yes' if comp['rerun_afsp_full'] else 'no'}")
    impact = artifact.get("afsp_sweep_objective_impact")
    if impact:
        print(
            "Existing AFSP sweep re-score: "
            f"pick {impact['legacy_recommendation']['tag']} -> "
            f"{impact['corrected_recommendation']['tag']} | "
            f"max register_fit delta={impact['max_abs_register_fit_delta']:.6f}"
        )


if __name__ == "__main__":
    main()
