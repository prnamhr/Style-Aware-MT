
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.eval.stylometrics import CENTROID_FEATURES, features, signed_z

_CENTROID_PATH = Path("results/stylometrics_centroid.json")
_TRAIN_TEMPLATE = Path("prompts/judge_train.txt")
_TEMPLATE_HASHES = Path("prompts/hashes.json")


_FLOOR_MARGIN = 1.0


@dataclass
class RewardConfig:
    """Weights and the feasibility band. Weights need not sum to 1."""

    w_bleu: float = 1.0
    w_kiwi: float = 1.0
    w_judge: float = 1.0
    # Feasible generated length, as a ratio of the reference's word count.
    len_min_ratio: float = 0.5
    len_max_ratio: float = 2.0
    on_violation: str = "floor"  # "floor" | "drop"
    # "bleu" (smoothed sentence-BLEU) or "chrf" (chrF++), the swappable grid cell.
    overlap_metric: str = "bleu"

    def __post_init__(self) -> None:
        if self.on_violation not in ("floor", "drop"):
            raise ValueError(f"on_violation must be 'floor' or 'drop', got {self.on_violation!r}")
        if self.overlap_metric not in ("bleu", "chrf"):
            raise ValueError(
                f"overlap_metric must be 'bleu' or 'chrf', got {self.overlap_metric!r}"
            )
        if not 0 < self.len_min_ratio <= self.len_max_ratio:
            raise ValueError(
                f"need 0 < len_min_ratio <= len_max_ratio, got "
                f"{self.len_min_ratio} and {self.len_max_ratio}"
            )

    @property
    def weights(self) -> dict[str, float]:
        return {"bleu": self.w_bleu, "kiwi": self.w_kiwi, "judge": self.w_judge}


# components 


def overlap_scores(hyps: list[str], refs: list[str], metric: str = "bleu") -> list[float]:
    """Smoothed sentence-BLEU, or chrF++ as the swappable alternative.

    Corpus BLEU is undefined per segment without smoothing; sacrebleu's exponential
    smoothing is used so a hypothesis with no 4-gram match still scores above zero and
    the group retains variance.
    """
    import sacrebleu

    if metric == "chrf":
        scorer = sacrebleu.CHRF(word_order=2)  # chrF++
        return [float(scorer.sentence_score(h, [r]).score) for h, r in zip(hyps, refs)]
    scorer = sacrebleu.BLEU(effective_order=True, smooth_method="exp")
    return [float(scorer.sentence_score(h, [r]).score) for h, r in zip(hyps, refs)]


def judge_scores(
    client,
    template: str,
    sources: list[str],
    refs: list[str],
    hyps: list[str],
    *,
    default: float = 1.0,
) -> list[float]:
    """Training-time Phi, one paid call per sample.

    An unparseable rating becomes ``default`` rather than being dropped: the group has a
    fixed size, and silently shrinking it would change the normalization denominator
    mid-training.
    """
    from src.eval.judge import _JUDGE_SYSTEM, build_prompt, parse_score

    out: list[float] = []
    for src, ref, hyp in zip(sources, refs, hyps):
        # Same system message as the evaluation judge, so the two differ only in rubric.
        text = client.complete(_JUDGE_SYSTEM, build_prompt(template, src, ref, hyp))
        score = parse_score(text)
        out.append(float(default if score is None else score))
    return out


def frozen_digest(
    path: str | Path, hashes_path: str | Path = _TEMPLATE_HASHES
) -> str:
    """The digest recorded for a rubric when it was frozen.

    Unrecorded raises rather than being read on trust: that is the `template_verified:
    false` gap of 2026-08-05, where rubric identity could only be asserted after the fact.
    """
    hashes_path = Path(hashes_path)
    if not hashes_path.exists():
        raise FileNotFoundError(
            f"{hashes_path} does not exist, so no run can prove which rubric it read. "
            f"Record each judge template's sha256 there before it is used."
        )
    record = json.loads(hashes_path.read_text(encoding="utf-8"))["templates"]
    name = Path(path).name
    if name not in record:
        raise KeyError(
            f"{name} has no freeze record in {hashes_path}. Freeze the template by "
            f"recording its sha256 there first; a digest recorded after the run is an "
            f"assertion about the rubric, not a verification of it."
        )
    return record[name]["digest"]


def load_train_template(
    path: str | Path = _TRAIN_TEMPLATE, hashes_path: str | Path = _TEMPLATE_HASHES
) -> str:
    """Read the frozen training-time rubric, verified against its freeze record."""
    from src.eval.judge import template_digest

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. The training-time judge rubric is a separate frozen "
            f"template from prompts/judge_eval.txt and must be written before RLSF runs; "
            f"reusing the evaluation rubric as the reward would make the Phi result circular."
        )
    text = path.read_text(encoding="utf-8")
    expected = frozen_digest(path, hashes_path)
    actual = template_digest(text)
    if actual != expected:
        raise ValueError(
            f"{path} has drifted from its freeze record: {hashes_path} holds {expected}, the "
            f"file on disk hashes to {actual}. Rewards computed under an edited rubric are not "
            f"comparable with those already collected. Restore the frozen text, or freeze the "
            f"new rubric deliberately and re-record its digest."
        )
    return text



def group_normalize(values: np.ndarray, valid: np.ndarray) -> np.ndarray:

    out = np.zeros_like(values, dtype=float)
    if valid.sum() < 2:
        return out
    subset = values[valid]
    sd = subset.std(ddof=0)
    if sd < 1e-9:
        return out
    out[valid] = (subset - subset.mean()) / sd
    return out


def length_feasible(hyps: list[str], refs: list[str], cfg: RewardConfig) -> np.ndarray:

    out = np.ones(len(hyps), dtype=bool)
    for i, (hyp, ref) in enumerate(zip(hyps, refs)):
        n_hyp = len(hyp.split())
        n_ref = len(ref.split())
        if n_hyp == 0:
            out[i] = False
        elif n_ref > 0:
            ratio = n_hyp / n_ref
            out[i] = cfg.len_min_ratio <= ratio <= cfg.len_max_ratio
    return out




def load_centroid(path: str | Path = _CENTROID_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def z_deviations(hyps: list[str], centroid: dict) -> dict[str, float]:
    """Mean signed z-deviation per centroid feature over a step's samples."""
    if not hyps:
        return dict.fromkeys(CENTROID_FEATURES, float("nan"))
    per_feature = {name: [] for name in CENTROID_FEATURES}
    for hyp in hyps:
        feats = features(hyp)
        for name in CENTROID_FEATURES:
            per_feature[name].append(feats[name])
    means = {name: float(np.mean(vals)) for name, vals in per_feature.items()}
    return signed_z(means, centroid)


@dataclass
class StepLog:
    """One training step's record. Written as a JSON line per step."""

    step: int
    n_samples: int
    n_feasible: int
    reward_mean: float
    reward_sd: float
    raw: dict[str, float] = field(default_factory=dict)
    normalized: dict[str, float] = field(default_factory=dict)
    length_mean: float = 0.0
    length_ratio_mean: float = 0.0
    z: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "n_samples": self.n_samples,
            "n_feasible": self.n_feasible,
            "reward_mean": self.reward_mean,
            "reward_sd": self.reward_sd,
            "raw": self.raw,
            "normalized": self.normalized,
            "length_mean": self.length_mean,
            "length_ratio_mean": self.length_ratio_mean,
            "z": self.z,
        }


def compute_rewards(
    sources: list[str],
    hyps: list[str],
    refs: list[str],
    *,
    cfg: RewardConfig,
    group_size: int,
    component_scores: dict[str, list[float]],
    centroid: dict,
    step: int = 0,
) -> tuple[np.ndarray, np.ndarray, StepLog]:

    n = len(hyps)
    if not (len(sources) == len(refs) == n):
        raise ValueError(f"ragged batch: {len(sources)} sources, {n} hyps, {len(refs)} refs")
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    if n % group_size:
        raise ValueError(f"batch of {n} is not a whole number of groups of {group_size}")
    missing = set(cfg.weights) - set(component_scores)
    if missing:
        raise ValueError(f"missing component scores: {sorted(missing)}")
    for name, values in component_scores.items():
        if len(values) != n:
            raise ValueError(f"component {name!r} has {len(values)} scores for {n} samples")

    feasible = length_feasible(hyps, refs, cfg)
    raw = {name: np.asarray(component_scores[name], dtype=float) for name in cfg.weights}

    rewards = np.full(n, np.nan, dtype=float)
    normalized_all = {name: np.zeros(n, dtype=float) for name in cfg.weights}
    for start in range(0, n, group_size):
        sl = slice(start, start + group_size)
        g_valid = feasible[sl]
        combined = np.zeros(group_size, dtype=float)
        for name, weight in cfg.weights.items():
            z = group_normalize(raw[name][sl], g_valid)
            normalized_all[name][sl] = z
            combined += weight * z
        group = np.full(group_size, np.nan, dtype=float)
        group[g_valid] = combined[g_valid]
        if cfg.on_violation == "floor" and (~g_valid).any():
            worst = np.nanmin(group) if g_valid.any() else 0.0
            group[~g_valid] = worst - _FLOOR_MARGIN
        rewards[sl] = group

    lengths = np.asarray([len(h.split()) for h in hyps], dtype=float)
    ref_lengths = np.asarray([max(len(r.split()), 1) for r in refs], dtype=float)
    finite = rewards[np.isfinite(rewards)]
    log = StepLog(
        step=step,
        n_samples=n,
        n_feasible=int(feasible.sum()),
        reward_mean=float(finite.mean()) if finite.size else float("nan"),
        reward_sd=float(finite.std(ddof=0)) if finite.size else float("nan"),
        raw={name: float(values.mean()) for name, values in raw.items()},
        normalized={name: float(values.mean()) for name, values in normalized_all.items()},
        length_mean=float(lengths.mean()) if n else float("nan"),
        length_ratio_mean=float((lengths / ref_lengths).mean()) if n else float("nan"),
        z=z_deviations(hyps, centroid),
    )
    return rewards, feasible, log
