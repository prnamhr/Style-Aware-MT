
from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.eval.stylometrics import features, signed_z

_CENTROID_PATH = Path("results/stylometrics_centroid.json")
_TRAIN_TEMPLATE = Path("prompts/judge_train.txt")
_TEMPLATE_HASHES = Path("prompts/hashes.json")


_FLOOR_MARGIN = 1.0

# Below this a group's rewards are flat, so its advantages are zero and it trains nothing.
_MIN_GROUP_SD = 1e-9


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
        # Keyed by the metric in use, so a chrF grid cell is not logged under a BLEU label.
        return {self.overlap_metric: self.w_bleu, "kiwi": self.w_kiwi, "judge": self.w_judge}


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


@dataclass
class JudgeTiming:
    """Wall clock of a judge block and the call time inside it."""

    wall_s: float = 0.0
    call_s: float = 0.0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, seconds: float) -> None:
        # Locked for the same reason Usage is: the fanned-out threads share one timing object.
        with self._lock:
            self.call_s += seconds
            self.calls += 1

    @property
    def achieved_parallelism(self) -> float:
        """Call time over wall time: what the fan-out bought, against max_workers asked for."""
        return self.call_s / self.wall_s if self.wall_s else 0.0

    def summary(self) -> dict:
        return {
            "wall_s": round(self.wall_s, 3),
            "call_s": round(self.call_s, 3),
            "mean_call_s": round(self.call_s / self.calls, 3) if self.calls else 0.0,
            "calls_per_s": round(self.calls / self.wall_s, 3) if self.wall_s else 0.0,
            "achieved_parallelism": round(self.achieved_parallelism, 2),
        }


def judge_scores(
    client,
    template: str,
    sources: list[str],
    refs: list[str],
    hyps: list[str],
    *,
    max_workers: int = 1,
    timing: JudgeTiming | None = None,
) -> list[float]:
    """Training-time Phi, one paid call per sample; nan where the verdict did not parse.

    The calls are latency-bound and independent, so ``max_workers`` above 1 fans them out
    over threads rather than holding the rented GPU idle for a step's worth of verdicts.
    A ``timing`` is filled in with the block's wall clock and the summed call time.
    """
    from src.eval.judge import _JUDGE_SYSTEM, build_prompt, parse_score

    prompts = [build_prompt(template, src, ref, hyp) for src, ref, hyp in zip(sources, refs, hyps)]

    def score_one(prompt: str) -> float:
        # Same system message as the evaluation judge, so the two differ only in rubric.
        started = time.perf_counter()
        score = parse_score(client.complete(_JUDGE_SYSTEM, prompt))
        if timing is not None:
            timing.record(time.perf_counter() - started)
        # An unreadable verdict is an unmeasured sample, not a 1; compute_rewards marks it
        # infeasible, as the evaluation path drops unparseable segments rather than scoring them.
        return float("nan") if score is None else float(score)

    started = time.perf_counter()
    if max_workers <= 1:
        scores = [score_one(prompt) for prompt in prompts]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # map yields in submission order, so a verdict cannot be attributed to another sample.
            scores = list(pool.map(score_one, prompts))
    if timing is not None:
        timing.wall_s = time.perf_counter() - started
    return scores


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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. The training-time judge rubric is a separate frozen "
            f"template from prompts/judge_eval.txt and must be written before RLSF runs; "
            f"reusing the evaluation rubric as the reward would make the Phi result circular."
        )
    text = path.read_text(encoding="utf-8")
    expected = frozen_digest(path, hashes_path)
    # Same digest as src.eval.judge.template_digest, inlined: hashing a text file should not
    # import torch and the provider SDKs. tests pin the two against each other.
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
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


def _measured_mean(values: np.ndarray) -> float:
    """Mean over the entries that were measured; nan when none were."""
    ok = np.isfinite(values)
    return float(values[ok].mean()) if ok.any() else float("nan")


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




def reward_degeneracy(rewards: np.ndarray, feasible: np.ndarray, group_size: int) -> dict:
    """Fraction of groups whose combined reward has no spread across feasible samples."""
    rewards = np.asarray(rewards, dtype=float)
    sds = []
    for start in range(0, len(rewards), group_size):
        sl = slice(start, start + group_size)
        # Infeasible entries are excluded, not floored in: `worst - 1.0` would manufacture
        # spread in a flat group. Below two feasible samples there is no spread to measure.
        values = rewards[sl][feasible[sl]]
        sds.append(float(values.std(ddof=0)) if values.size >= 2 else 0.0)
    degenerate = sum(sd < _MIN_GROUP_SD for sd in sds)
    return {
        "groups": len(sds),
        "degenerate": degenerate,
        "degenerate_frac": round(degenerate / len(sds), 3) if sds else 1.0,
        "min_group_sd": round(min(sds), 6) if sds else 0.0,
    }


def load_centroid(path: str | Path = _CENTROID_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def z_step_stats(
    hyps: list[str], centroid: dict, *, group_size: int = 1
) -> tuple[dict[str, float], dict[str, float]]:
    """Mean signed z-deviation per centroid feature over a step's samples, and its error."""
    names = centroid["features"]
    if not hyps:
        return dict.fromkeys(names, float("nan")), dict.fromkeys(names, float("nan"))
    if group_size < 1 or len(hyps) % group_size:
        raise ValueError(f"{len(hyps)} samples is not a whole number of groups of {group_size}")

    matrix = np.asarray([[features(h)[name] for name in names] for h in hyps], dtype=float)
    means = signed_z(dict(zip(names, matrix.mean(axis=0).tolist())), centroid)

    per_sample = (matrix - np.asarray(centroid["mean"], dtype=float)) / np.asarray(
        centroid["std"], dtype=float
    )
    group_means = per_sample.reshape(len(hyps) // group_size, group_size, len(names)).mean(axis=1)
    n_groups = group_means.shape[0]
    if n_groups < 2:
        return means, dict.fromkeys(names, float("nan"))
    ses = (group_means.std(axis=0, ddof=1) / np.sqrt(n_groups)).tolist()
    return means, dict(zip(names, ses))


def z_deviations(hyps: list[str], centroid: dict) -> dict[str, float]:
    """Mean signed z-deviation per centroid feature over a step's samples."""
    return z_step_stats(hyps, centroid)[0]


@dataclass
class StepLog:
    """One training step's record. Written as a JSON line per step."""

    step: int
    n_samples: int
    n_feasible: int
    reward_mean: float
    reward_sd: float
    # Samples dropped for a missing component score, not for a length violation.
    n_unmeasured: int = 0
    n_groups: int = 0
    # Groups whose combined reward is flat across their feasible samples: no gradient.
    degenerate_groups: int = 0
    degenerate_frac: float = 1.0
    min_group_sd: float = 0.0
    raw: dict[str, float] = field(default_factory=dict)
    normalized: dict[str, float] = field(default_factory=dict)
    length_mean: float = 0.0
    length_ratio_mean: float = 0.0
    z: dict[str, float] = field(default_factory=dict)
    z_se: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "n_samples": self.n_samples,
            "n_feasible": self.n_feasible,
            "reward_mean": self.reward_mean,
            "reward_sd": self.reward_sd,
            "n_unmeasured": self.n_unmeasured,
            "n_groups": self.n_groups,
            "degenerate_groups": self.degenerate_groups,
            "degenerate_frac": self.degenerate_frac,
            "min_group_sd": self.min_group_sd,
            "raw": self.raw,
            "normalized": self.normalized,
            "length_mean": self.length_mean,
            "length_ratio_mean": self.length_ratio_mean,
            "z": self.z,
            "z_se": self.z_se,
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

    raw = {name: np.asarray(component_scores[name], dtype=float) for name in cfg.weights}
    measured = np.logical_and.reduce([np.isfinite(values) for values in raw.values()])
    feasible = length_feasible(hyps, refs, cfg) & measured

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
    degeneracy = reward_degeneracy(rewards, feasible, group_size)

    lengths = np.asarray([len(h.split()) for h in hyps], dtype=float)
    ref_lengths = np.asarray([max(len(r.split()), 1) for r in refs], dtype=float)
    finite = rewards[np.isfinite(rewards)]
    z, z_se = z_step_stats(hyps, centroid, group_size=group_size)
    log = StepLog(
        step=step,
        n_samples=n,
        n_feasible=int(feasible.sum()),
        reward_mean=float(finite.mean()) if finite.size else float("nan"),
        reward_sd=float(finite.std(ddof=0)) if finite.size else float("nan"),
        n_unmeasured=int((~measured).sum()),
        n_groups=degeneracy["groups"],
        degenerate_groups=degeneracy["degenerate"],
        degenerate_frac=degeneracy["degenerate_frac"],
        min_group_sd=degeneracy["min_group_sd"],
        raw={name: _measured_mean(values) for name, values in raw.items()},
        normalized={name: float(values.mean()) for name, values in normalized_all.items()},
        length_mean=float(lengths.mean()) if n else float("nan"),
        length_ratio_mean=float((lengths / ref_lengths).mean()) if n else float("nan"),
        z=z,
        z_se=z_se,
    )
    return rewards, feasible, log
