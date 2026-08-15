# Style-Aware NMT of Low-Resource Texts

Style-aware neural machine translation of Persian and mixed Persian/Arabic Bahá’í scriptures into English, targeting the formal scriptural register of Shoghi Effendi’s authorized translations.

Undergraduate thesis project — BIHE, Department of Computer Engineering. Supervisor: Dr. Fares Hedayati.

> **Status:** in progress (last updated 2026-08-10).
> Six of seven conditions — the five prompting rungs and PEFT — are implemented, tuned, frozen, and scored on the **validation** split; their numbers are in [Results](#results-validation-split). RLSF is implemented — GRPO loop, reward path, spend caps, register-drift stop, best-of-N pool and ω grid — and has **not been run**: the only RLSF spend to date is the reward-path smoke. The test split is sealed and untouched, so every figure below is a validation figure and none is a final result.

---

## What this project does

Modern LLM-based MT is fluent but defaults to a neutral register. For literary and scriptural texts, register *is* meaning. This project compares three ways of adapting an open-source LLM to preserve Shoghi Effendi’s register on a low-resource, mixed-language corpus — under matched conditions, on the same base model and the same data.

The three adaptation strategies:

| Strategy | Mechanism | Updates weights? |
|---|---|---|
| **PEFT (LoRA)** | Supervised fine-tuning of LoRA adapters on parallel data | Yes (adapters only) |
| **AFSP** | Adaptive Few-Shot Prompting; retrieval of stylistically relevant exemplars into the prompt at inference time | No |
| **RLSF (GRPO)** | Reinforcement learning from a mixed reward combining COMET-Kiwi, BLEU, and an LLM-as-Judge style score | Yes (from PEFT init) |

Plus a fourth condition — the **unadapted base model** — as reference.

All four are to be evaluated on the same held-out test set with COMET, BLEU, objective stylometric features, and LLM-as-Judge scoring. Nothing has been generated on test yet; the figures reported here are on validation.

**Implementation status:** PEFT and the AFSP/prompting ladder are complete and scored on validation; RLSF is implemented but unrun. The current comparison is therefore three-way (base, prompting ladder, PEFT).

---

## Research questions

- **RQ1.** How do PEFT, AFSP, and RLSF compare on semantic adequacy, stylistic fidelity, and compute/data cost under matched conditions?
- **RQ2.** Do retrieval-based exemplars (AFSP) and reward-driven updates (RLSF) shift outputs toward the target register relative to PEFT, measurably via stylometrics *and* judge scores?
- **RQ3.** How sensitive is RLSF to the reward weights (ω₁, ω₂, ω₃) on COMET / BLEU / style?
- **RQ4.** When applied to the same outputs, where do COMET, stylometric features, and LLM-as-Judge agree and disagree?

Full hypotheses (H1–H4) and support criteria live in the thesis proposal ([`docs/proposal.pdf`](docs/proposal.pdf)); a long-form `docs/methodology.md` is **not yet written**. The project does **not** pre-commit to a ranking of the three adaptation methods.

---

## Repository layout

```
.
├── README.md                  ← you are here
├── data/
│   ├── raw/                   ← original sentence/paragraph TSVs (not redistributed)
│   ├── processed/             ← normalized, deduplicated, split by document
│   └── splits/                ← train/dev/test manifests with file hashes
├── src/
│   ├── data/                  ← preprocessing, alignment, split logic
│   ├── peft/                  ← LoRA training (PEFT condition)
│   ├── retrieval/             ← kNN retrieval index + prompt assembly (knn_fewshot baseline → AFSP)
│   ├── rlsf/                  ← GRPO loop, reward, best-of-N pool and ω grid
│   ├── eval/                  ← COMET, BLEU, stylometrics, LLM-as-Judge
│   └── infer/                 ← test-set inference for all four conditions
├── configs/                   ← YAML configs per condition + decoding settings
├── prompts/
│   ├── style_instruction.txt
│   ├── judge_train.txt        ← judge template used inside RLSF reward
│   ├── judge_eval.txt         ← separate judge template for final evaluation
│   └── hashes.json            ← freeze record: sha256 of both judge templates
├── models/                    ← trained LoRA adapters, one dir per sweep cell (+ epoch checkpoints)
├── outputs/                   ← <condition>_<split>.jsonl per run (split tag, e.g. _val)
│   ├── sweep/                 ← AFSP k × λ sweep cells
│   └── peft_sweep/            ← PEFT (r, lr, epoch) candidate
├── results/                   ← comet_<split>.json, judge_<split>.json, *_sweep/_verify records
├── archive/                  
├── docs/
│   ├── proposal.pdf           ← thesis proposal (H1–H4 live here)
│   ├── DEVLOG.md              ← engineering and decision log
│   └── afsp_strategies.md     ← AFSP mechanism specification
└── notebooks/                 ← Colab runbooks for the GPU stages (sweeps, training, inference)
```

---

## Data

- **Source:** Bahá’í scriptures originally in Persian and mixed Persian/Arabic, paired with Shoghi Effendi’s authorized English translations.
- **Granularity:** sentence-level for core experiments; paragraph-level as an optional extension.
- **Mixed-language handling:** source segments containing both Persian and Arabic are kept as single segments; the model is responsible for within-segment language mixing, mirroring how the authorized translations were produced.
- **No synthetic data** in training. Synthetic pairs from an unadapted LLM would dilute the target register signal, and validating their quality is out of scope.

### Splits

- Approximately **80 / 10 / 10** train / dev / test (85 / 5 / 10 acceptable if corpus size requires).
- **Split by document/section**, not by random row, to reduce stylistic leakage.
- The split is **fixed at the start** and frozen. File hashes are stored under `data/splits/`.
- Train: parameter updates only. Dev: all hyperparameter selection. Test: final evaluation only — unseen during RLSF and not used for any tuning.

### Preprocessing

Unicode NFC, diacritic handling, whitespace and punctuation normalization, removal of editorial metadata, deduplication on (src, tgt). No corpus-level tokenization — delegated to the base model’s tokenizer.

---

## Systems

### Base model
**`Qwen2.5-7B-Instruct`** — an open-source multilingual decoder-only Transformer in the 7B–8B range with documented Persian and Arabic coverage. It is **locked** and frozen across all four conditions so differences are attributable to adaptation, not model identity. Selection was vetted by a tokenizer-fertility check over the Arabic/Persian-script source: **2.70 subword tokens/word**, well clear of the catastrophic byte-fallback zone (see `docs/DEVLOG.md`, 2026-06-12).

### Reference (unadapted)
Base model, minimal style instruction (see `prompts/style_instruction.txt`), no exemplars, no fine-tuning. Provides the lower bound and the H1 comparison anchor.

### PEFT (LoRA)
LoRA adapters on all linear layers of the transformer (`q,k,v,o,gate,up,down`), base weights frozen. Trained with token-level MLE, completion-only loss, on the training partition. QLoRA is the memory-pressure fallback. The resulting checkpoint also serves as the **RLSF initialization**.

Rank, LR, and epoch count were tuned on dev (see [`configs/peft_sweep.yaml`](configs/peft_sweep.yaml)) over a four-cell (r, lr) grid — (16, 2e-4), (8, 2e-4), (32, 2e-4), (16, 1e-4), with α = 2r — trained for three epochs with **every epoch checkpoint kept**, so a candidate is an (r, lr, epoch) triple. `eval_loss` is used only as a pre-filter on which checkpoints are worth generating with; selection runs on the same axis as AFSP (chrF adequacy band, then register fidelity), confirmed on COMET + judge Φ. The sweep regime is byte-identical to the reported/inference regime so the selected hyperparameters transfer.

**Frozen configuration:** r = 32, α = 64, lr = 2e-4, **2 epochs** — adapter `models/peft_lora_r32_lr2e-4/checkpoint-1358`, 80.7 M trainable parameters (≈1.06 % of the base). Selecting on `eval_loss` instead would have picked a different and measurably less in-register adapter; see [`docs/DEVLOG.md`](docs/DEVLOG.md), 2026-07-25.

### kNN few-shot (baseline) and AFSP (retrieval-based ICL)
No parameter updates. At inference time:

1. Embed the source segment with a multilingual sentence-transformer.
2. Retrieve top-k relevant exemplars by nearest neighbour over an index built over the **Persian/Arabic (source-side)** training partition, then map each match back to its aligned English target.
3. Insert them into a fixed prompt template (system role + style instruction + k exemplars + new source).

**Ablation ladder.** The prompting conditions form a ladder, each rung adding one component over the previous one so any register shift is attributable to that component:

| Condition | Exemplars | Selection | Isolates |
|---|---|---|---|
| `zeroshot` | none | — | instruction only |
| `random_fewshot` | k | random (seeded) | having examples at all |
| `knn_fewshot` | k | cosine top-k | relevance-based retrieval |
| `afsp_margin` | k | margin + hub penalisation | AFSP margin (λ = 0) |
| `afsp_full` | k | margin + target-register rerank (λ > 0) | full AFSP method |

`knn_fewshot` is the baseline (plain top-k cosine), not the contribution; **AFSP** is the adaptive variant on the same index — margin-based scoring (hub penalisation), target-distribution-priority selection, demonstration ordering, and multi-view word-level weighting (see [`docs/afsp_strategies.md`](docs/afsp_strategies.md)). By construction `β = 0, λ = 0` reduces AFSP to `knn_fewshot`, so the rungs cleanly separate naive retrieval from the adaptive machinery. The register glossary (multi-view word pairs) is a **controlled prompt augmentation** set in the `prompt:` config block and applied uniformly to every few-shot rung, so it augments no single arm and can be toggled to measure its own effect.

The index is built over the **source** side (not the English targets) so that the AFSP margin — query–candidate similarity plus query and candidate hubness — is computed in one comparable space; target-side register is scored separately by the style rerank, from the exemplar text (see `docs/DEVLOG.md`, 2026-07-04).

**Register-fit objective.** The λ rerank scores an exemplar by a **band-pass** target: the distance to a point σ standard deviations along a *signed* register direction, rather than proximity to the corpus centroid (which selects register-bland text) or unbounded salience (which is direction-blind and has no target). The direction is dominated by archaic-marker rate and lexical density, with `root_ttr` loading negatively. `proximity` and `salience` remain selectable via `afsp.style_objective` for ablation. See [`docs/DEVLOG.md`](docs/DEVLOG.md), 2026-07-18.

**Sweep and freeze.** k × λ_style was swept on the full val split over **k ∈ {4, 8, 16} × λ ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0}** (18 cells, β fixed at 0.3), ranked on a chrF adequacy band then register fidelity, and the top three cells were re-confirmed on COMET and judge Φ before freezing. **Frozen: k = 8, λ_style = 0.75, β = 0.3, σ = 1.0.** Pattern: Tang et al. [AFSP, 2025]; related precedents in Wang et al. style-activation prompting and style-matching exemplar selection.

### RLSF (GRPO)
- **Algorithm:** group-relative policy optimization (`trl.GRPOTrainer`), not PPO. Each prompt is
  sampled G times and the reward is normalized within its group, so there is no value head and
  no clip range in the PPO sense; `configs/rlsf.yaml` carries GRPO field names for that reason
  (DEVLOG, 2026-08-08).
- **Init:** PEFT checkpoint.
- **Reference policy:** frozen copy of the PEFT checkpoint, used for KL regularization (β).
- **Reward:**
  ```
  r(y) = ω₁ · COMET-Kiwi(x, y) + ω₂ · BLEU(y, y*) + ω₃ · Φ(y, S_T)
  ```
  The adequacy term is **reference-free** (`wmt22-cometkiwi-da`), deliberately not the
  reference-based `wmt22-comet-da` that scores the final evaluation, so the arm is not trained
  on its own evaluation metric. `Φ` is an LLM-as-Judge style score using the **training-time**
  judge template. Each component is z-scored within its group before the ω weights apply, and
  the weight vector is rescaled to unit L2 norm (`src/rlsf/config.py:reward_config`) so that a
  grid cell changes the weighting and not the effective step size. Weights are tuned on the
  RLSF dev slice over four cells including a `ω₃ = 0` ablation, in
  [`configs/rlsf.yaml`](configs/rlsf.yaml).
- **Arms:** three are trained, at ω₃² = 0, 2/3 and 36/38 of the reward's variance —
  `w3_0.0` (RL-Metric, free, `--skip_judge`), `w3_2.0` and `w3_6.0`. The first two are
  pre-registered in [`docs/preregistration_rlsf.md`](docs/preregistration_rlsf.md); the third is
  added by that document's 2026-08-13 addendum, which also records β 0.05 → 0.01, the rollout at
  8 prompts × G = 4, and 300 rollouts per arm. `w3_6.0` is a training arm only and is not a
  candidate for the ω selection rule, which was pre-registered over the original four cells.
- **Reward judge:** `gpt-4o-mini`, **model-distinct from both evaluation raters** (Φ_A `claude-haiku-4-5`, Φ_B `gpt-5.6-terra`). RLSF is the only arm optimized against a judge, so it is the arm whose Φ most needs raters it was not trained against; training on either would spend one of them. It is also the only candidate honouring both `temperature: 0` and `seed: 42`, which matters here because group normalization turns rater noise into gradient noise. A Qwen judge is excluded as self-preference bias against a Qwen policy.
- **Bounded:** capped at **$25 of judge spend** for the arm, declared 2026-08-08 and derived in [`docs/budget.md`](docs/budget.md) from the smoke's measured $7.375e-5 per call. `JudgeBudget` opens at zero on every invocation, so what the config enforces is **$8 per run** — three paid runs, $24. The plan under it, re-priced 2026-08-13, is three arms of 300 rollouts plus the dev-slice best-of-N pool, $1.43; the step caps (600 final, 200 grid) bind first, at $2.83 for a single run at the group-size ceiling, and the call and dollar caps are a backstop against the per-call rate moving. `src/rlsf/config.py:assert_caps_declared` refuses to load the config if any cap is nulled again.
- **Register-drift stop:** the run halts if `marker_rate` leaves the regime it opened in — the direction the register reward can be gamed. The band is set from a simulated false-alarm rate rather than a per-check σ level (DEVLOG, 2026-08-09), and the operating point is pinned by a test. At the 8-prompt rollout the band widens to 0.933 centroid σ and power against a +0.35 step falls to about half; the rule was not retuned to recover it, and the addendum records what that costs.
- **Checkpoint selection:** `manage.py rlsf_select` scores every saved checkpoint of an arm on the dev slice and ranks on the chrF adequacy band then held-out register distance — the rule `src/peft/sweep.py` uses. No judge: the training rubric is what the paid arms optimize, and the evaluation raters are spent once, on val.
- **Per-step telemetry:** reward mean and spread, per-component raw and normalized means, feasibility and degenerate-group fraction, signed z per centroid feature with clustered error, the drift verdict, the trainer's own `kl`/`grad_norm`/`learning_rate`, and `adapter_delta` — how far the LoRA weights have travelled from the initialization. The last exists so a flat style score can be told apart from an optimizer that never moved.
- **Fallback:** if the GRPO run does not converge under budget, RLSF is reported using **best-of-N reranking** of PEFT-checkpoint samples, scored with the same reward.

---

## Evaluation

All four conditions, same split, same decoding settings (temperature, top-p, max-new-tokens fixed and logged before inference). Everything reported so far is on validation; the test split is reserved for the final pass.

| Axis | Metric | Notes |
|---|---|---|
| Stylistic fidelity (primary) | LLM-as-Judge Φ | **Evaluation-time** template, separate from training-time template |
| Semantic adequacy | COMET (`wmt22-comet-da`) | per-segment, paired bootstrap CI 95% |
| Lexical overlap | BLEU (`sacrebleu`) | corpus-level |
| Stylometrics | Lexical density, TTR, avg sentence length + variance, register-marker counts | per-segment; aggregated per condition |
| Stylometric distance | Distance vs. reference on standardized feature vector | per-condition |
| Cost | Trainable params, inference latency, RLSF API calls + spend | reported per condition |

### Judge circularity mitigation
- Two separate, fixed judge templates: training-time (for RLSF reward) vs. evaluation-time (for final scoring). Templates are frozen before their respective phases and never tuned against the test set. Both are hashed in [`prompts/hashes.json`](prompts/hashes.json) **at freeze time**, and `src/rlsf/reward.py:load_train_template` verifies the training rubric against that record on every load, so which rubric a run read is checkable from the artifacts rather than asserted from a config (DEVLOG, 2026-08-07).
- Test partition is unseen during RLSF.
- A **cross-family** confirmation pass with a judge from a different LLM family was run on 2026-08-05 (`gpt-5.6-terra`, same frozen rubric, all seven conditions, val); judge–judge agreement and contrast replication are reported below.

> **Open issue.** The evaluation-time judge is `claude-haiku-4-5`
> ([`configs/judge_eval.yaml`](configs/judge_eval.yaml)), which offers no seed control, so
> Φ is not byte-reproducible. Φ is the *primary* stylistic metric, so this remains the
> weakest link in the current results — Φ gaps on the order of 0.05 should not be treated
> as real. See [`docs/DEVLOG.md`](docs/DEVLOG.md), 2026-07-23.
>
> **The cross-family pass has now been run, and it narrows what Φ supports.** A second
> rater from a different family (`gpt-5.6-terra`,
> [`configs/judge_eval_gpt.yaml`](configs/judge_eval_gpt.yaml)) scored the same segments
> against the same frozen `prompts/judge_eval.txt`
> (`results/judge_gpt_val.json`, agreement report `results/judge_agreement_gpt_val.json`).
> Agreement is fair at best — quadratic-weighted κ = 0.384 [0.370, 0.398], exact 27.8 %,
> adjacent 77.2 % — and the second rater sits 0.95 [0.93, 0.97] rubric points higher on the
> same segments, so **Φ_A and Φ_B are not interchangeable as absolute values and are never
> averaged.** Of the nine pre-specified contrasts, **four of the five primary contrasts
> against the zero-shot floor do not replicate under the second rater** (two reverse sign),
> while all three AFSP-internal contrasts fail to separate under *both* raters. Which
> conclusions survive the rater swap is set out in
> [Threats to validity](#threats-to-validity); the full table is in
> [`docs/DEVLOG.md`](docs/DEVLOG.md), 2026-08-05.
>
> Separately, the judge and the commercial reference baseline are the *same model*
> (`claude-haiku-4-5`), so that baseline's Φ is a self-judged score. It does not affect
> any comparison among the six study conditions — none of which is generated by the judge
> — but it makes the reference baseline's Φ uninterpretable as a register measure. See
> [External reference baseline](#external-reference-baseline--commercial-zero-shot-not-a-condition-of-the-study).

### Statistics
- System-level comparisons: paired bootstrap at the segment level, α = 0.05. Primary: each adaptation vs. the zero-shot base. Secondary: pairwise among the adaptation conditions, and adjacent rungs of the prompting ablation ladder.
- Evaluation-component agreement (RQ4): pairwise Spearman correlation between COMET, stylometric distance, and LLM-as-Judge, with 95 % bootstrap CIs, computed at two levels and **reported at both** (`manage.py metric_agreement` → `results/metric_agreement_val.json`). Condition level is six points and is descriptive only. Segment level is the one with power, and the two do not agree: across the six study conditions ρ(Φ, `stylo_dist`) = −0.657 over six condition means, but pooled over 7,937 segments ρ(Φ, `centroid_dist`) = −0.045 [−0.068, −0.023] and ρ(Φ, `band_dist`) = +0.007 [−0.015, +0.030]. Φ tracks COMET far better than it tracks either register proxy (ρ = 0.453 [0.434, 0.472]). Quoting the condition-level figure alone would be an ecological correlation: what separates conditions on average does not order segments within a condition.

### Results *(validation split)*

All figures below are on the **validation** split (n = 1,323), same locked decoding
(greedy, seed 42, bf16, unquantized base), same evaluation files. **The test split is
sealed and has not been generated on.** Judge Φ is a 1–5 rubric scored by
`claude-haiku-4-5` at temperature 0.

| Condition | COMET | chrF | BLEU | Judge Φ | Lex. density | TTR | Stylo. dist. | Trainable params |
|---|---|---|---|---|---|---|---|---|
| Zero-shot | 0.6480 | 36.42 | 10.27 | 2.546 | 0.4025 | 0.8434 | 0.6518 | 0 |
| Random few-shot (k = 8) | 0.6644 | 37.52 | 11.64 | 2.633 | 0.4103 | 0.8444 | 0.4736 | 0 |
| kNN few-shot (baseline, k = 8) | 0.6839 | 39.82 | 13.99 | 2.748 | 0.4034 | 0.8367 | 0.4005 | 0 |
| AFSP-margin (k = 8, λ = 0) | 0.6824 | 39.68 | 13.69 | 2.763 | 0.4060 | 0.8387 | 0.3910 | 0 |
| AFSP-full (k = 8, λ = 0.75) | 0.6853 | 39.99 | 14.52 | **2.791** | 0.4105 | 0.8421 | 0.3698 | 0 |
| PEFT (LoRA, r = 32, 2 ep.) | **0.6986** | **41.58** | **16.90** | 2.744 | 0.4088 | 0.8515 | **0.2886** | 80.7 M (1.06 %) |
| RLSF (GRPO) | — | — | — | — | — | — | — | — |

Lower `Stylo. dist.` is better (standardized distance to the target-register centroid).
Latency is not instrumented; `outputs/*_usage.json` records token counts only.
The table lists **conditions of the study only**; the commercial zero-shot reference
baseline is reported separately below and is not one of them.

### What the paired bootstrap says

Segment-level paired bootstrap, 10,000 resamples, α = 0.05, seed 42, on the same
1,323 segments (1,322 for pairs involving `knn_fewshot`, whose judge coverage drops
one segment). Full tables in `results/bootstrap_{comet,judge,chrf,bleu}_val.json`.

**Significant.** Every rung of the retrieval ladder floor separates: zero-shot →
random few-shot → kNN few-shot is significant on all four metrics (Φ +0.087 and
+0.115; COMET +0.016 and +0.020). PEFT beats AFSP-full on COMET (+0.013,
95% CI [0.009, 0.018]), chrF (+1.06 [0.36, 1.77]) and BLEU (+2.13 [1.44, 2.84]).

> **The Φ column of the ladder result is rater-dependent.** Under the second, cross-family
> judge the same rungs do not separate on Φ (+0.027 and −0.014, both intervals spanning
> zero), and PEFT − zero-shot reverses sign. The COMET, chrF and BLEU separations are
> unaffected — they involve no judge. Read the ladder as separating on adequacy and overlap,
> and as separating on Φ *under one rater of two*. See
> [Threats to validity](#threats-to-validity).

**Not significant.** The comparisons the contribution rests on:

| Comparison | COMET | Judge Φ | chrF | BLEU |
|---|---|---|---|---|
| AFSP-full − kNN few-shot | +0.001 (p = .43) | +0.043 (p = .078) | +0.20 (p = .41) | +0.45 (p = .065) |
| AFSP-margin − kNN few-shot | −0.002 (p = .25) | +0.016 (p = .42) | −0.07 (p = .70) | −0.13 (p = .49) |
| AFSP-full − AFSP-margin | +0.003 (p = .077) | +0.028 (p = .23) | +0.27 (p = .23) | **+0.58 (p = .010)** |
| AFSP-full − PEFT | −0.013 (p < .001) | +0.047 (p = .12) | −1.06 (p = .003) | −2.13 (p < .001) |

Read against these intervals, three earlier readings of the table do not hold:

- **AFSP does not separate from its own baseline.** On no metric does AFSP-full beat
  `knn_fewshot` at α = 0.05. Φ (p = .078) and BLEU (p = .065) are near misses; COMET
  and chrF are flat. The adaptive layer is not yet demonstrated on val.
- **PEFT and AFSP-full do not "disagree".** PEFT's lead on COMET/chrF/BLEU is
  significant; AFSP-full's +0.047 Φ lead is not (95% CI [−0.012, 0.105]). This is one
  measure separating the families and the other failing to, not two measures ranking
  them oppositely. RQ4 needs the agreement analysis, not this pair.
- **The one place the register rerank separates is BLEU** (+0.58 over AFSP-margin,
  p = .010) — uncorrected across 56 tests, so it would not survive Holm–Bonferroni,
  and it is the weakest of the four metrics. Treat it as a lead to power up, not a
  result.

**The `Stylo. dist.` ladder is monotone but not separable.** The column falls
0.652 → 0.370 across the ladder, and `manage.py stylometrics_ci` resamples it by
recomputing the condition's mean feature vector inside each bootstrap replicate
(`results/stylometrics_ci_val.json`, 2,000 paired resamples, seed 42). Under those
intervals the two AFSP steps do not separate:

| Adjacent step | Δ `stylo_dist` | 95 % CI | p |
|---|---:|---|---:|
| AFSP-full − AFSP-margin | −0.021 | [−0.057, +0.012] | .232 |
| AFSP-margin − kNN few-shot | −0.009 | [−0.038, +0.021] | .577 |
| kNN few-shot − random few-shot | −0.073 | [−0.131, −0.017] | .004 |
| PEFT − AFSP-full | −0.080 | [−0.130, −0.030] | .002 |

So the retrieval floor separates and PEFT separates, while the adaptive layer does not —
the same pattern the COMET/Φ/chrF/BLEU bootstraps show. Read the monotone ordering as
descriptive; only the two marked steps are resolved.

**Detection floor for the RLSF arm.** The Φ CI half-width against PEFT at n = 1,323 is
≈0.058, and the COMET half-width ≈0.005. RLSF must clear those margins over its own
PEFT initialization to be reportable. For scale, the entire prompting ladder from
zero-shot to AFSP-full moves Φ by 0.245.

### External reference baseline — commercial zero-shot *(not a condition of the study)*

`commercial_haiku` is `claude-haiku-4-5` prompted zero-shot with the **same**
`prompts/style_instruction.txt` as the `zeroshot` rung, temperature 0, on the same 1,323
validation segments ([`configs/commercial_haiku_zeroshot.yaml`](configs/commercial_haiku_zeroshot.yaml)).
It is a **labelled external reference baseline**, not an arm of the comparison: it holds
neither the frozen base model nor a matched adaptation budget. It answers one question
only — what a general-purpose commercial model does on this corpus without adaptation.

| Condition | COMET | chrF | BLEU | Judge Φ | Lex. density | TTR | Stylo. dist. | Cost |
|---|---|---|---|---|---|---|---|---|
| *(ref.)* Commercial zero-shot (`claude-haiku-4-5`) | 0.7185 | 45.24 | 18.06 | 3.333 | 0.3928 | 0.8241 | 0.5557 | $0.63 / 1,323 calls |

Paired bootstrap against the study conditions, same estimator as above (10,000
resamples, α = 0.05, seed 42, n = 1,323; chrF/BLEU are segment-level means, so they do
not equal the corpus-level differences implied by the tables). Full records in
`results/bootstrap_{comet,judge,chrf,bleu}_val.json`.

| Comparison | COMET | Judge Φ | chrF | BLEU |
|---|---|---|---|---|
| Commercial − zero-shot | +0.070 [0.066, 0.075] | +0.788 [0.730, 0.844] | +9.39 [8.70, 10.08] | +7.25 [6.61, 7.91] |
| Commercial − PEFT | +0.020 [0.015, 0.025] | +0.590 [0.530, 0.648] | +3.76 [3.00, 4.53] | +1.26 [0.52, 2.03] |
| Commercial − AFSP-full | +0.033 [0.029, 0.037] | +0.543 [0.485, 0.599] | +4.82 [4.11, 5.51] | +3.39 [2.66, 4.12] |

All twelve intervals exclude zero (p < .001, except BLEU vs. PEFT at p = .0008). Three
things constrain how this may be read:

- **Φ here is self-judging and is not admissible as a register finding.** The generator
  and the evaluation-time judge are the *same model*, `claude-haiku-4-5`
  ([`configs/judge_eval.yaml`](configs/judge_eval.yaml)). The +0.590 Φ margin over PEFT is
  a model scoring its own output against a rubric, and known self-preference bias in
  LLM-as-Judge is a sufficient alternative explanation. **The cross-family pass now
  quantifies it as inflation, not fabrication.** The second rater sits 0.87–1.09 rubric
  points above `claude-haiku-4-5` on the six study conditions but only 0.641
  [0.605, 0.678] above it on `commercial_haiku` — i.e. the primary rater is 0.23–0.45
  points *less severe* on its own family's output than its treatment of everything else
  predicts. But the second rater, a different family, also places `commercial_haiku` first
  by a wide margin (Φ_B 3.981 [3.939, 4.022] against `afsp_full` 3.707 [3.661, 3.753]), so
  the lead is real and its size is overstated. Neither reading makes this Φ admissible as a
  register finding.
- **The objective register proxy points the other way.** At `Stylo. dist.` 0.556 the
  commercial baseline is *farther* from the target-register centroid than every few-shot
  rung (0.474–0.370) and far worse than PEFT (0.289) — worse than everything except the
  `zeroshot` floor (0.652) — while scoring the highest Φ of any system. Its signed
  z-deviations show why: lexical density −0.378 and TTR −0.276, the largest of any
  condition. It is producing longer, more function-word-dense, less lexically varied
  English than the authorized register.
  Admitting it to the
  correlation drops ρ(Φ, `stylo_dist`) across systems from −0.657 to −0.250, which is why
  [`src/eval/metric_agreement.py`](src/eval/metric_agreement.py) reports `study_only` and
  `with_reference` as separate scopes rather than pooling them
  (`results/metric_agreement_val.json`).

Per-segment scores are in `results/comet_val.json` and `results/judge_val.json`
(with per-condition segment caches under `results/judge_val_segments/`); the second
rater's are in `results/judge_gpt_val.json` and `results/judge_gpt_val_segments/`;
selection and freeze records are in `results/{afsp,peft}_{sweep,verify}_val.json`.

### Threats to validity

| Type | Threat | Status |
|---|---|---|
| Internal | **Hyperparameters were selected on the split the results are reported on.** The PEFT grid (`configs/peft_sweep.yaml:48`) and the AFSP k × λ sweep (`configs/afsp_sweep.yaml:42`) both rank candidates on `val.jsonl`, and the results table is val | **Open, unquantified.** The val figures for the two tuned conditions (PEFT, AFSP-full) are selection-optimistic by an unmeasured amount; the untuned rungs are not. This is what the sealed test split is held back for, so the final numbers must come from test with no further selection. Until then no val gap between a tuned and an untuned condition should be read as an unbiased effect size |
| Construct | **The register metric used to select is the register metric reported.** `results/stylometrics_centroid.json` drives the AFSP λ rerank, both sweeps' `register_fit` ranking, and the reported `Stylo. dist.` column | **Open by construction.** AFSP's λ = 0.75 and the PEFT checkpoint were chosen to minimise a function of that centroid, so `Stylo. dist.` is not an independent measure for them. `register_fit` (directional) and `stylo_dist` (undirected) are different functionals, which weakens but does not remove the circularity. Φ and COMET are unaffected — neither enters selection |
| Construct | Single-judge dependence for the primary metric Φ | **Measured, 2026-08-05.** Cross-family pass run (`gpt-5.6-terra`, same frozen rubric, 9,132 paired segments): κ = 0.384 [0.370, 0.398], severity offset −0.950 [−0.968, −0.932]. Four of five primary Φ contrasts do not replicate; the AFSP-vs-baseline null does. Bounds rater dependence — does not establish that either rater measures register correctly |
| Construct | Reward judge family-adjacent to Φ_B, for the RLSF row only | **Accepted, not yet incurred** (no RLSF run exists). The reward judge `gpt-4o-mini` is model-distinct from both raters but shares a provider family with Φ_B `gpt-5.6-terra`, so Φ_B is family-adjacent rather than fully clean for the one condition trained against a judge. Weaker than model-identical contamination; Φ_A stays clean. Any RLSF Φ claim states which rater it rests on |
| Internal | Judge non-determinism | **Unresolved for both raters.** `claude-haiku-4-5` exposes no seed; `gpt-5.6-terra` accepts `seed: 42` as best-effort only. Φ differences of order 0.05 are within measurement noise |
| Construct | Rubric identity across raters asserted, not verified | **Open for Φ.** Rater A's results predate `template_sha256` recording, so `template_verified: false` in the agreement report. Closable only by re-spending on Φ_A. Prevented from recurring on the RLSF side: both rubrics were hashed into `prompts/hashes.json` at freeze time on 2026-08-07 and the training rubric is verified on load |
| Measurement | 1.4 % of Φ_B ratings missing (129 of 9,261) | Non-random: a rating is missing because the response did not parse. No imputation; every contrast computed on the paired intersection, n stated |
| Conclusion | Multiplicity across 56 uncorrected tests | Acknowledged; Holm–Bonferroni applied per claim, and correction status stated. The one AFSP separation (BLEU +0.58, p = .010) does not survive it |
| Conclusion | Sample size below the detection floor for small effects | Quantified: Φ half-width ≈0.058, COMET ≈0.005 at n = 1,323. Effects below it are reported as unresolvable, not absent |
| Internal | Single training run per LoRA cell | Unmeasured within-cell variance; no seed replication |
| Internal | Greedy decoding drifted across sessions | 2 and 5 of 1,323 segments in two resumed runs; verify alignment before comparing cross-session artefacts |
| External | One corpus, one language pair, one base model | Scope stated under [Constraints](#constraints); not widened in prose |
| External | Register glossary hand-specified for this corpus | Toggleable in the `prompt:` config block, applied uniformly to every few-shot rung; effect measurable |

---

## Reproducibility

- Fixed random seeds at every stochastic stage (split, PEFT training, AFSP tie-breaking, GRPO rollouts, judge sampling where the provider honours one).
- Pinned library versions in `requirements.txt` (`transformers`, `peft`, `trl`, `sacrebleu`, `sentence-transformers`) and `requirements-comet.txt` (`unbabel-comet`). Retrieval is exact cosine over a stored embedding matrix (`data/knn_index/embeddings.npy`), so there is no ANN library and no index-approximation parameter to pin.
- Logged per run: base model + revision, all prompts (system, user, judge × 2), decoding params, LoRA config, GRPO config, reward weight grid and selected point, file hashes for splits and test outputs.
- Generation is greedy and deterministic, so a condition reproduces byte-for-byte given the same adapter and prompt. **The judge does not:** `claude-haiku-4-5` exposes no seed, so Φ is re-sampled on every run.
- Known gaps: LoRA cells are trained once (no seed replication), and the AFSP register direction is hard-coded in six configs with no derivation script (see [`docs/DEVLOG.md`](docs/DEVLOG.md), 2026-07-18).
- Greedy decoding reproduces byte-for-byte *within* a session but did not across sessions: two resumed runs (`zeroshot`, `afsp_full`) differ from their swept cells in 2 and 5 of 1,323 segments (see [`docs/DEVLOG.md`](docs/DEVLOG.md), 2026-07-31).

---

## Constraints

- **Compute:** Colab for every GPU stage (sweeps, training, full-split inference) — the 8 GB development GPU cannot hold `Qwen2.5-7B-Instruct` in bf16, and quantizing it would change the frozen-base definition. LoRA default; QLoRA fallback.
- **API budget:** judge calls and the commercial reference baseline are the only paid components so far — **$6.76 recorded to date**, of which $6.12 is the 2026-08-05 cross-family judge pass (9,261 calls, Batch API at 50 % discount). RLSF judge calls will dominate once implemented, and are capped at $25 (declared 2026-08-08). Spend, the derivation of that cap, the per-run rules and the fallback are in [`docs/budget.md`](docs/budget.md). If the cap is approached, switch to best-of-N reranking.
- **Scope:** the project does not claim to fully capture literary or sacred style. It documents trade-offs across three adaptation families on one specific corpus and language combination not previously addressed in published LLM-MT work.

---

## Setup

```bash
# clone
git clone <repo-url>
cd <repo>

# environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m venv .venv-comet
source .venv-comet/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu  # local/off-GPU; skip on Colab
pip install -r requirements-comet.txt
deactivate

# data prep (expects raw TSVs under data/raw/)
python -m src.data.preprocess
python -m src.data.split        # writes data/splits/ with hashes
```

### Running a condition

```bash
# Prompting ablation ladder (all share the source-side index)
python manage.py build_index --config configs/base_qwen.yaml
python manage.py stylometrics --build-centroid          # required by afsp_full
python manage.py infer --condition zeroshot       --config configs/base_qwen.yaml
python manage.py infer --condition random_fewshot --config configs/base_qwen.yaml
python manage.py infer --condition knn_fewshot    --config configs/base_qwen.yaml
python manage.py infer --condition afsp_margin    --config configs/base_qwen.yaml
python manage.py infer --condition afsp_full      --config configs/base_qwen.yaml

# AFSP hyperparameter selection (run BEFORE the ladder; freeze k/λ into base_qwen.yaml)
python manage.py afsp_sweep  --config configs/afsp_sweep.yaml               # proxy rank the grid
python manage.py afsp_verify --config configs/afsp_sweep.yaml --top 3 \
    --judge-config configs/judge_eval.yaml                                  # confirm on COMET + Φ

# PEFT: sweep -> verify -> freeze adapter_path -> generate
python manage.py peft_sweep  --config configs/peft_sweep.yaml               # train grid, rank candidates
python manage.py peft_verify --config configs/peft_sweep.yaml --top 3 \
    --judge-config configs/judge_eval.yaml
python manage.py infer --condition peft --config configs/peft_qwen.yaml
# (`manage.py peft --config configs/peft_qwen.yaml` trains a single adapter without the sweep)

python manage.py rlsf_dev --target 500 --seed 42             # dev slice: 499 / 10,361 (done)

# Reward-path smoke: Kiwi handshake, within-group reward spread, StepLog write, length band.
# Bounded by rlsf.pilot.judge_calls (80), separately from the training caps.
python manage.py rlsf_smoke --segments 4 --skip_judge             # free: judge held flat
python manage.py rlsf_smoke --segments 20 --group_size 4 --yes    # paid: 80 judge calls, $0.0059
python manage.py rlsf_smoke --hyps_file outputs/rlsf/smoke_hyps.jsonl --yes  # re-score, no sampling

# RLSF training and the ω grid. Implemented, not yet run.
# python manage.py rlsf_train --dry_run                     # CPU wiring check, 0.5B, no spend
# python manage.py drift_oc --steps 300                     # operating characteristic of the stop rule
# The 50-rollout smoke is where the geometry is checked before a paid arm starts. Read three
# things off it: `adapters ['default', 'ref']` (the KL anchors at the init, not the base), no
# offload assertion (the policy is wholly on the card), and the `peak VRAM` line against the
# rented card. It also writes 2 checkpoints, which is what the arm's disk cost scales from.
# python manage.py rlsf_train --cell w3_0.0 --steps 50 --skip_judge \
#   --out outputs/rlsf/smoke50_steps.jsonl --adapter_out models/rlsf_smoke50 --overwrite
# python manage.py rlsf_train --cell w3_0.0 --steps 300 --skip_judge   # RL-Metric, free
# python manage.py rlsf_train --cell w3_2.0 --steps 300 --yes          # 9,600 calls, $0.71
# python manage.py rlsf_train --cell w3_6.0 --steps 300 --yes          # 9,600 calls, $0.71
# python manage.py rlsf_pool  --yes                         # dev-slice best-of-N pool
# python manage.py rlsf_omega                               # re-argmax the pool per ω cell, free
# python manage.py rlsf_select --cell w3_2.0                # checkpoint selection on all 499 dev segments, free
# There is no `rlsf` inference condition: a trained adapter is scored through `peft`.
# python manage.py infer --condition peft --config configs/rlsf_eval_w3_2.0.yaml \
#   --out-name rlsf_w3_2.0
```

The GPU stages (sweeps, training, full-split inference) do not fit an 8 GB development
GPU with the base unquantized; they are run on Colab from the runbooks in
[`notebooks/`](notebooks/).

### Evaluation

The evaluation backbone scores any set of `<condition>_<split>.jsonl` files. Let
`CONDS = zeroshot random_fewshot knn_fewshot afsp_margin afsp_full peft`:

```bash
# Surface overlap + register proxy (BLEU, chrF, marker rate)
python manage.py eval          --conditions $CONDS --split val

# Learned adequacy (COMET wmt22-comet-da) -> results/comet_val.json  (per-segment)
# NOTE: run this one from the .venv-comet environment (see Setup), not .venv.
python manage.py comet         --conditions $CONDS --split val

# Register fidelity Φ, evaluation-time LLM-as-Judge -> results/judge_val.json
python manage.py judge         --conditions $CONDS --split val --config configs/judge_eval.yaml

# Stylometrics vs. the target-register centroid (per-condition feature table)
python manage.py stylometrics  --conditions $CONDS --split val

# Bootstrap CIs, paired adjacent differences and rank distributions for stylo_dist and the
# signed z-vector -> results/stylometrics_ci_val.json
python manage.py stylometrics_ci --conditions $CONDS --split val

# RQ4 metric agreement, condition level and segment level -> results/metric_agreement_val.json
python manage.py metric_agreement --conditions $CONDS --split val

# Paired-bootstrap 95% CIs for pairwise differences (α = 0.05), any metric
python manage.py bootstrap --metric chrf  --conditions $CONDS --split val --adjacent
python manage.py bootstrap --metric comet --conditions $CONDS --split val --adjacent
python manage.py bootstrap --metric judge --conditions $CONDS --split val --adjacent
```

`bootstrap` computes chrF/BLEU on the fly from the inference files and reads
COMET/judge from the JSON their own commands write, so run those first. Each
non-baseline condition is compared against the ladder floor (`zeroshot`),
`--adjacent` adds each consecutive-rung difference, `--pairs a:b` adds arbitrary
comparisons, and `--out` writes the table to
`results/bootstrap_<metric>_<split>.json`. The canonical run behind the results
section above is:

```bash
python manage.py bootstrap --metric $M --conditions $CONDS --split val --adjacent \
    --baseline zeroshot --out \
    --pairs afsp_full:knn_fewshot afsp_margin:peft afsp_full:peft \
            knn_fewshot:peft random_fewshot:peft
```

> **p-values are uncorrected.** The canonical run makes 14 comparisons per metric
> across four metrics. Any single result near α = 0.05 should be read against a
> family-wise correction before it is reported as a finding.

---

## Proposal and methodology

- Full thesis proposal: [`docs/proposal.pdf`](docs/proposal.pdf) — H1–H4 and support criteria
- Engineering and decision log: [`docs/DEVLOG.md`](docs/DEVLOG.md) — what was built, why, and how to reproduce it
- AFSP mechanism specification: [`docs/afsp_strategies.md`](docs/afsp_strategies.md)
- Compute and API budget: [`docs/budget.md`](docs/budget.md) — spend to date, the declared $25 RLSF cap and its derivation, per-run rules, fallback
- Not yet written: `docs/methodology.md` (long-form methodology)

---

## Citing

If this work is useful to you, please cite the thesis once available:

```
Mehri, P. (2025). Style-Aware Neural Machine Translation of Low-Resource Texts
Using Large Language Models and Reinforcement Learning. Undergraduate thesis,
Bahá'í Institute for Higher Education (BIHE).
```
