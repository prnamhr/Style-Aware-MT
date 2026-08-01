# Style-Aware NMT of Low-Resource Texts

Style-aware neural machine translation of Persian and mixed Persian/Arabic Bahá’í scriptures into English, targeting the formal scriptural register of Shoghi Effendi’s authorized translations.

Undergraduate thesis project — BIHE, Department of Computer Engineering. Supervisor: Dr. Fares Hedayati.

> **Status:** in progress (last updated 2026-07-31).
> Six of seven conditions — the five prompting rungs and PEFT — are implemented, tuned, frozen, and scored on the **validation** split; their numbers are in [Results](#results-validation-split). RLSF is not yet implemented. The test split is sealed and untouched. No testing has been run yet, so all reported differences are point estimates.

---

## What this project does

Modern LLM-based MT is fluent but defaults to a neutral register. For literary and scriptural texts, register *is* meaning. This project compares three ways of adapting an open-source LLM to preserve Shoghi Effendi’s register on a low-resource, mixed-language corpus — under matched conditions, on the same base model and the same data.

The three adaptation strategies:

| Strategy | Mechanism | Updates weights? |
|---|---|---|
| **PEFT (LoRA)** | Supervised fine-tuning of LoRA adapters on parallel data | Yes (adapters only) |
| **AFSP** | Adaptive Few-Shot Prompting; retrieval of stylistically relevant exemplars into the prompt at inference time | No |
| **RLSF (PPO)** | Reinforcement learning from a mixed reward combining COMET, BLEU, and an LLM-as-Judge style score | Yes (from PEFT init) |

Plus a fourth condition — the **unadapted base model** — as reference.

All four are evaluated on the same held-out test set with COMET, BLEU, objective stylometric features, and LLM-as-Judge scoring.

**Implementation status:** PEFT and the AFSP/prompting ladder are complete and scored on validation; RLSF is not yet implemented. The current comparison is therefore three-way (base, prompting ladder, PEFT).

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
│   ├── rlsf/                  ← PPO loop, reward, best-of-N fallback
│   ├── eval/                  ← COMET, BLEU, stylometrics, LLM-as-Judge
│   └── infer/                 ← test-set inference for all four conditions
├── configs/                   ← YAML configs per condition + decoding settings
├── prompts/
│   ├── style_instruction.txt
│   ├── judge_train.txt        ← judge template used inside RLSF reward
│   └── judge_eval.txt         ← separate judge template for final evaluation
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

### RLSF (PPO)
- **Init:** PEFT checkpoint.
- **Reference policy:** frozen copy of the PEFT checkpoint, used for KL regularization.
- **Reward:**
  ```
  r(y) = ω₁ · COMET(x, y, y*) + ω₂ · BLEU(y, y*) + ω₃ · Φ(y, S_T)
  ```
  with `Φ` an LLM-as-Judge style score using the **training-time** judge template. Weights are dev-tuned over a small grid that intentionally varies ω₃ relative to (ω₁, ω₂).
- **Bounded:** PPO step cap, batch cap, and judge API spend cap to be declared in `docs/budget.md` (not yet written) before training starts.
- **Fallback:** if PPO does not converge under budget, RLSF is reported using **best-of-N reranking** of PEFT-checkpoint samples, scored with the same reward.

---

## Evaluation

All four conditions, same held-out test set, same decoding settings (temperature, top-p, max-new-tokens fixed and logged before inference).

| Axis | Metric | Notes |
|---|---|---|
| Stylistic fidelity (primary) | LLM-as-Judge Φ | **Evaluation-time** template, separate from training-time template |
| Semantic adequacy | COMET (`wmt22-comet-da`) | per-segment, paired bootstrap CI 95% |
| Lexical overlap | BLEU (`sacrebleu`) | corpus-level |
| Stylometrics | Lexical density, TTR, avg sentence length + variance, register-marker counts | per-segment; aggregated per condition |
| Stylometric distance | Distance vs. reference on standardized feature vector | per-condition |
| Cost | Trainable params, inference latency, RLSF API calls + spend | reported per condition |

### Judge circularity mitigation
- Two separate, fixed judge templates: training-time (for RLSF reward) vs. evaluation-time (for final scoring). Templates are frozen before their respective phases and never tuned against the test set.
- Test partition is unseen during RLSF.
- Where budget permits, a **cross-family** confirmation pass with a judge from a different commercial LLM family is performed; judge–judge agreement is reported.

> **Open issue.** The evaluation-time judge is currently `claude-haiku-4-5`
> ([`configs/judge_eval.yaml`](configs/judge_eval.yaml)), which offers no seed control, so
> Φ is not byte-reproducible; and the cross-family confirmation pass has not been run. Φ is
> the *primary* stylistic metric, so this is the weakest link in the current results — Φ gaps
> on the order of 0.05 should not be treated as real until it is addressed. See
> [`docs/DEVLOG.md`](docs/DEVLOG.md), 2026-07-23.

### Statistics
- System-level comparisons: paired bootstrap at the segment level, α = 0.05. Primary: each adaptation vs. the zero-shot base. Secondary: pairwise among the adaptation conditions, and adjacent rungs of the prompting ablation ladder.
- Evaluation-component agreement (RQ4): pairwise Spearman correlation between COMET, stylometric distance, and LLM-as-Judge, with 95 % bootstrap CIs. Descriptive only.

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
| RLSF (PPO) | — | — | — | — | — | — | — | — |

Lower `Stylo. dist.` is better (standardized distance to the target-register centroid).
Latency is not instrumented; `outputs/*_usage.json` records token counts only.

### What the paired bootstrap says

Segment-level paired bootstrap, 10,000 resamples, α = 0.05, seed 42, on the same
1,323 segments (1,322 for pairs involving `knn_fewshot`, whose judge coverage drops
one segment). Full tables in `results/bootstrap_{comet,judge,chrf,bleu}_val.json`.

**Significant.** Every rung of the retrieval ladder floor separates: zero-shot →
random few-shot → kNN few-shot is significant on all four metrics (Φ +0.087 and
+0.115; COMET +0.016 and +0.020). PEFT beats AFSP-full on COMET (+0.013,
95% CI [0.009, 0.018]), chrF (+1.06 [0.36, 1.77]) and BLEU (+2.13 [1.44, 2.84]).

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

The `Stylo. dist.` monotonicity across the ladder (0.652 → 0.370) is descriptive and
was not bootstrapped; it is a distance between mean feature vectors, not a per-segment
score, so the CLI cannot resample it as it stands.

**Detection floor for the RLSF arm.** The Φ CI half-width against PEFT at n = 1,323 is
≈0.058, and the COMET half-width ≈0.005. RLSF must clear those margins over its own
PEFT initialization to be reportable. For scale, the entire prompting ladder from
zero-shot to AFSP-full moves Φ by 0.245.

Per-segment scores are in `results/comet_val.json` and `results/judge_val.json`
(with per-condition segment caches under `results/judge_val_segments/`); selection and
freeze records are in `results/{afsp,peft}_{sweep,verify}_val.json`.

---

## Reproducibility

- Fixed random seeds at every stochastic stage (split, PEFT training, AFSP tie-breaking, PPO rollouts, judge sampling).
- Pinned library versions (`transformers`, `peft`, `trl`, `unbabel-comet`, `sacrebleu`, `sentence-transformers`, `faiss`).
- Logged per run: base model + revision, all prompts (system, user, judge × 2), decoding params, LoRA config, PPO config, reward weight grid and selected point, file hashes for splits and test outputs.
- Generation is greedy and deterministic, so a condition reproduces byte-for-byte given the same adapter and prompt. **The judge does not:** `claude-haiku-4-5` exposes no seed, so Φ is re-sampled on every run.
- Known gaps: LoRA cells are trained once (no seed replication), and the AFSP register direction is hard-coded in six configs with no derivation script (see [`docs/DEVLOG.md`](docs/DEVLOG.md), 2026-07-18).
- Greedy decoding reproduces byte-for-byte *within* a session but did not across sessions: two resumed runs (`zeroshot`, `afsp_full`) differ from their swept cells in 2 and 5 of 1,323 segments (see [`docs/DEVLOG.md`](docs/DEVLOG.md), 2026-07-31).

---

## Constraints

- **Compute:** Colab for every GPU stage (sweeps, training, full-split inference) — the 8 GB development GPU cannot hold `Qwen2.5-7B-Instruct` in bf16, and quantizing it would change the frozen-base definition. LoRA default; QLoRA fallback.
- **API budget:** judge calls are the only paid component so far; RLSF judge calls will dominate once implemented. The declared cap lives in the proposal (`docs/budget.md` is not yet written); if approached, switch to best-of-N reranking.
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
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu  # local/off-GPU; skip on Colab
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

# RLSF — not yet implemented; neither the `rlsf` command nor configs/rlsf.yaml exists yet
# python manage.py rlsf  --config configs/rlsf.yaml
# python manage.py infer --condition rlsf --config configs/rlsf.yaml
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
- Not yet written: `docs/methodology.md` (long-form methodology) and `docs/budget.md` (declared compute and API caps)

---

## Citing

If this work is useful to you, please cite the thesis once available:

```
Mehri, P. (2025). Style-Aware Neural Machine Translation of Low-Resource Texts
Using Large Language Models and Reinforcement Learning. Undergraduate thesis,
Bahá'í Institute for Higher Education (BIHE).
```
