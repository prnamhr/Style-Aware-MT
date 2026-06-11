# Style-Aware NMT of Low-Resource Texts

Style-aware neural machine translation of Persian and mixed Persian/Arabic Bahá’í scriptures into English, targeting the formal scriptural register of Shoghi Effendi’s authorized translations.

Undergraduate thesis project — BIHE, Department of Computer Engineering. Supervisor: Dr. Fares Hedayati.

> **Status:** in progress. This README tracks the current experimental plan; results tables fill in as conditions are run.

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

---

## Research questions

- **RQ1.** How do PEFT, AFSP, and RLSF compare on semantic adequacy, stylistic fidelity, and compute/data cost under matched conditions?
- **RQ2.** Do retrieval-based exemplars (AFSP) and reward-driven updates (RLSF) shift outputs toward the target register relative to PEFT, measurably via stylometrics *and* judge scores?
- **RQ3.** How sensitive is RLSF to the reward weights (ω₁, ω₂, ω₃) on COMET / BLEU / style?
- **RQ4.** When applied to the same outputs, where do COMET, stylometric features, and LLM-as-Judge agree and disagree?

Full hypotheses (H1–H4) and support criteria are in [`docs/methodology.md`](docs/methodology.md). The project does **not** pre-commit to a ranking of the three adaptation methods.

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
│   ├── afsp/                  ← retrieval index + prompt assembly
│   ├── rlsf/                  ← PPO loop, reward, best-of-N fallback
│   ├── eval/                  ← COMET, BLEU, stylometrics, LLM-as-Judge
│   └── infer/                 ← test-set inference for all four conditions
├── configs/                   ← YAML configs per condition + decoding settings
├── prompts/
│   ├── style_instruction.txt
│   ├── judge_train.txt        ← judge template used inside RLSF reward
│   └── judge_eval.txt         ← separate judge template for final evaluation
├── outputs/                   ← <condition>_test.jsonl per run
├── results/                   ← metrics_<condition>.json, tables, plots
├── docs/
│   ├── proposal.pdf           ← thesis proposal
│   ├── methodology.md         ← long-form methodology, H1–H4 support criteria
│   └── budget.md              ← declared compute and API caps (§10.3)
└── notebooks/                 ← analysis, agreement plots, qualitative inspection
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
Open-source multilingual decoder-only Transformer in the **7B–8B** range with documented Persian and Arabic coverage. Final choice is decided by a small dev-set bake-off before any condition is trained, then frozen across all four conditions so differences are attributable to adaptation, not model identity.

### Reference (unadapted)
Base model, minimal style instruction (see `prompts/style_instruction.txt`), no exemplars, no fine-tuning. Provides the lower bound and the H1 comparison anchor.

### PEFT (LoRA)
LoRA adapters on the query and value projection layers of each attention block, base weights frozen. Trained with token-level MLE on the training partition. Rank, LR, and step count tuned on dev. QLoRA is the memory-pressure fallback. The resulting checkpoint also serves as the **RLSF initialization**.

### AFSP (retrieval-based ICL)
No parameter updates. At inference time:

1. Embed the source segment with a multilingual sentence-transformer.
2. Retrieve top-k stylistically relevant exemplars by nearest neighbour over a FAISS index built over the **English (target-side)** training partition.
3. Insert them into a fixed prompt template (system role + style instruction + k exemplars + new source).

Shot-count sensitivity on dev over **k ∈ {2, 4, 8}**, subject to base model context limits. Final k is fixed on dev before test inference. Pattern: Tang et al. [AFSP, 2025]; related precedents in Wang et al. style-activation prompting and style-matching exemplar selection.

### RLSF (PPO)
- **Init:** PEFT checkpoint.
- **Reference policy:** frozen copy of the PEFT checkpoint, used for KL regularization.
- **Reward:**
  ```
  r(y) = ω₁ · COMET(x, y, y*) + ω₂ · BLEU(y, y*) + ω₃ · Φ(y, S_T)
  ```
  with `Φ` an LLM-as-Judge style score using the **training-time** judge template. Weights are dev-tuned over a small grid that intentionally varies ω₃ relative to (ω₁, ω₂).
- **Bounded:** PPO step cap, batch cap, and judge API spend cap declared in `docs/budget.md` before training starts.
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

### Statistics
- System-level comparisons: paired bootstrap at the segment level, α = 0.05. Primary: each adaptation vs. Reference. Secondary: pairwise among the three adaptation conditions.
- Evaluation-component agreement (RQ4): pairwise Spearman correlation between COMET, stylometric distance, and LLM-as-Judge, with 95 % bootstrap CIs. Descriptive only.

### Results table *(filled as conditions complete)*

| Condition | COMET | BLEU | LLM-Judge Φ | Lex. density | TTR | Stylo. dist. | Latency (s/seg) | Trainable params |
|---|---|---|---|---|---|---|---|---|
| Reference | — | — | — | — | — | — | — | 0 |
| PEFT (LoRA) | — | — | — | — | — | — | — | — |
| AFSP (k = —) | — | — | — | — | — | — | — | 0 |
| RLSF (PPO) | — | — | — | — | — | — | — | — |

Detailed per-segment scores and bootstrap CIs land in `results/`.

---

## Reproducibility

- Fixed random seeds at every stochastic stage (split, PEFT training, AFSP tie-breaking, PPO rollouts, judge sampling).
- Pinned library versions (`transformers`, `peft`, `trl`, `unbabel-comet`, `sacrebleu`, `sentence-transformers`, `faiss`).
- Logged per run: base model + revision, all prompts (system, user, judge × 2), decoding params, LoRA config, PPO config, reward weight grid and selected point, file hashes for splits and test outputs.

---

## Constraints

- **Compute:** Colab for training. LoRA default; QLoRA fallback.
- **API budget:** RLSF judge calls are the dominant cost. Hard cap declared in `docs/budget.md`; if approached, switch to best-of-N reranking.
- **Scope:** the project does not claim to fully capture literary or sacred style. It documents trade-offs across three adaptation families on one specific corpus and language combination not previously addressed in published LLM-MT work.

---

## Setup

> Setup commands are placeholders for now; they’re finalized once the base model is selected.

```bash
# clone
git clone <repo-url>
cd <repo>

# environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# data prep (expects raw TSVs under data/raw/)
python -m src.data.preprocess
python -m src.data.split        # writes data/splits/ with hashes
```

### Running a condition

```bash
# Reference
python -m src.infer --condition reference --config configs/reference.yaml

# PEFT
python -m src.peft.train       --config configs/peft.yaml
python -m src.infer --condition peft --config configs/peft.yaml

# AFSP
python -m src.afsp.build_index --config configs/afsp.yaml
python -m src.infer --condition afsp --config configs/afsp.yaml

# RLSF
python -m src.rlsf.train       --config configs/rlsf.yaml
python -m src.infer --condition rlsf --config configs/rlsf.yaml
```

### Evaluation

```bash
python -m src.eval.run --condition reference peft afsp rlsf
python -m src.eval.agreement   # RQ4 pairwise Spearman + plots
```

Outputs land in `results/`.

---

## Proposal and methodology

- Full thesis proposal: [`docs/proposal.pdf`](docs/proposal.pdf)
- Long-form methodology with H1–H4 support criteria: [`docs/methodology.md`](docs/methodology.md)
- Declared compute and API budget: [`docs/budget.md`](docs/budget.md)

---

## Citing

If this work is useful to you, please cite the thesis once available:

```
Mehri, P. (2025). Style-Aware Neural Machine Translation of Low-Resource Texts
Using Large Language Models and Reinforcement Learning. Undergraduate thesis,
Bahá'í Institute for Higher Education (BIHE).
```
