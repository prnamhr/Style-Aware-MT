# Style-Aware NMT of Low-Resource Texts

Style-aware machine translation of Persian and mixed Persian/Arabic scriptures into English, targeting the formal scriptural register of Shoghi Effendi’s authorized translations.

Undergraduate thesis project — BIHE, Department of Computer Science. Supervisor: Dr. Fares Hedayati.

> **Status: in progress.** The data pipeline and an API-backed AFSP/reference inference loop are implemented and validated on a smoke slice. The local open-source model, the PEFT and RLSF stages, and the full evaluation harness (COMET, stylometrics, LLM-as-Judge, bootstrap CIs) are designed but **not yet implemented** — see [Implementation status](#implementation-status) for the exact line between built and planned.

---

## What this project does

Modern LLM-based MT is fluent but defaults to a neutral register. For literary and scriptural texts, register *is* meaning. This project compares ways of adapting an LLM to preserve Shoghi Effendi’s register on a low-resource, mixed-language corpus under matched conditions, on the same base model and the same data.

The adaptation strategies under study:

| Strategy | Mechanism | Updates weights? |
|---|---|---|
| **PEFT (LoRA)** | Supervised fine-tuning of LoRA adapters on parallel data | Yes (adapters only) |
| **AFSP** | Adaptive Few-Shot Prompting; retrieval of relevant exemplars into the prompt at inference time | No |
| **RLSF (PPO)** | Reinforcement learning from a mixed reward combining COMET, BLEU, and an LLM-as-Judge style score | Yes (from PEFT init) |

Plus a fourth condition — the **unadapted base model** — as reference.

All conditions are to be evaluated on the same held-out test set with COMET, BLEU, objective stylometric features, and LLM-as-Judge scoring. The project does **not** pre-commit to a ranking of the adaptation methods.

---

## Implementation status

This table is the authoritative record of what runs today. Anything marked *planned* is designed in the proposal but not yet in the codebase.

| Component | Module | Status |
|---|---|---|
| Corpus preprocessing & normalization | `src/data/preprocess.py` | ✅ implemented |
| Work-level split + cross-boundary dedup + leakage audit | `src/data/split.py` | ✅ implemented |
| AFSP retrieval index (embed + brute-force cosine) | `src/afsp/build_index.py`, `retrieve.py`, `embed.py` | ✅ implemented |
| API inference: `reference` + `afsp` conditions | `src/infer/run.py` (+ OpenAI/Anthropic clients) | ✅ implemented |
| Smoke scorer: BLEU, chrF, archaic-marker proxy | `src/eval/quick.py` | ✅ implemented |
| Command dispatcher | `manage.py` | ✅ implemented |
| **Open-source 7–8B local base model** | — | ⏳ planned (dev-set bake-off pending) |
| **PEFT (LoRA) training** | `src/peft/` | ⏳ planned (package stub only) |
| **RLSF (PPO) training + reward** | `src/rlsf/` | ⏳ planned (package stub only); feasibility is an open risk — see [Constraints](#constraints) |
| **Final eval harness**: COMET, stylometrics, LLM-as-Judge, paired bootstrap | `src/eval/` | ⏳ planned (`quick.py` is a smoke scorer, not the harness) |
| Cross-family judge agreement (RQ4) | — | ⏳ planned |

---

## Research questions

- **RQ1.** How do PEFT, AFSP, and RLSF compare on semantic adequacy, stylistic fidelity, and compute/data cost under matched conditions?
- **RQ2.** Do retrieval-based exemplars (AFSP) and reward-driven updates (RLSF) shift outputs toward the target register relative to PEFT, measurably via stylometrics *and* judge scores?
- **RQ3.** How sensitive is RLSF to the reward weights (ω₁, ω₂, ω₃) on COMET / BLEU / style?
- **RQ4.** When applied to the same outputs, where do COMET, stylometric features, and LLM-as-Judge agree and disagree?

Full hypotheses (H1–H4) and support criteria are in the thesis proposal: [`docs/proposal.pdf`](docs/proposal.pdf).

---

## Repository layout

Directories marked *(planned)* are part of the design but not yet populated.

```
.
├── README.md                  ← you are here
├── manage.py                  ← command dispatcher (python manage.py <command>)
├── data/
│   ├── raw/                   ← original sentence/paragraph TSVs (not redistributed)
│   ├── processed/             ← normalized, deduplicated jsonl/tsv
│   └── splits/                ← train/val/test manifests + hashes.json
├── src/
│   ├── data/                  ← preprocessing + work-level split logic       (implemented)
│   ├── afsp/                  ← retrieval index + embedding + prompt assembly (implemented)
│   ├── infer/                 ← API-backed inference (reference + afsp)       (implemented)
│   ├── eval/                  ← quick.py smoke scorer                         (implemented)
│   ├── peft/                  ← LoRA training                                 (planned)
│   └── rlsf/                  ← PPO loop, reward, best-of-N fallback          (planned)
├── configs/
│   ├── openai_smoke.yaml      ← OpenAI-backed smoke config
│   └── anthropic_smoke.yaml   ← Anthropic-backed smoke config
├── prompts/
│   └── style_instruction.txt  ← shared system prompt (target register)
├── docs/
│   ├── proposal.pdf           ← thesis proposal (full methodology, H1–H4)
│   ├── DEVLOG.md              ← engineering & decision log (authoritative)
│   └── afsp_strategies.md     ← AFSP research notes (some strategies not yet implemented)
├── notebooks/                 ← analysis / inspection
├── outputs/                   ← <condition>_test.jsonl per run (gitignored)
└── results/                   ← metrics, tables, plots (gitignored placeholder)
```

---

## Data

- **Source:** Bahá’í scriptures originally in Persian and mixed Persian/Arabic, paired with Shoghi Effendi’s authorized English translations.
- **Granularity:** sentence-level for core experiments; paragraph-level retained as an optional extension.
- **Mixed-language handling:** source segments containing both Persian and Arabic are kept as single segments; the model is responsible for within-segment language mixing, mirroring how the authorized translations were produced.
- **No synthetic data** in training. Synthetic pairs from an unadapted LLM would dilute the target register signal, and validating their quality is out of scope.

### Splits

- Target **80 / 10 / 10** train / val / test, split **by work**, not by random row, to reduce stylistic leakage. Exact ratios drift slightly because whole works are kept intact (the largest work is ≈16% of the corpus).
- After splitting, **cross-boundary deduplication** removes any val/test record whose normalized source *or* target already appears in train.
- A **leakage audit** asserts zero overlapping normalized keys between train and val/test; the run fails loudly otherwise.
- The split is **fixed and frozen**; SHA-256 hashes and full provenance are written to `data/splits/hashes.json`. Repeated runs produce byte-identical files.

Current split (sentence level): **train 10,860 · val 1,323 · test 1,322** across 28 works. The whole-work holdout means test works have *no* stylistic representation in train — a deliberately stringent generalization regime. See [`docs/DEVLOG.md`](docs/DEVLOG.md) for the full rationale and counts.

> **Known limitation.** Dedup is normalized-*exact*, not fuzzy. Because scripture is heavily formulaic, near-paraphrases can survive. This matters most for AFSP, where the retriever may surface a near-gold exemplar for a test query — inflating overlap metrics as leakage, not generalization. Fuzzy near-duplicate detection (MinHash / embedding-threshold) at retrieval time is a planned mitigation.

### Preprocessing

Unicode NFC/NFKC, removal of invisible characters, whitespace and English-punctuation normalization, and conservative Persian standardization (`ي→ی`, `ك→ک`, kashida removal). Arabic diacritics, hamza-bearing characters, and teh marbuta are **preserved** — ≈76% of source sentences carry harakat, and folding them corrupts the training signal and over-merges distinct verses. Tokenization is delegated to the model’s tokenizer, not applied at corpus level.

---

## Systems

### Base model *(planned)*
Open-source multilingual decoder-only Transformer in the **7B–8B** range with documented Persian and Arabic coverage. The final choice is decided by a small dev-set bake-off before any condition is trained, then frozen across all conditions so differences are attributable to adaptation, not model identity. This choice is the project’s quality ceiling and is on the critical path.

### Reference (unadapted) *(implemented for API stand-ins; local model pending)*
Base model, minimal style instruction ([`prompts/style_instruction.txt`](prompts/style_instruction.txt)), no exemplars, no fine-tuning. Provides the lower bound and the H1 comparison anchor.

### AFSP (retrieval-based ICL) *(implemented)*
No parameter updates. At inference time:

1. Embed the source segment with a multilingual sentence-transformer (`intfloat/multilingual-e5-large-instruct`).
2. Retrieve top-k exemplars by **cross-lingual cosine similarity** against an index built over the **English (target-side)** training partition. Retrieval is brute-force cosine over L2-normalized embeddings (exact and instant at ~10.8k rows; FAISS deferred).
3. Insert them into a fixed prompt template (system style instruction + k exemplars, most-similar last + new source).

Shot-count sensitivity over **k ∈ {2, 4, 8}** is planned **on the dev set**; final k is fixed before any test inference.

> **Scope note.** Retrieval is semantic, not stylistic — an embedding model matches by meaning. It yields style-bearing exemplars only because the entire exemplar pool is already in the target register. The advanced selection strategies in [`docs/afsp_strategies.md`](docs/afsp_strategies.md) (margin-based scoring, CAP, multi-view weighting) are research notes and are **not** yet implemented; current retrieval is plain top-k cosine.

### PEFT (LoRA) *(planned)*
LoRA adapters on the query/value projections of each attention block, base weights frozen, trained with token-level MLE on the training partition. Rank, LR, and step count tuned on dev. QLoRA is the memory-pressure fallback. The resulting checkpoint also serves as the **RLSF initialization**.

### RLSF (PPO) *(planned — at risk)*
- **Init:** PEFT checkpoint. **Reference policy:** frozen copy of it, for KL regularization.
- **Reward:** `r(y) = ω₁·COMET(x, y, y*) + ω₂·BLEU(y, y*) + ω₃·Φ(y, S_T)`, with `Φ` a training-time LLM-as-Judge style score. Weights dev-tuned over a small grid that varies ω₃ relative to (ω₁, ω₂).
- **Bounded:** PPO step cap, batch cap, and judge-API spend cap declared before training.
- **Fallback:** if PPO does not converge under budget, RLSF is reported using **best-of-N reranking** of PEFT-checkpoint samples under the same reward. (Best-of-N is a reward-guided decoding fallback, *not* reinforcement learning, and is reported as such.)

---

## Evaluation

### Implemented today — smoke scorer (`src/eval/quick.py`)
BLEU + chrF (sacrebleu) and a crude **archaic-marker rate** (counts of `thou/thee/thy/art/hast/doth/O …` per segment) as a direction-correct register proxy. Purpose: confirm the loop runs and give an early read on whether AFSP shifts register. **Not** the final harness.

### Planned — full harness
| Axis | Metric | Notes |
|---|---|---|
| Stylistic fidelity (primary) | LLM-as-Judge Φ | evaluation-time template, separate from any training-time template |
| Semantic adequacy | COMET (`wmt22-comet-da`) | per-segment, paired bootstrap CI 95% |
| Lexical overlap | BLEU (`sacrebleu`) | corpus-level |
| Stylometrics | lexical density, TTR, sentence-length mean/variance, register markers | per-segment, aggregated per condition |
| Stylometric distance | distance to reference on standardized feature vector | per condition |
| Cost | trainable params, inference latency, RLSF API calls + spend | per condition |

**Judge-circularity mitigation (planned):** separate fixed training-time vs evaluation-time judge templates; test partition unseen during RLSF; a cross-family judge confirmation pass with reported judge–judge agreement where budget permits.

**Statistics (planned):** system-level paired bootstrap at the segment level (α = 0.05), each adaptation vs reference; pairwise Spearman between COMET, stylometric distance, and judge scores for RQ4.

### Preliminary pipeline validation *(not thesis results)*
A harness check on **n = 25, k = 4** across three API stand-ins (not the project’s base model). Differences at this n are within noise and are **not** interpreted as findings — they only confirm the retrieve → prompt → generate → score loop works end-to-end. Two consistent directions: stronger base models score higher overall, and AFSP pulled the archaic-marker rate toward the target distribution on the two stronger models. A separate finding worth flagging: the blanket style instruction **over-archaizes** relative to the references (reference-condition marker rate ≈0.44–0.48 vs ≈0.28 in the gold targets), most harmfully on the weakest model. Full numbers and analysis are in [`docs/DEVLOG.md`](docs/DEVLOG.md).

---

## Setup

```bash
git clone https://github.com/prnamhr/Style-Aware-MT.git
cd Style-Aware-MT

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

API keys (for the inference smoke loop) are read from the environment or a project-root `.env`:

```bash
# .env
OPENAI_API_KEY=...        # for configs/openai_smoke.yaml
ANTHROPIC_API_KEY=...     # for configs/anthropic_smoke.yaml
```

### Commands (`manage.py`)

```bash
python manage.py --help            # list commands

# 1. Data pipeline (expects raw TSVs under data/raw/)
python manage.py preprocess        # -> data/processed/*
python manage.py split             # -> data/splits/* + hashes.json (work-level, deduped, audited)

# 2. AFSP index over the English training side
python manage.py build_index --config configs/openai_smoke.yaml

# 3. Inference (currently: reference + afsp, API-backed)
python manage.py infer --condition reference --config configs/openai_smoke.yaml
python manage.py infer --condition afsp      --config configs/openai_smoke.yaml
# Anthropic: same commands with --config configs/anthropic_smoke.yaml (index is shared)

# 4. Smoke scoring
python manage.py eval --conditions reference afsp
```

> **Warning.** The smoke configs read `data/splits/test.jsonl` with `data.limit: 25` for a one-shot harness check. **Any prompt tuning or k-sweep must switch `data.test_file` to `val.jsonl`** — the test split is reserved for final evaluation only. Output filenames are not provider-tagged, so running a second provider overwrites the first; tag the output path before any run you need to keep.

PEFT, RLSF, and the full evaluation harness will be added under `src/peft/`, `src/rlsf/`, and `src/eval/` with their own configs once the local base model is selected.

---

## Reproducibility

- Deterministic, frozen data split with a SHA-256 manifest (`data/splits/hashes.json`); a leakage audit gates every regeneration.
- Pinned dependencies in `requirements.txt` (`sentence-transformers`, `torch`, `openai`, `anthropic`, `sacrebleu`, `numpy`, `pandas`, `scikit-learn`, …).
- CI (`.github/workflows/ci.yml`) runs ruff lint + format checks and compiles all sources on every push/PR.
- Per inference run, the config records model, decoding/token caps, AFSP k and embed model, and writes a token/cost usage summary.

Seed control and full run-manifest logging for the training stages (PEFT/RLSF) are planned alongside those stages.

---

## Constraints

- **Compute:** Training runs on **Google Colab** (GPU tier depends on availability); the local machine is used only for data prep and inference smoke tests. LoRA is the default adaptation method; QLoRA is the fallback under memory pressure.
- **RLSF feasibility is an open risk.** PPO on a 7–8B model holds the policy, a frozen reference policy, a value head, and generation rollouts in memory at once, and the reward calls a commercial LLM-as-Judge *inside the training loop* — so per-step cost and time are high. PPO is validated with a minimal end-to-end run before the full RLSF stage is committed; if it does not converge under the declared budget, RLSF falls back to best-of-N reranking (reported as such).
- **API budget:** RLSF judge calls dominate cost. A hard cap is declared before training; approaching it triggers the best-of-N fallback.
- **Scope:** the project does not claim to fully capture literary or sacred style. It documents trade-offs across adaptation families on one specific corpus and language combination not previously addressed in published LLM-MT work, and contributes a leakage-controlled mixed Persian/Arabic → English scriptural parallel corpus.

---

## Documentation

- Thesis proposal (full methodology, hypotheses, budget): [`docs/proposal.pdf`](docs/proposal.pdf)
- Engineering & decision log (authoritative rationale for every pipeline choice): [`docs/DEVLOG.md`](docs/DEVLOG.md)

---

## Citing

This is an in-progress undergraduate thesis; please do not cite results until the evaluation is complete. To reference the project:

```
Mehri, P. (2026). Style-Aware Machine Translation of Low-Resource Texts Using
Large Language Models and Reinforcement Learning. Undergraduate thesis (in progress),
Bahá'í Institute for Higher Education (BIHE).
```