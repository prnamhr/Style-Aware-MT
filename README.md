# Style-Aware NMT of Low-Resource Texts

Can a language model preserve not only the meaning of a translation, but also the voice of a specific translator?

This project explores that question using Persian and mixed Persian/Arabic Bahá'í scripture translated into English. The target is the formal, scriptural register found in Shoghi Effendi's authorized translations.

It is an undergraduate Computer Engineering thesis project at BIHE, supervised by Dr. Fares Hedayati.

> **Status:** Core validation experiments are complete. The final test split is still reserved for the final pass, so every score in this README is a validation result.

## The short version

I compare three main ways of adapting the same open-source LLM:

| Method | What it changes | Weight updates? |
|---|---|---|
| **PEFT (LoRA)** | Learns the domain and translation style from parallel data | Yes, adapters only |
| **AFSP** | Retrieves useful examples and places them in the prompt at inference time | No |
| **RLSF (GRPO)** | Fine-tunes from PEFT using a reward that combines adequacy, overlap, and an LLM style judge | Yes |

I also test two hybrid conditions:

- **PEFT+KNN:** PEFT with ordinary retrieved examples
- **PEFT+AFSP:** PEFT with AFSP-selected examples

The main validation result is not that one method wins every metric.

Instead:

- **PEFT** gives the best score on the deliberately held-out style features.
- **PEFT+AFSP** gives the best observed full stylometric distance and the best COMET score among the study systems.
- **PEFT+KNN** gives the highest chrF, BLEU, and primary judge mean, although it is worse than PEFT+AFSP on objective full-register fit.
- **RLSF** improves some adequacy and judge scores, but continued optimization with a strong style-judge reward can push the model toward exaggerated style cues rather than closer target-register fidelity.

That last point became one of the most interesting findings in the project.

---

## Why this problem matters

General-purpose LLMs can translate Persian and Arabic fluently, but fluent is not always faithful.

In literary and scriptural translation, style carries part of the meaning. Vocabulary, cadence, formality, syntax, and recurring register choices all contribute to the reader's experience.

The goal here is therefore not simply:

> "Produce a correct English translation."

It is closer to:

> "Produce a semantically correct English translation that also behaves like the target translator's register."

That makes evaluation harder. BLEU or COMET alone cannot tell the whole story, so this project evaluates both **translation quality** and **style fidelity** from several angles.

---

## Research questions

**RQ1.** How do PEFT, AFSP, and RLSF compare on semantic adequacy, stylistic fidelity, and practical cost when the base model and data are held constant?

**RQ2.** Do retrieved examples and reward-driven updates move the model closer to the target register?

**RQ3.** How sensitive is RLSF to the weight placed on the LLM style judge?

**RQ4.** When COMET, stylometrics, and LLM-as-Judge evaluate the same translations, where do they agree and where do they disagree?

The original hypotheses and proposal are in [`docs/proposal.pdf`](docs/proposal.pdf).

---

## Experimental design

### Base model

All study conditions use **Qwen2.5-7B-Instruct**.

The base model is locked across the comparison so that changes can be attributed to the adaptation method rather than to a different model.

### Data

The corpus contains Persian and mixed Persian/Arabic Bahá'í texts paired with Shoghi Effendi's authorized English translations.

Key choices:

- sentence-level examples for the main experiments
- mixed Persian/Arabic passages are kept intact
- no synthetic training data
- document-aware splitting rather than random-row splitting
- fixed split hashes stored under `data/splits/`

The final test split is not used for model selection or for the results reported here.

### Prompting ladder

The prompting experiments are deliberately staged so each step adds one new mechanism.

| Condition | Exemplars | Selection method |
|---|---|---|
| `zeroshot` | none | style instruction only |
| `random_fewshot` | k examples | seeded random selection |
| `knn_fewshot` | k examples | cosine top-k retrieval |
| `afsp_margin` | k examples | AFSP margin and hub penalization |
| `afsp_full` | k examples | margin plus target-register reranking |

The frozen AFSP configuration is:

```text
k = 8
lambda_style = 0.75
beta = 0.3
sigma = 1.0
```

The source-side retrieval index is built from the training partition. Target-side English is used only after retrieval, when the candidate exemplar is scored for register fit.

More detail is in [`docs/afsp_strategies.md`](docs/afsp_strategies.md).

### PEFT

PEFT uses LoRA adapters while keeping the base model frozen.

Frozen configuration:

```text
rank = 32
alpha = 64
learning rate = 2e-4
epochs = 2
trainable parameters = 80.7M, about 1.06% of the base model
```

The selected PEFT checkpoint also becomes the initialization for RLSF.

### RLSF

The final implementation uses **GRPO**, not PPO.

For a generated translation \(y\), the reward is:

```text
r(y) = w1 * COMET-Kiwi(x, y)
     + w2 * BLEU(y, y*)
     + w3 * Phi_train(y, target_style)
```

Important details:

- adequacy reward: reference-free COMET-Kiwi
- lexical reward: BLEU
- style reward: training-time LLM-as-Judge
- reward judge: `gpt-4o-mini`
- evaluation judges: different models from the reward judge
- initialization: frozen PEFT checkpoint
- KL reference: frozen copy of PEFT
- reward components are normalized within each GRPO group
- reward weights are L2-normalized before use

Three RLSF arms were trained:

| Arm | Role |
|---|---|
| `w3_0.0` | metric-only RL control, no style-judge reward |
| `w3_2.0` | moderate style-judge pressure |
| `w3_6.0` | high style-judge pressure, diagnostic arm |

Selected validation checkpoints:

```text
w3_0.0 -> step 200
w3_2.0 -> step 200
w3_6.0 -> step 100
```

The high-pressure `w3_6.0` arm was added as a diagnostic training condition and is not treated as a new primary model-selection candidate.

The full engineering record, budget rules, drift checks, and preregistration are in:

- [`docs/DEVLOG.md`](docs/DEVLOG.md)
- [`docs/preregistration_rlsf.md`](docs/preregistration_rlsf.md)
- [`docs/budget.md`](docs/budget.md)

### PEFT+AFSP hybrid

The hybrid experiment asks a simple follow-up question:

> If PEFT learns the global domain and register in its weights, can AFSP still help by supplying useful local examples at inference time?

Two conditions were added without retraining the adapter:

| Condition | Model | Prompt examples |
|---|---|---|
| `peft_knn` | frozen PEFT | plain kNN examples |
| `peft_afsp` | frozen PEFT | AFSP-selected examples |

The KNN condition matters because it separates the value of **having examples at all** from the value of **AFSP's selection strategy**.

No PEFT or AFSP hyperparameter was re-tuned for this experiment.

---

## How the systems are evaluated

Style is not treated as one number.

| Axis | Metric | What it tells us |
|---|---|---|
| Semantic adequacy | COMET | how well meaning is preserved |
| Surface overlap | chrF, BLEU | overlap with the authorized reference |
| Perceived register | Phi_A, Phi_B | how two independent LLM judges rate the target style |
| Objective style | stylometric distance | distance from the target register across measured linguistic features |
| Independent style check | held-out distance | style distance on features not used in the RLSF reward |
| Linguistic diagnostics | lexical density, TTR, root TTR, marker rate, sentence statistics | which features are actually moving |

For the two LLM judges:

- **Phi_A:** `claude-haiku-4-5`
- **Phi_B:** `gpt-5.6-terra`

The two judges use the same frozen evaluation rubric, but their absolute scores are not interchangeable. They are reported separately and never averaged.

Lower stylometric distance is better.

---

## Validation results

All rows below use the same 1,323-segment validation split and locked greedy decoding.

Higher is better for COMET, chrF, BLEU, Phi_A, and Phi_B. Lower is better for the two style-distance columns.

| Condition | COMET | chrF | BLEU | Phi_A | Phi_B | Full style dist. | Held-out dist. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | 0.6480 | 36.42 | 10.27 | 2.546 | 3.648 | 0.6518 | 0.5837 |
| Random few-shot | 0.6644 | 37.52 | 11.64 | 2.633 | 3.633 | 0.4736 | 0.4200 |
| kNN few-shot | 0.6839 | 39.82 | 13.99 | 2.748 | 3.679 | 0.4005 | 0.2849 |
| AFSP-margin | 0.6824 | 39.68 | 13.69 | 2.763 | 3.667 | 0.3910 | 0.2936 |
| AFSP-full | 0.6853 | 39.99 | 14.52 | 2.791 | **3.707** | 0.3698 | 0.2990 |
| PEFT | 0.6986 | 41.58 | 16.90 | 2.744 | 3.613 | 0.2886 | **0.1707** |
| RLSF `w3=0`, step 200 | 0.7007 | 41.85 | 17.01 | 2.769 | 3.598 | 0.2960 | 0.1802 |
| RLSF `w3=2`, step 200 | 0.7007 | 42.04 | 17.00 | 2.791 | 3.653 | 0.3279 | 0.2133 |
| RLSF `w3=6`, step 100 | 0.6993 | 42.08 | 17.13 | 2.742 | 3.636 | 0.2990 | 0.1912 |
| PEFT+KNN | 0.7015 | **42.40** | **17.98** | **2.802** | 3.662 | 0.3152 | 0.2306 |
| PEFT+AFSP | **0.7033** | 42.12 | 17.77 | 2.795 | 3.637 | **0.2701** | 0.1908 |

### What I take from this table

There is no single metric winner.

**PEFT remains the strongest on the deliberately held-out style features.** Its held-out distance is 0.1707, compared with 0.1908 for PEFT+AFSP.

**PEFT+AFSP has the best observed full stylometric distance.** Its value of 0.2701 is the lowest point estimate in the study. However, its improvement over PEFT on that metric is not statistically resolved on validation.

**PEFT+AFSP also has the best COMET score among the study systems.** Compared with PEFT, it improves COMET, chrF, and BLEU significantly in paired validation tests.

**PEFT+KNN gets the highest chrF, BLEU, and Phi_A mean.** The two hybrid systems are not clearly separated by either LLM judge.

So the hybrid result is better described as:

> PEFT+AFSP is the strongest overall validation candidate, but it is not an unconditional winner on every definition of style.

---

## The RLSF finding: stronger style pressure can overshoot

The selected RLSF checkpoints alone did not explain what was happening, so I ran a matched checkpoint trajectory over:

```text
steps = 100, 200, 400, 800, 1200
arms  = w3_0.0, w3_2.0, w3_6.0
```

That produced 15 full-validation checkpoint outputs.

The main pattern is visible in the per-doubling slopes:

| Arm | Held-out style distance | Marker-rate z | COMET | chrF | BLEU |
|---|---:|---:|---:|---:|---:|
| `w3_0.0` | +0.0044 | -0.0001 | +0.0016* | +0.29* | +0.25* |
| `w3_2.0` | +0.0399* | +0.0442* | +0.0009* | +0.34* | +0.19* |
| `w3_6.0` | +0.0584* | +0.0656* | -0.0002 | +0.15* | -0.02 |

`*` means the paired 95% interval excludes zero.

The metric-only control stays roughly stable in independent register space.

The two judge-rewarded arms drift farther from the held-out target as training continues, and the high-pressure arm drifts fastest. At the same time, measured adequacy does not collapse.

That suggests the model is not simply "getting worse." Instead, stronger and longer optimization against the style reward appears to encourage more obvious style cues without producing a matching gain in independent register fidelity.

In other words:

> **More stylistic intensity is not automatically more stylistic fidelity.**

This is why the project reports the reward judge, independent judges, and objective style measurements separately.

Trajectory artifacts are in:

- `results/heldout_traj_val.json`
- `results/comet_traj_val.json`
- `docs/figures/`

---

## What the PEFT+AFSP follow-up adds

The hybrid experiment produced another useful distinction.

Compared with PEFT:

- PEFT+AFSP improves COMET, chrF, and BLEU
- full stylometric distance improves numerically
- held-out style distance does not significantly separate
- Phi_A rises slightly
- Phi_B also rises slightly, but does not separate statistically

Compared with PEFT+KNN:

- adequacy is essentially tied
- both LLM judges are essentially tied
- PEFT+AFSP has a clearly better full stylometric point estimate

This suggests that ordinary retrieval already gives the PEFT model useful local context, while AFSP's main added value may be in **which examples it chooses**, especially for objective register fit.

That interpretation is intentionally cautious because these are validation results, not final test results.

---

## External reference baseline

For context, I also score `claude-haiku-4-5` zero-shot on the same validation corpus.

This is **not a condition of the controlled study** because it changes the model family and compute budget.

| Condition | COMET | chrF | BLEU | Phi_A | Phi_B | Full style dist. |
|---|---:|---:|---:|---:|---:|---:|
| Commercial zero-shot | 0.7185 | 45.24 | 18.06 | 3.333 | 3.981 | 0.5557 |

The commercial model is much stronger on adequacy metrics, but much farther from the target-register centroid than PEFT or the hybrid systems.

Its Phi_A score also has a self-judging problem because the generator and primary judge are the same model family. Phi_B confirms that the commercial output is strongly preferred by the second judge too, but the size of the primary-judge advantage should not be read as an unbiased style effect.

---

## Statistical approach

The project uses paired bootstrap comparisons at the segment level, usually with:

```text
10,000 resamples
alpha = 0.05
seed = 42
```

Stylometric rank and distance uncertainty are also bootstrapped by recomputing the condition-level feature vector inside each resample.

A few rules matter when reading the results:

1. **Point estimates are not treated as proof.** If a confidence interval crosses zero, the comparison is reported as unresolved.
2. **Phi_A and Phi_B are never averaged.** Several small judge differences depend on which rater is used.
3. **Multiple comparisons matter.** Near-threshold p-values are not promoted to findings without considering the relevant correction family.
4. **Validation is not final evidence.** PEFT and AFSP were selected using validation, so the sealed test set is needed for the final unbiased comparison.

---

## Repository structure

```text
.
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── data/
│   ├── peft/
│   ├── retrieval/
│   ├── rlsf/
│   ├── eval/
│   └── infer/
├── configs/
├── prompts/
├── outputs/
├── results/
├── docs/
│   ├── proposal.pdf
│   ├── DEVLOG.md
│   ├── preregistration_rlsf.md
│   ├── afsp_strategies.md
│   └── budget.md
└── notebooks/
```

Raw corpus files and large model weights are not committed to Git.

---

## Reproducibility

The project records or freezes:

- random seeds
- split hashes
- prompt hashes
- base model and revision
- decoding parameters
- LoRA configuration
- RLSF reward configuration
- checkpoint selection rules
- judge templates
- adapter hashes for the RLSF trajectory
- per-run outputs and scoring artifacts

The detailed engineering record is in [`docs/DEVLOG.md`](docs/DEVLOG.md).

One reproducibility caveat is important: greedy generation is stable within a session, but a small amount of cross-session output drift has been observed. LLM judge scores are also not byte-reproducible, especially when the provider does not expose deterministic seed control.

---

## Setup

```bash
git clone <repo-url>
cd <repo>

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

COMET is kept in a separate environment:

```bash
python -m venv .venv-comet
source .venv-comet/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-comet.txt
```

Prepare the data:

```bash
python -m src.data.preprocess
python -m src.data.split
```

---

## Running the main conditions

Build the retrieval index and target-register statistics:

```bash
python manage.py build_index --config configs/base_qwen.yaml
python manage.py stylometrics --build-centroid
```

Generate the prompting ladder:

```bash
python manage.py infer --condition zeroshot       --config configs/base_qwen.yaml
python manage.py infer --condition random_fewshot --config configs/base_qwen.yaml
python manage.py infer --condition knn_fewshot    --config configs/base_qwen.yaml
python manage.py infer --condition afsp_margin    --config configs/base_qwen.yaml
python manage.py infer --condition afsp_full      --config configs/base_qwen.yaml
```

Generate PEFT:

```bash
python manage.py infer --condition peft --config configs/peft_qwen.yaml
```

Generate the stacked PEFT retrieval conditions:

```bash
python manage.py infer --condition peft_knn  --config configs/peft_afsp.yaml
python manage.py infer --condition peft_afsp --config configs/peft_afsp.yaml
```

<details>
<summary>RLSF training and selected-checkpoint inference</summary>

```bash
# Free metric-only control
python manage.py rlsf_train --cell w3_0.0 --steps 300 --skip_judge

# Paid style-judge arms
python manage.py rlsf_train --cell w3_2.0 --steps 300 --yes
python manage.py rlsf_train --cell w3_6.0 --steps 300 --yes

# Select checkpoints on the RLSF dev slice
python manage.py rlsf_select --cell w3_0.0
python manage.py rlsf_select --cell w3_2.0
python manage.py rlsf_select --cell w3_6.0

# A trained RLSF adapter is evaluated through the PEFT inference path
python manage.py infer --condition peft \
    --config configs/rlsf_eval_w3_2.0.yaml \
    --out-name rlsf_w3_2.0
```

The full GPU runbooks are under [`notebooks/`](notebooks/).

</details>

---

## Evaluation commands

Example:

```bash
CONDS="zeroshot random_fewshot knn_fewshot afsp_margin afsp_full peft peft_knn peft_afsp rlsf_w3_0.0 rlsf_w3_2.0 rlsf_w3_6.0"

python manage.py eval \
    --conditions $CONDS \
    --split val

python manage.py comet \
    --conditions $CONDS \
    --split val

python manage.py judge \
    --conditions $CONDS \
    --split val \
    --config configs/judge_eval.yaml

python manage.py stylometrics \
    --conditions $CONDS \
    --split val
```

The canonical paired stylometric ladder is:

```bash
python manage.py stylometrics_ci \
    --split val \
    --conditions zeroshot random_fewshot knn_fewshot afsp_margin afsp_full peft \
                 peft_knn peft_afsp rlsf_w3_0.0 rlsf_w3_2.0 rlsf_w3_6.0 commercial_haiku \
    --results_path results/stylometrics_ci_ladder_val.json
```

For full experiment provenance and the exact commands used for each run, see [`docs/DEVLOG.md`](docs/DEVLOG.md).

---

## Key files

- [`docs/proposal.pdf`](docs/proposal.pdf): original thesis proposal and hypotheses
- [`docs/DEVLOG.md`](docs/DEVLOG.md): engineering and decision log
- [`docs/preregistration_rlsf.md`](docs/preregistration_rlsf.md): RLSF preregistration and dated addenda
- [`docs/afsp_strategies.md`](docs/afsp_strategies.md): AFSP retrieval and reranking methodology
- [`docs/budget.md`](docs/budget.md): compute and API budget records and spending rules
- [`results/comet_val.json`](results/comet_val.json): validation COMET scores
- [`results/judge_val.json`](results/judge_val.json): primary LLM-as-Judge scores, Φ_A
- [`results/judge_gpt_val.json`](results/judge_gpt_val.json): second LLM-as-Judge scores, Φ_B
- [`results/stylometrics_ci_ladder_val.json`](results/stylometrics_ci_ladder_val.json): canonical full stylometric comparison with uncertainty estimates
- [`results/heldout_decomp_val.json`](results/heldout_decomp_val.json): RLSF held-out style decomposition
- [`results/heldout_traj_val.json`](results/heldout_traj_val.json): RLSF checkpoint-trajectory analysis
- [`results/heldout_decomp_peft_afsp_val.json`](results/heldout_decomp_peft_afsp_val.json): PEFT+AFSP held-out style analysis
---

## Current takeaway

The project started with the expectation that direct reinforcement learning for style might be the strongest approach.

The validation evidence turned out to be more complicated.

PEFT learns the target corpus well. Retrieval improves adequacy and perceived style. AFSP can improve the objective register fit of a PEFT-plus-retrieval system relative to plain KNN. RLSF can improve some translation and judge metrics, but strong and prolonged optimization against a style judge can also exaggerate the signals that the reward recognizes.

So the central lesson so far is not:

> "One adaptation method is always best."

It is:

> **Different adaptation methods improve different parts of style-aware translation, and a model can look more stylistic to a reward or judge without becoming more faithful to the target register.**

The final sealed-test evaluation will determine how well these validation findings generalize.

---

## Citation

If you use this work, please cite the thesis once the final version is available.

```text
Mehri, P. Style-Aware Neural Machine Translation of Low-Resource Texts
Using Large Language Models and Reinforcement Learning.
Undergraduate thesis, Bahá'í Institute for Higher Education (BIHE).
```
