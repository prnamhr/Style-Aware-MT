# Style-Aware NMT of Low-Resource Texts

Can a language model preserve meaning while also matching the register of a specific translator?

This project studies that question using Persian and mixed Persian/Arabic Bahá'í scripture translated into English. The target register is the formal scriptural style found in Shoghi Effendi's authorized translations.

This is an undergraduate Computer Engineering thesis project at BIHE, supervised by Dr. Fares Hedayati.

> **Status:** Test generation and scoring are complete. Validation results below document model development and selection; final thesis claims are based on the frozen held-out test evaluation in `results/confirmatory_test.json` and the associated test artifacts.

## Overview

The controlled study compares three ways of adapting the same base model, `Qwen2.5-7B-Instruct`.

| Method | What changes | Weight updates? |
|---|---|---|
| **PEFT (LoRA)** | Learns the domain and target register from the parallel corpus | Yes, adapter weights only |
| **AFSP** | Selects examples for the prompt at inference time | No |
| **RLSF (GRPO)** | Continues training from PEFT with a mixed adequacy and style reward | Yes |

Two hybrid conditions test whether retrieval still helps after PEFT:

- `peft_knn`: frozen PEFT with ordinary kNN examples
- `peft_afsp`: frozen PEFT with AFSP-selected examples

A later `sparse_knn` follow-up tests a different retrieval idea: use rare source-side terms to select part of the few-shot context, then fill the remaining prompt slots with ordinary cosine kNN.

The validation/development result is mixed rather than a single winner. RLSF `w3=2` gives the lowest held-out style distance, while `w3=6` gives the lowest full stylometric distance. PEFT+AFSP has the highest COMET score among the main study systems, and PEFT+KNN has the highest chrF and BLEU. Final thesis claims are separated from these development results and are reported from the frozen held-out test evaluation.

The interpretation of the RLSF results changed after a case-sensitivity bug was found in one stylometric feature. The corrected analysis shows the judge-conditioned RLSF arms moving closer to the measured target register through most of training, with possible over-stylization appearing only later in the strongest trajectory.

## Research problem

General-purpose LLMs can produce fluent Persian-to-English translations, but fluency alone is not enough for this corpus. The authorized translations use a recognizable register shaped by vocabulary, syntax, cadence, formality, and recurring scriptural constructions.

The task is therefore not simply to produce acceptable English. It is to preserve semantic content while producing output that behaves more like the target register.

That also changes the evaluation problem. COMET, chrF, and BLEU can measure useful parts of translation quality, but none of them directly measures whether a translation sounds like the target corpus. This project therefore evaluates semantic quality and style separately.

## Research questions

**RQ1.** How do PEFT, AFSP, and RLSF compare on semantic adequacy and stylistic fidelity when the base model and data are held constant?

**RQ2.** Do retrieved examples and reward-driven updates move the model closer to the target register?

**RQ3.** How sensitive is RLSF to the weight assigned to the LLM style judge?

**RQ4.** When COMET, stylometric measures, and LLM-as-Judge score the same translations, where do they agree and where do they disagree?

The original proposal and hypotheses are in [`docs/proposal.pdf`](docs/proposal.pdf).

## Experimental design

### Base model

All controlled study conditions use **Qwen2.5-7B-Instruct**.

Keeping the base model fixed makes the comparison about the adaptation method rather than differences between model families.

### Data

The corpus contains Persian and mixed Persian/Arabic Bahá'í texts paired with Shoghi Effendi's authorized English translations.

The current split is:

```text
train       10,860
validation   1,323
test         1,322
```

The main data choices are:

- sentence-level examples
- mixed Persian/Arabic passages kept intact
- no synthetic training data
- document-aware splitting rather than random row splitting
- fixed split artifacts under `data/splits/`

Validation is used for model development, hyperparameter and checkpoint selection, and diagnostic analysis. The held-out test split is reserved for the final evaluation after those choices are frozen; the thesis headline results should therefore be reported from test rather than validation.

### Prompting ladder

The prompting ladder adds one retrieval mechanism at a time.

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

The source-side retrieval index is built from training data. In `afsp_full`, target-side English is used only after retrieval to score candidate examples for register fit.

More detail is in [`docs/afsp_strategies.md`](docs/afsp_strategies.md).

### PEFT

PEFT uses LoRA adapters while the base model remains frozen.

```text
rank = 32
alpha = 64
learning rate = 2e-4
epochs = 2
trainable parameters = 80.7M, about 1.06% of the base model
```

The selected PEFT checkpoint is also used to initialize RLSF.

### RLSF


For a generated translation `y`, the training reward is:

```text
r(y) = w1 * COMET-Kiwi(x, y)
     + w2 * BLEU(y, y*)
     + w3 * Phi_train(y, target_style)
```

The reward combines:

- reference-free COMET-Kiwi for adequacy
- BLEU for lexical overlap
- `gpt-4o-mini` as the training-time style judge

The evaluation judges are separate from the reward judge. RLSF starts from the frozen PEFT checkpoint and uses a frozen PEFT copy as the KL reference. Reward components are normalized within each GRPO group, and the reward weights are L2-normalized before use.

Three arms were trained:

| Arm | Role |
|---|---|
| `w3_0.0` | metric-only RL control with no style-judge reward |
| `w3_2.0` | moderate style-judge weight |
| `w3_6.0` | high style-judge weight used as a diagnostic arm |

Selected checkpoints:

```text
w3_0.0 -> step 200
w3_2.0 -> step 200
w3_6.0 -> step 100
```

The `w3_6.0` arm is kept as a high-pressure diagnostic rather than a new primary selection candidate.

The training history, budget rules, and preregistration are in:

- [`docs/DEVLOG.md`](docs/DEVLOG.md)
- [`docs/preregistration_rlsf.md`](docs/preregistration_rlsf.md)
- [`docs/preregistration_test.md`](docs/preregistration_test.md)
- [`docs/budget.md`](docs/budget.md)

### PEFT with retrieval

The hybrid experiment tests whether retrieved examples still help after the model has already been adapted through PEFT.

| Condition | Model | Prompt examples |
|---|---|---|
| `peft_knn` | frozen PEFT | ordinary kNN examples |
| `peft_afsp` | frozen PEFT | AFSP-selected examples |

No PEFT or AFSP hyperparameter was retuned for this experiment.

### Sparse-KNN follow-up

`sparse_knn` is a secondary retrieval experiment. It uses the same base model, prompt format, `k = 8`, and greedy decoding as ordinary kNN.

The current run freezes exactly 500 source terms from the training pool. Terms must appear in at least 40 training sentences, then eligible terms are ranked from rarer to less rare by document frequency, with total frequency used to break ties.

For each query, the code checks which of those 500 terms are present, keeps at most the four rarest matches, retrieves one nearest training example for each selected term, and fills the remaining prompt slots with ordinary cosine kNN. The prompt always contains eight examples.

On validation, 563 queries use 4 rare-term examples, 639 use 1 to 3, and 121 use 8 ordinary kNN examples. The mean is 2.646 rare-term examples per prompt.

## Evaluation

Style is not reduced to one score.

| Axis | Metric | Purpose |
|---|---|---|
| Semantic adequacy | COMET | meaning preservation |
| Surface overlap | chrF, BLEU | overlap with the authorized reference |
| Perceived register | `Phi_A`, `Phi_B` | target-style ratings from two independent LLM judges |
| Objective style | full stylometric distance | distance from the training target-register centroid |
| Independent style check | held-out distance | distance on style features not used in the main RLSF reward |
| Diagnostics | lexical density, TTR, root TTR, marker rate, sentence statistics | shows which linguistic features are moving |

The evaluation judges are:

```text
Phi_A = claude-haiku-4-5
Phi_B = gpt-5.6-terra
```

They use the same frozen rubric, but their absolute scores are not treated as interchangeable and are never averaged.

Lower stylometric distance is better.

### Reporting convention

Validation and test results serve different purposes in this project. Validation is used to choose configurations and checkpoints and to inspect development behaviour. Once those choices are frozen, the held-out test split is used for the final performance claims in the thesis. This separation reduces selection bias from choosing a system on the same finite sample used to report its final performance, a problem discussed by [Cawley and Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html).

The validation table is retained in this README because it documents how the final systems were selected and interpreted during development. It should not be presented as the final generalization result.

### Marker-rate correction

One stylometric feature originally counted archaic forms such as `thou`, `thee`, `thy`, and `thine` case-sensitively. That undercounted capitalized reverential forms such as `Thou`, `Thee`, and `Thy` in the authorized references.

The feature was corrected to make the archaic marker class case-insensitive while keeping vocative `O` capital-sensitive.

This correction changes the objective style distances and the interpretation of the RLSF trajectory. It does not change the trained PEFT or RLSF adapters, the generated translations for conditions that do not use the stylometric centroid during retrieval, COMET, chrF, BLEU, or the existing LLM-judge scores.

AFSP is different because `afsp_full` uses the target-register centroid during exemplar reranking. For that reason, `afsp_full_casefix` and `peft_afsp_casefix` were generated as sensitivity-only diagnostics with the same frozen AFSP settings; they are not main study conditions. Those reruns changed many retrieved examples and individual translations, but their aggregate validation results stayed close to the original runs.

The AFSP register direction is no longer hard-coded in the configs. `python manage.py register_direction` derives it from the corrected training-target centroid and writes `results/register_direction.json`, which all seven AFSP/PEFT selection configs now read. The corrected coefficients are `marker_rate = 0.414116`, `lex_density = 0.321723`, `root_ttr = -0.273426`, and `ttr = 0.047505`. Against the former hard-coded vector, the direction has cosine similarity 0.9962, a 4.996 degree angle, no sign flips, and a maximum absolute normalized-weight-share shift of 0.0326. Under the comparison thresholds fixed in the derivation script, this is **near-identical**, not materially or very different. An offline re-score of the 19 already-generated AFSP sweep outputs changed `register_fit` by at most 0.00939 and kept the frozen recommendation at `k = 8`, `lambda_style = 0.75`. The direction correction therefore does not trigger a new `afsp_full` generation run. This conclusion is limited to the direction component; the marker-rate/centroid correction itself still changed retrieval choices in the earlier case-fix sensitivity runs.

## Final held-out test evidence

The final thesis claims use the 1,322-segment test split after model, retrieval, checkpoint, and decoding choices were frozen on development data. The pre-registered confirmatory family contains five contrasts evaluated on COMET and full stylometric distance.

Three predictions replicate under the pre-registered decision rule:

- **PEFT improves COMET over kNN few-shot:** +0.02909, 95% CI [0.02439, 0.03389].
- **RLSF `w3=2` improves full stylometric distance over PEFT:** -0.02836, 95% CI [-0.04140, -0.01563]. Lower distance is better.
- **PEFT+AFSP increases full stylometric distance relative to PEFT:** +0.08924, 95% CI [0.05730, 0.12284], reproducing the validation-side trade-off in which retrieval on top of PEFT moves the output farther from the target centroid.

The other seven pre-registered tests do not meet the corrected confirmatory criterion. They are reported as not replicated or not detected rather than as evidence of no effect. Full test statistics, correction thresholds, output hashes, and verdicts are in [`results/confirmatory_test.json`](results/confirmatory_test.json). Validation results remain useful for explaining model selection and development behaviour, but they are not substituted for this held-out test evidence in the thesis.

## Validation results

All main rows below use the same 1,323-segment validation split and locked greedy decoding. As a table, these are development results; the specific RLSF predictions fixed before GRPO are classified separately in `results/evidence_class.json`.

Higher is better for COMET, chrF, BLEU, `Phi_A`, and `Phi_B`. Lower is better for the two stylometric distances.

| Condition | COMET | chrF | BLEU | Phi_A | Phi_B | Full style dist. | Held-out dist. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | 0.6480 | 36.42 | 10.27 | 2.546 | 3.648 | 0.4481 | 0.3417 |
| Random few-shot | 0.6644 | 37.52 | 11.64 | 2.633 | 3.633 | 0.3162 | 0.2282 |
| kNN few-shot | 0.6839 | 39.82 | 13.99 | 2.748 | 3.679 | 0.3659 | 0.2337 |
| Sparse-KNN | 0.6846 | 40.08 | 14.31 | — | — | 0.3750 | 0.2404 |
| AFSP-margin | 0.6824 | 39.68 | 13.69 | 2.763 | 3.667 | 0.3394 | 0.2203 |
| AFSP-full | 0.6853 | 39.99 | 14.52 | 2.791 | **3.707** | 0.3032 | 0.2111 |
| PEFT | 0.6986 | 41.58 | 16.90 | 2.744 | 3.613 | 0.2890 | 0.1713 |
| RLSF `w3=0`, step 200 | 0.7007 | 41.85 | 17.01 | 2.769 | 3.598 | 0.2857 | 0.1626 |
| RLSF `w3=2`, step 200 | 0.7007 | 42.04 | 17.00 | 2.791 | 3.653 | 0.2793 | **0.1264** |
| RLSF `w3=6`, step 100 | 0.6993 | 42.08 | 17.13 | 2.742 | 3.636 | **0.2704** | 0.1423 |
| PEFT+KNN | 0.7015 | **42.40** | **17.98** | 2.802 | 3.662 | 0.3589 | 0.2874 |
| PEFT+AFSP | **0.7033** | 42.12 | 17.77 | 2.795 | 3.637 | 0.3244 | 0.2620 |

There is no single metric winner.

RLSF `w3=2` has the lowest corrected held-out distance, at 0.1264, compared with 0.1713 for PEFT. It also has higher means than PEFT under both evaluation judges.

RLSF `w3=6` has the lowest corrected full stylometric distance, at 0.2704. Because this is the high-pressure diagnostic arm, I treat it mainly as evidence about how strong style optimization changes the model rather than as a replacement for the selected moderate arm.

PEFT+AFSP has the highest COMET score among the main study conditions. PEFT+KNN has the highest chrF and BLEU. AFSP-full remains highest on `Phi_B` at 3.707. For the current Sparse-KNN run, COMET is 0.6846, chrF is 40.08, and BLEU is 14.31; none of its paired adequacy differences from ordinary kNN is statistically resolved. It has no rater score. Its full stylometric distance is 0.3750, compared with 0.3659 for ordinary kNN. The two PEFT retrieval hybrids are also close under both LLM judges.

The main result is therefore a trade-off, the method that looks best under semantic and reference-overlap metrics is not necessarily the one closest to the measured register.

## RLSF trajectory

The selected checkpoints do not show what happens later in training, so the three RLSF arms were also evaluated at matched checkpoints:

```text
steps = 100, 200, 400, 800, 1200
arms  = w3_0.0, w3_2.0, w3_6.0
```

This gives 15 full-validation checkpoint outputs.

The corrected per-doubling slopes are:

| Arm | Held-out style distance | Marker-rate z | COMET | chrF | BLEU |
|---|---:|---:|---:|---:|---:|
| `w3_0.0` | +0.0058 | -0.0010 | +0.0016* | +0.29* | +0.25* |
| `w3_2.0` | -0.0205* | +0.0328* | +0.0009* | +0.34* | +0.19* |
| `w3_6.0` | -0.0259* | +0.0514* | -0.0002 | +0.15* | -0.02 |

`*` means the paired 95% interval excludes zero.

The metric-only arm remains roughly stable in held-out register space. The two judge-conditioned arms move closer to the corrected target as training continues.

The high-weight arm also shows a limit. Its held-out distance falls from 0.1534 at step 100 to 0.0405 at step 800, where marker z is almost exactly on target at -0.004. By step 1200, marker z has crossed above the target to +0.076 and held-out distance rises to 0.0834.

I therefore read the trajectory as useful register adaptation followed by possible late over-stylization under prolonged high judge pressure. The trajectory analysis is exploratory and is not used to choose new official checkpoints.

Artifacts:

- `results/heldout_traj_val.json`
- `results/comet_traj_val.json`
- `docs/figures/`

## PEFT+AFSP follow-up

Compared with PEFT, PEFT+AFSP improves COMET, chrF, and BLEU, but its corrected full and held-out stylometric distances are higher than PEFT's. Its judge means are slightly higher, although the differences are not resolved consistently across raters.

Compared with PEFT+KNN, the two hybrid systems are close on adequacy and perceived register. PEFT+AFSP has the lower full stylometric distance, 0.3244 versus 0.3589.

The `afsp_full_casefix` and `peft_afsp_casefix` runs are sensitivity-only robustness checks, not main study conditions. Correcting the centroid changed many retrieved examples and many individual translations, but the aggregate validation profile stayed close to the original. That suggests AFSP is sensitive at the example-selection level without producing an equally large shift in corpus-level performance.


## External reference

For context, `claude-haiku-4-5` is also evaluated zero-shot on the same validation corpus. `gpt56_sparse_knn` is retained as a generator-family diagnostic using the frozen Sparse-KNN retrieval configuration. Neither is part of the controlled study because each changes the model family and compute budget.

| Condition | COMET | chrF | BLEU | Phi_A | Phi_B | Full style dist. |
|---|---:|---:|---:|---:|---:|---:|
| Commercial zero-shot | 0.7185 | 45.24 | 18.06 | 3.333 | 3.981 | 0.4873 |
| GPT-5.6 + Sparse-KNN | 0.7480 | 50.29 | 24.16 | 3.625 | — | 0.2455 |

The GPT-5.6 Sparse-KNN row is descriptive only. Its matched `gpt56_knn_fewshot` generation has not been fully scored, so this row cannot isolate the effect of sparse retrieval from the change in generator.

The commercial zero-shot model's `Phi_A` score has an additional caveat because the generator and the primary judge use the same model family. `Phi_B` gives a cross-family comparison for that row, but the size of the `Phi_A` advantage should not be read as an independent style effect.

## Statistical reporting

Most paired comparisons use segment-level bootstrap resampling with:

```text
10,000 resamples
alpha = 0.05
seed = 42
```

Full stylometric rank and distance uncertainty are also bootstrapped by recomputing the condition-level feature vector inside each resample.

A few reporting rules are kept throughout the project:

- a better point estimate is not treated as evidence of separation when its confidence interval crosses zero
- `Phi_A` and `Phi_B` are reported separately
- near-threshold p-values are interpreted with the relevant multiple-comparison family in mind
- validation results are not treated as final test evidence

## Reproducibility

The repository records the main pieces needed to reconstruct each experiment:

- random seeds and split hashes
- prompt hashes
- model revision and decoding settings
- LoRA configuration
- RLSF reward configuration and checkpoint choices
- judge templates
- adapter hashes for trajectory runs
- per-run outputs and scoring artifacts

The engineering history is kept in [`docs/DEVLOG.md`](docs/DEVLOG.md).

Greedy generation is stable within a session, but a small amount of cross-session output drift has been observed. Paid LLM judges are also not byte-reproducible when the provider does not expose deterministic seed control.

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
│   ├── preregistration_test.md
│   ├── afsp_strategies.md
│   └── budget.md
└── notebooks/
```

Raw corpus files and large model weights are not committed to Git.

## Setup

Create the main environment:

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

## Running the main conditions

Build the retrieval index and target-register statistics:

```bash
python manage.py build_index --config configs/base_qwen.yaml
python manage.py stylometrics --build-centroid
```

Generate the prompting ladder:

```bash
python manage.py infer --condition zeroshot        --config configs/base_qwen.yaml
python manage.py infer --condition random_fewshot --config configs/base_qwen.yaml
python manage.py infer --condition knn_fewshot    --config configs/base_qwen.yaml
python manage.py infer --condition afsp_margin    --config configs/base_qwen.yaml
python manage.py infer --condition afsp_full      --config configs/base_qwen.yaml
```

Generate Sparse-KNN:

```bash
python manage.py infer --condition sparse_knn --config configs/sparse_knn.yaml
```

Generate PEFT and the PEFT retrieval conditions:

```bash
python manage.py infer --condition peft      --config configs/peft_qwen.yaml
python manage.py infer --condition peft_knn  --config configs/peft_afsp.yaml
python manage.py infer --condition peft_afsp --config configs/peft_afsp.yaml
```

<details>
<summary>RLSF training and selected-checkpoint inference</summary>

```bash
# Metric-only control
python manage.py rlsf_train --cell w3_0.0 --steps 300 --skip_judge

# Style-judge arms
python manage.py rlsf_train --cell w3_2.0 --steps 300 --yes
python manage.py rlsf_train --cell w3_6.0 --steps 300 --yes

# Select checkpoints on the RLSF dev slice
python manage.py rlsf_select --cell w3_0.0
python manage.py rlsf_select --cell w3_2.0
python manage.py rlsf_select --cell w3_6.0

# Evaluate a selected RLSF adapter through the PEFT inference path
python manage.py infer --condition peft \
    --config configs/rlsf_eval_w3_2.0.yaml \
    --out-name rlsf_w3_2.0
```

The GPU runbooks are under [`notebooks/`](notebooks/).

</details>

## Evaluation commands

Example validation pass:

```bash
CONDS="zeroshot random_fewshot knn_fewshot sparse_knn afsp_margin afsp_full peft peft_knn peft_afsp rlsf_w3_0.0 rlsf_w3_2.0 rlsf_w3_6.0"

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

For exact experiment provenance, use [`docs/DEVLOG.md`](docs/DEVLOG.md).

## Key files

- [`docs/proposal.pdf`](docs/proposal.pdf): original thesis proposal and hypotheses
- [`docs/DEVLOG.md`](docs/DEVLOG.md): engineering and decision log
- [`docs/preregistration_rlsf.md`](docs/preregistration_rlsf.md): RLSF preregistration and dated addenda
- [`docs/preregistration_test.md`](docs/preregistration_test.md): the test-split condition set, confirmatory family, and correction
- [`docs/afsp_strategies.md`](docs/afsp_strategies.md): AFSP retrieval and reranking method
- [`docs/budget.md`](docs/budget.md): compute and API budget records
- [`results/comet_val.json`](results/comet_val.json): validation COMET scores
- [`results/judge_val.json`](results/judge_val.json): primary LLM-as-Judge scores, `Phi_A`
- [`results/judge_gpt_val.json`](results/judge_gpt_val.json): second LLM-as-Judge scores, `Phi_B`
- [`results/stylometrics_ci_ladder_val.json`](results/stylometrics_ci_ladder_val.json): corrected main stylometric comparison
- [`results/heldout_decomp_val.json`](results/heldout_decomp_val.json): selected-RLSF held-out style analysis
- [`results/heldout_traj_val.json`](results/heldout_traj_val.json): RLSF checkpoint trajectory
- [`results/stylometrics_ci_casefix_val.json`](results/stylometrics_ci_casefix_val.json): AFSP case-fix sensitivity analysis
- [`results/stylometrics_ci_sparse_knn_val.json`](results/stylometrics_ci_sparse_knn_val.json): Sparse-KNN vs kNN stylometric comparison
- [`results/judge_agreement_gpt_sparse_knn_val.json`](results/judge_agreement_gpt_sparse_knn_val.json): two-rater Sparse-KNN judge comparison, scored against the superseded `654e09e` generation and retained for provenance only
- [`results/sparse_selection_val.json`](results/sparse_selection_val.json): Sparse-KNN routing and selection diagnostics


## Citation

If you use this work, please cite the thesis once the final version is available.

```text
Mehri, P. Style-Aware Neural Machine Translation of Low-Resource Texts
Using Large Language Models and Reinforcement Learning.
Undergraduate thesis, Bahá'í Institute for Higher Education (BIHE).
```
