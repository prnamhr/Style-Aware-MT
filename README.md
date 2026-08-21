# Style-Aware NMT of Low-Resource Texts

Can a language model preserve both the meaning of a translation and the register of a specific translator?

This project studies that question using Persian and mixed Persian/Arabic Bahá'í scripture translated into English. The target is the formal scriptural register found in Shoghi Effendi's authorized translations.

This is an undergraduate Computer Engineering thesis project at BIHE, supervised by Dr. Fares Hedayati.

> **Status:** The main validation experiments and the marker-case correction audit are complete. The final test split is still sealed, so every score reported here is a validation result.

## At a glance

The controlled comparison uses the same open-source base model, `Qwen2.5-7B-Instruct`, with three adaptation strategies.

| Method | What changes | Weight updates? |
|---|---|---|
| **PEFT (LoRA)** | Learns from the parallel corpus through a small trainable adapter | Yes, adapter weights only |
| **AFSP** | Retrieves source-target examples and adds them to the prompt at inference time | No |
| **RLSF (GRPO)** | Starts from PEFT and continues training with a reward combining adequacy, lexical overlap, and an LLM style judge | Yes |

Two hybrid conditions test PEFT with retrieval:

- **PEFT+KNN:** frozen PEFT with ordinary nearest-neighbour examples
- **PEFT+AFSP:** frozen PEFT with AFSP-selected examples

The corrected validation results do not produce one winner on every metric. RLSF has the best objective register scores at the selected checkpoints, PEFT+KNN leads chrF, BLEU, and the primary evaluation judge, and the original PEFT+AFSP run has the highest COMET score. A later AFSP case-fix rerun changes many retrieved examples and individual translations but leaves aggregate performance broadly similar.

One evaluation bug materially changed the interpretation of the style results. The original `marker_rate` implementation counted archaic pronouns case-sensitively, which undercounted reverential capitalization in the training references. It was corrected before the test split was opened. The tables below use the corrected stylometric instrument.

## Research questions

**RQ1.** How do PEFT, AFSP, and RLSF compare on semantic adequacy and stylistic fidelity when the base model and data are held constant?

**RQ2.** Do retrieved examples and reward-driven updates move the model closer to the target register?

**RQ3.** How sensitive is RLSF to the weight placed on the LLM style judge?

**RQ4.** When COMET, stylometric measures, and LLM-as-Judge score the same translations, where do they agree and where do they disagree?

The original proposal and hypotheses are in [`docs/proposal.pdf`](docs/proposal.pdf).

## Experimental design

### Base model

All controlled study conditions use **Qwen2.5-7B-Instruct**. Keeping the base model fixed makes the comparison about the adaptation method rather than a change in model family.

### Data

The corpus contains Persian and mixed Persian/Arabic Bahá'í texts paired with authorized English translations by Shoghi Effendi.

The main experiments use sentence-level examples. Mixed Persian/Arabic passages are kept intact, no synthetic training data is added, and the train, validation, and test partitions are split by work rather than by random row. Cross-boundary duplicate checks and split hashes are stored with the project artifacts.

Current split sizes are:

```text
train       10,860
validation   1,323
test         1,322
```

The final test split is not used for the results in this README.

### Prompting ladder

The prompting experiments add one mechanism at a time.

| Condition | Exemplars | Selection method |
|---|---|---|
| `zeroshot` | none | style instruction only |
| `random_fewshot` | k examples | seeded random selection |
| `knn_fewshot` | k examples | cosine top-k retrieval |
| `afsp_margin` | k examples | margin and hub penalization |
| `afsp_full` | k examples | margin plus target-register reranking |

The frozen AFSP operating point is:

```text
k = 8
lambda_style = 0.75
beta = 0.3
sigma = 1.0
```

The retrieval index is built from the training source texts. For `afsp_full`, the target side of a retrieved candidate is used only during reranking to estimate its fit to the target register.

More detail is in [`docs/afsp_strategies.md`](docs/afsp_strategies.md).

### PEFT

PEFT uses LoRA while the base model remains frozen.

```text
rank = 32
alpha = 64
learning rate = 2e-4
epochs = 2
selected checkpoint = models/peft_lora_r32_lr2e-4/checkpoint-1358
trainable parameters = about 80.7M, or 1.06% of the base model
```

The selected PEFT adapter is also the initialization for RLSF.

### RLSF

The proposal originally planned PPO. The implemented experiment uses **GRPO**.

For a generated translation `y`, the training reward is:

```text
r(y) = w1 * COMET-Kiwi(x, y)
     + w2 * BLEU(y, y*)
     + w3 * Phi_train(y, target_style)
```

The adequacy component uses reference-free COMET-Kiwi, the lexical component uses BLEU, and the style component uses `gpt-4o-mini` with a frozen training-time rubric. The evaluation judges are separate from the reward judge.

Reward components are standardized within each GRPO group, and the weight vector is L2-normalized before use. The KL reference is a frozen copy of the PEFT initialization.

Three arms were trained:

| Arm | Role |
|---|---|
| `w3_0.0` | metric-only RL control, no paid style-judge reward |
| `w3_2.0` | moderate style-judge weight |
| `w3_6.0` | high style-judge weight used as a diagnostic arm |

The selected checkpoints remain:

```text
w3_0.0 -> step 200
w3_2.0 -> step 200
w3_6.0 -> step 100
```

The high-weight arm is a diagnostic training condition rather than an extra model-selection candidate.

The engineering history and the RLSF declarations are in:

- [`docs/DEVLOG.md`](docs/DEVLOG.md)
- [`docs/preregistration_rlsf.md`](docs/preregistration_rlsf.md)
- [`docs/budget.md`](docs/budget.md)

### PEFT with retrieval

The hybrid experiment asks whether retrieved examples still help after the model has already learned the corpus through PEFT.

| Condition | Model | Prompt examples |
|---|---|---|
| `peft_knn` | frozen PEFT | plain kNN examples |
| `peft_afsp` | frozen PEFT | AFSP-selected examples |

The KNN control separates the effect of adding examples from the effect of AFSP's reranking strategy. PEFT and AFSP hyperparameters were not tuned again for the hybrid experiment.

## Evaluation

The project separates translation quality from register fidelity rather than reducing both to one score.

| Measure | What it is used for |
|---|---|
| COMET | semantic adequacy |
| chrF, BLEU | reference overlap |
| `Phi_A`, `Phi_B` | perceived register under two independent evaluation judges |
| full stylometric distance | corpus-level distance from the training target-register centroid |
| held-out stylometric distance | distance on `ttr`, `root_ttr`, and `marker_rate`, none of which is part of the main RLSF reward |
| feature diagnostics | lexical density, TTR, root TTR, sentence statistics, and marker rate |

The evaluation judges are:

```text
Phi_A = claude-haiku-4-5
Phi_B = gpt-5.6-terra
```

Their absolute scores are reported separately and are never averaged.

Lower stylometric distance is better. Higher COMET, chrF, BLEU, `Phi_A`, and `Phi_B` are better.

### The corrected marker feature

The original marker expression matched forms such as `thou`, `thee`, `thy`, and `thine` case-sensitively. That was a poor fit for the corpus because the authorized translations often capitalize reverential forms such as `Thou`, `Thee`, and `Thy`.

The corrected implementation treats the archaic pronoun and verb class case-insensitively while keeping vocative `O` capital-sensitive:

```python
_PRONOUNS = r"thou|thee|thy|thine|art|hast|hath|dost|doth|shalt|wilt|unto|ye"
_MARKERS = re.compile(rf"(?i:\b({_PRONOUNS})\b)|\bO\b")
```

On the 10,860 training references, 52.1% of all markers were missed by the previous implementation. Correcting the count changed the target marker centroid from `0.032684` to `0.057349`.

The correction changes objective register scores and their interpretation. It does not change COMET, chrF, BLEU, the existing LLM-judge scores, the PEFT adapter, or the trained RLSF adapters.

## Validation results

The table below reports the original controlled generations with the **corrected stylometric scoring**. All conditions use the same 1,323-segment validation split.

| Condition | COMET ↑ | chrF ↑ | BLEU ↑ | Phi_A ↑ | Phi_B ↑ | Full style dist. ↓ | Held-out dist. ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | 0.6480 | 36.42 | 10.27 | 2.546 | 3.648 | 0.4481 | 0.3417 |
| Random few-shot | 0.6644 | 37.52 | 11.64 | 2.633 | 3.633 | 0.3162 | 0.2282 |
| kNN few-shot | 0.6839 | 39.82 | 13.99 | 2.748 | 3.679 | 0.3659 | 0.2337 |
| AFSP-margin | 0.6824 | 39.68 | 13.69 | 2.763 | 3.667 | 0.3394 | 0.2203 |
| AFSP-full | 0.6853 | 39.99 | 14.52 | 2.791 | **3.707** | 0.3032 | 0.2111 |
| PEFT | 0.6986 | 41.58 | 16.90 | 2.744 | 3.613 | 0.2890 | 0.1713 |
| RLSF `w3=0`, step 200 | 0.7007 | 41.85 | 17.01 | 2.769 | 3.598 | 0.2857 | 0.1626 |
| RLSF `w3=2`, step 200 | 0.7007 | 42.04 | 17.00 | 2.791 | 3.653 | 0.2793 | **0.1264** |
| RLSF `w3=6`, step 100 | 0.6993 | 42.08 | 17.13 | 2.742 | 3.636 | **0.2704** | 0.1423 |
| PEFT+KNN | 0.7015 | **42.40** | **17.98** | **2.802** | 3.662 | 0.3589 | 0.2874 |
| PEFT+AFSP | **0.7033** | 42.12 | 17.77 | 2.795 | 3.637 | 0.3244 | 0.2620 |

There is no single metric winner.

At the selected checkpoints, `w3_6.0` has the lowest full stylometric point estimate, while `w3_2.0` has the lowest held-out distance. Relative to PEFT, the held-out distance is significantly lower for `w3_2.0` (`Δ = -0.0432`, `p < .001`) and `w3_6.0` (`Δ = -0.0281`, `p = .0002`). The metric-only arm does not separate from PEFT on that measure.

`w3_2.0` is also the only selected RLSF arm that improves both evaluation judges relative to PEFT with paired intervals excluding zero. Its `Phi_A` gain is about `+0.047` (`p = .0018`) and its `Phi_B` gain is about `+0.039` (`p = .024`).

PEFT+KNN has the highest chrF, BLEU, and `Phi_A` mean among the controlled validation conditions. The original PEFT+AFSP generation has the highest COMET score, but it was generated before the marker correction and therefore used the earlier centroid during AFSP reranking. The sensitivity experiment below checks how much that matters.

## AFSP case-fix sensitivity

AFSP is the one adaptation method for which the marker bug affected generation, not only evaluation. `afsp_full` and `peft_afsp` used the target-register centroid when reranking retrieved examples.

Both conditions were therefore regenerated after the marker correction using the same frozen operating point:

```text
k = 8
lambda_style = 0.75
beta = 0.3
sigma = 1.0
```

No new sweep or tuning was performed.

| Condition | COMET ↑ | chrF ↑ | BLEU ↑ | Full style dist. ↓ | Held-out dist. ↓ |
|---|---:|---:|---:|---:|---:|
| AFSP-full, original generation | 0.6853 | 39.99 | 14.52 | 0.3032 | 0.2111 |
| AFSP-full, corrected rerank | 0.6836 | 39.88 | 14.17 | 0.2961 | 0.1979 |
| PEFT+AFSP, original generation | 0.7033 | 42.12 | 17.77 | 0.3244 | 0.2620 |
| PEFT+AFSP, corrected rerank | 0.7008 | 42.17 | 17.85 | 0.3290 | 0.2497 |

The corrected centroid changed exemplar selection substantially. For 1,323 validation segments, 967 AFSP selections changed their exemplar set, 311 changed only the order, and 45 stayed identical. The final translation changed on 1,132 AFSP-full segments and 953 PEFT+AFSP segments.

Even with those segment-level changes, the aggregate style differences between the original and corrected generations are unresolved. AFSP-full changes by about `-0.0064` in full stylometric distance (`p = .694`), and PEFT+AFSP changes by about `+0.0042` (`p = .730`). This sensitivity run therefore suggests that the reranking bug had a large effect on which examples and outputs were produced, but a much smaller effect on the aggregate validation profile.

The corrected PEFT+AFSP run has slightly lower COMET than the original (`0.7008` versus `0.7033`) while chrF and BLEU remain close. The corrected AFSP sensitivity conditions have not been rescored by `Phi_A` and `Phi_B`, so no judge values are inferred for them here.

Artifacts for this pass include:

```text
outputs/afsp_full_casefix_val.jsonl
outputs/peft_afsp_casefix_val.jsonl
outputs/afsp_casefix_manifest.json
results/stylometrics_ci_casefix_val.json
results/heldout_decomp_afsp_casefix_val.json
results/heldout_decomp_peft_afsp_casefix_val.json
results/bootstrap_comet_casefix_val.json
```

## RLSF trajectory after the correction

The selected checkpoints are the primary RLSF conditions. A later post-hoc analysis evaluates saved checkpoints at steps 100, 200, 400, 800, and 1200 for each arm. It is used to inspect training behavior, not to choose a new official checkpoint.

The corrected per-doubling slopes are:

| Arm | Held-out distance ↓ | Marker-rate z | COMET | chrF | BLEU |
|---|---:|---:|---:|---:|---:|
| `w3_0.0` | +0.0058 | -0.0010 | +0.0016* | +0.29* | +0.25* |
| `w3_2.0` | **-0.0205*** | +0.0328* | +0.0009* | +0.34* | +0.19* |
| `w3_6.0` | **-0.0259*** | +0.0514* | -0.0002 | +0.15* | -0.02 |

`*` means the paired 95% interval excludes zero.

The corrected interpretation is different from the original one. PEFT and the early RLSF checkpoints start below the training centroid on `marker_rate`. As judge weight and optimization continue, marker use rises and the two judge-conditioned arms move closer to the held-out target distribution rather than farther away from it.

The high-weight arm also shows why the trajectory should not be reduced to "more is always better." For `w3_6.0`, held-out distance falls from `0.1534` at step 100 to `0.0405` at step 800, where marker z is almost exactly on target at `-0.004`. At step 1200, marker z crosses above the target to `+0.076` and held-out distance rises to `0.0834`.

This is consistent with an under-target policy moving toward the measured register and then beginning to overshoot under prolonged high judge pressure. Because the trajectory analysis was written after the checkpoint results were visible, it is reported as exploratory rather than as a preregistered test.

Trajectory artifacts are in:

```text
results/heldout_traj_val.json
results/comet_traj_val.json
docs/figures/
```

## External reference baseline

`claude-haiku-4-5` is also evaluated zero-shot on the same validation split. It is an external reference rather than a controlled study condition because it changes the model family.

| Condition | COMET ↑ | chrF ↑ | BLEU ↑ | Phi_A ↑ | Phi_B ↑ | Full style dist. ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Commercial zero-shot | 0.7185 | 45.24 | 18.06 | 3.333 | 3.981 | 0.4873 |

The commercial model is stronger on the adequacy and judge metrics, but farther from the training target-register centroid than the adapted open-source conditions.

Its `Phi_A` value also has a self-judging caveat: the generator and the primary judge use the same model family. `Phi_B` provides a cross-family comparison, but the size of the `Phi_A` advantage should not be treated as an independent style effect.

## Statistical reporting

Paired comparisons over segment-level adequacy and held-out quantities usually use:

```text
10,000 bootstrap resamples
alpha = 0.05
seed = 42
```

The full stylometric ranking uses 2,000 paired resamples because each draw recomputes the condition-level feature vector and its distance from the centroid.

Point estimates are not treated as evidence of separation when the paired interval crosses zero. `Phi_A` and `Phi_B` remain separate because their absolute scales and some small pairwise effects differ. Multiple-comparison status is reported for comparison families where several endpoints are tested together.

Validation is still a model-development split. The final test pass is needed before the validation findings can be treated as the final system comparison.

## Interpreting the stylometric distances

The stylometric centroid is estimated from the training works. Because the split is work-level, the validation references do not have exactly the same corpus composition as the training references.

Under the corrected instrument, the authorized validation references themselves have a full distance of about `0.2156` and a held-out distance of about `0.2109` from the training centroid.

This means a smaller distance should be read as **closer to the training corpus centroid on the measured features**, not as "more Shoghi Effendi-like than the authorized translation." The metric is a corpus-level proxy for register, and work composition affects it.

The reward-side distance also needs a narrow interpretation. Its three features are lexical density, sentence-length mean, and sentence-length variance, but in some comparisons lexical density contributes most of the standardized distance. It should not be read as three equally informative independent style signals.

## Reproducibility

The repository records or freezes the main sources of experimental provenance:

- random seeds and split hashes
- prompt hashes
- base-model revision
- decoding settings
- LoRA configuration and selected checkpoint
- RLSF reward configuration and selected checkpoints
- judge templates
- adapter hashes for the RLSF trajectory
- centroid fingerprints
- per-run outputs, manifests, and scoring artifacts

Greedy generation is stable within a session, but some cross-session output drift has been observed. Paid LLM judges are also not byte-reproducible when the provider does not expose deterministic behavior.

The detailed engineering record is in [`docs/DEVLOG.md`](docs/DEVLOG.md).

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

Build the retrieval index and target-register centroids:

```bash
python manage.py build_index --config configs/base_qwen.yaml
python manage.py stylometrics --build-centroid
python manage.py stylometrics --build-split-centroid
```

## Running the main conditions

Generate the prompting ladder:

```bash
python manage.py infer --condition zeroshot        --config configs/base_qwen.yaml
python manage.py infer --condition random_fewshot --config configs/base_qwen.yaml
python manage.py infer --condition knn_fewshot    --config configs/base_qwen.yaml
python manage.py infer --condition afsp_margin    --config configs/base_qwen.yaml
python manage.py infer --condition afsp_full      --config configs/base_qwen.yaml
```

Generate PEFT and the hybrid retrieval conditions:

```bash
python manage.py infer --condition peft      --config configs/peft_qwen.yaml
python manage.py infer --condition peft_knn  --config configs/peft_afsp.yaml
python manage.py infer --condition peft_afsp --config configs/peft_afsp.yaml
```

RLSF training:

```bash
python manage.py rlsf_train --cell w3_0.0 --steps 300 --skip_judge
python manage.py rlsf_train --cell w3_2.0 --steps 300 --yes
python manage.py rlsf_train --cell w3_6.0 --steps 300 --yes
```

Checkpoint selection on the RLSF dev slice:

```bash
python manage.py rlsf_select --cell w3_0.0
python manage.py rlsf_select --cell w3_2.0
python manage.py rlsf_select --cell w3_6.0
```

A selected RLSF adapter is evaluated through the PEFT inference path, for example:

```bash
python manage.py infer --condition peft \
    --config configs/rlsf_eval_w3_2.0.yaml \
    --out-name rlsf_w3_2.0
```

The GPU runbooks are under [`notebooks/`](notebooks/).

### Reproducing the AFSP correction sensitivity

The corrected pass keeps the original conditions and configs but writes separate output names:

```bash
python manage.py infer \
    --condition afsp_full \
    --config configs/base_qwen.yaml \
    --out-name afsp_full_casefix

python manage.py infer \
    --condition peft_afsp \
    --config configs/peft_afsp.yaml \
    --out-name peft_afsp_casefix
```

The full runbook and provenance checks are in [`notebooks/afsp_casefix_gpu.ipynb`](notebooks/afsp_casefix_gpu.ipynb).

## Evaluation commands

Example validation pass:

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

The canonical full-register ladder is:

```bash
python manage.py stylometrics_ci \
    --split val \
    --conditions zeroshot random_fewshot knn_fewshot afsp_margin afsp_full peft \
                 peft_knn peft_afsp rlsf_w3_0.0 rlsf_w3_2.0 rlsf_w3_6.0 commercial_haiku \
    --results_path results/stylometrics_ci_ladder_val.json
```

For exact run provenance, use [`docs/DEVLOG.md`](docs/DEVLOG.md) rather than reconstructing commands from the README.

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

## Key files

- [`docs/proposal.pdf`](docs/proposal.pdf): original thesis proposal and hypotheses
- [`docs/DEVLOG.md`](docs/DEVLOG.md): engineering and decision log
- [`docs/preregistration_rlsf.md`](docs/preregistration_rlsf.md): RLSF preregistration and dated addenda
- [`docs/afsp_strategies.md`](docs/afsp_strategies.md): AFSP retrieval and reranking methodology
- [`docs/budget.md`](docs/budget.md): compute and API budget records
- [`results/comet_val.json`](results/comet_val.json): validation COMET scores, including the case-fix reruns
- [`results/judge_val.json`](results/judge_val.json): primary LLM-as-Judge scores, `Phi_A`
- [`results/judge_gpt_val.json`](results/judge_gpt_val.json): second LLM-as-Judge scores, `Phi_B`
- [`results/stylometrics_ci_ladder_val.json`](results/stylometrics_ci_ladder_val.json): corrected canonical stylometric comparison
- [`results/heldout_decomp_val.json`](results/heldout_decomp_val.json): corrected selected-RLSF held-out decomposition
- [`results/heldout_traj_val.json`](results/heldout_traj_val.json): corrected RLSF checkpoint trajectory
- [`results/stylometrics_ci_casefix_val.json`](results/stylometrics_ci_casefix_val.json): AFSP case-fix stylometric sensitivity
- [`results/heldout_decomp_afsp_casefix_val.json`](results/heldout_decomp_afsp_casefix_val.json): AFSP-full case-fix held-out comparison
- [`results/heldout_decomp_peft_afsp_casefix_val.json`](results/heldout_decomp_peft_afsp_casefix_val.json): PEFT+AFSP case-fix held-out comparison
- [`outputs/afsp_casefix_manifest.json`](outputs/afsp_casefix_manifest.json): corrected AFSP generation provenance

## Current takeaway

The original proposal expected the reinforcement-learning stage to provide the strongest stylistic control. The corrected validation evidence supports part of that expectation, but not a simple "RLSF wins" conclusion.

PEFT provides a strong domain-adapted starting point. Retrieval improves several adequacy and perceived-register measures, although the PEFT hybrids move farther from the training centroid on the held-out stylometric features. Judge-conditioned RLSF, in contrast, moves the selected systems closer on those held-out features, and the corrected trajectory shows that this improvement continues through much of training.

The high-weight RLSF trajectory also suggests a limit. By step 800 it is very close to the measured held-out target, while the later step begins to cross the marker target and move away again. That pattern is exploratory, but it is more consistent with useful style adaptation followed by possible late over-stylization than with the earlier interpretation of continuous register drift.

The marker-case correction is therefore part of the result, not just a code cleanup. It changed the sign of the main RLSF trajectory interpretation while leaving the adequacy metrics, trained adapters, and LLM-judge results untouched. The AFSP sensitivity rerun also shows that large changes in exemplar selection do not necessarily produce large changes in aggregate system performance.

The final sealed-test pass will determine which of these validation patterns generalize.

## Citation

If you use this work, please cite the thesis once the final version is available.

```text
Mehri, P. Style-Aware Neural Machine Translation of Low-Resource Texts
Using Large Language Models and Reinforcement Learning.
Undergraduate thesis, Bahá'í Institute for Higher Education (BIHE).
```
