# Pre-registration: the test-split family

Written 2026-08-26 at commit `ef1b510`, before any generation, metric, or judge call has been
made against `data/splits/test.jsonl`. It fixes the conditions that will be scored on test, the
metrics they will be scored on, which comparisons carry inferential weight and which do not, and
the multiple-comparison correction applied to the ones that do.

`docs/preregistration_rlsf.md` declares the three RLSF training arms and nothing else. The two
PEFT retrieval hybrids and the sparse retrieval arm entered the study after it was written, and
no document declares them. This one covers the whole test family.

Like `docs/budget.md` and the RLSF pre-registration, this is a control document. It is amended by
dated addendum, never by rewriting a commitment after the number it predicted is known. What was
visible to its author when it was written is disclosed at the end, and in this case that is a great
deal: the full validation table was in hand. This is a replication pre-registration, not a blind one.

## What the test split has already been read for

`data/splits/test.jsonl` holds 1,322 segments over 8 works, digest `3e24e90f` in
`data/splits/hashes.json`, untouched in git since `f78ec23`. Two reads are on record, both in the
DEVLOG:

- 2026-08-20: marker-capitalization and marker-density statistics of the test targets, computed
  during the case-sensitivity investigation.
- 2026-08-23: the pool quarantine audit, which flagged 28 pool rows against test and built
  `data/knn_index_clean` from them. That quarantine was rebuilt from the val flags alone and
`results/leakage_test.json` was deleted; `src/retrieval/leakage.py:135` now refuses `test` without
`--unseal-test`.

Neither read reached a selection decision that survives in the pipeline. No generation, metric,
judge, or bootstrap artifact derives from the test split.

## Condition set

Fourteen rows are generated and scored. Twelve are the controlled study, matching
`STUDY_CONDITIONS` in `src/eval/metric_agreement.py`; two are external diagnostics that change the
generator family and are never used to support a study claim.

| Row | Config | Generator | Paid generation |
|---|---|---|---|
| `zeroshot` | `configs/base_qwen.yaml` | Qwen2.5-7B-Instruct | no |
| `random_fewshot` | `configs/base_qwen.yaml` | Qwen2.5-7B-Instruct | no |
| `knn_fewshot` | `configs/base_qwen.yaml` | Qwen2.5-7B-Instruct | no |
| `sparse_knn` | `configs/sparse_knn.yaml` | Qwen2.5-7B-Instruct | no |
| `afsp_margin` | `configs/base_qwen.yaml` | Qwen2.5-7B-Instruct | no |
| `afsp_full` | `configs/base_qwen.yaml` | Qwen2.5-7B-Instruct | no |
| `peft` | `configs/peft_qwen.yaml` | PEFT `checkpoint-1358` | no |
| `peft_knn` | `configs/peft_afsp.yaml` | PEFT `checkpoint-1358` | no |
| `peft_afsp` | `configs/peft_afsp.yaml` | PEFT `checkpoint-1358` | no |
| `rlsf_w3_0.0` | `configs/rlsf_eval_w3_0.0.yaml` | RLSF step 200 | no |
| `rlsf_w3_2.0` | `configs/rlsf_eval_w3_2.0.yaml` | RLSF step 200 | no |
| `rlsf_w3_6.0` | `configs/rlsf_eval_w3_6.0.yaml` | RLSF step 100 | no |
| `commercial_haiku` | `configs/commercial_haiku_zeroshot.yaml` | `claude-haiku-4-5` | yes |
| `gpt56_sparse_knn` | `configs/commercial_gpt56_sparse_knn.yaml` | `gpt-5.6-sol` | yes |

`afsp_full_casefix` and `peft_afsp_casefix` are val-only sensitivity diagnostics and are not
generated on test. `gpt56_knn_fewshot` is not generated on test: without a matched, fully scored
companion it cannot isolate a retrieval effect from a generator change, which is the same reason
it was cut from the val results.

The seal is opened by pointing `data.eval_file` at `data/splits/test.jsonl`. `manage.py infer`
carries no test guard, so the opening is a config edit and is recorded as one.

## Retrieval pool

All fourteen rows retrieve from `data/knn_index`, the unquarantined 10,860-row pool that every
validation run used. Substituting the quarantined index would change the method between the two
splits, which is a larger problem than the near-duplicate rate it removes.

The test-time leakage audit is run once, with `--unseal-test`, as a diagnostic. Its trigger is
declared here: if more than 2.56% of test segments are flagged as near-duplicates of a pool row -
twice the val rate of 1.28%, 17 of 1,323 - the retrieval conditions are regenerated on a
test-quarantined index and reported as a sensitivity check. That rerun, if it happens, is
exploratory. The primary table is the unquarantined one either way.

## Metrics

| Metric | Estimator | Artifact |
|---|---|---|
| COMET | `Unbabel/wmt22-comet-da`, segment-level | `results/comet_test.json` |
| chrF, BLEU | sacrebleu 2.6.0; `manage.py eval` prints the corpus scores | `results/bootstrap_chrf_test.json`, `results/bootstrap_bleu_test.json` |
| `stylo_dist` | standardized Euclidean distance to the training-target centroid over `lex_density`, `ttr`, `root_ttr`, `marker_rate` | `results/stylometrics_ci_ladder_test.json` |
| held-out distance | the same distance over `ttr`, `root_ttr`, `marker_rate` | `results/heldout_decomp_test.json` |
| `Phi_A` | `claude-haiku-4-5`, `prompts/judge_eval.txt` | `results/judge_test.json` |
| `Phi_B` | `gpt-5.6-terra`, same rubric, Batch API | `results/judge_gpt_test.json` |

The centroid is `results/stylometrics_centroid.json`, fingerprint `fd5aec8d69454b02`, built from
the 10,860 training targets under the corrected case-insensitive archaic-marker regex. It is a
training-side object and the seal does not touch it. It is not rebuilt for this pass.

All inference is paired at the segment level with 10,000 resamples, alpha = 0.05, seed 42, and
two-sided p-values. `manage.py stylometrics_ci` defaults to 2,000 resamples and the val ladder was
run at that setting; the test pass uses 10,000 because Holm thresholds reach 0.005 and a p-value
quantized to 1/2,000 cannot be read against them. The estimator is unchanged.

## Provenance

Every score is bound to the bytes it scored. `manage.py judge` and `manage.py judge_batch` record
the SHA-256 of `outputs/<condition>_test.jsonl` in the segment cache's `_meta.json` under
`outputs.<condition>`, and refuse to extend a cache whose recorded digest no longer matches the
file. `manage.py comet` writes the same digest into each condition's record in
`results/comet_test.json`. The judge cache resumes on the source text alone, and greedy decoding
is not byte-stable across sessions, so without this a regenerated condition would inherit the
scores of the translations it replaced. That failure is silent, the raters are paid, and the test
pass is run once.

The validation artifacts carry no such digest. They were scored before the binding existed and it
is not backfilled: recording a digest now would assert a link that was never checked.

## Confirmatory family

Five contrasts, each scored on COMET and on `stylo_dist`. Ten tests. The predicted sign and the
validation value it comes from are recorded for each.

| # | Contrast | Question | Metric | Predicted sign | Validation value |
|---|---|---|---|---|---|
| 1 | `peft` − `knn_fewshot` | RQ1 | COMET | + | +0.0147, point difference only |
| 2 | `peft` − `knn_fewshot` | RQ1 | `stylo_dist` | − | −0.0753 [−0.1231, −0.0277], p = .002 |
| 3 | `rlsf_w3_2.0` − `peft` | RQ1, RQ3 | COMET | + | +0.0021 [+0.0004, +0.0039], p = .015 |
| 4 | `rlsf_w3_2.0` − `peft` | RQ1, RQ3 | `stylo_dist` | − | −0.0098 [−0.0267, +0.0071], p = .273 |
| 5 | `afsp_full` − `knn_fewshot` | RQ2 | COMET | + | +0.0014 [−0.0020, +0.0046], p = .433 |
| 6 | `afsp_full` − `knn_fewshot` | RQ2 | `stylo_dist` | − | −0.0622 [−0.0983, −0.0270], p < .001 |
| 7 | `peft_afsp` − `peft` | hybrid | COMET | + | +0.0047 [+0.0011, +0.0081], p = .010 |
| 8 | `peft_afsp` − `peft` | hybrid | `stylo_dist` | + | +0.0342 [−0.0024, +0.0714], p = .069 |
| 9 | `sparse_knn` − `knn_fewshot` | sparse | COMET | none | +0.0007 [−0.0017, +0.0031], p = .567 |
| 10 | `sparse_knn` − `knn_fewshot` | sparse | `stylo_dist` | none | +0.0089 [−0.0162, +0.0339], p = .492 |

Negative is better for `stylo_dist`, positive for COMET. Prediction 8 therefore says that
retrieval on top of PEFT moves the output away from the target centroid while improving COMET.
That is the shape of the val hybrid result and it is predicted to repeat.

Test 1 carries no interval because the paired COMET bootstrap for that pair was overwritten by a
later run. 0.6986 against 0.6839 in the val table is the whole of what is recorded for it.

A prediction counts as replicated only when both conditions hold: the sign on test matches the
sign above, and the Holm-adjusted test rejects. A rejection in the opposite direction is a failed
prediction and is reported as one, not as a finding.

## Correction

Holm-Bonferroni across all ten tests at alpha = 0.05, using `holm_bonferroni` in
`src/eval/metric_agreement.py`. One family, not two: the ten tests answer one question set about
whether the val ordering holds on held-out works, and splitting them by metric would buy power by
asserting an independence the design does not have. Thresholds run from 0.005 for the smallest
p-value to 0.05 for the largest.

What that is expected to yield, stated before the pass rather than after it. At val effect sizes
only tests 2 and 6 - the two register contrasts against `knn_fewshot`, at p = .002 and p < .001 -
are likely to clear a 0.005 or 0.006 threshold. The COMET effects that separated on val are
0.002 to 0.005 COMET points at p = .010 to .015, and Holm over ten will most likely drop them.
A thin confirmatory yield is the expected outcome of this family, not a surprise, and it is not
evidence that those effects are absent.

Uncorrected p-values and intervals are reported alongside the corrected verdicts for every test,
as they are in the val tables.

## What the family excludes

Everything below is exploratory. It is reported, it is not corrected as part of the confirmatory
family, and no claim in the thesis rests on it alone.

- Both LLM raters. `Phi_A` and `Phi_B` run on all fourteen rows and are reported per rater, never
  averaged. The rater-dependence found on val is the reason: a measure whose contrasts flip sign
between raters is not a measure a confirmatory claim should be built on.
- Held-out distance, chrF, BLEU, per-feature signed z, rank distributions, and the 91 pairwise
  intervals the full ladder produces.
- The RQ4 metric-agreement analysis over twelve conditions and both raters.
- The two external rows, `commercial_haiku` and `gpt56_sparse_knn`.
- Any comparison among the three RLSF arms, and the checkpoint trajectory, which is val-only and
  stays val-only.

One exclusion costs something and is named so it is not smuggled back later. The strongest
validation result for RLSF is on held-out distance - `rlsf_w3_2.0` at 0.1264 against PEFT's 0.1713
- and held-out distance is not in the confirmatory family. Test 4 uses the four-feature distance,
where the same contrast was −0.0098 and did not separate. The held-out prediction is recorded here
so it is fixed in advance and carries no weight: `rlsf_w3_2.0` − `peft` on held-out distance is
predicted negative, and `rlsf_w3_6.0` − `peft` likewise.

## Predicted nulls

Tests 9 and 10 predict no separation, and tests 4 and 5 predict a direction that val did not
resolve. No equivalence bounds are declared for any of them, so a test interval that again
contains zero is a failure to detect and is reported in those words. It is not evidence that the
sparse arm and ordinary kNN are equivalent. What would be a surprise is a resolved effect in
either direction on test where val had none; that would be reported as a discrepancy between the
splits, and the works the two splits contain are the first place to look for its cause.

## Order of operations

1. Freeze. The commit at which the test pass runs is recorded in each output manifest.
2. Authorization for the paid lines below is recorded in `docs/budget.md`, dated, before step 5.
3. Leakage audit on test with `--unseal-test`; record the flagged rate against the 2.56% trigger.
4. Generate the twelve local rows with `data.eval_file: data/splits/test.jsonl`.
5. Generate the two external rows. Both start with a `--limit 20` pilot.
6. Local scoring: `eval`, `comet`, `stylometrics`, `stylometrics_ci`.
7. Paid raters, each after a `--limit 20` pilot: `judge` for `Phi_A`, `judge_batch` for `Phi_B`.
8. The ten confirmatory tests, Holm applied, verdicts recorded.
9. Everything exploratory.

After step 4 no selection of any kind occurs. No checkpoint, no retrieval cell, no decoding
setting, and no rater is chosen on a test figure. Every such choice was made on val or on the dev
slice and is frozen below.

```bash
CONDS="zeroshot random_fewshot knn_fewshot sparse_knn afsp_margin afsp_full peft \
       peft_knn peft_afsp rlsf_w3_0.0 rlsf_w3_2.0 rlsf_w3_6.0 commercial_haiku gpt56_sparse_knn"

python manage.py eval    --conditions $CONDS --split test
python manage.py comet   --conditions $CONDS --split test
python manage.py stylometrics --conditions $CONDS --split test
python manage.py stylometrics_ci --split test --n_resamples 10000 \
    --conditions $CONDS --results_path results/stylometrics_ci_ladder_test.json
python manage.py bootstrap --metric comet --split test --n_resamples 10000 \
    --conditions knn_fewshot sparse_knn afsp_full peft peft_afsp rlsf_w3_2.0 \
    --pairs peft:knn_fewshot afsp_full:knn_fewshot sparse_knn:knn_fewshot \
            peft_afsp:peft rlsf_w3_2.0:peft
```

## Frozen settings

| Component | Value | Where |
|---|---|---|
| Base model | `Qwen2.5-7B-Instruct`, bf16, greedy, `max_tokens: 1024`, seed 42 | `configs/base_qwen.yaml` |
| Retrieval | `intfloat/multilingual-e5-large-instruct`, `data/knn_index`, k = 8, `most_similar_last` | `configs/base_qwen.yaml` |
| AFSP | beta 0.3, lambda_style 0.75, bandpass, target sigma 1.0 | `configs/base_qwen.yaml` |
| Register direction | derived, `results/register_direction.json` | seven configs read the artifact |
| Sparse | `min_df: 40`, `freeze_n: 500`, m = 4, rarity digest `8fa5b0b2` | `configs/sparse_knn.yaml` |
| PEFT | LoRA r32 alpha64, `models/peft_lora_r32_lr2e-4/checkpoint-1358` | `configs/peft_qwen.yaml` |
| RLSF checkpoints | `w3_0.0` step 200, `w3_2.0` step 200, `w3_6.0` step 100 | `configs/rlsf_eval_*.yaml` |
| Centroid | `results/stylometrics_centroid.json`, `fd5aec8d69454b02` | training targets |
| Rubric | `prompts/judge_eval.txt`, digest-verified on load | both raters |

## Budget

No paid call has been made against the test split. The projection uses the measured per-call rates
from the most recent val passes rather than the older planning rates.

| Line | Calls | Rate | Cost |
|---|---:|---:|---:|
| `Phi_A`, `claude-haiku-4-5`, 14 conditions | 18,508 | $1.010e-3 to $1.020e-3 | $18.70 to $18.88 |
| `Phi_B`, `gpt-5.6-terra`, Batch 50%, 14 conditions | 18,508 | $6.589e-4 to $6.619e-4 | $12.20 to $12.25 |
| `gpt56_sparse_knn` generation | 1,322 | $7.765e-3 | $10.27 |
| `commercial_haiku` generation | 1,322 | $4.774e-4 | $0.63 |
| **Total** | **39,660** |  | **$41.80 to $42.03** |

This is larger than any single authorization in `docs/budget.md` - the val rater pass was
authorized at $5.74 - and it is separate from the $25 RLSF training authorization, which covers
the reward judge only. Step 2 of the order of operations exists for this reason.

Two levers are recorded now so that pulling one later is a budget decision and not a redesign.
Dropping `gpt56_sparse_knn` removes $10.27 of generation and about $2.20 of rater calls; that row
is descriptive and supports no confirmatory test. Restricting the raters to the six conditions the
confirmatory contrasts name would remove roughly $13 more. Neither lever touches the ten tests.

## Disclosure: information visible at the time of writing

The complete validation results table was in hand, including every quantity the ten tests will
measure. The predicted signs above are the val point estimates, and the two contrasts expected to
survive correction are the two that already had the smallest val p-values. This document does not
claim the evidential standing of a pre-registration written before the effects were seen.

What it does fix, and what it is worth: the condition set cannot grow to include a row that looks
good on test; the confirmatory family cannot be re-cut after the p-values are known; the correction
cannot be relaxed from Holm to per-metric families or to FDR once it drops a result; and a null on
tests 9 and 10 cannot be reported as equivalence. The 2026-08-01 DEVLOG entry asked for exactly
this - a comparison family declared before the test run so the correction is decided in advance
rather than after seeing which comparisons came close. That is what is declared here.

Held-out distance, `Phi_A`, `Phi_B`, chrF, and BLEU were also read on val, which is part of why
they sit outside the family rather than in it.
