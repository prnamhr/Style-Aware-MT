# Compute and API Budget

This document tracks paid API use for the project. It records the spending limit set for RLSF judge calls, the cost of completed paid runs, and the rules used before any new paid run starts.

GPU time is handled separately. AFSP sweeps, LoRA training, RLSF training, and full-split inference run on Colab from the runbooks under `notebooks/`. The 8 GB development GPU cannot hold `Qwen2.5-7B-Instruct` in bf16, and quantizing the model would change the frozen base used in the experiments. Local work such as data preparation, retrieval-index construction, most evaluation code, inferential analysis, and the test suite does not incur API cost.

The test split is still sealed. No paid call listed here was made on the test split.

## Current status

The three RLSF training arms have run. `w3_0.0` used `--skip_judge` and therefore incurred no judge cost. The two judge-conditioned arms, `w3_2.0` and `w3_6.0`, each ran for 300 rollouts.

The original planning rate came from an 80-call smoke test. The two completed paid arms later provided a larger measurement over 19,200 calls. That measured rate is now used for future RLSF pricing in this document.

The declared RLSF judge-spend cap remains **$25 across the paid RLSF work covered by the authorization dated 2026-08-08**. The runtime config uses an **$8 per-run cap**, because the cap enforced inside one process is not the same thing as the total project-level authorization.

## Declared cap

The RLSF judge-spend cap was set at **$25 on 2026-08-08**.

`configs/rlsf.yaml` enforces the following limits per run:

```text
caps.max_judge_spend_usd: 8.0
max_steps: 600
max_grid_steps: 200
max_judge_calls: 340000
```

The distinction between the $25 total authorization and the $8 runtime limit matters. `JudgeBudget` is created separately for each process, so one process cannot see spend from another process. The $8 limit therefore applies to a single run. The $25 figure is tracked across the full set of authorized paid RLSF runs in this document and in the DEVLOG.

The proposal itself does not state a numeric API budget. An earlier version of this file said the cap had to be copied from the proposal, but there was no number to copy. The $25 cap was therefore set here as a dated spending decision rather than chosen during a run.

Evaluation-rater spend is authorized separately from RLSF training spend.

## Spend recorded so far

The figures in this table come from provider-reported token counts stored by the inference clients, except for the aborted `w3_6.0` attempt, which had to be reconstructed from its surviving step log.

| Date | Purpose | Model | Calls | Cost | Record |
|---|---|---|---:|---:|---|
| 2026-08-04 | `commercial_haiku` generation, external reference baseline | `claude-haiku-4-5` | 1,323 | $0.6316 | `outputs/commercial_haiku_val_usage.json` |
| 2026-08-05 | Cross-family second judge Phi_B, 7 conditions, Batch API at 50% discount | `gpt-5.6-terra` | 9,261 | $6.1173 | `results/judge_gpt_val_usage.json` |
| 2026-08-07 | RLSF reward-path smoke, 20 dev segments x G=4 | `gpt-4o-mini` | 80 | $0.0059 | `outputs/rlsf/smoke_usage.json` at `a988c57` |
| 2026-08-08 | Same smoke rerun after timing the judge block | `gpt-4o-mini` | 80 | $0.0059 | `outputs/rlsf/smoke_usage.json` |
| 2026-08-14 | RLSF arm `w3_2.0`, 300 rollouts x 32 calls | `gpt-4o-mini` | 9,600 | $0.7423 | `outputs/rlsf/steps_w3_2.0_usage.json` |
| 2026-08-14 | RLSF arm `w3_6.0`, 300 rollouts x 32 calls | `gpt-4o-mini` | 9,600 | $0.7431 | `outputs/rlsf/steps_w3_6.0_usage.json` |
| **Recorded total** |  |  | **29,944** | **$8.2461** |  |
| 2026-08-14 | First `w3_6.0` attempt, killed at rollout 208 | `gpt-4o-mini` | 6,656 | ~$0.5149 | reconstructed, see below |
| **Including reconstructed row** |  |  | **36,600** | **$8.7610** |  |

`w3_0.0` ran 300 rollouts with `--skip_judge` on 2026-08-13. It made no paid judge calls, so there is no usage row for that arm.

The aborted `w3_6.0` run is the only line above that is not provider-reported. The usage sidecar is written after `trainer.train()` returns (`src/rlsf/train.py:675`), so a kernel kill can prevent that file from being created. The surviving file, `outputs/rlsf/aborted/steps_w3_6.0_kernelkill_208.jsonl`, records 208 rollouts and 6,656 samples. Each sample corresponds to one judge call. The cost is reconstructed using the realized per-call rate described below.

For the RLSF reward judge alone, spend covered by the $25 authorization is **$2.0121 over 26,016 calls**. This includes the two smoke tests, the aborted attempt, and the two completed paid arms.

A few paid calls are not included in that number because no usage artifact exists for them:

- **Phi_A validation pass:** `claude-haiku-4-5`, stored in `results/judge_val.json`. This pass happened before judge usage files were added on 2026-08-05, so its cost is unrecorded rather than estimated.
- **Early exemplar-count probe:** `outputs/gpt-4o-mini__n{2,4,8,10}_val.jsonl`, committed at `50c7477` on 2026-07-17. These 85 generations also predate usage recording. They are not used in a reported result.
- **The 2026-08-05 Phi_B amount is cumulative:** the full pass cost $6.0996 over 9,236 calls. The remaining 25 calls and $0.0177 came from the diagnostic pilot that preceded it. The batch pass used 3,796,497 prompt tokens and 386,804 completion tokens at `pricing: [2.00, 12.00]` with the 50% Batch API discount.

## Realized RLSF judge rate

The smoke test gave the planning rate. The two completed paid arms provide the larger measurement.

| Source | Calls | Prompt tokens/call | Completion tokens/call | $/call |
|---|---:|---:|---:|---:|
| Smoke, 20 dev segments | 80 | 416.6 | 18.5 | $7.375e-5 |
| `w3_2.0` | 9,600 | 442.7 | 18.2 | $7.732e-5 |
| `w3_6.0` | 9,600 | 443.2 | 18.2 | $7.741e-5 |
| **Realized, blended** | **19,200** | **443.0** | **18.2** | **$7.737e-5** |

The realized rate is 1.049 times the planning rate. The difference comes from prompt length: about 443 prompt tokens per call in the completed arms versus about 417 in the smoke test. The rubric did not change. The training segments were simply longer on average than the 20 segments used for the smoke.

At the realized rate, the three-arm plan would price at about $1.50 instead of the original $1.43. The step-capped worst case moves from about $2.83 to about $2.97. Both remain below the $8 per-run spend cap.

Future RLSF estimates in this file use **$7.737e-5 per call**. Historical entries keep the rate that was available when each decision was made.

## Validation judge pass for the RLSF conditions

The planned validation pass scores three RLSF conditions on 1,323 validation segments:

```text
1,323 segments x 3 conditions = 3,969 calls per rater
7,938 calls across two raters
```

This is separate from the reward-judge budget used during training.

For Phi_B, the closest measured rate comes from the 2026-08-05 validation pass. That pass used `gpt-5.6-terra` through the Batch API and averaged 409.8 prompt tokens and 41.8 completion tokens per call.

The initial point estimates were:

| Rater | Transport | Calls | Per call | Cost |
|---|---|---:|---:|---:|
| Phi_B, `gpt-5.6-terra` | Batch, 50% discount | 3,969 | $6.604e-4 | **$2.62** |
| Phi_A, `claude-haiku-4-5` | synchronous | 3,969 | $6.187e-4 | **$2.46**, estimated |
| **Total** |  | **7,938** |  | **$5.08** |

On 2026-08-15, this was widened to a pricing band rather than treated as a single exact figure. The reason is simple: candidate length can change the prompt length, and the Phi_A estimate also crosses tokenizers.

| Rater | Calls | Per call | Cost |
|---|---:|---:|---:|
| Phi_B, `gpt-5.6-terra`, Batch 50% | 3,969 | $6.604e-4 to $6.809e-4 | **$2.62 to $2.70** |
| Phi_A, `claude-haiku-4-5`, synchronous | 3,969 | $6.187e-4 to $6.391e-4 | **$1.96 to $3.04**, estimated |
| **Total** | **7,938** |  | **$4.58 to $5.74** |

The authorization is based on the top of that range, **$5.74**.

The two ranges are not equally certain. Phi_B is based on a measured token profile, so its range only allows for candidate-length growth. Phi_A has no historical usage artifact. Its estimate applies Claude pricing, `[1.00, 5.00]`, to Phi_B's measured token counts and then allows a 20% tokenizer and output-length band.

A full paid pass still starts with a `--limit N` pilot. `manage.py judge --limit N` keeps the diagnostic cache but does not write the final results file, so the pilot is not mixed into the reported score.

The test split is not included in these estimates.

## Per-run cap semantics

`JudgeBudget` is created inside `src/rlsf/train.py:run` and `src/rlsf/pool.py` once per invocation. It starts with:

```text
calls: 0
spend_usd: 0.0
```

It does not load an accumulated project total from disk. This means `max_judge_spend_usd` is a **per-process limit**, not a project-level total.

The config now uses:

```text
max_judge_spend_usd: 8.00
```

The intended geometry was:

| Constraint | Figure |
|---|---:|
| Paid runs covered by the plan, `w3_2.0`, `w3_6.0`, and the dev-slice pool | 3 |
| 3 x $8.00 against the $25 project-level cap | $24.00 <= $25 |
| One run at the step-capped worst case, 600 rollouts x 8 prompts x G=8 | 38,400 calls, $2.83 at the planning rate |
| $8.00 compared with that worst case | 2.8x |

`w3_0.0` uses `--skip_judge` and therefore does not add a fourth paid process.

`max_judge_calls` remains 340,000. At the measured rate, that many calls would cost about $25.08, so the $8 dollar cap would stop a run first. The call limit is therefore loose, but it still provides a second guardrail.

The audited project total comes from the spend table above. The per-run config limits do not replace that accounting.

## Historical authorization records

The following entries are kept because they show what was known when earlier spending decisions were made. They should not be read as the current run configuration.

### 2026-08-13: revised three-arm geometry

This entry superseded the 2026-08-08 run volumes. The preregistration addendum moved from two arms to three and reduced the rollout batch from 16 prompts to 8.

At the planning rate of $7.375e-5 per judge call, one rollout used 32 judge calls and cost about $0.0024.

| Line item | Judge calls | Cost |
|---|---:|---:|
| RLSF-0, `w3_0.0`, 300 rollouts with `--skip_judge` | 0 | $0.00 |
| RLSF-Mid, `w3_2.0`, 300 rollouts | 9,600 | $0.71 |
| RLSF-High, `w3_6.0`, 300 rollouts | 9,600 | $0.71 |
| 50-rollout smoke with `--skip_judge` | 0 | $0.00 |
| 3-rollout judge-path smoke | 96 | $0.01 |
| Checkpoint selection on the dev slice | 0 | $0.00 |
| Omega grid rescored over the cached pool | 0 | $0.00 |
| **Planned total** | **19,296** | **$1.43** |

The three-arm setup cost less than the earlier two-arm plan because it used 300 rollouts rather than 500 and 32 judge calls per rollout rather than 64.

At the operative group size, 800 capped rollouts corresponded to 25,600 calls and about $1.89. At the authorized group-size ceiling of 8, the same cap allowed 51,200 calls and about $3.78.

The following config limits stayed unchanged:

```text
max_steps: 600
max_grid_steps: 200
max_judge_calls: 340000
```

`max_judge_spend_usd` was tightened from $25 to $8 per run after the per-process behavior of `JudgeBudget` was audited.

Judge latency was the more practical constraint. At `judge.concurrency: 8`, a 32-call block took about 4.3 seconds of GPU idle time, or about 0.36 GPU-hours across 300 paid rollouts.

### 2026-08-08: measured smoke and original cap calculation

This entry is superseded by the 2026-08-13 geometry above. It is kept as the record behind the original cap.

The smoke test ran twice and measured:

```text
80 calls
about 417 prompt tokens per call
about 18 completion tokens per call
$0.0059 total
$7.375e-5 per call
```

That rate was below the earlier estimated range of $1.14e-4 to $1.44e-4 per call.

At the run geometry used in this earlier plan, one rollout was 16 prompts x group size 4, or 64 judge calls.

| Line item | Judge calls | Cost |
|---|---:|---:|
| Final RL run, 500 rollouts | 32,000 | $2.36 |
| Dev-slice best-of-N pool, 499 segments x 8 samples | 3,992 | $0.29 |
| Omega grid rescored offline over the same pool | 0 | $0.00 |
| **Planned total** | **35,992** | **$2.65** |

The grid itself added no paid calls because it reused completions already generated for the pool.

The caps recorded at that point were:

| Cap | Value | Worst case it admitted |
|---|---:|---|
| `max_steps` | 600 | final run plus 100 rollouts of headroom |
| `max_grid_steps` | 200 | 4 cells x 50 if the grid were trained online |
| `max_judge_calls` | 340,000 | about $25.08 at the measured smoke rate |
| `max_judge_spend_usd` | 25.0 | $25.00 |

The step limits were expected to bind before the call or dollar limits. At the operative group size, the capped 800 rollouts would have used 51,200 calls and about $3.78. At the group-size ceiling of 8, they would have used 102,400 calls and about $7.55.

A "step" in these budget calculations means a **rollout**, because that is what creates new judge calls. It is not the same as TRL's optimizer-step count. GRPO can reuse one rollout across multiple optimizer iterations without creating new judge calls.

The large call and dollar ceilings were meant as protection against a rate change rather than permission to extend the planned training length. A new model, longer rubric, longer completions, more than 600 rollouts, or group size above 8 would require a new authorization.

The 2026-08-08 timing pass also measured judge latency: 80 calls took 10.63 seconds at `judge.concurrency: 8`, with a mean individual-call latency of 1.01 seconds and 7.63x achieved parallelism.

### 2026-08-07: initial estimated envelope

This entry predates the measured smoke rate and is fully superseded by the later entries.

| Quantity | Operative | Ceiling |
|---|---:|---:|
| Planned RL rollouts, 4 x 50 RQ3 grid cells plus 150 final | 350 | 350 |
| Prompts per rollout | 16 | 16 |
| Group size | 4 | **8** |
| Judge calls | 22,400 | **44,800** |
| Estimated cost | $2.55 to $3.23 | **$5.11 to $6.45** |

The estimate used `gpt-4o-mini` pricing of `[0.15, 0.60]`, with an assumed prompt length of 600 to 800 tokens and about 40 completion tokens.

These were estimates, not measurements. The planned pilot used 80 judge calls, 20 segments at group size 4, to obtain the rate later used in the 2026-08-08 entry.

The group-size ceiling of 8 was declared before the run so that moving from 4 to 8 would still fall inside the authorized envelope. Going above 8 or above the declared rollout count required a new pricing decision.

The reward judge was chosen separately from both evaluation raters. `gpt-4o-mini` was used for the RLSF reward, while Phi_A and Phi_B used different models.

## Spending rules

1. **State the volume before spending.** A paid run must have an expected call count and a priced upper bound before it starts.

2. **Run a pilot first.** A full paid evaluation pass begins with `--limit N`. The pilot may write its cache, but it must not create the final reported results file.

3. **Do not widen a paid run after authorization.** Adding a condition, rater, split, or larger sampling geometry counts as a new run and needs a new cost estimate.

4. **Price every model explicitly.** If a model is not in the built-in pricing table, its config must set `pricing: [input, output]`. Otherwise a run can report `$0.00` even when calls were paid.

5. **Use batch transport when waiting is acceptable.** The 2026-08-05 Phi_B pass used the OpenAI Batch API at a 50% discount. The same pass would have cost about $12.20 synchronously.

6. **Keep the fallback method explicit.** Before RLSF training was completed, the declared fallback was best-of-N reranking of PEFT-checkpoint samples scored with the same reward. Using that fallback would count as a method change and would need to be recorded in the DEVLOG rather than substituted silently.

## Related records

- `README.md`: project constraints and the Phi reliability note
- `docs/DEVLOG.md`: dated records for the runs and decisions summarized here
- `docs/preregistration_rlsf.md`: RLSF design and addenda
- `results/judge_gpt_val_usage.json`: measured Phi_B validation usage
- `outputs/*_val_usage.json`: generation usage artifacts
- `outputs/rlsf/*_usage.json`: RLSF judge usage artifacts

Where a usage artifact and this summary disagree, the usage artifact is the primary record.
