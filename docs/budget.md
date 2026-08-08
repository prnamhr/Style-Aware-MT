# Compute and API Budget

This document records the declared spend cap for paid API calls, the spend incurred to
date, and the fallback if the cap is approached. It is a control document: no paid run is
started before its expected call volume is stated here or in the DEVLOG entry that
authorises it, and no paid run is widened after the fact.

GPU compute is not metered in dollars. Every GPU stage — AFSP sweeps, LoRA training,
full-split inference — runs on Colab from the runbooks under `notebooks/`, because the
8 GB development GPU cannot hold `Qwen2.5-7B-Instruct` in bf16 and quantizing it would
redefine the frozen base (see `README.md`, Constraints). Local stages cost nothing:
data preparation, retrieval index construction, all of `src/eval` except COMET,
inferential analysis, and the test suite.

## Declared cap

**$25 of judge spend on the RLSF arm, declared 2026-08-08.** It is set in
`configs/rlsf.yaml` as `caps.max_judge_spend_usd`, together with `max_steps: 600`,
`max_grid_steps: 200` and `max_judge_calls: 340000`, so
`src/rlsf/config.py:assert_caps_declared` no longer refuses to load. The derivation is in
*RLSF arm — caps declared* below.

This page previously said the cap had to be copied verbatim from `docs/proposal.pdf` and
that an approximation of it was not a cap. The proposal has since been read on this point:
**it states no numeric spend cap.** There is nothing to transcribe, so the figure is set
here instead. What the verbatim rule was protecting — that a cap be a recorded commitment
rather than a number chosen at the moment of spending — is met by deriving it from a
measured rate and dating it, which is what the section below does. No other paid arm has a
cap on this page; the per-run confirmation rule still governs those.

## Spend to date

All figures are provider-reported token counts priced through
`src/infer/openai_client.py` / `src/infer/anthropic_client.py` and persisted at run time.
Every paid call to date is on the **validation** split or on the RLSF dev slice carved
from `train.jsonl`; the test split is sealed and has had no paid call made against it.

| Date | Purpose | Model | Calls | Cost | Record |
|---|---|---|---:|---:|---|
| 2026-08-04 | `commercial_haiku` generation (external reference baseline) | `claude-haiku-4-5` | 1,323 | $0.6316 | `outputs/commercial_haiku_val_usage.json` |
| 2026-08-05 | Cross-family second judge Φ_B, 7 conditions, Batch API at 50 % discount | `gpt-5.6-terra` | 9,261 | $6.1173 | `results/judge_gpt_val_usage.json` |
| 2026-08-07 | RLSF reward-path smoke, 20 dev segments × G=4 | `gpt-4o-mini` | 80 | $0.0059 | `outputs/rlsf/smoke_usage.json` @ `a988c57` |
| 2026-08-08 | Same smoke re-run once the judge block was timed | `gpt-4o-mini` | 80 | $0.0059 | `outputs/rlsf/smoke_usage.json` |
| **Recorded total** | | | **10,744** | **$6.7607** | |

Two qualifications on that total:

* **The primary judge pass is not in it.** Φ_A (`claude-haiku-4-5`,
  `results/judge_val.json`, 7 × 1,323 calls plus the earlier sweep-verification judge
  calls) predates usage recording — `results/judge_<stem>_usage.json` was only added on
  2026-08-05 together with `--tag` — so no usage artefact exists for it and its cost is
  **unrecorded**. It is not estimated here; an estimate would be indistinguishable in the
  table from a measured figure. Any future re-run of Φ_A will write a usage file and can be
  added.
* **The 2026-08-05 figure is cumulative, not incremental.** `session.cost_usd` for the full
  pass was $6.0996 over 9,236 calls; the extra 25 calls and $0.0177 are the diagnostic
  25-segment pilot that gated it. The batch transport applied a 50 % discount to
  3,796,497 prompt and 386,804 completion tokens at `pricing: [2.00, 12.00]`
  (`configs/judge_eval_gpt.yaml`).

## What remains to be spent

**RLSF is the dominant future cost and has not yet been run.** Its reward includes a
training-time judge term (`ω₃ · Φ`), so the loop consumes one judge call per sampled
completion: cost scales as rollout steps × rollout batch size, not with the size of the
validation split. The loop exists as of 2026-08-08 (`src/rlsf/train.py`) and enforces the
caps below at run time through `JudgeBudget`, which refuses a judge block that would cross
`max_judge_calls` and refuses to start one once `judge.usage` reports spend at
`max_judge_spend_usd`.

### RLSF arm — caps declared (2026-08-08)

The smoke has now run twice and reports a **measured** per-call rate of **$7.375e-5**
(`outputs/rlsf/smoke_usage.json`: 80 calls, 417 prompt and 18 completion tokens each,
$0.0059). That is 0.65× the low end of the $1.14e-4 – $1.44e-4 range the 2026-08-07
envelope assumed: the rubric plus segment came in near 417 prompt tokens rather than
600–800, and the justification ran to 18 completion tokens rather than 40. Every figure
below is priced at the measured rate.

At the operative rollout — 16 prompts × group size 4 — a step is 64 judge calls, $0.0047.

| Line item | Judge calls | Cost |
|---|---:|---:|
| Final PPO run, 500 steps | 32,000 | $2.36 |
| Dev-slice best-of-N pool, 499 segments × 8 samples | 3,992 | $0.29 |
| ω grid, re-scored offline over that pool | 0 | $0.00 |
| **Planned total** | **35,992** | **$2.65** |

The ω grid is the reason this is cheaper than the 2026-08-07 envelope. Ranking the four
cells over one fixed pool of sampled completions re-weights scores that were already paid
for, so the grid costs nothing beyond the pool; the envelope had priced it as 200 PPO
steps of its own.

The caps are set above that plan, at two different distances and for two different reasons:

| Cap | Value | Worst case it admits |
|---|---:|---|
| `max_steps` | 600 | final run, 100 steps of headroom |
| `max_grid_steps` | 200 | 4 cells × 50, spent only if the grid is trained online after all |
| `max_judge_calls` | 340,000 | $25.08 |
| `max_judge_spend_usd` | 25.0 | $25.00 |

**The step caps bind first and are the operative constraint.** 800 capped steps at the
operative group size are 51,200 calls and $3.78; at the authorized ceiling of group size 8
they are 102,400 calls and $7.55 (`src/rlsf/config.py:worst_case_judge_calls` computes the
call figure at the ceiling, not at the operative size).

A step here is a **rollout**, which is what costs money: 16 prompts × G completions, each
one judge call. It is not TRL's `max_steps`. GRPO reuses a rollout `num_iterations` times,
so the optimizer-step count handed to the trainer is 4× the figures in this table while the
judge-call count is unchanged (`src/rlsf/config.py:optimizer_steps`).

The call and dollar caps sit ~3.3× above that, which is deliberate: they are not a second
opinion on the step count, they are a backstop against the *rate* moving. A model swap, a
longer rubric, or a completion-length regression that tripled the per-call price would
still satisfy every step cap while tripling the bill, and nothing else in the loop would
notice. $25 is 9.4× the planned $2.65 and 3.3× the step-capped worst case. `max_judge_calls`
is rounded up past its own $25 line, so the dollar cap is the one that actually trips.

Going beyond 600 steps, beyond group size 8, or beyond $25 is a new authorisation
requiring a re-priced entry here, not a config edit.

**Dollars are still not what constrains this arm.** The 2026-08-08 pass timed the judge
block: 80 calls in 10.63 s wall clock at `judge.concurrency: 8`, mean call 1.01 s, 7.63×
achieved parallelism. A 64-call step therefore holds the rented GPU idle for roughly 8.5 s,
and the planned 500 steps for about 1.2 GPU-hours of judge latency alone. That is the
number to watch, and it is bounded by the same step caps.

### RLSF arm — authorized envelope (2026-08-07)

**Superseded by the entry above.** It is kept because it is the record the 2026-08-07
decisions were made against. Two things changed: the ω grid moved off PPO and onto the
best-of-N pool, and the per-call price stopped being an estimate. Its group-size ceiling of
8 carries forward unchanged.

| Quantity | Operative | Ceiling |
|---|---:|---:|
| PPO steps (4 × 50 RQ3 grid cells + 150 final) | 350 | 350 |
| Prompts per step | 16 | 16 |
| Group size (samples per prompt) | 4 | **8** |
| Judge calls (one per sampled completion) | 22,400 | **44,800** |
| Priced worst case | $2.55 – $3.23 | **$5.11 – $6.45** |

Priced at `gpt-4o-mini` `[0.15, 0.60]` — in the built-in table at
`src/infer/openai_client.py:17`, so rule 4 needs no `pricing:` override. The range spans
600–800 prompt tokens (the ~382-token frozen rubric plus source, reference, and
candidate) and ~40 completion tokens (a ≤15-word justification and the verdict line).
**These are estimates over assumed token counts, not measurements.** The pilot in the
config's `pilot:` block measures the real per-call rate, and the measured figure replaces
this range. `pilot.judge_calls` is **80** — 20 segments at group size 4, ~$0.01, sized to
the reward-path smoke's brief rather than the smoke trimmed to fit a round number.

The ceiling is stated at group size **8** while the config operates at **4**, so that
raising the group size later is covered by this authorisation rather than being a
widening of an authorised run under rule 3. Going beyond 8, or beyond 350 steps, is a
new authorisation requiring a re-priced entry here.

For scale: the ceiling is under a dollar of the $6.75 spent to date, so **dollars are not
the binding constraint on this arm** — GPU rental hours and in-loop judge latency are.
Latency was unmeasured when this section was written; the 2026-08-08 pass measured it.

The reward judge was selected on validity grounds rather than cost: `gpt-4o-mini` is
model-distinct from both evaluation raters, so neither Φ_A nor Φ_B is spent on the one
condition trained against a judge. See DEVLOG 2026-08-07.

The terminal test-split evaluation will require one further judge pass per condition on
1,323 sealed segments. Both raters are priced above, so that cost is estimable from the
validation pass once the condition set is final.

## Rules

1. **State the volume before spending.** Any paid run states its expected call volume and
   priced worst-case cost, and obtains explicit confirmation, before it starts.
2. **Pilot first.** A paid pass over a full split is preceded by a `--limit N` diagnostic
   pilot. `manage.py judge --limit N` deliberately writes the segment cache but *not* the
   results file, so a pilot cannot contaminate a reported figure.
3. **Never widen a paid run unprompted.** Adding a condition, a rater, or a split to a run
   already authorised is a new run requiring new confirmation.
4. **Price every model explicitly.** A model absent from the built-in pricing table reports
   $0.00 and silently understates spend. Configs for such models set `pricing: [in, out]`
   (`src/infer/openai_client.py:35`, `src/infer/run.py:72`).
5. **Prefer batch transport where latency permits.** The 2026-08-05 pass used the OpenAI
   Batch API at a 50 % discount; the same pass synchronously would have been ≈$12.20.
6. **Fallback if the cap is approached.** RLSF is reported using **best-of-N reranking** of
   PEFT-checkpoint samples scored with the same reward, instead of PPO. This is the declared
   fallback in the proposal and in `README.md`; it is a change of method to be recorded in the
   DEVLOG, not a silent substitution.

## Related records

* `README.md` — Constraints, and the Φ reliability open issue.
* `docs/DEVLOG.md` — 2026-08-08 (smoke green, caps declared), 2026-08-05 (both entries),
  2026-08-04, for the runs priced above.
* `results/judge_gpt_val_usage.json`, `outputs/*_val_usage.json` — the primary artefacts;
  authoritative over this summary.
