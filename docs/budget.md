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

**Not transcribed here yet.** The cap is stated in `docs/proposal.pdf`, which could not be
read in the environment where this file was written (no PDF text extraction available).
It must be copied here verbatim from the proposal rather than reconstructed, because the
figure is a pre-registered commitment and an approximation of it is not a cap. Until that
transcription happens, this file records actual spend only, and the cap is enforced by the
per-run confirmation rule below rather than by a number on this page.

## Spend to date

All figures are provider-reported token counts priced through
`src/infer/openai_client.py` / `src/infer/anthropic_client.py` and persisted at run time.
Every paid call to date is on the **validation** split; the test split is sealed and has
had no paid call made against it.

| Date | Purpose | Model | Calls | Cost | Record |
|---|---|---|---:|---:|---|
| 2026-08-04 | `commercial_haiku` generation (external reference baseline) | `claude-haiku-4-5` | 1,323 | $0.6316 | `outputs/commercial_haiku_val_usage.json` |
| 2026-08-05 | Cross-family second judge Φ_B, 7 conditions, Batch API at 50 % discount | `gpt-5.6-terra` | 9,261 | $6.1173 | `results/judge_gpt_val_usage.json` |
| **Recorded total** | | | **10,584** | **$6.7489** | |

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

**RLSF is the dominant future cost and is not yet implemented.** Its reward includes a
training-time judge term (`ω₃ · Φ`), so PPO consumes one judge call per sampled
completion: cost scales as PPO steps × rollout batch size, not with the size of the
validation split.

### RLSF arm — authorized envelope (2026-08-07)

The shape of the arm and its priced worst case, as required above. **The caps in
`configs/rlsf.yaml` are null and `src/rlsf/config.py:assert_caps_declared` refuses to
load until they are set**, so recording this envelope does not itself authorise a run:
rule 1 still requires the declared cap to be transcribed from `docs/proposal.pdf` first,
and that transcription has not happened.

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
* `docs/DEVLOG.md` — 2026-08-05 (both entries), 2026-08-04, for the runs priced above.
* `results/judge_gpt_val_usage.json`, `outputs/*_val_usage.json` — the primary artefacts;
  authoritative over this summary.
