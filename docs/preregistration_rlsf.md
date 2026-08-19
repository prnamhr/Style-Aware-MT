# Pre-registration: the RLSF arms

Written **2026-08-12**, before the ω pool re-ranking of that date and before any GRPO step
has run. It fixes the two training arms, the outcome predicted for each, the rule that halts
a run, and the condition under which the test split is opened. Like `docs/budget.md` this is
a control document: it is amended by dated addendum, never by rewriting a commitment after
the number it predicted is known.

What was visible to its author when it was written is disclosed at the end.

## The two arms

`configs/rlsf.yaml` declares a four-cell ω grid. Two of those cells are trained:

| Arm | Cell | ω as declared | ω at unit norm (bleu, kiwi, judge) |
|---|---|---|---|
| RL-Metric | `w3_0.0` | (1, 1, 0) | (0.7071, 0.7071, 0.0) |
| RLSF-Judge | `w3_2.0` | (1, 1, 2) | (0.4082, 0.4082, 0.8165) |

`w3_0.5` and `w3_1.0` are read off the cached pool and are not trained. The question these
arms answer is whether a rubric judge in the reward buys register fidelity or buys its
proxy, and that contrast needs the term present and absent, not a dose-response curve at
four points. Two arms is also what the GPU time and the $25 judge cap in `docs/budget.md`
support.

Both configs pass through `unit_omega()` (`src/rlsf/config.py:97`), so the two arms differ
in the direction of ω and not in its length: the weight vector cannot double as a step size.
Components are standardized within their group before the weights apply
(`normalization: group_z`, `src/rlsf/reward.py:286`), so a weight is a share of reward
variance rather than of raw scale. In RLSF-Judge the judge term carries ω₃² = 2/3 of the
reward's variance, four times either metric term. In RL-Metric it carries none.

Everything else is held identical and is not a free parameter of the comparison: the same
frozen PEFT initialization (`models/peft_lora_r32_lr2e-4/checkpoint-1358`), the same
reference policy and KL coefficient β = 0.05, the same prompt assembly as the PEFT
condition, rollout at T = 1.0 / top-p 0.95, 16 prompts × G = 4, `loss_type: dapo`,
lr 1e-6, μ = 4, ε = 0.2, seed 42, the same length band [0.5, 2.0] with `on_violation: floor`,
and the same frozen training rubric `prompts/judge_train.txt`, digest-verified on load.

**RL-Metric runs with `--skip_judge` and spends nothing.** At ω₃ = 0 the judge score cannot
enter the reward, so paying for it would buy only a matched feasibility mask: an unparseable
verdict marks a sample infeasible in RLSF-Judge (`src/rlsf/reward.py:474`) and, with the
judge held flat, does not in RL-Metric. The size of that asymmetry is fixed by the
unparseable rate, which is measured on the pool and reported with the arms rather than
assumed to be zero.

## Predicted outcome of the judge arm

Stated in advance and directionally. The initialization is the PEFT condition, which on val
sits at `marker_rate` z = +0.136 [0.082, 0.194], `stylo_dist` 0.2886, COMET 0.6986, Φ 2.744
(`results/stylometrics_ci_val.json`, `README.md`). It is already above the target centroid
on `marker_rate`, so any further rise moves away from the target rather than toward it.

For **RLSF-Judge**, relative to that initialization:

1. Training-time Φ, the mean of the reward's judge component per step, rises over the run.
2. `marker_rate` signed z rises with it, in the same direction and over the same steps.
   These two are predicted to move **together**; that pairing, not either alone, is the claim.
3. COMET-Kiwi is flat: its step means stay inside the run's own step-to-step spread, and at
   evaluation the arm is no better than its PEFT initialization on COMET.
4. Distance to the target register over the held-out features `ttr`, `root_ttr`,
   `marker_rate` (`results/stylometrics_centroid_split.json`) **worsens** — for the arm
   against its own initialization, and against RL-Metric.

For **RL-Metric**: no rise in `marker_rate` z beyond the drift band, and no reason for
training-time Φ to move, since nothing optimizes it.

The pattern predicted is therefore adequacy bought nothing, the rubric score bought a great
deal, and the register measure the rubric does not see got worse. That is Goodhart's law
with the reward named: Φ is the measure, register fidelity is the target, and the prediction
is that optimizing the first detaches it from the second.

**What would contradict this.** Φ rises, `marker_rate` stays inside the band, and held-out
distance improves or holds — the judge term buys register fidelity without inflating its
proxy, and the argument above fails. So does Φ rising while held-out distance improves and
`marker_rate` rises: that would mean the marker rise moved toward the target register, which
the initialization's position makes unlikely but does not forbid.

**What would be uninformative.** Neither arm moves — β = 0.05 holds the policy at its
initialization, or lr 1e-6 is too small to travel in the step budget. Then no prediction here
is tested, and the run is reported as a null run of the optimizer, not as evidence about the
reward. This outcome is written down now so it cannot later be read as confirmation.

## Stop rule

Both arms run under one register-drift rule, unchanged between them:
`feature: marker_rate`, `baseline_steps: 20`, `window: 5`, `k_sigma: 4.0`, `min_delta: 0.23`
(`configs/rlsf.yaml:70`, defaults mirrored at `src/rlsf/stop.py:18`, pinned by
`tests/test_rlsf_config.py`). The run's opening 20 steps set the baseline; the rule fires
when all five steps of the trailing window sit above baseline + 0.23 and the window mean
exceeds baseline by `max(0.23, 4.0 · se)`.

The operating point comes from simulation at the smoke's measured error, not from a σ level
(`manage.py drift_oc`, DEVLOG 2026-08-09): 1.3 % of drift-free runs halt, against 88.8 % of
runs with a +0.35 step and 90.6 % of runs ramping to +0.50. Against a +0.23 step it fires
45 % of the time; that is accepted, because a drift that size is measured in the val table
whether or not the monitor fires.

Three commitments follow. A halt is a result: the arm is reported at the step it halted, with
that step named, and is not restarted under a loosened rule. The rule is not edited after a
trip. And the asymmetry is the point — RLSF-Judge is predicted to trip it, RL-Metric is not,
so a trip in RL-Metric is either a 1.3 % false alarm or evidence against the attribution of
drift to the judge term, and will be reported as the second possibility before the first.

## The seal

**The test split stays sealed until Week 3.** Nothing in this pre-registration touches
`data/splits/test.jsonl`: ω selection reads the dev slice carved from `train.jsonl`
(`data/splits/rlsf_dev.jsonl`, 499 segments), and both arms are trained on
`data/splits/rlsf_train.jsonl` and scored on `val.jsonl`. No generation, no judge call and
no metric pass runs against test before that point, and `docs/budget.md` records that no
paid call has ever been made against it.

Week 3 opens the seal only when both arms have finished or halted, both have been scored on
val, and the reported comparison has been written. After the seal is opened no further
selection of any kind may occur — no cell, no checkpoint, no decoding setting chosen on a
test figure. The final pass is a single scoring of already-frozen systems.


## What will be reported

Per arm: the step log (`outputs/rlsf/steps.jsonl`), reward mean and spread, per-component
means, degenerate-group fraction, signed z per centroid feature with its clustered error,
the drift verdict at every step, and judge calls and dollars spent. At evaluation, the val
row of the results table on the same footing as the six existing conditions, plus distance
over the reward and held-out feature subsets separately.

Dev-slice figures select ω and are never reported as results. The four val figures the
predictions above name — training-time Φ, `marker_rate` z, COMET, held-out distance — are
the ones the predictions are scored against, and they are scored as stated here regardless
of which direction they move.

## Disclosure: what was visible when this was written

`outputs/rlsf/pool_omega.json` existed on disk, dated 2026-08-11, and its `selection` block
had been read: it selects `w3_0.0` at a register distance of 0.567 and reports that cell's
`marker_rate` pick shift as +0.02 ± 0.02, inside the 0.23 band. That file predates commit
`87659be` and is stale for the reasons the DEVLOG entry of 2026-08-12 gives — it ranks at
N = 8 rather than at the training group size, and takes flat-group picks in sampling order.

No reading of `w3_2.0`, and no per-cell row of that file other than the selected one, was
read before the predictions above were written. The pool's own completions and their cached
component scores were not inspected at all.

## Addendum, 2026-08-12: what the regenerated pool says

Added after `outputs/rlsf/pool_omega.json` was regenerated under `87659be`, and after the
predictions above were fixed. It records a consequence of that reading for the arms; it
changes no commitment.

The pre-registered selection rule ranks `w3_0.0` first and `w3_2.0` **last** of the four
cells, on register distance at G = 4 (0.5581 against 0.6645). The two trained arms are
therefore the cell the rule selects and the cell it ranks worst. That is intended.
RLSF-Judge is not run because it is the better reward; it is run because the argument is
about what a rubric term does to a policy, and the arm that demonstrates it is the one the
selection rule rejects. Its reported figures are evidence about the reward, never a
recommendation of it.

One prediction is sharpened rather than changed. Prediction 3 said Kiwi would be flat. At
selection time it is not quite flat: it declines monotonically as ω₃ rises, by 0.013 across
the grid. Flat remains the prediction for the trained arms — the decline is small and the
selection-time reading is a different quantity from a training trajectory — but the
direction of that decline is recorded now so it is not read afterwards as having been
predicted.

## Third arm, and the settings the run is made at

Written before any GRPO step has run. It adds an arm, changes seven training settings, and
records what those changes cost the stop rule. The two arms above are unchanged in ω, in
initialization and in what they are predicted to do, and predictions 1–4 for RLSF-Judge are
scored exactly as written.

### The third arm

| Arm | Cell | ω as declared | ω at unit norm (bleu, kiwi, judge) | ω₃² |
|---|---|---|---|---|
| RL-Metric | `w3_0.0` | (1, 1, 0) | (0.7071, 0.7071, 0.0) | 0 |
| RLSF-Judge | `w3_2.0` | (1, 1, 2) | (0.4082, 0.4082, 0.8165) | 2/3 |
| RLSF-Judge-High | `w3_6.0` | (1, 1, 6) | (0.1622, 0.1622, 0.9733) | 36/38 |

The two-arm design was chosen because the argument needs the judge term present and absent
rather than a dose-response curve. That is still true of the argument, and it is not true of
the outcome the arms are being run to locate. Two arms can show that a rubric term moves the
policy; they cannot separate *how far it moves* from *how much semantic quality that costs*,
because with two points every trade-off is a line. `w3_6.0` is the third point. It is
predicted, in the same direction and for the same reason as `w3_2.0`, to raise training-time Φ
and `marker_rate` z further and to worsen held-out register distance further, and — unlike
`w3_2.0`, for which prediction 3 says Kiwi is flat — to show adequacy falling: at ω₃² = 36/38
the metric terms hold 1/19 of the reward's variance and are no longer a guardrail in any
meaningful sense. If instead `w3_6.0` matches `w3_2.0` on both style and adequacy, the
reward has saturated and the ω axis is not the axis that governs the outcome.

`w3_6.0` did not exist when the pre-registered ω grid was ranked on the cached pool, so it has
no selection-rule ranking and is not a candidate for selection. It is a training arm only.
Re-running `manage.py rlsf_omega` over the five cells re-ranks a cached pool and buys no new
judge calls; the four-cell ranking recorded in the addendum above is the one the selection
rule was pre-registered over and remains the one reported as selection.

### Settings changed

| Setting | Was | Is | Why |
|---|---|---|---|
| `reference.beta` | 0.05 | 0.01 | the KL anchor is a frozen adapter, not the base; at 0.05 the "uninformative outcome" clause below was the likelier reading of a null result than any statement about the reward |
| `rollout.prompts_per_step` | 16 | 8 | GPU budget for three arms rather than two |
| `train.per_device_train_batch_size` | 4 | 1 | memory; `grpo_args` derives grad accum 32, and `loss_type: dapo` normalizes by the generation batch's token count, so the gradient averages the same 8 groups at any micro-batch. 1 is the value the 22.04 GiB measurement was taken at |
| `train.lr_scheduler_type` | unset, so HF's `linear` | `constant` | a fixed rollout budget, not a fixed token budget; see below |
| `generator.max_tokens` | 256 | 192 | still 3× the dev reference p95 of 61 tokens, so nothing feasible is truncated |
| `train.save_every_rollouts` | 10 | 25 | 12 checkpoints over 300 rollouts |
| rollouts per arm | 500 | 300 | three arms inside the same envelope |

β is the one of these that changes what is being tested rather than what it costs. Lowering it
loosens the constraint holding the policy at its initialization, which makes movement more
likely in both arms — including in RL-Metric, which is predicted not to move on `marker_rate`.
That prediction is therefore now made under a weaker anchor and is correspondingly easier to
falsify. This is stated in advance so a `w3_0.0` trip cannot afterwards be attributed to β.

The schedule is the second of these that changes training dynamics rather than cost, and it is
recorded here on the same argument. It was never declared: `grpo_args` set `learning_rate` and
left the schedule to TRL, which takes HF's default of linear decay to zero. The 50-rollout smoke
therefore stepped at 9.9e-07 at its first optimizer step and 3e-08 at its last, and
`adapter_delta.rel` sat at 0.0021 for its final ten rollouts. Over 300 rollouts the same schedule
would spend the last hundred on steps too small to move an adapter, which makes "300 rollouts" a
count of generations rather than of learning. `constant` holds every rollout at 1e-6, so the
budget means what it says. There is no warmup: the smoke's opening rollouts ran within 1% of the
peak rate without instability, which is the evidence that none is needed, and a ramp would spend
part of a fixed budget below the rate the rest of it uses. All three arms share the schedule.
Two consequences are stated in advance rather than discovered afterwards. Movement in any arm is
now made under a step size that does not decay, so a `w3_0.0` trip cannot be attributed to the
schedule any more than to β. And a constant rate does not anneal, so the final adapter is
wherever the last rollout left the policy rather than a settled point; this is what
`save_every_rollouts: 25` and the dev-slice checkpoint selection below are for, and it is now
their reason rather than an extra.

The "uninformative outcome" clause stands unchanged: if neither arm moves, the run is reported
as a null run of the optimizer and not as evidence about the reward. `adapter_delta_rel` in the
step log is what distinguishes the two, and it is read before any statement about style is
made.

### What the geometry costs the stop rule

The rule is unchanged: `baseline_steps: 20`, `window: 5`, `k_sigma: 4.0`, `min_delta: 0.23`.
Its operating characteristic is not. At the smoke's measured per-prompt error, halving the
rollout raises the per-step standard error by √2 and the band with it, from 0.660 to 0.933
centroid standard deviations (`manage.py drift_oc`, 1,000 runs, seed 42):

| Geometry | Band | Null (false alarm) | Step +0.23 | Step +0.35 |
|---|---:|---:|---:|---:|
| 16 prompts, 500 steps (as pre-registered) | 0.660 | 1.3 % | 45.4 % | 88.8 % |
| 8 prompts, 500 steps | 0.933 | 2.1 % | 30.4 % | 64.7 % |
| 8 prompts, 300 steps (the run) | 0.933 | 1.5 % | 21.4 % | 49.7 % |

The false-alarm rate is what the operating point was chosen on and it holds. Power does not:
against a +0.35 step the rule now fires about half the time rather than nine times in ten. The
rule was not retuned to recover it, because retuning a stop rule to a power target after
choosing the geometry that lost it is the kind of edit this document exists to prevent. The
consequence is recorded instead: **a run that does not halt is now much weaker evidence that it
did not drift than it was under the pre-registered geometry.** Drift is measured in the val
table and in the step log's `marker_rate` z either way; the monitor is a safety halt, not the
measurement. The three commitments above are unchanged — a halt is a result, the rule is not
edited after a trip, and a `w3_0.0` trip is reported as evidence against the attribution of
drift to the judge term before it is reported as a false alarm.

The `ramp +0.50` scenario is omitted from the table above deliberately: `src/rlsf/drift_oc.py`
hard-codes the ramp to reach its endpoint at step 500, so at 300 steps the simulated drift only
reaches +0.28 and the resulting figure measures the simulator, not the rule.

### Disclosure: the pool reading taken after these predictions were written

`manage.py rlsf_omega` was re-run over the five cells after the predictions above were fixed,
to a scratch path; `outputs/rlsf/pool_omega.json` still holds the four-cell selection the
pre-registration cites and was not overwritten. The reading buys no judge calls — it re-argmaxes
completions already paid for.

At G = 4, register distance rises monotonically across the grid: `w3_0.0` 0.5581, `w3_0.5`
0.5995, `w3_1.0` 0.6195, `w3_2.0` 0.6645, `w3_6.0` 0.6909. The `marker_rate` shift of each
cell's picks over their own groups rises with it, `w3_6.0` at +0.187 [+0.125, +0.248] between
`w3_2.0` (+0.158) and `judge_only` (+0.206). The selection rule still ranks `w3_0.0` first and
`w3_6.0` now last.

This is consistent with the direction predicted for `w3_6.0` and is recorded so it is not later
read as having been predicted from the training run. It is a selection-time reading over
best-of-N picks, which is a different quantity from a training trajectory: it says the reward
prefers those completions, not that a policy optimized against it arrives there.

### Everything else held

Same frozen PEFT initialization, same reference adapter, same prompt assembly as the PEFT
condition, rollout at T = 1.0 / top-p 0.95, G = 4, `loss_type: dapo`, lr 1e-6 constant, μ = 4, ε = 0.2,
seed 42, length band [0.5, 2.0] with `on_violation: floor`, `normalization: group_z`,
`scale_rewards: none`, unquantized BF16 policy, and the frozen training rubric
`prompts/judge_train.txt` digest-verified on load. All three arms are trained under one set of
these, and `steps_<cell>_manifest.json` records the installed `torch`, `transformers`, `trl`,
`peft`, `accelerate` and CUDA versions with each.

Checkpoint selection is added and is a dev-slice operation: `manage.py rlsf_select` scores each
saved checkpoint on `data/splits/rlsf_dev.jsonl` and ranks on the chrF adequacy band then
held-out register distance, the rule `src/peft/sweep.py` already uses. It carries no judge —
ranking on the rubric the paid arms optimize would be circular, and the evaluation raters are
spent once, on val. Selection reads no val and no test figure, so the seal is untouched.

## Addendum, 2026-08-19: the checkpoint ladder, declared post-hoc

The checkpoint ladder is not in this document. Everything above is about the three arms at their
final adapters, and the four val figures the predictions are scored against are named in *What
will be reported*. Regenerating five checkpoints per arm on val and reading them against optimizer
step is a different measurement, decided after those figures existed. **T1–T4 below are
exploratory and post-hoc**, and are labelled so wherever they are reported. Nothing here amends a
commitment above, and no prediction above is scored against them.

The disclosure that matters is the ordering, and it is unflattering. Every quantity T1–T4 concern
had already been read when they were written: held-out distance, `marker_rate` z and COMET were in
`results/heldout_traj_val.json` from `81ce2b4`, and corpus chrF and BLEU per checkpoint were in the
scoring notebook's stored output from the same commit. These are therefore descriptions of numbers
in hand, written down so that they are stated once and reported as stated, rather than predictions.
They carry none of the evidential weight of the four predictions above.

**T1** Within an arm, held-out register distance grows with optimizer step, and the three growth
rates rank in ω₃ order.

**T2** Signed z on `marker_rate` rises over the same steps, on the same ordering.

**T3** Reference-based adequacy does not fall along the ladder: the per-doubling slope of COMET,
chrF and BLEU is positive or indistinguishable from zero in every arm.

**T4** T1 and T3 hold in the same arm over the same steps. Register drift is not paid for out of
measured adequacy, so along this ladder the two axes are decoupled.

T4 is the only one of the four that is a claim rather than a description, and it is the weakest of
them, because "does not fall" is an absence and an absence is not evidence of decoupling — it is
the failure to detect coupling at this sample size. Three metrics of two different kinds agreeing
on that null is worth more than one, and is still a null. What would contradict T4: a significant
fall on any of the three metrics in an arm whose register distance climbs.

T1 and T2 are also weaker than they look. Five checkpoints is five points on a line fitted in
log-2 of the step, the ladder was saved at `save_every_rollouts: 25` rather than chosen, and the
matched-checkpoint noise floor measured in the scoring notebook — 0.0136 in held-out distance —
bounds what any single step of the ladder can be said to show. The slope over the whole ladder is
the readable quantity; a step-to-step move is not.

The seal is untouched. The ladder is generated and scored on `val.jsonl` only, and the checkpoint
selection the arms are reported at was made on the dev slice before any of this was read.
