# Engineering and Decision Log

This document records the pipeline stages, architectural decisions, implementation
changes, and rationale for the Style-Aware Machine Translation project. Entries are
ordered in reverse chronological order, most recent first.

Each entry answers four questions: what changed, why the change was made, how the
result can be reproduced, and what limitations or risks should be monitored. This log
is the authoritative record of why the dataset, pipeline, and generated artifacts have
their current form. It is an engineering record, not a results document: numbers appear
here to support decisions, and the reported figures for the thesis are those in
`README.md` and the files under `results/`.

## Conventions

* Dates are written in absolute form: `YYYY-MM-DD`. An entry's date is the date of the
  work it records, taken from the git history of the commits it cites. Where an entry
  covers a range of days, the range is stated in the entry.
* Entries are ordered strictly by that date, most recent first.
* Section headings are drawn from a fixed set: **Summary**, **What changed**,
  **Rationale**, **Verification**, **Reproduction**, **Limitations and risks**. Entries
  may add descriptive headings for sub-analyses, and early entries may omit sections
  that did not apply.
* Code references use the format `path:line`. Line numbers are valid as of the entry's
  date and are not updated when later commits move the code.
* Commands are recorded as exact command-line invocations. Fenced blocks that list
  files or artifact paths rather than commands are not shell input.
* When a pipeline stage writes data, the entry identifies the artifact paths and
  describes how integrity was verified, for example through hashes or manifests.

### Notation

* **Φ** is the evaluation-time LLM-as-Judge register-fidelity score: the mean over
  segments of a 1–5 rubric rating of a candidate translation against the authorized
  reference (`prompts/judge_eval.txt`; see the 2026-07-05 entry). Higher is better.
  It is distinct from the training-time reward judge used by RLSF.
* **`stylo_dist`** is the undirected standardized Euclidean distance from a condition's
  mean feature vector to the target-register centroid; **`register_fit`** is the
  directional band-pass distance introduced on 2026-07-18. Lower is better for both.
* **`eval_loss`** is completion-only token-level cross-entropy on `val.jsonl`.

### Stage and phase labels

Two labelling schemes appear in the titles. Entries of 2026-06-12 use *Phase N*;
entries from 2026-07-03 onward use *Stage N*; the 2026-06-08 entries and several later
ones use neither. The schemes are historical artifacts of the project and are not one
numbering — *Phase 2* (stylometrics) and *Stage 2* are unrelated. **Stage 2 has no
entry**: the demonstration-ordering flag (commit `4e7647d`, 2026-07-04, PR #4
`feat/afsp-stage2-fixes`) is referenced only in passing by the Stage 3 entry.

### Provenance of these entries

The four entries dated 2026-07-18, 2026-07-23, 2026-07-25, and 2026-07-31 were written
**retrospectively on 2026-07-31**, reconstructed from committed artifacts, configs, and
git history rather than at the time of each change. The 2026-07-24 PEFT-implementation
entry and all entries before 2026-07-18 were written at the time of the change. Where a
rationale was not recorded in the repository it is marked as unrecorded rather than
inferred.

### Known documentation gaps

Changes visible in the git history that no entry records. They are listed here so the
absence is deliberate rather than an oversight; none has a recorded rationale.

* `270b96c` (2026-07-16) re-cut the sweep grid to k ∈ {1, 2, 4, 8, 16} and added the
  zero-shot anchor row. The 2026-07-18 entry describes the grid it replaced, not this
  intermediate grid.
* `568ba73` (2026-07-19) restored λ = 0.5 to `DEFAULT_LAMBDAS`, completing the six-value
  λ axis that the 2026-07-23 run used.
* `446eca8` (2026-07-18) removed the lazy `comet` import from `src/eval/comet.py`,
  which the 2026-07-13 entry had relied on. See that entry's limitations.
* The trained adapter's output location moved from `outputs/peft/adapter`
  (2026-07-24 entry) to `models/peft_lora_r{r}_lr{lr}/checkpoint-{step}`
  (`configs/peft_sweep.yaml`, `sweep.output_base`). No entry records the move.
* The judge provider switch of 2026-07-20 has no recorded rationale; this is noted in
  the 2026-07-23 entry.

These gaps were open as of the audit. Nothing in this list has been closed since.

### Audit

This log was audited for internal consistency against the committed artifacts, configs,
and git history on **2026-08-01**. Corrections made in that pass are marked inline as
*Correction (2026-08-01)*; they amend the claims of earlier entries but do not restate
their history.

---

## 2026-08-07 — RLSF dev slice carved from train.jsonl

### Summary

`manage.py rlsf_dev --target 500 --seed 42` was run. It carved 499 segments across four
works out of `train.jsonl` as the RLSF weight-selection slice, leaving 10,361 for PPO
updates. This is a data operation only: no model was loaded, no judge call made, no
reward computed. The reported splits are untouched and no figure changes.

### What changed

Three new files under `data/splits/`:

| File | Segments | sha256 |
|---|---:|---|
| `rlsf_dev.jsonl` | 499 | `45b45e40b03ea3c3…` |
| `rlsf_train.jsonl` | 10,361 | `3c2dbc43d11311c1…` |
| `rlsf_dev_manifest.json` | — | selection record |

The dev works are `Lawh-i-Burhan` (117), `Lawh-i-Dunya` (183),
`Lawh-i-Siyyid-i-Mihdiy-i-Dahaji` (77) and `Suriy-i-Vafa` (122). Selection is whole-work
and deterministic: the subset minimising `|total − target|`, ties toward fewer works then
lexicographic. At target 500 the best achievable is 499. The 193 unlabelled segments are
forced to the remainder rather than being eligible for the slice.

`configs/rlsf.yaml` and the README command block were amended, both of which said these
files did not exist.

### Rationale

Reward weights are a selection decision, so they need a partition that is neither the one
used for updates nor the one used for reporting. Validation is the reported split for
every other condition and using it here would make the RLSF row's figures selection-
contaminated relative to the rest of the table. Carving from train instead keeps
validation clean.

The slice is whole-work for the same reason the top-level split is: partitioning by row
would leak an author's stylistic habits across the boundary, and register is the thing
being optimized.

### Verification

```bash
.venv/bin/python manage.py rlsf_dev --target 500 --seed 42 --dry_run   # plan
.venv/bin/python manage.py rlsf_dev --target 500 --seed 42
```

Checked after the write:

* `train.jsonl`, `val.jsonl` and `test.jsonl` still hash to the digests in
  `data/splits/hashes.json`. The frozen split was read, not modified.
* `rlsf_dev + rlsf_train` equals `train.jsonl` exactly as a multiset over
  `(input, output)`, and record order is preserved within each part.
* Input overlap between the two parts is 0, and no dev work appears in the remainder.
* All three digests in `rlsf_dev_manifest.json` match the files on disk.

### Limitations and risks

* The slice is not unseen by the model. `models/peft_lora_r32_lr2e-4/checkpoint-1358`,
  which RLSF initializes from, was trained on all of `train.jsonl` including these four
  works. Dev-slice figures select reward weights; they do not measure generalization and
  are never reported as a result.
* `results/stylometrics_centroid.json` was built over all 10,860 train targets, these
  works included, so the `z` field in the step log is measured against a centroid the
  slice contributed to. It is a training diagnostic, not evidence.
* 499 segments is small for ranking four grid cells. What separation it can resolve is
  unquantified, and no interval has been computed on it. If the cells land close, the
  slice cannot break the tie and saying so is the correct outcome rather than picking the
  leader.
* Removing these four works changes what PPO trains on relative to what PEFT trained on.
  The two arms are no longer matched on training data, which is a difference between
  conditions that the RQ1 comparison has to carry.

---

## 2026-08-07 — RLSF condition config written; reward judge selected; caps deliberately null

### Summary

`configs/rlsf.yaml` (new) specifies the RLSF arm: policy init, rollout sampling, reward
weights and the RQ3 weight grid, spend caps, and the training-time judge block.
`src/rlsf/config.py` (new) loads and validates it. **The config is not runnable as
committed**: every spend cap is `null` and the loader raises until they are set, because
budget rule 1 forbids starting a paid run before its volume is recorded and the declared
cap is still untranscribed from `docs/proposal.pdf`. **Implemented and statically
verified only: no PPO step, no judge call, no artefact written.** No reported figure
changes. RLSF remains an empty row in the results table.

### What changed

* `configs/rlsf.yaml` (new) — the condition config. Policy initializes from the frozen
  PEFT cell `models/peft_lora_r32_lr2e-4/checkpoint-1358`, which is also the KL reference.
* `src/rlsf/config.py` (new) — `load_config` (validating; `require_caps=False` inspects
  without spending), `assert_caps_declared`, `assert_group_size_within_ceiling`,
  `reward_config`, `grid_reward_configs`, `worst_case_judge_calls`, `priced_worst_case`.
* `docs/budget.md` — the RLSF arm's authorized envelope, the section's own outstanding
  requirement since 2026-08-05.
* `tests/test_rlsf_config.py` (new) — 18 cases (157 in the suite).

Reward-block field names match `src.rlsf.reward.RewardConfig` exactly, so
`reward_config` is a filtered splat rather than a translation layer; the `kiwi:`
sub-block is ignored by that filter and read by the scorer.

### Rationale

**The reward judge is `gpt-4o-mini`, model-distinct from both evaluation raters.** Φ_A is
`claude-haiku-4-5` and Φ_B is `gpt-5.6-terra`. Training against either spends that rater
on the single condition that most needs two independent ones — RLSF is the only arm
optimized against a judge, so it is the arm whose Φ most needs a rater it was not trained
against. A third model keeps both. It is also the only candidate honouring `temperature:
0` *and* `seed: 42`: Haiku exposes no seed and Terra treats it as best-effort. That
matters more at training time than at evaluation time, because under group normalization
a rater that randomly flips a 3 to a 4 inverts a sample's advantage sign — rater noise
becomes gradient noise, not just measurement noise.

A local Qwen judge was excluded outright rather than on cost: the policy is
`Qwen2.5-7B-Instruct`, so a Qwen judge is self-preference bias by construction — the same
objection that makes `commercial_haiku`'s Φ inadmissible as anything but a labelled
external reference.

**The caps are null on purpose.** RLSF spends one judge call per sampled completion, so
an uncapped run has no upper bound and a step-count typo is a spend event. Writing a
plausible number here would have substituted an estimate for a pre-registered commitment;
the cap in `docs/proposal.pdf` is the commitment, and it still cannot be read in this
environment. So the keys exist, hold `null`, and the loader raises with the rule-1
reasoning in the message. The envelope is nonetheless priced in `docs/budget.md`, because
recording the arithmetic and authorising the spend are separate acts and only the first
can happen today.

**The ceiling is priced at group size 8 while the config operates at 4.** Judge calls
scale one-for-one with group size, so raising it after authorisation would be a widening
under budget rule 3. Pricing the envelope at 8 makes that headroom part of the original
authorisation instead. `assert_group_size_within_ceiling` enforces the boundary.

**Rollout sampling is deliberately not the locked greedy decoding.** `temperature: 1.0`,
`top_p: 0.95`. At temperature 0 every sample in a group is identical, the group's reward
sd is 0, and `group_normalize` (`reward.py:115`) returns all zeros — the reward would be
identically flat and no update informative. The locked greedy setting still governs the
reported inference pass that scores this arm; the matched-decoding requirement is a
property of evaluation, not of rollout.

**The weight grid varies ω₃ against a fixed adequacy pair**, per the README's RLSF
section, with a `w_judge: 0.0` ablation cell so the style term's contribution is
identifiable rather than inferred. Grid cells are short proxy runs on the dev slice under
the §7 selection protocol — sweep, verify, freeze — and a cell is never itself a reported
result.

### Verification

No PPO step, no judge call, no network, no GPU.

```bash
.venv/bin/python -m ruff check src tests manage.py    # All checks passed
.venv/bin/python -m pytest tests/ -q                  # 157 passed
```

The new cases assert the properties that would otherwise be prose: the committed config
refuses to load; every undeclared cap is named, not just the first; a non-positive cap is
rejected; a group size above the ceiling raises citing rule 3; worst-case volume defaults
to the ceiling rather than the operative group size (44,800 against 22,400); the grid
varies only ω₃ and inherits the base feasibility band; `template_file` is the training
rubric and not the evaluation one; and the reward judge's model appears in neither
`configs/judge_eval.yaml` nor `configs/judge_eval_gpt.yaml`.

### Reproduction

```bash
.venv/bin/python -m pytest tests/test_rlsf_config.py -q
python -c "from src.rlsf.config import load_config; load_config()"   # must raise
```

### Limitations and risks

* **Nothing here has been run, and no trainer exists.** `src/rlsf/` holds the reward, the
  COMET-Kiwi worker, and now the config loader. There is no PPO loop, no `rlsf` entry in
  `manage.py`, and no `infer` support for the condition. This config specifies an arm that
  cannot yet execute.
* **The data files do not exist.** `data/splits/rlsf_train.jsonl` and `rlsf_dev.jsonl` are
  produced by `manage.py rlsf_dev`, which has not been run. Its manifest records that the
  dev slice is held out from PPO updates *only* — the PEFT checkpoint RLSF initializes
  from was trained on all of `train.jsonl`, this slice included, so the slice is not
  unseen by the model and dev-slice figures select weights rather than measure them.
* **The cost figures are estimates over assumed token counts.** The $5.11–$6.45 ceiling
  spans 600–800 prompt tokens; at the 400-token assumption it is ~$2.96. The 50-call pilot
  measures the real rate. Either way the arm is under a dollar of spend to date, so
  dollars are not the binding constraint — GPU hours and in-loop judge latency are.
* **Family adjacency, logged.** `gpt-4o-mini` shares a provider family with Φ_B
  (`gpt-5.6-terra`). This is weaker than model-identical contamination but not nothing:
  Φ_B is family-adjacent rather than fully clean for the RLSF row specifically. Added to
  the threats register.
* **Group size 4 is statistically thin.** `group_normalize` returns all zeros when fewer
  than 2 samples in a group are length-feasible (`reward.py:110`), so at G = 4 a group
  with three length-band violations contributes no gradient signal at all. How often that
  happens is unmeasured; the step log's `n_feasible` field is what will show it, and the
  ceiling permits raising to 8 without re-authorisation if it does.
* **`learning_rate: 1e-6`, `kl_coef: 0.05`, `clip_range: 0.2`, `ppo_epochs: 4` are
  conventional defaults, not tuned values.** They have no support from this project's
  data and should not be described as selected.

---

## 2026-08-07 — Training-time judge rubric written and frozen with a recorded digest

### Summary

`prompts/judge_train.txt` — the Φ term of the RLSF reward, referenced by
`src/rlsf/reward.py:13` since the first RLSF stage but never written — now exists, is
frozen, and its sha256 is recorded in a new freeze manifest `prompts/hashes.json`
**before** any reward call reads it. `src/rlsf/reward.py:load_train_template` verifies
the file against that record on every load and refuses a mismatch. **Implemented and
statically verified only: no judge call has been made, no RLSF training has run, and
nothing under `results/` or `outputs/` was written.** No reported figure changes, and
the rubric's behaviour as a reward signal is entirely unmeasured.

### What changed

* `prompts/judge_train.txt` (new) — the training-time rubric. Same construct as the
  evaluation rubric (register fidelity, 1–5, scored against the authorized reference)
  and the same `source` / `reference` / `prediction` fields `build_prompt` fills, but
  independently worded, with its own band anchors and its own output contract.
  sha256 `8eaa11ff341c86ca…`, digest `8eaa11ff341c86ca`.
* `prompts/hashes.json` (new) — freeze record for both rubrics, named and shaped after
  `data/splits/hashes.json`. Records role, freeze date, full sha256, and the 16-character
  `digest` that `src/eval/judge.py:74 template_digest` computes and writes into each judge
  artefact as `template_sha256`.
* `src/rlsf/reward.py:96 frozen_digest` (new) — reads a rubric's recorded digest; raises
  if the manifest is absent or the rubric has no entry in it.
* `src/rlsf/reward.py:123 load_train_template` — now takes `hashes_path`, hashes the file
  it read, and raises `ValueError` if it differs from the freeze record. The
  missing-file `FileNotFoundError` and its circularity message are unchanged.
* `tests/test_rlsf_reward.py` — 6 new cases (43 in the file, 139 in the suite).

The frozen eval rubric is untouched: `prompts/judge_eval.txt` still hashes to
`ffd6dad41acb0512…`, the digest rater B's 2026-08-05 artefacts carry, so recording it
here is a transcription of the current file, not a change to it.

### Rationale

**Why the digest is recorded at freeze time rather than at run time.** The 2026-08-05
agreement pass reports `template_verified: false` because rater A's artefacts predate
per-condition digest recording: that both raters read the same rubric rests on the
configs, not on the artefacts, and is closable only by re-spending on Φ_A. That gap
opened because the rubric was frozen and the digest was recorded at two different times,
with runs in between. Recording the digest as part of the freeze, before the first
consumer exists, removes the window in which that can recur for the RLSF reward.

**Why the loader verifies rather than logs.** A recorded hash that nothing checks is the
same class of evidence as a config assertion. Verification on load means an edited rubric
fails loudly at the start of training instead of producing rewards that are silently
incomparable with those already collected.

**Why a separate rubric at all.** §Judge circularity mitigation in `README.md` and the
2026-07-05 entry commit the study to two fixed templates, training-time and
evaluation-time. Scoring the reward with the same rubric that scores the reported Φ would
make the Φ result circular — the policy would be optimized directly against its own
evaluation instrument. `load_train_template`'s error message has carried that reasoning
since the first RLSF stage; this entry supplies the template it was pointing at.

Three rubric decisions are load-bearing and revisable **only until the first RLSF run
consumes them**:

1. **Degenerate output is floored at 1.** Empty, truncated, self-repeating, non-English,
   and source-script output score 1 regardless of any elevated wording, as does archaic
   vocabulary applied ungrammatically. The evaluation rubric has no such clause because it
   never sees policy samples mid-training; a reward that does not floor these is a reward
   that pays for scattering `Thou` and `hath` into modern prose.
2. **A ≤15-word justification precedes the verdict.** It costs completion tokens on every
   sample. The alternative — a bare `Score: N` — is cheaper, but rater B compressed to a
   0.10 range across six conditions under `reasoning_effort: none`, and a judge component
   with no within-group variance is silently zeroed by `group_normalize` (sd < 1e-9 → all
   zeros), which would drop the style term from the reward without any error surfacing.
3. **Register only; adequacy is explicitly out of scope.** Adequacy enters the reward
   through the BLEU/chrF and COMET-Kiwi terms, and the length band gates feasibility.
   Double-counting it in the judge term would confound ω₃ with ω₁ and ω₂ and make the RQ3
   weight sweep uninterpretable.

### Verification

No judge calls, no network, no GPU. Nothing was run against a model.

```bash
.venv/bin/python -m ruff check src tests manage.py    # All checks passed
.venv/bin/python -m pytest tests/ -q                  # 139 passed
sha256sum prompts/judge_eval.txt prompts/judge_train.txt
```

The new cases cover: the committed rubric loads and verifies against the committed
manifest; it fills through `build_prompt` and is brace-safe against stray braces in
inserted text; the two rubrics' digests differ (the circularity control, asserted as a
test rather than as prose); an edited rubric raises; a rubric absent from the manifest
raises; a missing manifest raises.

`prompts/judge_eval.txt` hashing to `ffd6dad41acb0512…` was checked against the digest
recorded in rater B's artefacts, confirming the eval rubric has not drifted since the
2026-08-05 pass.

### Reproduction

```bash
sha256sum prompts/judge_eval.txt prompts/judge_train.txt   # must match prompts/hashes.json
.venv/bin/python -m pytest tests/test_rlsf_reward.py -q
```

### Limitations and risks

* **This does not close the `template_verified: false` gap retrospectively.** That gap is
  a property of Φ_A's existing artefacts and remains open in the threats register;
  only re-spending on Φ_A closes it. What this entry closes is the possibility of the
  same gap opening for the RLSF reward.
* **The rubric is unvalidated.** No call has been made against it. Whether it
  discriminates register at all, whether it discriminates it *within* a PPO group of
  samples from one policy — a much narrower spread than the seven conditions the eval
  rubric separates — and whether it is gameable by the policy are all unmeasured. A
  `--limit N` pilot on existing outputs is the cheapest first check and has not been run.
* **The freeze is revisable now and expensive later.** Once reward values collected under
  this rubric exist, changing it invalidates them. Any revision to the three decisions
  above should happen before the first run, not after.
* **Cost is unestimated.** The rubric is ~330 tokens of prompt plus source, reference, and
  candidate, and one call per sample. Neither the PPO step cap, batch cap, nor judge spend
  cap is fixed in `docs/budget.md` yet, so the total is still undeclared — that
  precondition is unchanged by this entry.
* **The training judge model is unselected.** No `configs/rlsf.yaml` exists; which model
  reads this rubric, and whether it is the same family as the evaluation judge, is an open
  decision with its own circularity implications.

---

## 2026-08-05 — Cross-family second judge run: four of five primary Φ contrasts are rater-dependent

### Summary

The cross-family confirmation pass was executed. `gpt-5.6-terra` scored all seven
conditions on val through the OpenAI Batch API — 9,261 calls, 9,132 segments scored,
**$6.12** — against the same frozen `prompts/judge_eval.txt` the primary rater reads.
The result is the one the threats register has been waiting on, and it is not
reassuring: the two raters order the systems similarly at the top but **four of the five
pre-specified primary contrasts on Φ do not replicate under the second rater**, two of
them with the sign reversed. The central negative result *does* replicate — AFSP-full
still fails to separate from `knn_fewshot` under both raters. No reported figure in
`README.md` changes, because Φ_A is unchanged; what changes is how much weight Φ_A alone
can carry.

### What changed

Relative to the implementation entry below (same date), three things moved before the run
could happen:

* **Model identity resolved.** `configs/judge_eval_gpt.yaml` names `gpt-5.6-terra`, not the
  unconfirmed `gpt-5.6` placeholder, with real `pricing: [2.00, 12.00]` replacing the
  placeholder `[5.00, 30.00]` (commit `7e5ee52`). `reasoning_effort: none` and
  `max_tokens: 256` bound the output side, which the implementation entry flagged as
  un-estimable.
* **Batch transport added** (commit `bb5243e`) — `src/infer/openai_batch.py` (submit /
  poll / collect, `openai_batch.py:78`, `:106`, `:135`) and `src/eval/judge_batch.py`
  (`manage.py judge_batch`), which shards a condition into request files under
  `results/judge_<tag>_<split>_batch/`, records job ids in
  `results/judge_<tag>_<split>_batch_state.json`, and appends into the same segment cache
  the synchronous path writes (`judge_batch.py:59` `resume_point`, `:71`
  `append_results`). This halved the pass: the same 9,236-call session priced
  synchronously is $12.20.
* **`src/eval/judge_ci.py` added** (`manage.py judge_ci`) — per-condition Φ with bootstrap
  interval, score histogram, and a bootstrap rank distribution per rater, on shared
  resample draws (`judge_ci.py:74`). Written for both raters:
  `results/judge_ci_val.json` (A) and `results/judge_ci_gpt_val.json` (B).

Artefacts written: `results/judge_gpt_val.json`, `results/judge_gpt_val_segments/`
(7 conditions + `_meta.json` sentinel), `results/judge_gpt_val_usage.json`,
`results/judge_ci_val.json`, `results/judge_ci_gpt_val.json`,
`results/judge_agreement_gpt_val.json`. `results/judge_val.json` and its segment cache
were **not** touched, which is what `assert_results_identity` and `assert_cache_identity`
exist to guarantee.

Repository hygiene in the same commit range: the 9,261 lines of batch *request payloads*
under `results/judge_gpt_val_batch/` are untracked and gitignored — they are regenerable
from `outputs/` plus the frozen rubric, whereas the job ids, the scores and the usage
record are tracked. The accidental nested clone at `notebooks/Style-Aware-MT` was
committed as a gitlink (`24e81c5`, pointing at the pre-fork origin) and has been removed
from the index with `git rm --cached`; the working directory is untouched.

### Cost

| Item | Figure |
|---|---|
| Calls (cumulative, incl. 25-call pilot) | 9,261 |
| Prompt / completion tokens | 3,796,497 / 386,804 |
| Batch discount | 0.5 |
| **Cost** | **$6.1173** ($6.0996 for the 9,236-call full session) |
| Full-price equivalent | $12.24 |

Recorded in `docs/budget.md`, written today. That file was the outstanding precondition
this entry's predecessor flagged; the declared cap itself still has to be transcribed
from `docs/proposal.pdf`, which cannot be read in the current environment.

### Coverage

9,132 of 9,261 segments returned a parseable rating — 98.6 %, between 1,297 and 1,310 per
condition against 1,323 offered. The 129 unscored segments are a property of rater B only;
Φ_A coverage is unchanged (1,323 everywhere, 1,322 for `knn_fewshot`). Every agreement and
contrast figure below is computed on the intersection of both raters and both conditions,
so each contrast carries its own n (1,275–1,295) and none is padded or imputed.

`template_verified` is **false** in the report: rater A's results predate per-condition
digest recording, so while B's `ffd6dad41acb0512` is recorded, *that both raters read the
same rubric is asserted from the config, not verified from the artefacts*.

### Rater agreement

Pooled over the six study conditions (n = 7,823):

| Quantity | Value |
|---|---|
| Quadratic-weighted κ | 0.384 [0.370, 0.398] |
| Spearman ρ | 0.592 [0.577, 0.607] |
| Exact agreement | 27.8 % |
| Adjacent (±1) agreement | 77.2 % |
| Severity offset (A − B) | −0.950 [−0.968, −0.932] |

Two readings follow. First, **the raters are not interchangeable in absolute terms**:
B sits almost a full rubric point above A on the same segments (Φ_A 2.55–2.79 against
Φ_B 3.61–3.71), so a table averaging the two, or quoting one as a check on the other's
level, would be uninterpretable. A constant offset cannot manufacture or destroy a
contrast, which is why the deliverable is contrast replication rather than κ. Second,
**κ ≈ 0.38 is fair agreement at best** — the two raters place the same segment in the same
rubric band under 28 % of the time. Per-condition κ rises monotonically with how far up
the ladder the condition sits, 0.302 (`zeroshot`) → 0.518 (`peft`), so agreement is worst
exactly where the outputs are worst.

### System ordering

`afsp_full` is the highest-scoring study condition under **both** raters (modal rank 2 of
7 including the reference, p = 0.838 under A and 0.874 under B), which is the one ordering
claim that survives the swap intact. Below it the orderings diverge: A ranks
`afsp_margin` > `knn_fewshot` > `peft` > `random_fewshot` > `zeroshot`, while B ranks
`knn_fewshot` > `afsp_margin` > `zeroshot` > `random_fewshot` > `peft`. **PEFT falls from
5th to last.** Spearman over the six study conditions is ρ = 0.714, p = 0.111 — with six
systems the permutation floor is p = 0.0028, so this neither separates nor could it
plausibly; n = 6 is simply too small to test ordering agreement, and the coefficient is
reported as descriptive.

### Contrast replication

The deliverable. Each of the nine pre-specified contrasts re-run under both raters on one
common segment set, Holm correction applied within each rater's family of nine.

| Contrast | Judge A (haiku) | Judge B (gpt) | Verdict |
|---|---|---|---|
| random few-shot − zero-shot | +0.089, p = .0008 ✓Holm | −0.014, p = .574 | **rater-dependent, sign flips** |
| kNN few-shot − zero-shot | +0.193, p < .001 ✓Holm | +0.027, p = .281 | **rater-dependent** |
| AFSP-margin − zero-shot | +0.212, p < .001 ✓Holm | +0.018, p = .487 | **rater-dependent** |
| AFSP-full − zero-shot | +0.244, p < .001 ✓Holm | +0.055, p = .030 ✗Holm | same sign, both separate uncorrected |
| PEFT − zero-shot | +0.202, p < .001 ✓Holm | −0.027, p = .403 | **rater-dependent, sign flips** |
| AFSP-full − kNN few-shot | +0.046, p = .056 | +0.031, p = .166 | neither separates |
| AFSP-margin − kNN few-shot | +0.013, p = .518 | −0.010, p = .596 | neither separates |
| AFSP-full − AFSP-margin | +0.029, p = .218 | +0.041, p = .059 | neither separates |
| PEFT − AFSP-full | −0.043, p = .169 | −0.097, p = .0018 ✓Holm | **rater-dependent** |

Three consequences, in order of how much they matter:

1. **The "every rung of the ladder separates on Φ" reading is rater-dependent.** Under A,
   all five primary contrasts against the zero-shot floor separate and survive Holm. Under
   B, one of the five separates uncorrected and none survives Holm. Two — random few-shot
   and PEFT — reverse sign. Rater B still recovers the ladder's *direction* for the
   retrieval rungs (all three retrieval contrasts positive), but at a tenth the magnitude
   and with intervals spanning zero. The ladder's separation on COMET, chrF and BLEU is
   untouched by any of this; only the Φ column is affected.
2. **The central negative result replicates.** All three AFSP-internal secondary
   contrasts fail to separate under *both* raters, `afsp_full − knn_fewshot` included
   (A p = .056, B p = .166). The finding that the adaptive layer is not demonstrated on
   validation therefore does not depend on rater identity — the one place where a
   negative result being robust is genuinely useful.
3. **PEFT − AFSP-full is rater-dependent against PEFT.** Under B, PEFT scores 0.097 below
   AFSP-full on Φ and this survives Holm. This does not license "AFSP beats PEFT on
   register": it is one rater of two, against the other rater finding nothing (p = .169),
   and PEFT's significant lead on COMET, chrF and BLEU is unaffected. It is a lead, and it
   points the opposite way from PEFT's `stylo_dist` advantage (0.289, the best of any
   condition) — which is itself an RQ4 disagreement worth following.

### Self-preference of the primary rater

The sharpest reason for this pass was that rater A and the `commercial_haiku` condition are
the same model family. The severity offset makes the effect visible: A sits 0.87–1.09 below
B on the six study conditions, but only **0.641 [0.605, 0.678] below B on
`commercial_haiku`** — that is, A is 0.23–0.45 rubric points *less severe* on its own
family's output, relative to B, than it is on everything else. Exact agreement on that
condition is 42.7 % against 27.8 % pooled, and adjacent agreement 91.4 %, so the two raters
converge there while diverging elsewhere, consistent with the shared-family reading.

But the reference's lead is not an artefact of self-judging. Rater B, a different family,
also places `commercial_haiku` first by a wide margin: Φ_B = 3.981 [3.939, 4.022] against
`afsp_full` at 3.707 [3.661, 3.753]. The correct statement is that the +0.590 Φ_A margin
over PEFT is **inflated** by self-preference, not that it is manufactured by it. This does
not rehabilitate that figure as a register measure — `commercial_haiku` remains a
diagnostic external reference, its `stylo_dist` of 0.556 still points the other way, and
its Φ is still not admissible as a register finding.

### Verification

* `ruff check src tests manage.py` clean; `pytest tests/ -q` **96 passed** (19 added since
  the implementation entry's 77, chiefly `tests/test_judge_batch.py`).
* The 25-call pilot ran first and wrote only the segment cache, per `--limit` semantics;
  its cost is the $0.0177 difference between the session and cumulative figures.
* `results/judge_val.json` compared before and after: unchanged. The tagged paths kept
  Φ_A intact, which was the design requirement.
* Bootstrap parameters identical to every prior inferential run — 10,000 resamples,
  α = 0.05, seed 42, paired at segment level — so the A-side figures in the replication
  table reproduce `results/bootstrap_judge_val.json` where the segment sets coincide.
* Rank distributions in both `judge_ci` reports are computed on shared resample draws, so
  the per-condition ranks within one rater are mutually consistent.

### Reproduction

```bash
python manage.py judge --conditions zeroshot --split val \
    --config configs/judge_eval_gpt.yaml --limit 25          # pilot; cache only
python manage.py judge_batch --conditions zeroshot random_fewshot knn_fewshot \
    afsp_margin afsp_full peft commercial_haiku \
    --split val --config configs/judge_eval_gpt.yaml
python manage.py judge_ci --split val --n_resamples 10000
python manage.py judge_ci --split val --tag gpt --n_resamples 10000
python manage.py judge_agreement --tag_b gpt --split val --n_resamples 10000
```

The batch request payloads are gitignored; `judge_batch` regenerates them from `outputs/`
and the frozen rubric. Re-running the judge stage costs money again — the segment caches
under `results/judge_{val,gpt_val}_segments/` are what make the analysis stages free.

### Limitations and risks

* **This bounds rater dependence; it does not validate Φ.** Neither rater is ground truth.
  κ = 0.384 with a −0.95 offset is equally consistent with two differently-biased raters as
  with one correct and one noisy. What the pass establishes is which conclusions survive
  changing the rater, and the answer is: the AFSP-versus-baseline null, and `afsp_full` at
  the top of the ladder. Not the primary contrasts against the floor.
* **Φ_B is not seed-controlled either.** `seed: 42` is set in the config but documented as
  best-effort by the provider, so a re-run will not reproduce byte-for-byte. The pass
  removes single-family dependence; it does not make Φ reproducible, and the standing
  caution that Φ differences of order 0.05 are within measurement noise still holds — note
  that seven of the nine contrasts under B are smaller than 0.05 in absolute value.
* **Rubric identity is asserted, not verified, for rater A.** Until Φ_A is re-run under a
  build that records `template_sha256`, the claim that both raters read the same rubric
  rests on the configs rather than on the artefacts. This is cheap to close only by
  re-spending on Φ_A, so it stays open.
* **129 segments (1.4 %) have no Φ_B.** They are not random with respect to content —
  a rating is missing because the response did not parse, which correlates with the
  output being unusual. The effect on the reported figures is bounded by 1.4 % of the
  sample but is not zero, and no imputation was performed.
* **The ordering-agreement test is underpowered by construction.** At n = 6 systems the
  Spearman permutation floor is p = 0.0028; ρ = 0.714 could not have separated whatever
  the data. Do not report the orderings as "statistically consistent" — report the
  rankings and let them speak.
* **Rater B's compression is unexplained.** B's ratings span 3.61–3.71 across the six study
  conditions, a range of 0.10, against A's 0.25. Whether that is genuine insensitivity to
  register, a ceiling effect from `reasoning_effort: none`, or the rubric reading
  differently to a different family is not established, and it is the obvious next
  diagnostic: a small pilot at a higher reasoning effort would distinguish the ceiling
  explanation from the others.

## 2026-08-05 — Cross-family second judge implemented (not run)

### Summary

The evaluation-time judge pipeline can now score the same outputs with a second rater
from a different provider family, and a new `judge_agreement` stage measures how far a
Φ-based conclusion depends on rater identity. **Implemented and statically verified
only: no judge call has been made, no OpenAI artifact exists, and nothing under
`results/` or `outputs/` was written.** No reported figure changes.

### What changed

* `src/eval/judge.py` — `--tag` routes a second judge's artifacts to
  `results/judge_<tag>_<split>.json` and `results/judge_<tag>_<split>_segments/`,
  defaulting to the config's `tag:`. Untagged behaviour is unchanged, so the existing
  `judge_val.json` and its segment cache keep their paths.
* `src/eval/judge.py:104` `assert_results_identity` — refuses to write a results file
  already scored by a different judge model unless `--allow_model_overwrite` is passed.
* `src/eval/judge.py:79` `assert_cache_identity` — binds a segment cache to one
  `(model, tag, rubric digest)` via a `_meta.json` sentinel and refuses a mismatch.
* `src/eval/judge.py` — records `judge_tag` and `template_sha256` per condition; writes
  `results/judge_<stem>_usage.json` with per-session and cumulative token/cost figures;
  adds `--limit N` for a diagnostic pilot, which writes the segment cache but
  deliberately **not** the results file.
* `src/eval/judge_agreement.py` (new, `manage.py judge_agreement`) — quadratic-weighted
  κ, Spearman ρ, exact/adjacent rates, the paired severity offset, system-ordering
  comparison, and re-runs of the nine pre-specified contrasts under each rater.
* `configs/judge_eval_gpt.yaml` (new) — the second rater. `template_file` is the same
  frozen `prompts/judge_eval.txt`.
* `src/infer/openai_client.py:35`, `src/infer/run.py:72` — optional `pricing: [in, out]`
  from config, so a model absent from the built-in table reports real cost instead of
  silently costing zero.
* `notebooks/judge_second_pass_colab.ipynb` (new) — the runbook.
* `tests/test_judge_agreement.py` (new) — 22 cases.

### Rationale

Single-judge dependence has been the open construct-validity threat since the judge was
introduced, and the threats register carries it as *cross-family pass unrun*. It is
sharper than a generic reliability concern here: the primary rater `claude-haiku-4-5` is
the same model family as the `commercial_haiku` condition it scores, so the one system
scoring highest on Φ is scored by its own family.

Three design points are load-bearing:

1. **Same frozen rubric.** Both raters read `prompts/judge_eval.txt`. Re-tuning the
   prompt for the second rater would confound rater identity with rubric identity and
   void the comparison; the digest is recorded per condition and `judge_agreement`
   raises if the two disagree.
2. **Contrast replication is the deliverable, not κ.** Two raters can agree poorly
   segment-by-segment and still order the systems identically. `contrast_replication`
   therefore re-runs each pre-specified contrast under both raters on one common segment
   set — the intersection over both conditions and both raters — and labels each
   contrast `both separate`, `neither separates`, or `RATER-DEPENDENT`. Holm correction
   is applied within each rater's family of nine.
3. **The primary Φ must be unclobberable.** Running the second config without a tag
   would have overwritten `results/judge_val.json` in place, and pointing it at the
   existing segment cache would have found the cache complete, made zero calls, and
   returned judge A's scores labelled as judge B's. Both paths now raise.

### Verification

Static only, as nothing was run:

* `ruff check src tests manage.py` clean; `pytest tests/ -q` 77 passed (22 new).
* `quadratic_weighted_kappa` agrees with `sklearn.metrics.cohen_kappa_score(weights=
  'quadratic')` to 12 decimal places over five random 200-point samples. The bootstrap
  is vectorised through `bincount` because a per-resample Python loop at the pooled
  n ≈ 9,000 would be ~10⁸ iterations.
* End-to-end plumbing check at full scale in a scratch directory, with a **synthetic**
  second rater derived by perturbing judge A's scores: 9,260 paired segments, report
  renders, `knn_fewshot`'s n = 1,322 propagates correctly, 11 s at 2,000 resamples.
  Those numbers are plumbing evidence and carry no empirical status whatsoever.
* Notebook pre-flight and cost-estimate cells executed against the real repository;
  the post-run cells executed against the scratch artifacts.

### Reproduction

```bash
python manage.py judge --conditions zeroshot --split val \
    --config configs/judge_eval_gpt.yaml --limit 25          # diagnostic pilot first
python manage.py judge --conditions zeroshot random_fewshot knn_fewshot \
    afsp_margin afsp_full peft commercial_haiku \
    --split val --config configs/judge_eval_gpt.yaml
python manage.py judge_agreement --tag_b gpt --split val --n_resamples 10000
```

### Limitations and risks

* **Unrun, so it establishes nothing yet.** A first execution would establish the
  agreement coefficients, the severity offset, and — the only part bearing on the
  thesis — which pre-specified contrasts survive the rater swap. Until then
  single-judge dependence remains a standing limitation and the threats register entry
  stays open.
* **Cost is the gating risk, and it is not yet known.** The full pass is 7 × 1,323 =
  9,261 calls over ~15.9 M prompt characters (≈4.0–5.3 M input tokens). The output side
  is dominated by hidden reasoning tokens and cannot be estimated from the prompt, so
  the notebook requires a 25-call pilot and an explicit confirmation before the full
  run. `docs/budget.md` is still not written; the cap belongs there first.
* **Model identity unconfirmed.** `configs/judge_eval_gpt.yaml` names `gpt-5.6` with
  placeholder `pricing: [5.00, 30.00]`. Both must be checked against the provider's
  current model list before spending. A wrong price only misreports cost; a wrong model
  id fails the run.
* **Φ_B is not seed-controlled either.** The config sets `seed: 42`, but the provider
  documents it as best-effort, so the second rater does not resolve the
  non-reproducibility of Φ — it only removes the single-family dependence.
* **Agreement does not validate Φ.** A high κ is equally consistent with two
  similarly-biased raters. The result bounds rater dependence; it does not establish
  that either rater measures register correctly.
* **The severity offset must not be averaged away.** If the two raters differ in mean
  severity, Φ_A and Φ_B are not interchangeable in a table of absolute values. A
  constant offset cannot create or destroy a contrast, but a combined "mean Φ" column
  would be uninterpretable.

## 2026-08-04 — Commercial zero-shot reference baseline reported in the README

### Summary

`commercial_haiku` has existed as an artifact since the 2026-07-31 six-condition run and
its paired-bootstrap comparisons were already in
`results/bootstrap_{comet,judge,chrf,bleu}_val.json`, but no figure from it appeared in
`README.md`. It is now reported, in a section of its own, explicitly outside the study
table. No new inference or scoring was run and no existing figure changed.

### What changed

* `README.md:252-303` — new subsection *External reference baseline — commercial
  zero-shot (not a condition of the study)*: system-level table, the three paired
  bootstrap contrasts against `zeroshot`, `peft`, and `afsp_full` with 95 % intervals,
  and three constraints on how it may be read.
* `README.md:206-207` — the study-conditions table now states that it lists study
  conditions only and points to the new subsection.
* `README.md:176-182` — the judge *Open issue* callout now records that the judge and
  this baseline are the same model.

### Reported figures

System level, val, n = 1,323: COMET 0.7185, chrF 45.24, BLEU 18.06, Φ 3.3333
(coverage 1.00), lexical density 0.3928, TTR 0.8241, `stylo_dist` 0.5557. Cost
$0.6316 over 1,323 calls (`outputs/commercial_haiku_val_usage.json`).

Paired bootstrap, 10,000 resamples, α = 0.05, seed 42, segment level. Against `peft`:
COMET +0.0199 [0.0153, 0.0245], Φ +0.5896 [0.5298, 0.6478], chrF +3.76 [3.00, 4.53],
BLEU +1.26 [0.52, 2.03] (p = .0008; the other eleven at p < .001). Against `zeroshot`
and `afsp_full` the direction and significance are the same.

### Rationale

Protocol requires the evidence class of a cited number to be named in the same sentence
and diagnostic runs to stay out of the study tables. `commercial_haiku` is neither a
study condition nor purely diagnostic: it is a labelled external reference baseline, and
`src/eval/metric_agreement.py:43` already treats it that way by reporting `study_only`
and `with_reference` scopes separately. Omitting it from the README entirely was the
wrong resolution — it hid the one system in the artifact set that scores highest on the
primary style metric — so it is reported under its own heading with its class in the
heading itself.

### Verification

Figures recomputed from artifacts rather than copied from prose:
`src.eval.quick.score` for chrF/BLEU, `src.eval.stylometrics.aggregate` +
`distance_to_centroid` for the stylometric columns, `results/comet_val.json` and
`results/judge_val.json` for COMET and Φ, and the `comparisons` arrays in the four
bootstrap files for the intervals. Nothing under `results/` or `outputs/` was written.

### Reproduction

```bash
python manage.py eval --conditions commercial_haiku --split val
python -c "import json;from pathlib import Path;from src.eval._io import load_condition;\
from src.eval.stylometrics import aggregate,distance_to_centroid;\
c=json.loads(Path('results/stylometrics_centroid.json').read_text());\
m=aggregate(load_condition('outputs','commercial_haiku','val')[1])['mean'];\
print(m['lex_density'],m['ttr'],distance_to_centroid(m,c))"
```

### Limitations and risks

* **The Φ figure is self-judged.** The generator is `claude-haiku-4-5`
  (`configs/commercial_haiku_zeroshot.yaml`) and so is the evaluation-time judge
  (`configs/judge_eval.yaml`). Its Φ 3.333 — the highest of any system, +0.590 over PEFT
  — is a model rating its own output, and self-preference bias is a sufficient
  alternative explanation. This does not touch the six study conditions, none of which
  the judge generated, but it makes this one Φ inadmissible as a register claim. The
  README says so at the point of use.
* **The two style measures invert on this system.** Highest Φ of any system, yet
  `stylo_dist` 0.5557 is second-worst overall, behind every few-shot rung and well behind
  PEFT (0.2886). Signed z-deviations: lexical density −0.378 and TTR −0.276, the largest
  magnitudes in the condition set. Admitting it to the RQ4 correlation moves
  ρ(Φ, `stylo_dist`) from −0.657 (n = 6) to −0.250 (n = 7).
* **Not a matched comparison.** Different base model, no adaptation budget, closed
  weights, and a price per call. It cannot enter the single-factor ladder and does not
  bear on RQ1. The adequacy margins are a scope statement about the 7 B open base.
* **Multiplicity.** These three contrasts sit inside the same 14-comparisons-per-metric
  family as the rest of the bootstrap run. Their intervals are wide of zero by margins
  that Holm–Bonferroni would not reach, but the correction status is stated nowhere in
  the new subsection beyond the shared estimator note; if any of these figures is ever
  used to support a claim rather than to bound a baseline, the correction must be applied
  explicitly.

---

## 2026-08-01 — λ dose-response over the AFSP sweep: the rerank has a direction, not a distance

### Summary

The bootstrap entry below left the adaptive layer unshown at one λ. This entry reads
the whole knob instead: all 18 `k × λ` cells already in `outputs/sweep`, scored on the
four centroid features and on COMET, with λ treated as a dose. The motivating claim was
that a monotone fall in `stylo_dist` across six λ values would be stronger evidence than
the single pairwise test at λ = 0.75 that failed (Φ, p = .078).

**`stylo_dist` shows no dose-response, and its per-k trends contradict each other.**
Pooled over all 18 cells, Spearman ρ = −0.10 (p = .68). Per k: ρ = +0.77 (p = .072) at
k = 4 — distance *grows* with λ — ρ = −0.20 (p = .70) at k = 8, and ρ = −0.94 (p = .005)
at k = 16. A knob whose sign flips with k is not a dose-response. The headline metric
does not answer RQ3 affirmatively.

**The signed per-feature view does, and it explains why the norm hides it.** λ moves
three of the four features monotonically, in the same direction, at every k:

| Feature | z at λ=0 (k=16) | ρ vs λ, k=4 | k=8 | k=16 | Reading |
|---|---|---|---|---|---|
| `lex_density` | −0.275 | +0.89 (p=.019) | +0.77 (p=.072) | +0.83 (p=.042) | undershoots → λ **helps** |
| `ttr` | −0.188 | +0.83 (p=.042) | +0.71 (p=.111) | +0.94 (p=.005) | undershoots → λ **helps** |
| `marker_rate` | +0.153 | +0.71 (p=.111) | +0.77 (p=.072) | +0.89 (p=.019) | overshoots → λ **hurts** |
| `root_ttr` | −0.177 | −0.09 (p=.87) | −0.03 (p=.96) | −0.43 (p=.40) | inert |

λ does not move outputs *toward* the target register. It turns register intensity *up*:
denser, more varied, more archaic-marked prose, monotonically in λ. Two of those
movements close a gap and one widens one that is already open — every condition in the
project overshoots `marker_rate`, and λ pushes it further out, the opposite of the
predicted correction. `stylo_dist` is the norm of the four, so the opposing components
partly cancel, which is what makes it flat pooled and sign-flipping per k: at k = 4 the
`marker_rate` overshoot is largest (+0.265 at λ = 0) and dominates, so distance grows;
at k = 16 it starts smallest (+0.153) and has headroom, so the two gains dominate and
distance falls. The k-dependence is a consequence of the starting overshoot, not of the
rerank behaving differently.

**No style–adequacy trade-off at λ = 1.** COMET spans 0.6746–0.6886 across all 18 cells,
and within each k the λ trend is null (ρ = −0.49, +0.37, +0.03; all p > .32). k, not λ,
sets adequacy: k = 16 is above k = 8 above k = 4 at every λ. Pure register rerank with
semantic similarity switched off (λ = 1) costs 0.0023 COMET at k = 4 and 0.0000 at
k = 16 relative to λ = 0. The anticipated trade-off curve is a null; the figure is kept
because the null is the finding.

**Φ at the three judged cells moves with `stylo_dist`, on two usable points.** k = 8
λ = 0.75 → λ = 1: Φ 2.797 → 2.763 while `stylo_dist` worsens 0.371 → 0.411 — same
direction, but the judge CIs (±0.035) overlap and the third judged cell is at a λ
already occupied. This is consistent with agreement, not evidence of it.

**Everything is inside the bootstrap CIs.** Cell-level 95% CIs on `stylo_dist` are
≈±0.045 (e.g. k = 16 λ = 1: 0.3808 [0.3378, 0.4344]); every cell overlaps every other
cell. The trends above are rank correlations over point estimates, which is what makes
the consistency across k — three independent k series agreeing on the sign for three
features — the load-bearing evidence rather than any single cell.

### What changed

* `src/eval/sweep_curves.py` (new), registered as `python manage.py sweep_curves`.
  Computes per-cell aggregates, signed z-deviations, segment bootstrap CIs, and
  Spearman trend tests; writes `results/sweep_curves_val.json` and four figures.
* `results/sweep/comet_val.json` (new): COMET for all 18 cells, scored locally on the
  RTX 4060. `afsp_k16_l0.75` reproduces the 2026-07-23 verify value 0.6871 exactly.
* `docs/figures/` (new): `lambda_stylo_dist.png`, `lambda_z_deviations.png`,
  `lambda_style_adequacy.png`, `lambda_judge_overlay.png`.
* `matplotlib` added to the environment; it was not previously a dependency.

### Rationale

`stylo_dist` was the only stylometric number the sweep reported per cell, and it is an
undirected norm — it cannot distinguish a condition approaching the centroid from one
trading an improvement on one feature against a regression on another. The λ sweep is
exactly the case where that matters, because the rerank turns out to move features in
opposite senses relative to target. The signed decomposition is four numbers the sweep
already had the data for and did not report.

Spearman over six λ points per k is a deliberately weak test (its p floor at n = 6 is
.0028) and is not used here to establish significance. It summarizes direction; the
argument rests on three k series independently agreeing.

### Reproduction

```
python manage.py comet --conditions afsp_k4_l0 afsp_k4_l0.1 afsp_k4_l0.25 afsp_k4_l0.5 \
  afsp_k4_l0.75 afsp_k4_l1 afsp_k8_l0 afsp_k8_l0.1 afsp_k8_l0.25 afsp_k8_l0.5 \
  afsp_k8_l0.75 afsp_k8_l1 afsp_k16_l0 afsp_k16_l0.1 afsp_k16_l0.25 afsp_k16_l0.5 \
  afsp_k16_l0.75 afsp_k16_l1 \
  --split val --out_dir outputs/sweep --results_dir results/sweep --batch_size 8
python manage.py sweep_curves
```

The COMET pass loads `Unbabel/wmt22-comet-da` once and reuses it across all 18 cells;
it fits in the 8 GB card at `--batch_size 8`. `sweep_curves` is CPU-only. The bootstrap
is seeded (`--seed 42`, 2,000 resamples), so the CIs are reproducible.

### Limitations and risks

The frozen `afsp_full` outputs are **not** byte-identical to their sweep cell
`afsp_k8_l0.75`: 5 of 1,323 predictions differ, worth 0.0011 in `stylo_dist`
(0.3698 vs 0.3709). `afsp_margin` and `afsp_k8_l0` are identical on all 1,323. The
frozen condition was therefore re-run rather than copied, and decoding is not fully
deterministic across runs. The scale of that drift is ≈2% of the bootstrap CI
half-width, so it does not affect any conclusion here, but the frozen artifact and the
sweep cell are two samples, not one.

This is validation-split evidence about the *mechanism* of the rerank. It does not
license a claim that AFSP beats its baseline — the bootstrap below still stands, and
no test-split number exists. In particular, the `marker_rate` finding argues that the
current register objective is mis-targeted for this corpus rather than that λ should be
raised; acting on it means revisiting the band-pass target of 2026-07-18, not retuning λ.

Six λ values at three k is 18 points with no replication per cell. The per-k Spearman
values cannot separate a real k × λ interaction from three noisy draws; the shared
direction across k is the only claim the design supports.

---

## 2026-08-01 — Paired bootstrap run: AFSP does not separate from its baseline on val

### Summary

The paired bootstrap that every entry since 2026-07-05 has deferred was run on all
six conditions and all four metrics. It is the gate the 2026-07-31 entry named: until
it ran, no ordering in the results table could be called a difference. Two of the
three observations that entry recorded do not survive it.

**The contribution does not separate from its own baseline.** On no metric does
`afsp_full` beat `knn_fewshot` at α = 0.05:

| Comparison | COMET | Φ | chrF | BLEU |
|---|---|---|---|---|
| `afsp_full` − `knn_fewshot` | +0.001 (p = .43) | +0.043 (p = .078) | +0.20 (p = .41) | +0.45 (p = .065) |
| `afsp_margin` − `knn_fewshot` | −0.002 (p = .25) | +0.016 (p = .42) | −0.07 (p = .70) | −0.13 (p = .49) |
| `afsp_full` − `afsp_margin` | +0.003 (p = .077) | +0.028 (p = .23) | +0.27 (p = .23) | **+0.58 (p = .010)** |
| `afsp_full` − `peft` | −0.013 (p < .001) | +0.047 (p = .12) | −1.06 (p = .003) | −2.13 (p < .001) |

**The PEFT/AFSP "disagreement" was an artifact of reading point estimates.** The
2026-07-31 entry framed it as the objective and subjective register measures ranking
the two adaptation families oppositely, and called it the RQ4 question. The COMET half
is significant; the Φ half is not (+0.047, 95% CI [−0.012, 0.105], p = 0.12). One
measure separates the families and the other fails to — which is a null result on the
Φ side, not a disagreement between measures. RQ4 must be answered by the agreement
analysis the README specifies (pairwise correlation across metrics), not by this pair.

**The prompting ladder's floor is solid.** `zeroshot` → `random_fewshot` →
`knn_fewshot` is significant on all four metrics (Φ +0.087 [0.037, 0.137] and
+0.115 [0.066, 0.163]; COMET +0.016 and +0.020, both p < .001). Retrieval works; the
adaptive layer over it is what remains unshown.

**Detection floor for RLSF.** The Φ CI half-width against `peft` at n = 1,323 is
≈0.058 and the COMET half-width ≈0.005. Since RLSF initializes from the PEFT adapter,
those are the margins an RL gain must clear to be reportable. For scale, the whole
prompting ladder moves Φ by 0.245, so RLSF needs roughly a quarter of the ladder's
total effect to register. Judge test–retest spread (≈0.002, measured in the
2026-07-31 entry) is an order of magnitude below the sampling interval, so sampling
noise, not judge noise, is the binding constraint.

### What changed

`src/eval/bootstrap.py` gained three things needed to make the run reproducible
rather than a terminal session:

* `--out [PATH]` writes the comparison table as JSON — bare `--out` uses
  `results/bootstrap_<metric>_<split>.json`. Each file records the metric, split,
  conditions, baseline, `n_resamples`, `alpha`, `seed`, and per-comparison `n`,
  `mean_a`, `mean_b`, `diff`, `ci_low`, `ci_high`, `p_value`, `significant`. Before
  this the command printed and exited, which is why no `results/bootstrap_*` artifact
  existed despite the command having shipped on 2026-07-05.
* `--pairs A:B` adds arbitrary comparisons to the baseline and `--adjacent` sets.
  The comparison this project exists to make — `afsp_full` against `knn_fewshot` — is
  neither a comparison against the ladder floor nor an adjacent-rung pair, so it was
  unreachable from the CLI as shipped. `_parse_extra_pairs` rejects unknown
  conditions and self-comparisons.
* `_pairs` now deduplicates, so a pair reachable two ways is resampled once.

No scoring code changed; `paired_bootstrap` is untouched.

### Verification

`ruff check src`, `ruff format --check src`, and `python -m compileall -q src` pass.
The four artifacts were written in one pass per metric from the same command; the
`comet`/`judge` runs read the persisted per-segment vectors and the `chrf`/`bleu` runs
recompute from the inference files. Alignment is enforced by the existing
`_assert_aligned` guard, which compares the stored `sources` lists index for index —
it passed for all six conditions, confirming the paired assumption holds.

`n = 1,322` rather than 1,323 on judge pairs involving `knn_fewshot` is the single
unparseable judge segment recorded in the 2026-07-31 entry; `paired_bootstrap` drops
the pair rather than imputing it.

### Reproduction

```bash
CONDS="zeroshot random_fewshot knn_fewshot afsp_margin afsp_full peft"
EXTRA="afsp_full:knn_fewshot afsp_margin:peft afsp_full:peft knn_fewshot:peft random_fewshot:peft"
for m in comet judge chrf bleu; do
    python manage.py bootstrap --metric "$m" --conditions $CONDS --split val \
        --baseline zeroshot --adjacent --pairs $EXTRA --out
done
```

Deterministic at `seed 42`: the resample indices come from
`np.random.default_rng(seed)`, so re-running reproduces every interval exactly.

### Limitations and risks

**The p-values are uncorrected.** The canonical run makes 14 comparisons per metric
over four metrics, 56 tests in total, at α = 0.05 each. The lone significant AFSP
result — `afsp_full` − `afsp_margin` on BLEU, p = .010 — would not survive
Holm–Bonferroni over that family, and BLEU is the weakest of the four metrics. It is
a lead worth powering up, not a finding. A pre-registered comparison family should be
declared before the test-split run so the correction is decided in advance rather than
after seeing which comparisons came close.

Two near misses (Φ p = .078 and BLEU p = .065 for `afsp_full` − `knn_fewshot`) mean
the null result is a failure to detect, not a demonstration of no effect. At this
effect size the val split is underpowered; the honest options are to report the
intervals as they stand, or to increase power by a route that does not touch the
sealed test split — repeated judge passes to average down judge variance, or a
larger exemplar-selection effect.

The chrF and BLEU intervals resample **sentence-level** scores, whereas the results
table reports corpus-level chrF and BLEU. The two are not the same statistic, so a
CI here does not bound the corpus figure in the table. COMET and Φ have no such gap:
both are means over per-segment scores in both places.

`stylo_dist` is not bootstrappable through this CLI. It is a distance between a
condition's *mean* feature vector and the centroid, not a mean of per-segment scores,
so the monotone ladder in the results table (0.652 → 0.370) is still descriptive
only. Reporting it as an ordering requires either a per-segment formulation or a
bootstrap over the feature-vector statistic itself.

The judge remains a single non-reproducible model. These intervals quantify sampling
uncertainty at fixed judge output; they do not include the judge's own re-sampling
variance, so the true uncertainty on any Φ difference is wider than the interval
printed here.

---

## 2026-07-31 — PEFT frozen and run; six-condition val evaluation complete

### Summary

The PEFT condition was trained, tuned, confirmed, and generated on the full val
split, which completes the **val** evaluation for all six implemented conditions
— the five prompting rungs plus PEFT. For the first time the project has COMET
and LLM-as-Judge Φ numbers for every condition on the same 1,323-segment split,
so the README results table is no longer empty.

The frozen PEFT configuration is **r = 32, α = 64, lr = 2e-4, 2 epochs**, LoRA on
all linear layers, adapter `models/peft_lora_r32_lr2e-4/checkpoint-1358`
(80,740,352 trainable parameters across 392 adapter tensors, ≈1.06% of the 7.62 B
base). It was selected by `src.peft.sweep` and confirmed by `src.peft.verify`
(2026-07-25 entry), and is frozen into `configs/peft_qwen.yaml` as
`generator.adapter_path`.

**These are val results. The test split remains sealed** — `data/splits/test.jsonl`
(1,322 segments) has not been generated on by any condition.

| Condition | COMET | chrF | BLEU | Judge Φ | lex_density | TTR | stylo_dist |
|---|---|---|---|---|---|---|---|
| `zeroshot` | 0.6480 | 36.42 | 10.27 | 2.546 | 0.4025 | 0.8434 | 0.6518 |
| `random_fewshot` | 0.6644 | 37.52 | 11.64 | 2.633 | 0.4103 | 0.8444 | 0.4736 |
| `knn_fewshot` | 0.6839 | 39.82 | 13.99 | 2.748 | 0.4034 | 0.8367 | 0.4005 |
| `afsp_margin` | 0.6824 | 39.68 | 13.69 | 2.763 | 0.4060 | 0.8387 | 0.3910 |
| `afsp_full` | 0.6853 | 39.99 | 14.52 | **2.791** | 0.4105 | 0.8421 | 0.3698 |
| `peft` | **0.6986** | **41.58** | **16.90** | 2.744 | 0.4088 | 0.8515 | **0.2886** |

Three observations, all descriptive and none yet significance-tested:

1. The prompting ladder is monotone on COMET and chrF from `zeroshot` to
   `knn_fewshot`, and `stylo_dist` falls monotonically across the whole ladder
   (0.652 → 0.370) — each rung sits closer to the target-register centroid than
   the one below it.
2. **PEFT and AFSP disagree about which is more in-register.** *(Not supported by
   the bootstrap of 2026-08-01: the Φ half of this comparison is non-significant,
   p = 0.12.)* PEFT wins every
   surface and adequacy metric (COMET +0.013, chrF +1.6, BLEU +2.4 over
   `afsp_full`) and has the smallest stylometric distance (0.289 vs 0.370), but
   `afsp_full` scores highest on judge Φ (2.791 vs 2.744). The objective and
   subjective register measures rank the two adaptation families oppositely. This
   is exactly the RQ4 agreement/disagreement question and should be reported as
   such, not resolved by picking the flattering metric.
3. `afsp_margin` does **not** beat `knn_fewshot` on COMET (0.6824 vs 0.6839) —
   the margin/hub rung is flat-to-slightly-negative on adequacy — while the
   register rerank rung (`afsp_full`) adds both adequacy and Φ over it. On this
   split the gain attributable to AFSP comes from the register rerank, not from
   margin/hub penalization. *(Not supported by the bootstrap of 2026-08-01: neither
   AFSP rung separates from the rung below it, and `afsp_full` does not separate from
   `knn_fewshot` on any metric.)*

### What changed

No source changes. This entry records generation and scoring artifacts:

* `outputs/peft_val.jsonl` (1,323 rows) + `outputs/peft_val_usage.json` — the
  frozen adapter generated on full val under the locked decoding contract
  (greedy, `temperature 0.0`, `seed 42`, `max_new_tokens 1024`, bf16, **not**
  quantized).
* `results/comet_val.json` — per-segment `Unbabel/wmt22-comet-da` for all six
  conditions, `n = 1,323` each, with the shared `sources` list persisted so the
  paired bootstrap can verify segment alignment.
* `results/judge_val.json` + `results/judge_val_segments/*.jsonl` — evaluation-time
  judge Φ for all six conditions. Coverage is 1.0 everywhere except
  `knn_fewshot` (0.9992 — one segment unparseable and dropped).
* `results/peft_sweep_val.json`, `results/peft_verify_val.json` — the selection
  and confirmation records (see the 2026-07-25 entry).
* `configs/peft_qwen.yaml` — `adapter_path`, `r: 32`, `alpha: 64`,
  `learning_rate: 0.0002` frozen to the confirmed cell.

The run was executed on Colab across three phases recorded in
`notebooks/peft_full_run_colab.ipynb` (train/sweep, verify, final generation);
the 8 GB development GPU cannot hold the bf16 base, unchanged from Stage 0.

### Verification

The surface and stylometric numbers in the table above were **recomputed locally
on 2026-07-31** from the committed inference files. chrF matches the sweep/verify
records cell for cell (`peft` 41.58 = the `peft_r32_lr2e-4_e2` sweep cell;
`afsp_full` 39.99 = the frozen `afsp_k8_l0.75` cell), confirming that the shipped
`peft`/`afsp_full` outputs were generated from the frozen configs and not from
some other cell:

```bash
python manage.py eval          --conditions zeroshot random_fewshot knn_fewshot afsp_margin afsp_full peft --split val
python manage.py stylometrics  --conditions zeroshot random_fewshot knn_fewshot afsp_margin afsp_full peft --split val
```

COMET and Φ were read from the committed JSON, not re-run. The adapter parameter
count was read from the safetensors header of
`models/peft_lora_r32_lr2e-4/checkpoint-1358/adapter_model.safetensors`; its
`adapter_config.json` confirms `r: 32`, `lora_alpha: 64`, and all seven linear
projections (`q,k,v,o,gate,up,down`) as targets.

**Correction (2026-08-01): `stylo_dist` does not match cell for cell, because two
of the four comparable runs are not byte-identical to their swept cell.** Comparing
`outputs/sweep/*.jsonl` against the shipped ladder files segment by segment:

| Shipped run | Swept cell | Differing segments | `stylo_dist` (cell → shipped) |
|---|---|---:|---|
| `afsp_margin` | `afsp_k8_l0` | 0 / 1,323 | 0.3910 → 0.3910 |
| `peft` | `peft_r32_lr2e-4_e2` | 0 / 1,323 | 0.2886 → 0.2886 |
| `zeroshot` | `afsp_zeroshot` | 2 / 1,323 | 0.6515 → 0.6518 |
| `afsp_full` | `afsp_k8_l0.75` | 5 / 1,323 | 0.3709 → 0.3698 |

The seven differing segments are whole-sentence divergences, not token-level jitter.
The split tracks the usage records exactly: the two runs that differ are the two that
were resumed across sessions (`zeroshot` 1,303 calls, `afsp_full` 1,318, against
`n = 1,323`), and the two that match byte for byte each record the full 1,323 calls.
That is consistent with divergence **across** generation sessions rather than within
one, so the locked greedy contract of the 2026-07-03 Stage 0 entry buys byte-for-byte
reproducibility within a session but has not held across sessions on this hardware. The reported figures are the shipped ladder files, which are the ones
COMET and the judge scored; the discrepancy affects `stylo_dist` in the fourth
decimal and changes no ordering, but the decoding contract should be restated to
claim only within-session determinism.

### Reproduction

```bash
# Selection and confirmation (Colab-class GPU; see the 2026-07-25 entry)
python manage.py peft_sweep  --config configs/peft_sweep.yaml
python manage.py peft_verify --config configs/peft_sweep.yaml --top 3 \
    --judge-config configs/judge_eval.yaml

# Final generation with the frozen adapter
python manage.py infer --condition peft --config configs/peft_qwen.yaml

# Scoring (COMET from the .venv-comet environment)
CONDS="zeroshot random_fewshot knn_fewshot afsp_margin afsp_full peft"
python manage.py eval         --conditions $CONDS --split val
python manage.py comet        --conditions $CONDS --split val
python manage.py judge        --conditions $CONDS --split val --config configs/judge_eval.yaml
python manage.py stylometrics --conditions $CONDS --split val
```

### Limitations and risks

**No significance testing has been run.** `manage.py bootstrap` exists and the
per-segment COMET/Φ vectors it needs are persisted, but no `results/bootstrap_*`
artifact exists. Every ordering above — including the 0.047 Φ gap between
`afsp_full` and `peft` and the 0.0015 COMET gap between `knn_fewshot` and
`afsp_margin` — is currently an unqualified point estimate. None of them may be
described as a difference until the paired bootstrap is run.

*Superseded (2026-08-01): the bootstrap has now been run; see that entry. Both gaps
named here are non-significant, and observations 2 and 3 of this entry's summary do
not survive it. The observations are left standing as written, because they record
what the point estimates showed on 2026-07-31 and why the bootstrap was the next
step — not as claims that may be cited.*

The judge is `claude-haiku-4-5` at `temperature 0.0`, a single model from a single
family with no seed control (see the 2026-07-23 entry). Φ differences on the order
of 0.05 are plausibly within judge noise, and the cross-family confirmation pass
the methodology calls for has not been run. Φ is also the *primary* stylistic
metric for the thesis, which makes this the weakest link in the current results.

**The size of that judge noise is now measurable, and it is not negligible.** Two
conditions were scored twice by the same judge on the same 1,323 segments — once
during confirmation and once in the final evaluation pass — and the two passes
disagree: `afsp_full` scores 2.797 in `results/afsp_verify_val.json` against 2.791
in `results/judge_val.json`, and `peft` scores 2.741 in
`results/peft_verify_val.json` against 2.744 in `results/judge_val.json`. For
`afsp_full` the shipped file also differs from the confirmed cell in 5 segments, so
that pair is not a pure re-scoring; the `peft` pair is, and it moves by 0.002 on
identical inputs. Test–retest spread of that order sits inside the 0.047 Φ gap this
entry reports between `afsp_full` and `peft`, and the freeze decisions of the
2026-07-23 and 2026-07-25 entries turned on Φ gaps of 0.019 and 0.009. Repeated
judge passes, not only a cross-family pass, are needed before any Φ ordering is
reported.

Cost and latency are not instrumented. `outputs/*_usage.json` records token counts
only (`cost_usd` is 0 for local weights), and the `calls` field is below
`n = 1,323` for the resumed runs (1,184–1,318), so it counts generation calls
actually issued, not segments — it cannot be used as a per-condition cost measure.
The README reports trainable parameters but leaves inference latency unmeasured,
which is one of the three components its cost row promises.

RLSF remains unimplemented (`src/rlsf/` holds only an empty `__init__.py`). RQ1
compares four conditions — the prompting baseline, AFSP, PEFT, and RLSF — so it is
a three-condition comparison today.

---

## 2026-07-25 — PEFT tuning as a first-class sweep (`src/peft/sweep.py`, `src/peft/verify.py`)

*Scope: the sweep and verify drivers were written on 2026-07-24 and 2026-07-25
(`de81c23`, `d813146`, `deb4298`, `1de6059`, `d2452be`, `c0efb67`); the smoke runs
that validated them ran on 2026-07-25 and 2026-07-26, and the sweep and verify runs
whose results this entry reports ran on 2026-07-27 to 2026-07-30. The entry is dated
to the driver work and reports the run it selected.*

### Summary

The 2026-07-24 entry left the LoRA hyperparameters as placeholder defaults with an
explicit warning that they had to be dev-tuned before the reported run. That tuning
was implemented as a two-stage driver that mirrors the AFSP `sweep → verify → freeze`
pattern, so **both adaptation arms are selected under the same rule on the same
axis**. Without this, PEFT would have been selected on `eval_loss` and AFSP on
register fidelity, and a PEFT-vs-AFSP comparison would have confounded the
adaptation method with the selection criterion.

The consequential design decision is that **`eval_loss` is not the selector**. It
is demoted to a free pre-filter that decides which checkpoints are worth generating
with; the reported adapter is picked on chrF adequacy band + register fidelity, then
confirmed on COMET + Φ.

### What changed

`src/peft/sweep.py` (new, `manage.py peft_sweep`). Trains the grid in
`configs/peft_sweep.yaml` — four (r, lr) cells: (16, 2e-4) as the labelled anchor,
(8, 2e-4), (32, 2e-4), (16, 1e-4), all with α = 2r — for the full three-epoch budget.
`enumerate_candidates` then treats **every saved epoch checkpoint as its own
candidate**, so a candidate is an (r, lr, epoch) triple, not a cell.
`prune_by_eval_loss(candidates, keep)` keeps the `keep` lowest-`eval_loss`
checkpoints per cell (default 2, `--epochs-keep -1` disables) purely to bound
generation cost, and logs every pruned candidate so the cap is never silent.
Surviving candidates are generated on val and scored on the AFSP-matched proxy axis
(chrF via `src.eval.quick`, `register_fit` via `src.infer.afsp_sweep._register_fit_fn`),
then ranked into `results/peft_sweep_val.json`.

Making epoch a tuned axis required regime changes in `configs/peft_sweep.yaml`:
`save_total_limit: null` (keep every epoch checkpoint) and
`load_best_model_at_end: false` (do not collapse the run to the `eval_loss`-best
epoch). A per-cell `epoch_checkpoints.json` manifest maps epoch → checkpoint path.

`src/peft/verify.py` (new, `manage.py peft_verify`). The analogue of
`src/infer/afsp_verify.py`: regenerates the top-N proxy candidates on full val,
scores them with COMET and — only when `--judge-config` is supplied — the judge,
and applies the identical freeze rule (keep cells within `--comet-adequacy` of the
best COMET, pick the highest Φ among them). Writes `results/peft_verify_val.json`
with per-segment COMET and the shared source list.

`src/peft/train.py`: LoRA `target_modules` moved from `[q_proj, v_proj]` to
`all-linear`, matching the README's stated design and `peft_sweep.yaml`. A
`enable_input_require_grads()` call was added on the bf16 path — with LoRA plus
gradient checkpointing and a frozen base, the checkpointed inputs carry no
`requires_grad`, so backward produced no gradients and training silently did
nothing until this was fixed (`1de6059`). Further config-plumbing fixes landed in
`d2452be` / `c0efb67`.

New configs: `configs/peft_sweep.yaml` (grid + `register:` block carrying
`select_target_sigma: 0.5` and the register direction), `configs/peft_smoke.yaml`
(loop validation), `configs/peft_anchor_e3.yaml` (three-epoch anchor cell).

### Why `eval_loss` is not the selector — and what the sweep showed

Completion-only MLE loss falls as the adapter fits target tokens; the thesis
objective is register fidelity in generated text. On this grid the two ordered
**oppositely**:

| Candidate | eval_loss | chrF | register_fit (lower = better) |
|---|---|---|---|
| `peft_r16_lr2e-4_e1` (anchor) | **1.5312** (best) | 41.96 | **1.0829** (worst) |
| `peft_r8_lr2e-4_e1` | 1.5336 | 41.40 | 1.0776 |
| `peft_r16_lr1e-4_e1` | 1.5379 | 41.10 | 1.1015 |
| `peft_r32_lr2e-4_e1` | 1.5398 | 42.18 | 1.0736 |
| `peft_r16_lr1e-4_e2` | 1.5553 | 41.66 | 1.0449 |
| `peft_r8_lr2e-4_e2` | 1.5583 | 41.57 | 1.0616 |
| `peft_r16_lr2e-4_e2` | 1.5601 | 41.73 | 1.0386 |
| `peft_r32_lr2e-4_e2` | **1.5733** (worst) | 41.58 | **1.0265** (best) |

Every epoch-2 checkpoint has a worse `eval_loss` and a better `register_fit` than
its epoch-1 counterpart, and the selected candidate is the single worst cell by
`eval_loss`. Selecting on `eval_loss`, as the placeholder config implied, would
have frozen the *least* in-register adapter of the eight. All eight sit inside a
1.1-point chrF band (41.10–42.18), so adequacy does not discriminate between them
and the register axis decides.

`src.peft.verify` then confirmed the proxy pick on the reported metrics
(`results/peft_verify_val.json`, `proxy_pick_held: true`):

| Candidate | COMET | Φ |
|---|---|---|
| `peft_r32_lr2e-4_e2` **(frozen)** | 0.6986 | **2.741** |
| `peft_r16_lr2e-4_e2` | **0.7016** | 2.732 |
| `peft_r16_lr1e-4_e2` | 0.6978 | 2.697 |

The three sit within 0.004 COMET — inside the 0.01 adequacy band — so the freeze
fell to Φ, which kept the proxy pick.

### Verification

Two Colab smoke runs preceded the real sweep: a first sweep smoke on 2026-07-25
(32-segment cell, retained under `archive/peft-smoke/`) and a multi-epoch smoke on
2026-07-26 (`notebooks/peft_multiepoch_smoke_colab.ipynb`). Per project findings
discipline these are loop validation and produce no thesis numbers. The real sweep
and verify runs are recorded in `notebooks/peft_full_run_colab.ipynb`.

### Reproduction

```bash
python manage.py peft_sweep  --config configs/peft_sweep.yaml --dry-run     # grid plan, trains nothing
python manage.py peft_sweep  --config configs/peft_sweep.yaml               # train + generate + rank
python manage.py peft_sweep  --config configs/peft_sweep.yaml --score-only  # re-rank, no GPU
python manage.py peft_verify --config configs/peft_sweep.yaml --top 3 \
    --judge-config configs/judge_eval.yaml
```

### Limitations and risks

The grid is small and not a cross-product: four (r, lr) cells with α tied to 2r, so
α/r is never varied independently and the (8, 1e-4) and (32, 1e-4) corners are
unexplored. Epoch 3 checkpoints were trained and remain on disk
(`checkpoint-2037`) but were pruned from selection by the default
`--epochs-keep 2`; since `register_fit` improved monotonically from epoch 1 to
epoch 2 in every cell, a third epoch might have improved it further and was never
generated with. That pruning is an `eval_loss`-driven cost cap on an axis the entry
above argues `eval_loss` should not govern — it is logged, but it is the one place
`eval_loss` still constrains the outcome.

Each cell was trained once at `seed: 42`; there is no seed replication, so
`register_fit` gaps of ~0.01 between adjacent cells are not distinguishable from
training noise. The COMET spread across the three verified candidates (0.004) is
likewise well inside the paired-bootstrap noise floor that has not yet been
computed, so "r = 32 at 2 epochs is the best adapter" is a selection decision, not
a demonstrated result.

---

## 2026-07-24 — PEFT (LoRA) condition: supervised fine-tuning implemented (not run)

*Date corrected on 2026-08-01: this entry was filed under 2026-07-23; the commit it
records, `fcb0834` "feat: implement PEFT (LoRA) fine-tuning condition", is dated
2026-07-24, and the entry has been moved to its position in the ordering.*

### Summary

The PEFT condition — the third of the four conditions to be built, leaving only
RLSF unimplemented — was implemented. `src/peft/` had held only an
empty `__init__.py`;
it now contains `src/peft/train.py`, a LoRA supervised fine-tuning entry point
wired into `manage.py` as the `peft` command and paired with a new inference
condition `peft` in `src/infer/run.py`. This entry records the implementation and
its static verification only. **No training was run**: the frozen thesis base is
`Qwen2.5-7B-Instruct`, which does not fit on the 8 GB development GPU (RTX 4060
Laptop) in bf16, so training — like the AFSP sweep — is deferred to a
larger-memory environment (Colab) to keep the base unquantized. Nothing here is a
thesis finding; the PEFT results row stays empty.

### What changed

`src/peft/train.py` implements LoRA SFT on the locked base. LoRA adapters are
applied to the query/value projections (`target_modules: [q_proj, v_proj]`,
`src/peft/train.py:142`), base weights frozen, trained with token-level MLE on
`data/splits/train.jsonl`. The training prompt is byte-identical to the `peft`
inference condition: the locked style instruction as system, the zero-shot
translate directive as user (`_ZEROSHOT_USER`, `src/peft/train.py:34`, replicated
from `run.build_zeroshot_user` to avoid pulling the commercial-API client imports
into training), and the reference translation as the assistant turn. Loss is
**completion-only** — the prompt prefix is tokenized separately and its tokens
masked to `-100` (`build_example`, `src/peft/train.py:43`) — so only the target
tokens are supervised. Model selection is by `eval_loss` on `val.jsonl`
(`load_best_model_at_end`, `metric_for_best_model="eval_loss"`); the best adapter
+ tokenizer and a `train_metrics.json` are saved to `outputs/peft/adapter`
(git-ignored). QLoRA is the memory-pressure fallback (`load_in_4bit` →
`BitsAndBytesConfig` + `prepare_model_for_kbit_training`). The saved adapter is
also the intended RLSF initialization.

Inference wiring: `LocalChatClient` gained an optional `adapter_path`
(`src/infer/local_client.py`); when set, the LoRA adapter is layered onto the
frozen base via `PeftModel.from_pretrained` (lazy import). `make_client` passes
`generator.adapter_path` through, and `run.py` adds `peft` to `CONDITIONS` — it
reuses the zero-shot prompt (no exemplars; the register lives in the weights) and
raises if `adapter_path` is unset. Decoding stays locked (greedy, seed 42,
max_new_tokens 1024). Config: `configs/peft_qwen.yaml`. Dependency `peft==0.18.0`
added to `requirements.txt`.

### Reproduction

    python manage.py peft  --config configs/peft_qwen.yaml            # train (Colab / A100)
    python manage.py infer --condition peft --config configs/peft_qwen.yaml  # generate on val

### Verification (static only — no model load, no network)

`ruff check src`, `ruff format --check src`, `python -m compileall -q src` all
pass. The completion-masking arithmetic in `build_example` was checked offline
against a stub tokenizer with synthetic data: only the assistant/target tokens
are supervised, the prompt is fully masked, and truncation below the prompt
length degrades to an all-masked example without crashing.

### Limitations and risks

LoRA rank, learning rate, and epoch/step count in `configs/peft_qwen.yaml` are
placeholder defaults (r=16, lr=2e-4, 3 epochs) — they must be **dev-tuned** on
`eval_loss` before the frozen PEFT run, exactly as `k`/`λ` were swept for AFSP.
*(The 2026-07-25 entry carried out that tuning and rejected the `eval_loss`
criterion proposed here: on this grid `eval_loss` ranks candidates in the opposite
order to register fidelity.)*
The completion mask relies on the prompt-only chat-template render being a strict
token prefix of the full render; this holds for the Qwen2.5 template (clean
`<|im_start|>assistant\n` boundary) but should be re-checked if the base or
template changes.

---

## 2026-07-23 — AFSP run end to end on val: sweep → verify → freeze (k = 8, λ = 0.75)

*Scope: the sweep, the confirmation, the judge-provider change, and the ladder
generation span 2026-07-19 to 2026-07-23. The entry is dated to the last of them
(`93906ab`, `7230065`, `4812465`, `9fbc249`).*

### Summary

The AFSP sweep and confirmation described as *not run* in the 2026-07-06 and
2026-07-16 entries were executed on Colab, and the resulting (k, λ) was frozen.
The proxy sweep ranked the grid, `afsp_verify` confirmed the top three cells on
full val under COMET and the judge, the proxy pick held, and **k = 8,
λ_style = 0.75** was frozen into `configs/base_qwen.yaml`. All five prompting
rungs were then generated on the full val split. The evaluation-time judge was
also changed to a different provider during this period (below), which supersedes
a decision recorded in the 2026-07-05 entry.

### The sweep

`results/afsp_sweep_val.json` — 19 rows: the k ∈ {4, 8, 16} × λ ∈ {0, 0.1, 0.25,
0.5, 0.75, 1.0} grid (18 cells) plus a `afsp_zeroshot` reference row, each on all
1,323 val segments, β fixed at 0.3.

Selection ran the documented rule — keep cells within `adequacy_margin` 1.0 chrF
of the grid best (40.44), then take the best `register_fit` — and recommended
`afsp_k8_l0.75` (chrF 39.99, `register_fit` 0.9611, `stylo_dist` 0.3709).

Two things in the table are worth recording:

* **The zero-shot row has a better `register_fit` (0.9555) than every AFSP cell
  inside the adequacy band**, the best of which is the selected `afsp_k8_l0.75`
  at 0.9611. It is excluded only by that band (chrF 36.42 vs 40.44). The register
  objective alone would have selected no exemplars at all; the adequacy band is
  what makes the selection meaningful, so its width is load-bearing, not cosmetic.
  *Correction (2026-08-01): the original entry claimed zero-shot beat every AFSP
  cell in the grid. It does not. `afsp_k4_l1` (0.9217) and `afsp_k4_l0.75`
  (0.9273) both score better, and both fall outside the band — which strengthens
  rather than weakens the point, since the two cells with the best register
  fidelity in the whole grid are also the two the adequacy band has to reject.*
* **k and λ trade off against each other.** k = 16 dominates on chrF (up to 40.44)
  but has the worst `register_fit` at low λ (1.069 at λ = 0.1); k = 4 has good
  `register_fit` (0.922 at λ = 1.0) but the worst chrF (38.81). k = 8 at λ = 0.75
  wins by sitting inside the adequacy band with the best fidelity available there.

### The confirmation

`results/afsp_verify_val.json` regenerated the top three proxy cells on full val
and scored them with COMET and the judge (`proxy_pick_held: true`):

| Cell | chrF | COMET | Φ | coverage |
|---|---|---|---|---|
| `afsp_k8_l0.75` **(frozen)** | 39.99 | 0.6853 | **2.797** | 1.0 |
| `afsp_k8_l1` | 39.76 | 0.6824 | 2.763 | 1.0 |
| `afsp_k16_l0.75` | 40.44 | **0.6871** | 2.778 | 1.0 |

All three lie within 0.005 COMET — inside the 0.01 band — so the freeze fell to Φ
and kept the proxy pick, even though `afsp_k16_l0.75` is nominally better on both
chrF and COMET.

`configs/base_qwen.yaml` was frozen to `retrieval.k: 8`, `afsp.lambda_style: 0.75`,
and the five ladder rungs were generated on full val into
`outputs/{zeroshot,random_fewshot,knn_fewshot,afsp_margin,afsp_full}_val.jsonl`.

*Correction (2026-08-01): the original entry attributed the freeze to commit
`6ae191a`. That commit is dated 2026-07-17, its diff touches `.gitignore` only, and
it predates both the band-pass objective and the grid re-cut of 2026-07-18 — so it
cannot be the freeze that this sweep produced, despite its message. The commit that
actually wrote `retrieval.k: 8` and `afsp.lambda_style: 0.75` into
`configs/base_qwen.yaml` is `220bba2` (2026-07-17, message "fix: add new req"), and
no later commit touches either line. **The frozen (k, λ) therefore entered the
config before the sweep this entry reports, and was never rewritten in response to
it.** The values coincide with the post-band-pass recommendation, but the config is
not evidence of that recommendation having been applied; the sweep confirms the
setting rather than having produced it. This should be stated that way in the
methodology, or the freeze should be re-applied from
`results/afsp_verify_val.json` in a commit that says so.*

### Judge provider change (supersedes the 2026-07-05 decision)

`configs/judge_eval.yaml` was changed repeatedly on 2026-07-20 and settled on
`provider: anthropic`, `model: claude-haiku-4-5`, `temperature: 0.0`,
`thinking: false`, having passed through Gemini Flash variants from the originally
documented OpenAI judge. **No rationale for the switch is recorded in the
repository**; only the config history shows it.

This directly contradicts the Stage 4 entry (2026-07-05), which chose an OpenAI
judge specifically because `temperature=0` + `seed` gave a reproducible primary,
and stated that "the Anthropic cross-family judge is non-deterministic (no
temperature control in the client) and should be treated as a confirmation pass,
not a reproducible primary." The Anthropic judge is now the primary. Every Φ in
`results/judge_val.json` comes from it.

Half of that objection was addressed in the same window, though no entry says so:
commit `b0fb732` (2026-07-20) added temperature plumbing to
`src/infer/anthropic_client.py`, which now forwards `temperature` to
`messages.create` whenever `thinking` is off — the configuration
`configs/judge_eval.yaml` uses. So the Stage 4 statement was accurate when written
and is now outdated on temperature. **It remains accurate on determinism**: the
Anthropic client exposes no seed, so the primary judge is temperature-0 but not
reproducible, and the 2026-07-31 entry measures the resulting test–retest spread at
about 0.002 Φ on identical inputs. The methodology must therefore either
re-establish a seeded primary or defend a non-reproducible one with repeated-pass
variance reported alongside every Φ; leaving the choice undocumented is not an
option, since the switch itself has no recorded rationale.

### 4-bit flag (transient, did not affect reported runs)

On 2026-07-18 `load_in_4bit` was flipped to `true` in `base_qwen.yaml` and
`afsp_sweep.yaml`, and a `configs/base_qwen_t4_test.yaml` was added, to fit the
base on a T4. This would have violated the frozen-unquantized-base contract of the
Stage 0 entry. It was reverted the same day (`446eca8`), the T4 config no longer
exists in the tree, and every commit from 2026-07-19 onward carries
`load_in_4bit: false`. The sweep, verify, and ladder generations all postdate the
revert, and the run notebooks contain no 4-bit override, so **the reported AFSP
outputs are bf16 on the unquantized base.**

### Housekeeping

`4812465` archived the pre-band-pass sweep outputs to
`archive/pre-band-pass-fix/` (26 stale cells from the old k ∈ {1, 2, 4, 8, 16}
grid plus the notebook that produced them) with a README marking them
non-canonical. `9fbc249` restored `ruff` lint/format compliance across `src`.

### Reproduction

```bash
python manage.py build_index   --config configs/base_qwen.yaml
python manage.py stylometrics  --build-centroid
python manage.py afsp_sweep    --config configs/afsp_sweep.yaml
python manage.py afsp_verify   --config configs/afsp_sweep.yaml --top 3 \
    --judge-config configs/judge_eval.yaml
# then freeze retrieval.k / afsp.lambda_style into configs/base_qwen.yaml
for c in zeroshot random_fewshot knn_fewshot afsp_margin afsp_full; do
    python manage.py infer --condition "$c" --config configs/base_qwen.yaml
done
```

### Limitations and risks

The freeze between `afsp_k8_l0.75` and `afsp_k16_l0.75` rests on a Φ gap of 0.019
from a single judge with no seed control, inside a COMET spread of 0.005. That is
almost certainly below the noise floor of both metrics; the pick is defensible as a
pre-registered rule mechanically applied, but it is not evidence that k = 8 beats
k = 16.

β was held at 0.3 throughout and is still unswept, so β × (k, λ) interactions
remain unseen. The sweep's own zero-shot row shows the register objective is
satisfiable by degenerate means, which is a standing argument for reporting the
adequacy band alongside any register claim.

---

## 2026-07-18 — AFSP register objective: band-pass target replaces unbounded salience

### Summary

The register-fit term that drives AFSP's λ rerank was changed from **salience**
(mean absolute z-score, unbounded) to a **band-pass** objective: the distance to a
target point sitting σ standard deviations along a *signed* register direction.
The sweep's ranking axis changed with it, from the undirected `stylo_dist` to the
new directional `register_fit`. The k × λ grid was simultaneously re-cut. All
sweep outputs produced before this change were invalidated and later archived.

### Rationale

The 2026-07-04 Stage 1 entry adopted salience precisely to fix the opposite error —
toward-centroid reranking had been selecting the register-*bland* corpus average —
and flagged in its own limitations that salience "is unbounded and length-sensitive
in isolation", safe only because the candidate pool bounds it. Two problems remain
with it as a *ranking* axis:

* Salience is the mean **absolute** z-score, so it is direction-blind: an exemplar
  that deviates from the centroid the wrong way scores as well as one that deviates
  the right way.
* Being unbounded, "more" is always "better", so a λ sweep pushes monotonically
  toward the extreme rather than toward a target register.

Band-pass fixes both: the target is an explicit point, not an extremum, and the
direction is signed.

### What changed

`src/eval/stylometrics.py` gained `register_band_distance(feats, centroid, sigma,
direction)`. It z-scores the segment's centroid features, forms the target
`sign(d) · σ`, and returns the direction-magnitude-weighted standardized distance
to it — `sqrt(Σ wᵢ(zᵢ − targetᵢ)² / (Σw / n))` with `w = |d|`, so features that load
weakly on the register direction contribute proportionally less. `distance_to_centroid`
and `register_salience` are retained: the former undirected for reporting, both
still selectable.

`src/retrieval/afsp.py` gained `STYLE_OBJECTIVES = ("bandpass", "proximity",
"salience")` with **`bandpass` as the default**, plus `style_target_sigma`,
`style_register_direction`, and `_resolve_direction` (which accepts a dict keyed by
centroid feature name — order-independent and validated against the centroid's own
feature list, raising on a missing or mis-sized spec). `_register_fit` dispatches on
the objective; `proximity` reproduces the pre-Stage 1 behaviour and `salience` the
Stage 1 (2026-07-04) behaviour, so all three are still reachable for ablation.

`src/infer/afsp_sweep.py` now computes a per-cell `register_fit` via
`_register_fit_fn` and **ranks on it instead of `stylo_dist`** (`446eca8`):
`recommend` takes `min(register_fit)` within the chrF band, ties broken toward
higher chrF then smaller k; `stylo_dist` is retained in the table for reporting
only. `src/infer/run.py` passes the same three settings through to the retriever.

`configs/base_qwen.yaml` set the objective and the direction:

```yaml
style_objective: bandpass
style_target_sigma: 1.0
style_register_direction:
  marker_rate: 0.514
  lex_density: 0.355
  root_ttr: -0.296
  ttr: 0.098
```

Read out: the register direction is dominated by archaic-marker rate and lexical
density, with **`root_ttr` loading negatively** — on this corpus the target register
is *less* lexically varied per unit length, not more. This is why a direction-blind
objective mismeasures it: salience rewards deviation in `root_ttr` in either
direction, while the corpus says only one direction is register-bearing.

The grid was re-cut in the same window. `97c1744` cut the default k axis to
{4, 8, 16}, dropping k = 1 and k = 2 as too few exemplars, and `6169de9` densified λ
at the low end by adding 0.1.

*Correction (2026-08-01): the original entry described the grid being replaced as
k ∈ {1, 2, 3, 4} × λ ∈ {0, 0.25, 0.5, 0.75, 1.0}. That was the Stage 5 grid of
2026-07-06. The grid actually in force on 2026-07-18 was k ∈ {1, 2, 4, 8, 16}
(`PREVIOUS_KS` in `src/infer/afsp_sweep.py`), set by `270b96c` on 2026-07-16 along
with the zero-shot anchor row — a change no entry records. So k = 16 was not added
here, and the re-cut narrowed an existing five-value k axis to three rather than
widening a four-value one. The 26 cells archived by `4812465` (25 grid cells plus
the anchor) are from that intermediate grid, as the housekeeping note of the
2026-07-23 entry correctly states. The λ axis also did not reach its final form
here: `843fe38`
repaired a malformed `DEFAULT_LAMBDAS` tuple later the same day, leaving five
values, and λ = 0.5 was restored only on 2026-07-19 by `568ba73`. The 18-cell
grid the 2026-07-23 run used dates from that commit, not from this one.*

### Verification

Not separately verified at the time beyond the static checks; the objective's
effect is visible in the sweep table recorded in the 2026-07-23 entry, where
`register_fit` and the undirected `stylo_dist` rank cells differently.
`afsp_k16_l1` has the second-best `stylo_dist` of the 18 cells at 0.3808 but only
the tenth-best `register_fit` at 0.9811 — the clearest single case that the two
axes disagree. *(Correction, 2026-08-01: the original entry called this the
seventh-best `register_fit`; recounted against `results/afsp_sweep_val.json` it is
tenth of 18.)*

### Limitations and risks

**The provenance of `style_register_direction` is not recorded anywhere in the
repository.** The four coefficients are hard-coded in `configs/base_qwen.yaml` and
duplicated in `configs/peft_sweep.yaml`; no script derives them and no entry
explains how they were obtained. They look like a normalized loading vector, and
their signs agree with the correlations reported in the 2026-07-04 Stage 1 entry
(distance–marker +0.51, distance–lex +0.36), but that is inference, not a record.
Since this vector defines what the project *means* by "the target register", it
needs a derivation script or a documented source before it appears in the thesis.

*Correction (2026-08-01): the original entry stated that `style_target_sigma` is 1.0
in `base_qwen.yaml` but 0.5 in `configs/peft_sweep.yaml`, and concluded that the AFSP
and PEFT arms target register points at different distances. They do not — the two
values are different parameters, not two settings of one. `style_target_sigma`
(1.0) is the σ the **retriever** uses when reranking exemplars during generation;
`select_target_sigma` (0.5) is the σ the **selection metric** uses when scoring a
finished cell's `register_fit`. `configs/afsp_sweep.yaml` carries both, at 1.0 and
0.5 respectively, and `configs/peft_sweep.yaml` carries only the latter, also at 0.5.
The PEFT arm is therefore selected on exactly the AFSP selection axis, σ included,
and the 2026-07-25 entry's "matched axis" claim holds.*

The real σ mismatch is **inside** the AFSP arm: exemplars are reranked toward a
target 1.0 σ along the register direction while cells are scored against a target at
0.5 σ, so the objective the retrieval optimizes is not the objective the sweep ranks
on. That is defensible — reranking and evaluation need not share a target — but it
is undocumented, and the 0.5/1.0 split appears to be inherited rather than chosen.

The direction vector is duplicated verbatim in **six** configs — `base_qwen.yaml`,
`afsp_sweep.yaml`, `qwen_smoke.yaml`, `peft_sweep.yaml`, `peft_anchor_e3.yaml`, and
`peft_smoke.yaml` — with no single source. Editing one will silently desynchronize
the arms.

---

## 2026-07-16 — Stage 5b hardening: judge-gating the freeze, persisting per-segment COMET

### Summary

Two correctness fixes to `src/infer/afsp_verify.py` before the confirmation is run
for real on the A100, both aimed at making the freeze decision trustworthy rather
than merely runnable.

First, a **judge-gating** fix. Supplying `--judge-config` is a statement of intent:
*confirm the pick on both axes — adequacy (COMET) and register fidelity (judge Φ).*
But `_freeze_pick` decides "do I have judge scores?" from the data, not from that
intent: if the judge pass came back empty (every call failed) or too thin to trust
(a handful of segments parsed), `have_judge` silently went `False` and the code fell
through to freezing on the **best-COMET cell alone**. That path is indistinguishable
in the output from a deliberate COMET-only run, so a broken or under-covered judge
pass could quietly freeze a (k, λ) on adequacy while the run looked like a two-axis
confirmation. The fix refuses instead of falling back.

Second, **persisting per-segment COMET**. `comet.score` already returns the
segment-level vector, but the verifier kept only the system score, so the paired
bootstrap the thesis uses for significance had nothing aligned to resample. The
verifier now persists the per-segment COMET for each confirmed cell, plus the shared
source list, and guards that the cells are actually segment-aligned before it does.

Still not run here (needs the Qwen base + COMET, ~15 GB); static checks pass and the
sweep is grinding on the A100 as this lands, so `afsp_verify` is correct by the time
its results are consumed.

### What changed

`src/infer/afsp_verify.py`:

* After the judge pass, when `--judge-config` was supplied, the run now checks every
  top cell has a usable Φ — `mean is not None` **and** `coverage >= --min-judge-coverage`
  (new flag, default 0.9). Any thin cell raises `SystemExit` with the offending tags,
  a pointer to the resumable per-segment cache under `results/judge_<split>_segments/`,
  and the explicit alternative (rerun without `--judge-config` to freeze on COMET
  deliberately). `_freeze_pick` is unchanged; its COMET-only branch is now reached
  only when no judge was requested, which is the one case it is correct for.
* The COMET loop keeps `res["segments"]` alongside `res["system"]`. Because a paired
  bootstrap requires segment i to mean the same source across cells, the loop asserts
  every top cell loads an identical source list (same val rows, same order) and raises
  `SystemExit` on any mismatch rather than pairing misaligned segments.
* `results/afsp_verify_<split>.json` gains a top-level `sources` (persisted once) and
  a per-cell `comet_segments`, so the bootstrap can re-verify alignment and resample
  without re-running COMET.

### Verification

No generation, no COMET load. Static checks pass:

```bash
python -m py_compile src/infer/afsp_verify.py
ruff check src/infer/afsp_verify.py
```

Reasoned through the gate: with `--judge-config` and a full-coverage judge pass the
run freezes on Φ within the COMET band as before; with a judge pass that fails or
falls below `--min-judge-coverage` it now exits non-zero instead of freezing on COMET;
without `--judge-config` the COMET-only freeze is untouched. The alignment guard fires
before COMET is scored on the second cell, so a misaligned regeneration is caught
early rather than silently paired.

### Limitations and risks

The coverage threshold (0.9) is a judgement call — a legitimately hard cell that the
judge refuses on many segments will now block the freeze rather than freeze on a thin
mean; that is the intended trade (fail loud), but it means a genuinely low-coverage
val split needs `--min-judge-coverage` lowered consciously, not by accident. Persisted
`comet_segments` assume the COMET checkpoint is fixed across cells (it is — one model
load), so the stored vectors are only comparable within a single verify run, not across
runs on different checkpoints. The paired bootstrap that consumes these segments is not
yet written.

---

## 2026-07-16 — Stage 5b: Confirming the sweep pick on full val (COMET/judge)

### Summary

The Stage 5 sweep ranks the 20-cell k × λ grid on **free, local proxies** — a
chrF adequacy band, then stylometric distance to the register centroid — and may
be run on a reduced selection subset to keep generation compute affordable. Those
proxies are crude next to the COMET and LLM-as-Judge metrics the thesis actually
reports, and the selection set may be a sample rather than the full split. Either
gap could let the proxy-picked (k, λ) be a false winner. Stage 5b closes both with
a first-class command, `src/infer/afsp_verify.py` (`manage.py afsp_verify`): it
takes the top-N proxy cells, **regenerates only those on the full val split**,
scores them with COMET (free/local) and — only when a judge config is supplied —
the paid judge, and reports whether the proxy pick holds up on the reported
metrics or a runner-up overtakes it. The freeze decision is made here, on full
val under the reported metrics; the sealed test split is never touched.

This makes the methods claim defensible: *hyperparameters were pre-ranked with
inexpensive proxies (optionally on a selection subset), and the top candidates
were confirmed on full val under COMET and the judge before freezing; all reported
results are on the full val and test splits.* That is stronger than a full-val
proxy sweep, which spends the compute on more proxy estimates while still selecting
on a metric the thesis does not report.

Like Stage 5, this was **not run**: the confirmation regenerates cells with the
Qwen-7B base (bf16, ~15 GB) and loads COMET, neither of which the 8 GB dev box can
hold, so a live run is deferred to a Colab-class GPU. Everything that runs without
the base model and COMET — the candidate ranking, the freeze rule, the test-split
seal, the CLI/dispatch — was verified offline and the CI-equivalent static checks
pass.

### What changed

`src/infer/afsp_sweep.py`: the per-cell generation loop was extracted into
`generate_cells(cfg, cells, ...)`, which takes an explicit list of (k, λ) cells;
`generate_grid` now delegates to it with the full cross-product, so the sweep's
behaviour is byte-for-byte unchanged (same shared index/retriever reuse, same
atomic-write + cell-level resume, same `test` seal). `afsp_verify` reuses
`generate_cells` to regenerate just the top cells on a different (full) split.
A new `ranked_cells(rows, adequacy_margin)` exposes the sweep's own selection
ordering — in-adequacy-band cells first, register fidelity (lower `stylo_dist`)
within each group, ties toward higher chrF then smaller k — so the verifier pulls
exactly the candidates `recommend` would have chosen from.

`src/infer/afsp_verify.py` (new). It loads `results/afsp_sweep_<split>.json`
(derived from the sweep config's `eval_file`, or `--sweep-result`), ranks the
cells with `ranked_cells` at the sweep's own `adequacy_margin`, takes `--top`
(default 3), and regenerates them on `--val-file` (default `data/splits/val.jsonl`,
with the same `test`-seal refusal). It COMET-scores each cell with one model load
reused across cells, and — only if `--judge-config` is given — runs the resumable
judge (per-cell segment cache under `results/judge_<split>_segments/`), so the paid
metric is opt-in. The freeze rule (`_freeze_pick`) is the real-metric analogue of
the sweep's proxy rule: with judge scores present, keep cells within
`--comet-adequacy` (default 0.01) of the best COMET and pick the highest register
Φ among them; without the judge, take the best COMET cell. It prints a
COMET-ordered table marking both the proxy pick and the freeze pick, states
whether the pick held, writes `results/afsp_verify_<split>.json`, and prints the
exact `base_qwen.yaml` edits to freeze. `manage.py` registers the `afsp_verify`
command.

### Verification

No generation and no COMET load (no Qwen base, no GPU). CI-equivalent static checks
pass with `ruff==0.14.2`:

```bash
ruff check src/infer/afsp_sweep.py src/infer/afsp_verify.py manage.py
ruff format --check src/infer/afsp_sweep.py src/infer/afsp_verify.py
python -m compileall -q src/infer/afsp_sweep.py src/infer/afsp_verify.py manage.py
```

Offline behavioural checks on synthetic proxy tables: `ranked_cells` leads with the
in-band lowest-`stylo_dist` cell and sinks an out-of-band cell despite its better
fidelity; `_freeze_pick` returns the best-COMET cell with no judge, switches to the
highest-Φ cell when all candidates sit within the COMET band, and re-excludes a cell
that falls outside a tightened band before Φ is consulted. `manage.py afsp_verify
--help` dispatches and argparses, the command appears in the dispatcher listing, and
pointing `--val-file` at `test.jsonl` raises the seal (`refusing to verify on the
sealed test split`) before any model is loaded.

### Reproduction

```bash
# After a sweep has written results/afsp_sweep_<split>.json.
# COMET-only confirmation of the top 3 proxy cells on full val (free/local, needs a GPU for Qwen):
python manage.py afsp_verify --config configs/afsp_sweep.yaml --top 3

# Add the paid judge pass (top cells only) to decide within the COMET band.
# --judge-config takes any YAML carrying a `judge:` block; the canonical one is
# configs/judge_eval.yaml, which every later entry uses.
python manage.py afsp_verify --config configs/afsp_sweep.yaml --top 3 \
    --judge-config configs/judge_eval.yaml
```

Then read `results/afsp_verify_val.json`, freeze the reported `retrieval.k` and
`afsp.lambda_style` into `configs/base_qwen.yaml`, and only then run the final
`afsp_full` condition on `test.jsonl`.

### Limitations and risks

The confirmation has not been run; there are no COMET/judge numbers and no frozen
(k, λ) yet. It confirms only the top-N proxy cells, so a strong cell that the proxy
ranked below N is never re-scored — widen `--top` if the proxy table is flat near
the top. The COMET-band-then-Φ freeze rule is a modelling choice mirroring the
sweep's proxy rule; `--comet-adequacy` defaults to 0.01 (COMET-DA system scores are
~0–1) and should be set against the paired-bootstrap noise floor, not eyeballed. The
judge is a paid commercial API and, per the project's findings discipline, any judge
number is a sanity signal for selection, not a reported thesis result — the reported
judge numbers still come from the frozen conditions on val/test. β stays fixed at
0.3, unchanged from the sweep.

---

## 2026-07-13 — Resolving the COMET dependency conflict (separate venv)

### Summary

`unbabel-comet==2.2.6` was split out of `requirements.txt` into a new
`requirements-comet.txt`, installed in its own virtualenv. The Stage-4 log flagged
this as a risk; it is a hard, three-way conflict, so `pip install -r requirements.txt`
as written never resolves.

### What changed

- `requirements.txt`: removed the `unbabel-comet==2.2.6` line (replaced with a comment
  pointing to the new file). The generation stack now installs cleanly.
- `requirements-comet.txt` (new): `unbabel-comet==2.2.6` plus the two conflict-critical
  transitive pins from the verified resolution (`transformers==4.57.6`, `numpy==1.26.4`),
  with a header documenting the local (CPU torch) and Colab (GPU torch) setup.
- `README.md`: Setup adds the `.venv-comet` environment; the Evaluation section notes
  `manage.py comet` must run from it.

### Rationale

COMET 2.2.6's metadata pins `transformers>=4.17,<5.0`, `numpy>=1.20,<2.0`, and
`huggingface-hub<1.0`. The generation stack pins `transformers==5.12.1` and
`numpy==2.4.1`. All three constraints conflict, so co-resolution is impossible — this
is not a version-bump-away problem. COMET only scores the inference `*.jsonl` files
post-hoc (`src/eval/comet.py:19` imports `comet` lazily; `src/eval/_io.py` is
stdlib-only), so the eval backbone already decouples and a second venv is the clean fix.

### Reproduction and verification

```bash
python -m venv .venv-comet && source .venv-comet/bin/activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-comet.txt
python manage.py comet --help   # imports comet + torch; exits 0
```

Verified 2026-07-13 on Python 3.11: both `requirements.txt` (minus COMET) and
`requirements-comet.txt` resolve independently via `pip install --dry-run`; COMET
imports and `manage.py comet --help` runs under the isolated stack.

### Limitations and risks

- The COMET stack pulls its own `transformers`/`numpy`/`torch` (~CPU torch locally, GPU
  torch on Colab); keep the two venvs strictly separate — never `pip install` one file
  into the other's environment.
- `transformers==4.57.6` / `numpy==1.26.4` are pinned to the resolution verified today;
  if COMET is ever upgraded, re-verify these against its new metadata.

**Correction (2026-08-01): the decoupling this entry relies on no longer holds.**
Commit `446eca8` (2026-07-18) moved `from comet import download_model,
load_from_checkpoint` out of `load_model` and up to module scope, so
`src/eval/comet.py` now requires the `comet` package merely to be imported. The
consequence reaches beyond that module: `src/infer/afsp_verify.py:13` imports
`src.eval.comet` at module scope, so `manage.py afsp_verify` — which also needs the
Qwen base to regenerate cells — can no longer be loaded from the generation venv at
all, and the two-environment split of this entry cannot satisfy it. `src/peft/verify.py:79`
imports COMET inside its function and is unaffected. The confirmation runs of
2026-07-23 and 2026-07-25 postdate the change but did not surface it, because they
were executed on Colab, where one environment carries both stacks. Either restore
the lazy import or document that `afsp_verify` requires a combined environment.

---

## 2026-07-06 — Stage 5: The Real AFSP k × λ Sweep (val, Qwen)

### Summary

The thesis hyperparameter sweep was implemented as `src/infer/afsp_sweep.py`
(`manage.py afsp_sweep`) with its own config `configs/afsp_sweep.yaml`. It runs
the AFSP condition over the grid k ∈ {1, 2, 3, 4} × λ_style ∈ {0, 0.25, 0.5,
0.75, 1.0} — 20 cells — on the **val** split, on the frozen local Qwen base, and
recommends a single (k, λ) for a human to freeze into `configs/base_qwen.yaml`
before `test.jsonl` is ever touched. This is deliberately separate from
`src/infer/sweep.py`, the model × shot-count *smoke* sweep that drives commercial
APIs and produces no findings.

The sweep was **not run**: generating 20 × 1,323 val translations needs a GPU
that holds Qwen-7B in bf16 (~15 GB), which the development machine (8 GB) cannot
provide — the same constraint recorded for the Stage 0 local generator. Generation
is therefore deferred to a Colab-class GPU. Everything that runs without the base
model — cell tagging, the free/local scoring pass, the selection rule, the
test-split seal, the resumable-generation skip logic, and the CLI/dispatch — was
verified offline, and the CI-equivalent static checks pass.

### What changed

`src/infer/afsp_sweep.py` (new). `generate_grid` loads one shared source-side
index and a single `AFSPRetriever` (with the register centroid loaded) and reuses
them across all cells, mutating only `lambda_style` and the selection `k`, so the
candidate-hubness and register caches are computed once. Each cell reuses the
Stage 3 prompt path exactly — `_load_configured_glossary`, `order_exemplars`,
`build_fewshot_user` — so a swept cell is byte-identical in prompt construction to
the corresponding `afsp_margin`/`afsp_full` rung (λ = 0 is margin-only; λ > 0 adds
the register rerank). Cells are written to `outputs/sweep/afsp_k{k}_l{λ}_{split}.jsonl`
in the standard inference schema (plus `k` and `lambda_style` fields), and existing
cells are skipped unless `--overwrite`, making a multi-hour GPU sweep resumable
across interrupted sessions.

`generate_grid` refuses to run when the eval file's split stem is `test`, so the
sweep cannot accidentally consume the sealed split. Selection uses only free,
local signals — chrF/BLEU against the gold targets (via `src.eval.quick`) and the
standardized stylometric distance to the target-register centroid (via
`src.eval.stylometrics`); the expensive COMET / LLM-as-Judge metrics are reserved
for the final frozen conditions, not spent on 20 exploratory cells.

`recommend` encodes the thesis objective — register preservation without
sacrificing adequacy — as: keep cells whose chrF is within `--adequacy-margin`
(default 1.0) of the grid's best chrF, then pick the smallest register distance
among them, breaking ties toward higher chrF and then smaller k (cheaper prompt).
It falls back to max chrF when no centroid is available. The recommended (k, λ),
the full cell table, and the margin are written to `results/afsp_sweep_<split>.json`,
and the exact `base_qwen.yaml` edits to freeze are printed. `--score-only`
re-scores and re-selects over already-generated cells with no GPU.

`configs/afsp_sweep.yaml` (new) mirrors `base_qwen.yaml`'s locked decoding and
retrieval, pins the val split, and documents that `retrieval.k` / `afsp.lambda_style`
are placeholders overridden per cell by the grid. `manage.py` registers the
`afsp_sweep` command.

### Verification

No generation (no Qwen base loaded, no GPU). CI-equivalent static checks pass with
`ruff==0.14.2`:

```bash
ruff check src
ruff format --check src
python -m compileall -q src
python tests/test_stylometrics.py    # 7/7
```

Offline behavioural checks: `cell_tag` produces collision-free, filesystem-safe
tags with `%g`-formatted λ (`afsp_k2_l0.25`, `afsp_k4_l1`); `recommend` picks the
lowest register distance strictly within the chrF band, widens its choice as the
margin grows, breaks ties toward smaller k, falls back to max chrF without a
centroid, and returns `None` on an empty grid; `generate_grid` raises on a `test`
eval file (seal intact). The `--score-only` CLI was then run end to end over
synthetic `outputs/sweep/*.jsonl` cells: it scored the present cells, skipped the
absent grid positions with a note, printed the register-fidelity-ordered table
with the recommendation marked, and wrote `results/afsp_sweep_val.json`.
`manage.py afsp_sweep --help` dispatches and argparses.

### Reproduction

```bash
# On a Colab-class GPU (Qwen-7B bf16). Full grid over val:
python manage.py afsp_sweep --config configs/afsp_sweep.yaml

# Re-score / re-select over already-generated cells anywhere (no GPU):
python manage.py afsp_sweep --config configs/afsp_sweep.yaml --score-only
```

Then read `results/afsp_sweep_val.json`, freeze the recommended `retrieval.k` and
`afsp.lambda_style` into `configs/base_qwen.yaml`, and only then run the final
`afsp_full` condition on `test.jsonl`.

### Limitations and risks

The sweep has not been run; there are no val numbers and therefore no frozen
(k, λ) yet. The selection rule is a defensible default (register-first within an
adequacy band) but is a modelling choice; the chrF adequacy proxy and the
stylometric register distance are both crude relative to the COMET/judge metrics
reserved for the frozen conditions, so the recommendation should be sanity-checked
against a COMET/judge pass on the top few cells before freezing. β is held fixed
at 0.3 across the sweep — it is not a swept axis here, so an interaction between β
and (k, λ) would go unseen. The full grid is 20 × |val| generations; `--overwrite`
is off by default and cells are skipped when present, but the wall-clock and the
requirement to keep the base unquantized (no `load_in_4bit`) both point to Colab.

---

## 2026-07-05 — Stage 4: Evaluation Backbone (COMET, LLM-as-Judge, paired bootstrap)

*Date corrected on 2026-08-01: this entry was filed under 2026-07-04; the commit it
records, `0c1f81c`, is dated 2026-07-05. References to "the 2026-07-04 entry"
elsewhere in this log that concern the judge, COMET, or the bootstrap refer to this
entry and have been updated to 2026-07-05.*

### Summary

The evaluation backbone needed to report any AFSP result was implemented: learned
adequacy (COMET), register-fidelity LLM-as-Judge with a frozen evaluation-time
template, paired-bootstrap confidence intervals for pairwise condition
comparisons, and a reusable per-segment chrF/BLEU entry point so every metric can
be computed over any `<condition>_<split>.jsonl`. A shared loader keeps all
scorers aligned segment-for-segment, which the bootstrap requires. No COMET
checkpoint was downloaded and no judge API calls were made; the pieces that can be
checked without a model or network — bootstrap statistics, per-segment chrF/BLEU,
judge parsing/aggregation (via a mock client), lazy imports, and CLI wiring — were
verified offline, and the CI-equivalent static checks pass.

### What changed

`src/eval/_io.py` (new) centralizes loading: `load_condition(out_dir, condition,
split)` returns aligned `(sources, predictions, references)` from the inference
JSONL. `quick.py`'s `score` was refactored onto it, and `quick.py` gained
`segment_scores(preds, refs, metric)` — sentence-level chrF (default) or BLEU
(`effective_order=True`) — so chrF is now a reusable library entry point rather
than living only in the smoke sweep (addresses the Stage-4 chrF item).

`src/eval/comet.py` (new, `manage.py comet`) scores each condition with
`Unbabel/wmt22-comet-da` against the gold targets and writes **per-segment**
scores to `results/comet_<split>.json`. The `comet` package and the ~2.3 GB
checkpoint are imported/downloaded lazily inside `load_model`, so importing the
module needs neither; GPU is auto-detected with CPU fallback. `unbabel-comet==2.2.6`
was pinned in `requirements.txt`.

`src/eval/judge.py` (new, `manage.py judge`) + `prompts/judge_eval.txt` (new)
implement the **evaluation-time** register-fidelity judge, deliberately separate
from the RLSF training-time reward judge. The frozen template rates a candidate's
register against the authorized reference on a 1-5 rubric; the judge model is read
from a `judge:` block (`configs/judge_eval.yaml`, new — an OpenAI model at
temperature 0 + seed for determinism) and driven through the existing
`make_client`. `parse_score` prefers an explicit `Score: N` verdict and falls back
to the last lone 1-5 integer, rejecting `12`/`3.5` while allowing a sentence-final
`3.`; unparseable/refused segments become `None` and are dropped rather than
mis-scored. Per-segment scores, mean, and coverage are written to
`results/judge_<split>.json`. Cross-family confirmation is a second run with a
different-family judge block (commented in the config).

`src/eval/bootstrap.py` (new, `manage.py bootstrap`) implements segment-level
paired resampling (Koehn, 2004): `paired_bootstrap(a, b, n_resamples, alpha, seed)`
returns the observed mean difference, the two-sided 95% CI, a bootstrap p-value,
and a significance flag, dropping NaN pairs first. The CLI works on any metric —
chrF/BLEU computed on the fly, COMET/judge read from their JSON — compares each
condition against the ladder floor (`zeroshot`), and with `--adjacent` reports
each consecutive-rung difference. `manage.py` registers `comet`, `judge`, and
`bootstrap`; the README evaluation section documents the full flow.

### Verification

No COMET download, no judge API calls. CI-equivalent static checks pass with
`ruff==0.14.2`:

```bash
ruff check src
ruff format --check src
python -m compileall -q src
python tests/test_stylometrics.py    # 7/7
```

Offline behavioural checks (no model, no network): `paired_bootstrap` gives
`diff = 0` with a zero-width CI and non-significance on identical inputs, a
positive significant difference with CI excluding 0 on clearly-separated inputs,
is reproducible under a fixed seed, and drops NaN pairs (n falls accordingly);
`segment_scores` ranks a close match above a distant one for both chrF and BLEU
and rejects unknown metrics; `parse_score` handles the `Score: N`, bare-integer,
`3.`, `3.5`, `12`, refusal, and empty cases; `build_prompt` fills the frozen
template and is brace-safe against stray braces in inserted text; `score_condition`
+ `_aggregate` produce the right scores/mean/coverage against a mock client with a
simulated refusal; the `comet` module imports without `comet` installed and
`load_model` raises only when called. The `bootstrap` CLI was then run end to end
over synthetic fixtures for both the on-the-fly chrF path and the stored-JSON
COMET/judge path, and `manage.py` was confirmed to dispatch and argparse the three
new commands.

### Reproduction

```bash
pip install -r requirements.txt        # pulls unbabel-comet (heavy)
CONDS="zeroshot random_fewshot knn_fewshot afsp_margin afsp_full"
python manage.py eval         --conditions $CONDS --split val
python manage.py comet        --conditions $CONDS --split val
python manage.py judge        --conditions $CONDS --split val --config configs/judge_eval.yaml
python manage.py stylometrics --conditions $CONDS --split val
python manage.py bootstrap --metric comet --conditions $CONDS --split val --adjacent
```

### Limitations and risks

`unbabel-comet==2.2.6` was pinned but not installed or resolved in this
environment; COMET depends on `pytorch-lightning` and a `transformers` range that
may conflict with the pinned `transformers==5.12.1`, so the first real install
should confirm the resolution (and may need a separate environment). COMET was not
run, so no adequacy numbers exist for any condition.

The judge is an intentional commercial-API evaluator, not smoke, but it has not
been run: no register-fidelity scores exist yet, and the rubric/template will need
a small human-agreement check before the numbers are trusted. Judge determinism
relies on an OpenAI model honouring `temperature=0` + `seed`; the Anthropic
cross-family judge is non-deterministic (no temperature control in the client) and
should be treated as a confirmation pass, not a reproducible primary.

The paired bootstrap assumes conditions are aligned segment-for-segment (identical
eval order and count); the CLI asserts equal counts but cannot detect a reordering,
so all conditions must be generated from the same eval file. The two-sided p-value
is the standard twice-the-smaller-tail estimate and is granular at small
`n_resamples`; the default 10,000 is adequate for α = 0.05 reporting.

---

## 2026-07-04 — Stage 3: Ablation Ladder as Discrete Conditions + Glossary Decoupling

### Summary

The prompting arm was restructured from three conditions (`reference`,
`knn_fewshot`, `afsp`) into a five-rung **ablation ladder** of discrete
conditions, and the register glossary was decoupled from the AFSP arm so it no
longer confounds the selection-mechanism comparison. Two changes:
(1) `reference` was renamed to `zeroshot`, and `random_fewshot`, `afsp_margin`,
and `afsp_full` were added, so each rung adds exactly one component over the one
before it; (2) the `[Terms]` glossary line, previously emitted only inside
`build_afsp_user`, was moved into a single shared few-shot prompt builder and
promoted to a controlled `prompt:` flag applied uniformly to every few-shot
rung. No inference was run. The change was verified with the CI-equivalent
static checks, the stylometrics test, and an offline exercise of the new
selection and prompt-assembly paths.

### 1. The ladder

`src/infer/run.py` now defines `CONDITIONS = (zeroshot, random_fewshot,
knn_fewshot, afsp_margin, afsp_full)` and dispatches on it:

| Condition | Exemplars | Selection | β | λ | Isolates |
|---|---|---|---|---|---|
| `zeroshot` | none | — | — | — | instruction only |
| `random_fewshot` | k | random (seeded) | — | — | having examples at all |
| `knn_fewshot` | k | cosine top-k | 0 | 0 | relevance-based retrieval |
| `afsp_margin` | k | margin + hub | β>0 | 0 | AFSP margin/hub penalty |
| `afsp_full` | k | margin + register rerank | β>0 | λ>0 | full AFSP method |

`afsp_margin` and `afsp_full` are both served by the existing `AFSPRetriever`
via a `_select_afsp(..., rerank)` helper: `afsp_margin` forces `lambda_style = 0`
and passes `centroid=None`, so it exercises margin/hub selection only and does
**not** require the target-register centroid to be built; `afsp_full` uses the
configured `beta` and `lambda_style` and loads the centroid. This preserves the
Stage-0 ablation property that `β = 0, λ = 0` reduces AFSP to `knn_fewshot`, and
adds an intermediate rung isolating the margin from the register rerank.
`random_fewshot` samples `k` exemplars per source from the training pool via the
shared seeded `random.Random`, so it is reproducible across reruns.

`reference` is renamed `zeroshot` because "reference" was ambiguous with the gold
reference target used for scoring; "zeroshot" names precisely what the rung is
(instruction only, no in-context examples).

### 2. Glossary confound

`_term_line` was called only inside `build_afsp_user`, so the multi-view word
pairs rode along with AFSP alone. A win of `afsp` over `knn_fewshot` could
therefore have come from the glossary rather than the adaptive selection — a
confound. The two prompt builders (`build_knn_fewshot_user`, `build_afsp_user`)
were unified into one `build_fewshot_user(source, exemplars, glossary=None)`
shared by every few-shot rung; when `glossary` is empty its output is
byte-identical to the old baseline builder. The glossary is now loaded once by
`_load_configured_glossary` from a controlled `prompt:` block flag
(`word_pairs` + `glossary_file`, with a fallback to the legacy `afsp:` location)
and applied uniformly across `random_fewshot`, `knn_fewshot`, `afsp_margin`, and
`afsp_full`. It is therefore held constant across the selection ladder — no
longer an AFSP-only rider — and can be toggled as its own controlled factor.
`configs/base_qwen.yaml` moved `word_pairs`/`glossary_file` from the `afsp:`
block to the `prompt:` block accordingly.

`src/infer/sweep.py` (smoke harness) was updated to the new builder/label names,
and `src/eval/quick.py`'s default/example conditions were relabelled.

### Verification

No inference was run. CI-equivalent static checks pass with the pinned
`ruff==0.14.2`:

```bash
ruff check src
ruff format --check src
python -m compileall -q src
python tests/test_stylometrics.py    # 7/7
```

Two pre-existing one-line-docstring format deviations (in `run.py` from the
demonstration-ordering commit and in `stylometrics.py` from the Stage-1 commit),
which had left this branch's `ruff format --check` red, were also collapsed so
the format gate is green.

Offline (no generator, no embedding model, no API): the shared builder was shown
to omit `[Terms]` when the glossary is empty and to inject it when non-empty;
`_select_random` was confirmed seeded, per-source, and of length `k`;
`_load_configured_glossary` was confirmed to read the `prompt:` block, fall back
to the legacy `afsp:` location, and default to disabled; and `run("reference", …)`
was confirmed to raise, listing the five valid rungs.

### Reproduction

```bash
python -m src.retrieval.build_index --config configs/base_qwen.yaml
python manage.py stylometrics --build-centroid            # required by afsp_full only
python -m src.infer.run --condition zeroshot       --config configs/base_qwen.yaml
python -m src.infer.run --condition random_fewshot --config configs/base_qwen.yaml
python -m src.infer.run --condition knn_fewshot    --config configs/base_qwen.yaml
python -m src.infer.run --condition afsp_margin    --config configs/base_qwen.yaml
python -m src.infer.run --condition afsp_full      --config configs/base_qwen.yaml
python -m src.eval.quick --conditions zeroshot random_fewshot knn_fewshot afsp_margin afsp_full --split val
```

### Limitations and risks

Holding the glossary constant across the ladder resolves the confound but does
not by itself measure the glossary's effect; that requires an explicit contrast
(the same condition with `word_pairs` on vs. off), which the flag now makes
possible but which has not been run. `random_fewshot` draws from the whole
training pool without a topical floor, so at very small `k` its exemplars may be
off-topic — that is the intended contrast against `knn_fewshot`, but its variance
across seeds should be reported. The ladder has been exercised only offline on
the selection and prompt-assembly paths; no translations were generated, so no
register results exist for any rung.

---

## 2026-07-04 — Stage 1: AFSP Coherence Fixes (retrieval space, margin, register fit)

### Summary

The AFSP mechanisms implemented on 2026-07-03 were built but diverged from the
intended design in ways that would have silently degraded the method. This entry
records four corrective changes: (1) the retrieval index is committed to the
**source** side so the margin runs in one comparable space; (2) the margin terms
are confirmed consistent with that space; (3) the two heavy-tailed sentence-length
features are dropped from the register-fit centroid where they were dead weight;
and (4) the register rerank is changed from *toward-centroid* (which surfaced the
register-bland corpus average) to a directed **register-salience** score that
prefers register-bearing exemplars. No thesis inference was run. The changes were
verified with static checks, the stylometrics test, an offline corpus-geometry
analysis, and a handful of real validation queries through `AFSPRetriever.select`.

### 1. Source-side index (the space the margin lives in)

The margin in `src/retrieval/afsp.py` scores a candidate as
`cos(x, y) − β·½(hub(x) + hub(y))`, with all three terms read from
`self.index.embeddings`. That is only coherent if the index is a **single**
embedding space. The committed `build_index.py` embedded the English **target**
side, which made `cos(x, y)` and the query hubness `hub(x)` cross-lingual
(source→target) while the candidate hubness `hub(y)` was monolingual English —
three quantities in two different spaces, added together. This is the margin of
Artetxe and Schwenk (2019) applied incoherently.

A verification pass showed the on-disk `data/knn_index/embeddings.npy` was in fact
already **source**-side (cosine 1.00 to the source of pair 0, 0.86 to its target),
and its `meta.json` carried an `indexed_side: "source"` key that no committed code
ever wrote — the artifact had been rebuilt source-side out of band while the code
and docs still described a target-side, cross-lingual index.

Decision: **commit to the source-side index.** Retrieval is now monolingual on the
source side — a Persian/Arabic query is matched against the embedded training
sources, and each match maps back to its aligned English target. This makes the
margin's similarity, query hubness, and candidate hubness one comparable
source-side space, and it preserves the ablation property that `β = 0, λ = 0`
reduces AFSP exactly to the `knn_fewshot` baseline (both become source-side
top-k). Target-side register is scored separately by the style rerank from the
exemplar's English *text*, which needs no embedding, so nothing register-related
is lost by not embedding the targets. The alternative — keeping a cross-lingual
variant — was rejected: it would require a second (source) embedding index anyway
to compute a coherent candidate hubness, and it conflates semantic similarity with
translation-pair alignment quality.

`src/retrieval/build_index.py` now embeds `pairs[i]["input"]` and records
`indexed_side: "source"`. `src/retrieval/embed.py`'s query instruction was
rewritten from "Retrieve English translations …" (cross-lingual) to a
source-to-source instruction; on pair 0 this raised the self-similarity of the
query embedding against its own stored row from 0.929 to 0.967 and produced a
cleaner top-3 neighbourhood, confirming the old instruction was mismatched to the
source index. `retrieve.py`, `afsp.py`, the README, and `configs/*.yaml` retrieval
comments were updated to describe source-side retrieval.

### 2. Margin formula vs. the committed space

With the index committed to the source side, the existing margin is correct as
written: `sims`, `qhub` (`_topk_mean(sims)`), and `chub` (`_candidate_hubness`)
are all computed from the source embeddings, so `qhub` and `chub` are the two
halves of one symmetric same-space margin. No formula change was needed; the fix
was to guarantee the space, not to rewrite the algebra. The cached
`hubness_top{m}.npy` is source-space (built from the source embeddings) and stays
valid; per the cache-key caveat, it must be cleared only if the index is rebuilt.

### 3. Inert sentence-length features in the register centroid

In the target-register centroid the length features had centroid std ≈ 16.6
(`sent_len_mean`) and ≈ 58.4 (`sent_len_var`) against means of 24.7 and 6.7. Under
the z-scoring in `distance_to_centroid` such a large denominator makes their
standardized contribution vanish, so they were dead weight in any `lambda_style`
sweep, and they measure segment structure rather than register.

`stylometrics.py` now defines `CENTROID_FEATURES = [lex_density, ttr, root_ttr,
marker_rate]` and builds the centroid and the register-fit distance over that
subset. The two length features remain in `FEATURE_NAMES`, so the reporting table
and the H2 across-segment variance analysis still see them; only the register-fit
distance drops them. `distance_to_centroid` already iterated `centroid["features"]`
and needed no change; the `--build-centroid` printout was fixed to iterate the
centroid's own feature list rather than the full `FEATURE_NAMES`. The centroid was
rebuilt (n = 10,860, four features).

### 4. Register rerank: salience, not toward-centroid

`_style_fit` previously returned `−distance_to_centroid`, i.e. it preferred
exemplars **closest to the centroid**. Because the centroid is the corpus *mean*,
this selects the register-bland average and demotes strongly-styled exemplars. An
offline analysis over the 10,860 training targets confirmed it empirically:

* closest-5%-to-centroid mean marker rate 0.024 and closest-25% 0.019, both
  **below** the corpus mean of 0.033;
* the farthest-5% mean marker rate 0.136, and the top-5% highest-marker
  (most register-bearing) exemplars sit at the **93rd** distance percentile;
* `corr(distance, marker_rate) = +0.51`, `corr(distance, lex_density) = +0.36`.

So minimizing distance to the centroid actively pushes the most register-bearing
exemplars to the bottom — the opposite of the "target-distribution-priority" and
anti-neutral-register intent in `docs/afsp_strategies.md`.

The rerank now uses `register_salience`: the mean z-score of an exemplar's register
features relative to the centroid, in the register-positive direction (every
`CENTROID_FEATURES` entry — marker rate, lexical density, ttr, root_ttr — marks
stronger register when higher). `_style_fit` returns this salience directly (no
negation); larger = more strongly in-register. `distance_to_centroid` is retained,
undirected, for reporting how far a system's output distribution sits from the
target centre.

On a handful of validation queries through `AFSPRetriever.select`, raising
`lambda_style` from 0.0 to 0.5 shifted the mean marker rate of the selected
exemplars from 0.0197 to 0.0270 — toward stronger register, as intended. Unbounded
salience over the *whole* corpus tops out on degenerate two-word fragments
("O Lord.", marker rate 0.5), but these never enter the rerank: it operates only
within the `pool_mult·k` source-similarity pool, whose members are topically
relevant and of realistic length, so the pool constraint bounds the effect.

### Verification

No inference was run. CI-equivalent static checks pass on the touched sources:

```bash
ruff check src
ruff format --check src
python -m compileall -q src
python tests/test_stylometrics.py    # 7/7 (features() and FEATURE_NAMES unchanged)
```

Offline (no generator, no API): the source/target identity of `embeddings.npy` was
confirmed against the embedding model; the source-to-source query instruction was
shown to improve self-similarity and neighbourhood quality; the toward-centroid
bias was quantified over all training targets; and `AFSPRetriever.select` was run
on validation queries with `lambda_style ∈ {0, 0.5}` to confirm the salience
rerank shifts selection toward register-bearing exemplars.

### Reproduction

```bash
python manage.py build_index --config configs/base_qwen.yaml   # now source-side
python manage.py stylometrics --build-centroid                 # four register features
python manage.py infer --condition afsp --config configs/base_qwen.yaml
```

If `data/knn_index/` already holds a target-side `embeddings.npy` from before this
change, delete the directory (including `hubness_top*.npy`) and rebuild so the
index and its hubness cache are source-side.

### Limitations and risks

The register-salience direction assumes each `CENTROID_FEATURES` entry is
monotonically register-positive on this corpus, which held empirically here
(distance–marker and distance–lex correlations are positive) but is a
corpus-specific assumption to revisit if the feature set changes. Salience is
unbounded and length-sensitive in isolation; it is only safe because it reranks
within the source-similarity pool — a much larger `pool_mult` would begin to admit
short high-marker fragments and should be swept with that in mind. `lambda_style`
and `beta` still need a dev-split sweep; these fixes make that sweep meaningful
(non-inert features, coherent margin, correctly-signed register term) but do not
themselves select values.

---

## 2026-07-03 — Stage 0: Local Open-Source Generator (Qwen2.5-7B-Instruct)

### Summary

The local generator client was implemented, completing the last model-agnostic
gap in the inference pipeline. Until now the canonical base configuration
(`configs/base_qwen.yaml`) declared `provider: local` as a placeholder, and no
open-source generator existed: the pipeline could be exercised only through the
commercial-API smoke clients, which produce no thesis findings. `LocalChatClient`
now runs `Qwen/Qwen2.5-7B-Instruct` through HuggingFace `transformers`, exposing
the same `complete(system, user)` interface as the OpenAI and Anthropic clients,
so the four thesis conditions (reference, PEFT, knn_fewshot/AFSP, RLSF) can be
generated on the locked open-source base rather than a commercial API.

The decoding contract for every thesis run was fixed and documented as part of
this stage. The client and its wiring were verified end to end on the
`Qwen2.5-0.5B-Instruct` sibling, which shares the Qwen2.5 chat template and
tokenizer family. The full 5-segment reference smoke on the 7B base was **not**
run locally: the development GPU (RTX 4060 Laptop, 8 GB VRAM) cannot hold the 7B
base in `bfloat16` (approximately 15 GB), so that run is deferred to a
larger-memory environment (Colab) in order to keep the frozen base unquantized.
This entry records the implementation and the decoding decision; it reports no
translation quality results, and a 5-segment slice would in any case be
loop-validation, not a finding.

### What changed

`src/infer/local_client.py` was added, providing `LocalChatClient`
(`src/infer/local_client.py:26`). It is a dataclass mirroring the existing
clients: a `complete(system, user) -> str` method
(`src/infer/local_client.py:77`) and a shared `Usage` accumulator. `torch` and
`transformers` are imported lazily inside `__post_init__`
(`src/infer/local_client.py:43`) and `complete`, so importing the module — for
example from `make_client` — does not require the heavy dependencies or a GPU.

Generation applies the model's chat template to a system/user message pair
(`src/infer/local_client.py:84`) with `add_generation_prompt=True`, generates
with `torch.no_grad`, and decodes only the newly generated tokens by slicing off
the prompt prefix. Prompt and completion token counts are recorded through
`Usage`; because local weights carry no per-token price, the pricing table is
empty and `cost_usd` remains zero while token counts still accumulate for
throughput and reproducibility bookkeeping.

In `src/infer/run.py`, a `local` branch was added to `make_client`
(`src/infer/run.py:41`) that constructs `LocalChatClient` from the `generator`
block, reading `temperature`, `top_p`, `seed`, `dtype`, `device_map`, and
`load_in_4bit` with defaults. The error message for an unrecognized provider now
lists `local` (`src/infer/run.py:54`).

`configs/base_qwen.yaml` was updated: the `provider: local` placeholder comment
and the "client not implemented" note were removed, and the locked decoding
settings were made explicit and documented in the `generator` block.

`configs/qwen_smoke.yaml` was added as a loop-validation configuration. It mirrors
the base model and locked decoding but sets `data.limit: 5` and runs the
`reference` condition; it is labelled, like the commercial-API smoke configs, as
producing no thesis findings, the distinction being that it exercises the real
thesis base rather than a commercial API.

`requirements.txt` gained `transformers==5.12.1`, `accelerate==1.14.0` (required
for `device_map: auto`), and `bitsandbytes==0.49.2` (required only for the 4-bit
fallback).

### Decoding decision

Decoding is fixed identically across all four conditions so that cross-condition
differences are attributable to adaptation rather than to sampling. The locked
settings, recorded in `configs/base_qwen.yaml`, are:

```text
temperature   = 0.0     greedy decoding, fully deterministic (do_sample=False)
top_p         = 1.0     inert under greedy; used only if temperature is raised
seed          = 42      fixes any residual kernel nondeterminism / sampling RNG
max_new_tokens = 1024   covers a paragraph-length translation
```

Greedy decoding was chosen as the reproducible default: it removes sampling
variance entirely, so a run can be reproduced byte-for-byte. `temperature` and
`top_p` remain configurable for later sampling experiments, in which case `seed`
governs the RNG through `transformers.set_seed`.

`dtype`, `device_map`, and `load_in_4bit` are treated as hardware-placement
settings, not part of the decoding contract. The frozen thesis base is
full-precision `bfloat16`; `load_in_4bit` (NF4, approximately 5 GB) exists only as
a fallback for GPUs with 8 GB of VRAM or less and is off by default, because
enabling it would quantize the base model and therefore change the frozen-base
definition that underpins cross-condition attribution.

### Verification

The touched sources pass the CI-equivalent static checks:

```bash
ruff check src/infer configs
python -m py_compile src/infer/local_client.py src/infer/run.py
```

Wiring was validated without loading weights by parsing `configs/base_qwen.yaml`
and `configs/qwen_smoke.yaml` through `make_client`, confirming `provider: local`
dispatch, the propagated decoding settings, and the `complete(self, system, user)`
signature.

The generation path was then exercised end to end on `Qwen/Qwen2.5-0.5B-Instruct`,
the smallest member of the same model family and therefore sharing the Qwen2.5
chat template and tokenizer. Using the project style instruction and
`build_reference_user`, the client produced a correct English translation of an
Arabic test source, recorded non-zero prompt and completion token counts through
`Usage`, and — under the locked greedy configuration — produced byte-for-byte
identical output across two calls, confirming determinism. The 0.5B run validates
the client code path (chat-template application, generation, prompt-prefix
slicing, token accounting, and decoding); it does not characterize 7B behaviour.

The 7B smoke on the locked base was not run. `Qwen/Qwen2.5-7B-Instruct` in
`bfloat16` requires roughly 15 GB of weights and does not fit the development
GPU's 8 GB of VRAM.

### Reproduction

The 5-segment loop-validation smoke, intended for a GPU with at least 16 GB of
VRAM (for example a Colab T4/L4/A100) so the base loads in full `bfloat16`:

```bash
pip install -r requirements.txt
python -m src.infer.run --condition reference --config configs/qwen_smoke.yaml
```

The full development run uses the canonical configuration, which reads the entire
validation split (`data.limit: null`):

```bash
python -m src.infer.run --condition reference --config configs/base_qwen.yaml
```

`configs/base_qwen.yaml` sets `device_map: auto`, which requires `accelerate`. On
a GPU with 8 GB of VRAM or less, set `load_in_4bit: true` (requiring
`bitsandbytes`); this fits the model at the cost of quantizing the base, which
must then be documented as the base-model definition. The Qwen weights are fetched
from the HuggingFace Hub on first run; no `HF_TOKEN` is required for these public
weights.

### Limitations and risks

The end-to-end verification used the 0.5B sibling, not the 7B thesis base. It
confirms the client is correct as code but not that the 7B base loads and
generates within a given memory budget; that is confirmed only when the smoke runs
on adequate hardware.

The development GPU cannot run the locked base in full precision. Until the smoke
is run on a larger-memory environment, `configs/base_qwen.yaml` is implemented but
not yet exercised on its own base model. Enabling the `load_in_4bit` fallback to
run locally would change the frozen base to a quantized model and must not be
treated as equivalent to the `bfloat16` base.

A 5-segment run validates the loop only and is not an evaluation; no translation
quality numbers from `configs/qwen_smoke.yaml` may be reported as thesis findings.

The `accelerate` and `bitsandbytes` pins were selected as current releases
compatible with `transformers==5.12.1` but have not been installed and exercised
together in this environment; the first full `pip install -r requirements.txt` on
the target machine should confirm the resolution.

---

## 2026-07-03 — AFSP: Adaptive Few-Shot Prompting Implementation

### Summary

The AFSP method was implemented as an adaptive selection layer over the existing
`knn_fewshot` retrieval index, and an `afsp` condition was added to the
provider-agnostic inference pipeline. AFSP had until now been specified only in
`docs/afsp_strategies.md`, with an empty results row; the corresponding code now
exists.

This entry records the implementation only. The AFSP condition was not run: no
inference was performed, no API calls were made, and no results were produced.
The AFSP row in the README remains empty until the method is run on the
development split and, subsequently, the sealed test split.

### What changed

`src/retrieval/afsp.py` was added, providing `AFSPRetriever` and
`load_centroid`. In `src/infer/run.py`, the `afsp` condition was added together
with its prompt builder `build_afsp_user`, the glossary loader `load_glossary`,
and the helper `_term_line`; `afsp` was registered as a choice in the
`run.py` argument parser. `eval/quick.py` and `eval/stylometrics.py` accept
conditions as arbitrary strings and therefore require no change.

The implementation follows the four mechanisms of `docs/afsp_strategies.md`,
operating over the shared index of English training targets.

1. Margin-based scoring with hub penalization. The candidate score is
   `cos(x, y) − β · ½(hub(x) + hub(y))`, where `hub` is the mean cosine
   similarity of a row to its `knn_hubness` nearest neighbours, self-similarity
   excluded. This is the distance-form margin of Artetxe and Schwenk (2019) with
   a tunable `β`. Candidate hubness is precomputed in row batches and cached to
   `data/knn_index/hubness_top{m}.npy`.

2. Target-distribution-priority selection. A candidate pool of size `pool_mult·k`
   is drawn by margin score, and each candidate's English target is then scored
   by its standardized distance to the target-register centroid
   (`results/stylometrics_centroid.json`), reusing `stylometrics.features` and
   `distance_to_centroid`. The final ranking combines margin similarity and
   register fit, weighted by `lambda_style`.

3. Demonstration ordering. `select` returns the exemplars with the
   highest-scoring one last, adjacent to the test source. `build_afsp_user`
   preserves this order and does not reverse it, in contrast to the baseline
   builder.

4. Multi-view word-level pairs. A curated seed glossary
   `prompts/register_glossary.tsv` (`source_term<TAB>target_term`) is
   substring-matched against each exemplar and the query source, and the matches
   are inserted as a `[Terms]` line to emphasize register-bearing terminology.

A new `afsp:` block in `configs/base_qwen.yaml` exposes `beta`, `knn_hubness`,
`pool_mult`, `lambda_style`, `centroid_file`, `word_pairs`, and `glossary_file`.

By construction, `beta = 0` together with `lambda_style = 0` reduces AFSP to the
`knn_fewshot` baseline. The two parameters therefore isolate the contribution of
the adaptive components from that of naive retrieval, and support a direct
ablation.

### Verification

No inference was run. The touched sources pass the CI-equivalent static checks:

```bash
ruff check src
ruff format --check src
python -m compileall -q src
```

A separate offline check, loading no embedding model and making no API calls,
exercised the numeric helpers and prompt assembly against synthetic arrays and a
small in-memory index. It confirmed min-max normalization and top-k mean, the
batched candidate-hubness computation (which ranked a near-duplicate cluster as
hubs), the target-register fit (which ranked an archaic exemplar above a plain
one using the real stylometric centroid), glossary parsing, and the ordering and
`[Terms]` injection of the assembled prompt.

### Reproduction

Not yet run. AFSP requires the retrieval index and the target-register centroid
to be built beforehand:

```bash
python manage.py build_index --config configs/base_qwen.yaml
python manage.py stylometrics --build-centroid
python manage.py infer --condition afsp --config configs/base_qwen.yaml
python manage.py eval --conditions knn_fewshot afsp --split val
python manage.py stylometrics --conditions knn_fewshot afsp --split val
```

`configs/base_qwen.yaml` specifies `provider: local`, whose client is not yet
implemented, so an end-to-end run requires either that client or a
commercial-API generator block. Development tuning should sweep `beta`,
`lambda_style`, and `k` before any test-set run.

### Limitations and risks

The register glossary is a hand-curated seed list rather than a statistical word
alignment. Matching is plain substring matching over undiacritized forms and may
miss diacritized forms or over-match short terms; a subsequent revision should
align and extend the list against the corpus. The mechanism is inactive when the
file is absent or `word_pairs` is false.

Candidate hubness is an `O(N²)` precomputation over the index. It is batched and
cached, and is inexpensive at roughly 10,860 rows, but would need revisiting at a
larger scale. The cache key encodes only `knn_hubness`, so an index rebuilt at
the same path requires its `hubness_top*.npy` files to be cleared.

The target-register centroid must exist before the AFSP condition runs;
`load_centroid` raises an explicit error when it is missing rather than degrading
silently.

AFSP has not been evaluated against reference translations. Its effect on
register and adequacy relative to the `knn_fewshot` baseline is unknown until the
condition is run.

---

## 2026-07-03 — Model × Shot-Count Smoke Sweep Harness

### Summary

A sweep harness was added to compare commercial-API models across few-shot sizes in a single run, score each combination with the existing quick metrics, and emit artifacts for manual human evaluation. It runs the `reference` condition at `n = 0` and the `knn_fewshot` condition at each `n > 0`, so a single invocation covers the shot-count sensitivity sweep that the configs and README flag for the dev split.

Important: like the `*_smoke.yaml` configs, this is smoke and loop-validation only. The thesis base model is the local `Qwen/Qwen2.5-7B-Instruct`; nothing produced by this harness is a thesis finding. Its purpose is picking a sensible commercial default and eyeballing output quality.

### What changed

Added:

```bash
src/infer/sweep.py
```

and registered a `sweep` command in `manage.py`.

The module iterates over a set of models and a set of shot counts `n`, running every `(model, n)` combination on the same eval slice. It reuses the existing pipeline rather than duplicating it:

* prompt assembly via `build_reference_user` and `build_knn_fewshot_user` from `src/infer/run.py`;
* provider dispatch via `make_client`, so OpenAI and Anthropic models are driven through one interface;
* scoring via `sacrebleu` BLEU/chrF and the shared `_MARKERS` regex from `src/eval/stylometrics.py`, keeping metric definitions identical to the quick scorer.

A `MODEL_REGISTRY` maps `--models` names to generator blocks. The cheap models (`gpt-4o-mini`, `claude-haiku-4-5`, `claude-sonnet-4-6`) are the defaults; the flagships (`gpt-5.5`, `claude-opus-4-8`) are present but opt-in because they are far pricier.

Retrieval is model-independent, so exemplars for each `n` are retrieved once and reused across all models. A fresh client is constructed per `(model, n)` combination so the `Usage` accumulator reports isolated per-run token counts and cost.

Output filenames are now tagged with the model and shot count:

```bash
outputs/<model>__n<k>_<split>.jsonl
```

This directly addresses the overwrite risk recorded in the 2026-06-11 smoke entry, where the fixed `knn_fewshot_val.jsonl` filename would clobber earlier predictions across runs.

Three result artifacts are written per sweep:

```bash
results/sweep_leaderboard_<split>.json   # metrics + cost per (model, n), for ranking
results/human_eval_<split>.tsv           # one row per (segment × run) with blank score columns
results/human_eval_<split>.md            # readable side-by-side, grouped by segment
```

The TSV carries empty `adequacy_1to5`, `style_1to5`, and `notes` columns for hand scoring; the Markdown groups every system's output under each source and reference for quick reading. A leaderboard table sorted by chrF is also printed to the terminal.

### Rationale

The configs and README call for a shot-count sensitivity sweep over `k ∈ {2, 4, 8}` on the dev split, and for choosing a commercial default before the local pipeline exists. Doing this by editing YAML and re-running `src/infer/run.py` once per combination both overwrites outputs and makes cross-model comparison manual. A single harness that fans out over models and shot counts, tags its outputs, and aggregates a leaderboard removes that friction.

The human-evaluation artifacts exist because BLEU, chrF, and marker rate are only proxies. A side-by-side dump lets the register and adequacy of candidate outputs be judged directly before committing to a model or shot count.

### Verification

The harness was validated end-to-end on a cheap four-call slice:

```bash
python manage.py sweep --models gpt-4o-mini --n 0 2 --limit 2
```

This exercised retrieval, both conditions, tagged output writing, scoring, the leaderboard, and both human-eval files.

CI-equivalent checks pass on the touched sources:

```bash
ruff check src
ruff format --check src
python -m compileall -q src
```

The retrieval index required by `n > 0` was rebuilt first and reported `embeddings (10860, 1024)` over 10,860 pairs.

### Reproduction

```bash
python manage.py build_index --config configs/base_qwen.yaml   # if data/knn_index/ is absent
python manage.py sweep --models gpt-4o-mini claude-haiku-4-5 claude-sonnet-4-6 --n 0 2 4 8 --limit 25
```

Defaults match that full command, so `python manage.py sweep` is equivalent. Add `gpt-5.5` and `claude-opus-4-8` to `--models` for the flagship comparison. Required environment variables are `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`, which may live in a project-root `.env`.

### Limitations and risks

These are smoke numbers on commercial APIs, not thesis findings, and must never be reported as results.

API cost and wall-clock scale with `models × n × limit`; the flagship models are excluded from the defaults for this reason.

The leaderboard sorts by chrF purely as a display convenience. It is not a principled "best model" criterion — register fidelity in particular is better judged from the human-eval artifacts than from chrF or the crude marker-rate proxy.

The retrieval index is reloaded once per shot count when prompts are built, a minor inefficiency that is negligible at this corpus size but would matter if the sweep grew large.

Shot counts are bounded by the base model context window; the registry and defaults assume `n ≤ 8`.

The human-eval TSV score columns are intentionally blank and must be filled by a human; no automatic judge is applied at this stage.

---

## 2026-06-12 — Phase 2: Stylometrics Module

### Summary

A new stylometric evaluation module was added to measure whether model outputs match the target scriptural register. This module provides per-segment style features, builds a target-register centroid from the training targets, and reports the distance of system outputs from that centroid.

### What changed

Added `src/eval/stylometrics.py`, which implements the objective style-measurement layer promised in the README. This module also supports two later stages of the project: style-based reranking and the H2 variance hypothesis.

The module computes the following per-segment features:

* `lex_density`
* `ttr`
* `root_ttr`
* `sent_len_mean`
* `sent_len_var`
* `marker_rate`

The function `feature_vector(text)` returns these features in the fixed order defined by `FEATURE_NAMES`, so downstream reranking code can consume a consistent vector format.

A target-register centroid can now be built from the 10,860 English training targets and saved to:

```bash
results/stylometrics_centroid.json
```

The centroid stores per-feature means and standard deviations. Standard deviations are floored at `1e-9` so later z-scoring cannot fail because of a zero-variance feature.

The reporting functions compute per-condition feature means, across-segment standard deviations, and `stylo_dist`, the standardized Euclidean distance between a condition’s mean feature vector and the target-register centroid.

The archaic-marker regex `_MARKERS` is now defined in `stylometrics.py` and imported by `src/eval/quick.py`. This makes the smoke scorer and the stylometric report use the same marker definition, preventing metric drift.

The `stylometrics` command was registered in `manage.py`, and `tests/test_stylometrics.py` was added as a pytest-free assertion script. The tests include a hand-labelled lexical-density check.

### Rationale

The project requires a way to evaluate style preservation beyond BLEU and chrF. Standard MT metrics measure lexical overlap with references, but they do not directly measure whether a translation preserves the target register.

The stylometrics module provides a deterministic, lightweight alternative to full NLP parsing. Lexical density is computed using a register-aware function-word list rather than POS tagging. A word is treated as content unless it appears in `FUNCTION_WORDS`, which combines standard English function words with archaic and scriptural forms such as:

```text
thou, thee, thy, thine, ye, hath, doth, hast, art, unto, verily, whilst, wherefore
```

This avoids adding spaCy or NLTK dependencies while still capturing the register-specific grammatical forms that matter for this project.

Both raw type-token ratio and root type-token ratio are reported because scripture segments vary widely in length. Raw TTR is length-sensitive, while root TTR provides a more length-robust vocabulary-diversity signal.

### Verification

`ruff check` passed on all touched files.

The command below passed all seven sanity checks:

```bash
python tests/test_stylometrics.py
```

This included a hand-labelled excerpt from training target #0, where the expected values were:

```text
42 tokens
18 content words
31 unique types
lex_density = 18/42
```

The centroid was built over 10,860 training segments and produced plausible means:

```text
lex_density = 0.4344
ttr = 0.8540
marker_rate = 0.0327
```

Gold training targets reported `stylo_dist = 0.0` against their own centroid, as expected.

The per-condition reporting path was also exercised against quarantined smoke outputs for `reference` and `knn_fewshot` with `n = 25`.

The existing quick evaluator still runs, confirming that the `_MARKERS` import refactor did not break the smoke scorer.

### Reproduction

```bash
python tests/test_stylometrics.py
python manage.py stylometrics --build-centroid
python manage.py stylometrics --targets-split train
python manage.py stylometrics --conditions reference knn_fewshot --split val
```

### Limitations and risks

`lex_density` is approximate and should be described in the thesis as a function-word-ratio measure with a register-aware list, not as a POS-based linguistic metric.

Because archaic pronouns and auxiliary forms are counted as function words, a heavily archaic passage may score lower in lexical density. Therefore, `lex_density` and `marker_rate` are complementary signals, not redundant measures.

The `_MARKERS` regex is case-sensitive. As inherited from the quick scorer, sentence-initial `Thou` is not matched, while capital `O` is matched only as the vocative form. This was left unchanged to preserve existing marker-rate behaviour.

Sentence splitting uses a simple regex. Single-sentence segments receive `sent_len_var = 0`.

---

## 2026-06-12 — Phase 1: Repository Hygiene, Test Sealing, and Baseline Relabelling

### Summary

A repository hygiene pass was completed before implementing the full AFSP contribution. The goal was to prevent earlier commercial-API smoke results from being mistaken for thesis findings, protect the sealed test set, and correctly relabel the implemented retrieval method as a kNN few-shot baseline rather than AFSP.

### What changed

The repository was re-pointed to the selected open-source base model by adding:

```bash
configs/base_qwen.yaml
```

This config names the locked base model:

```text
Qwen/Qwen2.5-7B-Instruct
```

The local generator client is not yet implemented, so `provider: local` is currently a placeholder for the upcoming local-inference phase. The two commercial-API configs were clearly relabelled as smoke-only configs and should not be treated as thesis conditions or findings.

The test split was sealed. Earlier smoke configs pointed to:

```bash
data/splits/test.jsonl
```

The data key was renamed from `test_file` to `eval_file` and re-pointed to:

```bash
data/splits/val.jsonl
```

The inference script now derives the split tag from the eval-file name and writes outputs such as:

```bash
outputs/reference_val.jsonl
outputs/knn_fewshot_val.jsonl
```

This prevents validation or development runs from being mislabelled as test results. The real `test.jsonl` is reserved for the final evaluation run only.

Commercial-API smoke outputs were quarantined under:

```bash
outputs/smoke_api_quarantine/
```

They were renamed to reflect their actual status. For example:

```bash
afsp_test.jsonl → knn_fewshot_test.smoke.jsonl
```

A warning banner was also added to the earlier development-log entry explaining that the BLEU/chrF numbers are smoke checks only, not thesis findings.

The implemented retrieval method was relabelled from `afsp` to `knn_fewshot`, because it is a plain top-k cosine nearest-neighbour baseline rather than the full adaptive AFSP method.

The following renames were made:

```text
src/afsp/ → src/retrieval/
data/afsp_index/ → data/knn_index/
AfspIndex → RetrievalIndex
build_afsp_user → build_knn_fewshot_user
config block afsp: → retrieval:
condition afsp → knn_fewshot
```

The README now separates the `knn_fewshot` baseline from the planned AFSP contribution.

The notebook below was deleted because it used a misleading random row-level split:

```bash
notebooks/SplitData.ipynb
```

The canonical splitter is now:

```bash
src/data/split.py
```

### Rationale

The earlier API-backed runs were useful for validating the pipeline, but they should not be interpreted as thesis results. They used commercial APIs rather than the selected open-source model, and they touched the sealed test split.

Relabelling the current retrieval implementation as `knn_fewshot` keeps the experimental comparison honest. The future AFSP contribution should be evaluated against this baseline, not confused with it.

Sealing the test set also enforces the project’s intended train/dev/test discipline. Any prompt tuning, retrieval tuning, or debugging should happen on the validation split, not the final test set.

### Verification

`src/data/split.py` was re-run and reproduced the committed splits byte-identically. SHA-256 hashes of `train.jsonl`, `val.jsonl`, and `test.jsonl` matched the digests stored in:

```bash
data/splits/hashes.json
```

Split counts matched the manifest:

```text
train = 10,860
val   = 1,323
test  = 1,322
```

Cross-boundary deduplication removed:

```text
val drops  = 27
test drops = 35
```

The leakage audit passed.

After the rename, the following imports worked cleanly:

```python
from src.retrieval.retrieve import RetrievalIndex
from src.retrieval.build_index import build_index
```

The `run.py` and `quick.py` CLIs now expose the conditions:

```text
reference
knn_fewshot
```

A grep check found no residual `afsp` or `AfspIndex` references in `src/`, `configs/`, or `manage.py`, except in `docs/afsp_strategies.md`, where AFSP remains the planned method.

### Reproduction

```bash
python -m src.data.split
python -m src.retrieval.build_index --config configs/base_qwen.yaml
python -m src.infer.run --condition knn_fewshot --config configs/openai_smoke.yaml
python -m src.eval.quick --conditions reference knn_fewshot --split val
```

### Limitations and risks

`configs/base_qwen.yaml` is not yet runnable end-to-end because the local generator client has not been implemented.

The smoke configs remain the only runnable inference path at this stage, and they are still API-backed. Their smoke-only labels must remain intact, and their numbers should not be reported as thesis results.

`data/knn_index/` is git-ignored and must be rebuilt with `build_index`.

AFSP itself is still unimplemented. The README and development log should continue to describe AFSP as the planned contribution, and the AFSP results row should remain empty until the method exists.

---

## 2026-06-12 — Tokenizer Fertility Vetting of Qwen2.5-7B-Instruct

### Summary

Tokenizer fertility was measured to determine whether `Qwen/Qwen2.5-7B-Instruct` is a suitable open-source local base model for Arabic/Persian-script source text.

### What changed

Added:

```bash
src/eval/fertility.py
```

and registered a `fertility` command in `manage.py`.

The module loads the source side of one or more splits, tokenizes each sentence with a Hugging Face tokenizer using:

```python
add_special_tokens=False
```

It reports:

* corpus-level fertility
* mean per-sentence fertility
* median fertility
* p90 fertility
* maximum fertility

The script also prints a threshold-based verdict:

```text
< 2.5      lock
2.5–4.0    borderline
4.0+       catastrophic
```

### Rationale

Before committing to an open-source 7–8B local model, the project needed to verify that the model’s tokenizer handles Arabic/Persian-script source text efficiently.

Fertility is used as a proxy for tokenizer efficiency:

```text
fertility = subword tokens / whitespace words
```

Very high fertility indicates byte-fallback behaviour, which inflates sequence length and can degrade model quality. The decision rule was set in advance: a value near or below 2.5 would be acceptable, while a value around 4–5 or higher would be unacceptable.

### Result

`Qwen/Qwen2.5-7B-Instruct` was tested on the full corpus:

```text
13,505 sentences
212,140 source words
```

Results:

| Metric            | Full corpus | Train split |
| ----------------- | ----------: | ----------: |
| Corpus fertility  |      2.7000 |      2.6987 |
| Mean per-sentence |      2.7383 |      2.7393 |
| Median            |      2.7000 |      2.6991 |
| p90               |      3.2500 |      3.2500 |
| Max               |      7.0000 |      7.0000 |

The final fertility value was:

```text
2.70 subword tokens/source word
```

This is slightly above the ideal 2.5 target but far below the catastrophic 4–5+ range. The distribution is also tight, with the median close to the corpus mean and p90 at 3.25.

Decision:

```text
Qwen2.5-7B-Instruct is locked as the local base model.
```

### Reproduction

```bash
python manage.py fertility --model Qwen/Qwen2.5-7B-Instruct --split all
python manage.py fertility --model Qwen/Qwen2.5-7B-Instruct --split train
```

This requires `transformers`. The Qwen tokenizer is fetched from the Hugging Face Hub on first run. No `HF_TOKEN` is needed for this public tokenizer.

### Limitations and risks

The source corpus is largely Arabic/Persian-script rather than Persian only. The methodology section should describe it accordingly.

Fertility measures tokenizer efficiency only. It does not measure translation quality or register preservation. Those must be evaluated through the downstream translation and evaluation pipeline.

Word count uses whitespace splitting, which may slightly undercount Persian orthographic words joined by ZWNJ. This can slightly inflate fertility and therefore makes the decision conservative.

The maximum fertility value of 7.0 appears to come from a very short sentence, where ratios are naturally unstable.

---

## 2026-06-11 — API-Backed Smoke Pipeline

### Summary

A complete provider-pluggable inference and scoring loop was added to validate the retrieval-prompting pipeline before building the local open-source model pipeline.

Important: this entry describes smoke testing only. These runs are not thesis findings.

### What changed

The following components were added:

```bash
prompts/style_instruction.txt
configs/openai_smoke.yaml
configs/anthropic_smoke.yaml
src/infer/usage.py
src/infer/openai_client.py
src/infer/anthropic_client.py
src/infer/run.py
src/eval/quick.py
```

The smoke configs define provider-specific generator settings and shared retrieval settings.

The inference pipeline supports two conditions:

```text
reference
knn_fewshot
```

The `reference` condition uses the shared style instruction and a direct translation prompt.

The `knn_fewshot` condition retrieves similar training examples and inserts them into the prompt as few-shot demonstrations.

The API clients expose a common interface:

```python
complete(system, user) -> str
```

Each client also records usage through a shared `Usage` object, including token counts and estimated cost.

The quick evaluator computes:

* BLEU
* chrF
* archaic marker rate
* reference marker rate

### Rationale

The supervisor requested validating the approach against commercial APIs before building the local 7–8B pipeline.

This smoke phase de-risked the model-agnostic parts of the project:

```text
retrieval → prompt assembly → generation → output writing → scoring
```

Using API models allowed the pipeline to be tested without first implementing local inference or GPU-based training.

### Method and dependencies

Retrieval used brute-force cosine similarity with NumPy rather than FAISS. At approximately 10,860 rows and 1024 dimensions, retrieval is only a matrix multiplication, so FAISS was deferred.

The OpenAI client supports reasoning-style request behaviour by using:

```python
max_completion_tokens
```

and by omitting unsupported parameters when necessary.

The Anthropic client similarly omits parameters unsupported by newer Claude models and steers output formatting through the prompt.

Dependencies added and pinned:

```text
openai==2.41.1
anthropic==0.109.1
sacrebleu==2.6.0
PyYAML==6.0.3
python-dotenv==1.2.2
```

The following paths are git-ignored:

```bash
.env
outputs/
data/knn_index/
```

### Smoke result

The smoke run used `n = 25` and `k = 4`, on the **first 25 segments of
`data/splits/test.jsonl`** — at this date the smoke configs still pointed at the
test split, which is the defect the 2026-06-12 Phase 1 entry corrected by sealing it.
The outputs were written as `*_test.jsonl` and are now quarantined under
`outputs/smoke_api_quarantine/` with `.smoke.` in their names.

These results are not thesis findings. They are retained only as evidence that the pipeline executed end-to-end.

| Provider / model            | Condition   |  BLEU |  chrF | Marker rate | Reference marker rate |
| --------------------------- | ----------- | ----: | ----: | ----------: | --------------------: |
| OpenAI `gpt-4o-mini`        | reference   | 20.83 | 47.61 |        0.44 |                  0.28 |
| OpenAI `gpt-4o-mini`        | knn_fewshot | 19.56 | 47.30 |        0.48 |                  0.28 |
| Anthropic `claude-opus-4-8` | reference   | 22.30 | 50.00 |        0.48 |                  0.28 |
| Anthropic `claude-opus-4-8` | knn_fewshot | 23.72 | 51.12 |        0.32 |                  0.28 |
| OpenAI `gpt-5.5`            | reference   | 26.16 | 52.14 |        0.40 |                  0.28 |
| OpenAI `gpt-5.5`            | knn_fewshot | 25.15 | 51.90 |        0.36 |                  0.28 |

The smoke results showed that stronger base models achieved higher overlap scores. The retrieval condition sometimes moved archaic-marker rate closer to the reference distribution, especially on the stronger models. However, because this run used commercial APIs and an early test-set smoke setup, the numbers should not be interpreted as research findings.

### Reproduction

The commands below are written against the post-sealing layout of the 2026-06-12
Phase 1 entry, in which the smoke configs read `data/splits/val.jsonl` and the
inference script derives the split tag from the eval file. Re-running them therefore
reproduces the pipeline but not the table above, which was produced on the test
split before sealing and must not be regenerated there.

```bash
python manage.py build_index --config configs/openai_smoke.yaml
python manage.py infer --condition reference --config configs/openai_smoke.yaml
python manage.py infer --condition knn_fewshot --config configs/openai_smoke.yaml
python manage.py eval --conditions reference knn_fewshot --split val
```

For Anthropic, use:

```bash
python manage.py build_index --config configs/anthropic_smoke.yaml
python manage.py infer --condition reference --config configs/anthropic_smoke.yaml
python manage.py infer --condition knn_fewshot --config configs/anthropic_smoke.yaml
python manage.py eval --conditions reference knn_fewshot --split val
```

Required environment variables:

```bash
OPENAI_API_KEY
ANTHROPIC_API_KEY
```

These can also be stored in a project-root `.env` file.

### Limitations and risks

The smoke configs are API-backed and do not represent the final open-source model condition.

The quick evaluator is only a smoke scorer. The final evaluation still needs COMET, paired-bootstrap confidence intervals, LLM-as-Judge, and stronger style analysis.

The marker-rate regex is a crude register proxy.

Output filenames should include provider/model tags before retaining future cross-provider results, otherwise later runs may overwrite earlier predictions.

---

## 2026-06-08 — Django-Style `manage.py` Command Dispatcher

### Summary

A repository-level `manage.py` dispatcher was added to provide a single command interface for the project pipeline.

### What changed

A lightweight command dispatcher was added at the repository root.

It maps user-facing commands to pipeline modules. Initially, it included:

```text
preprocess → src.data.preprocess
split      → src.data.split
```

Additional commands were later registered for retrieval, inference, evaluation, fertility, and stylometrics.

The dispatcher removes the subcommand from `sys.argv` before importing the target module. This preserves each module’s existing `argparse` behaviour and allows all flags to pass through unchanged.

### Rationale

This provides a single, discoverable command interface:

```bash
python manage.py <command>
```

This is easier than requiring users to remember full module paths such as:

```bash
python -m src.data.preprocess
python -m src.data.split
```

It also makes future extension simple: adding a new pipeline stage only requires adding one entry to the `COMMANDS` dictionary.

### Reproduction

```bash
python manage.py --help
python manage.py preprocess
python manage.py split --seed 7
```

### Limitations and risks

This is not a Django application entry point. It does not introduce Django settings, apps, or project configuration.

It is only a dispatch shim.

New commands must be registered in the `COMMANDS` dictionary, and each target module must expose a callable `main()` function.

---

## 2026-06-08 — Preservation of Arabic Diacritics and Hamza-Bearing Characters

### Summary

Source-side normalization was revised to avoid destructive transformations of Arabic orthography.

### What changed

Two normalization procedures were updated.

First, in `src/data/preprocess.py`, `PERSIAN_STANDARDIZATION_MAP` was reduced to three safe mappings:

```text
ي → ی
ك → ک
ـ → removed
```

The pipeline no longer folds hamza-bearing characters such as:

```text
أ, إ, ؤ, ٱ
```

It also no longer normalizes:

```text
ة, ۀ
```

Second, in `src/data/split.py`, the `normalize_key()` function used for cross-boundary deduplication no longer strips Arabic harakat.

`PUNCT_RE` was expanded to preserve Arabic combining marks because Python’s `\w` class does not include combining marks and could remove diacritics unintentionally during punctuation stripping.

The now-redundant `is_source` parameter was removed.

### Rationale

The source corpus contains mixed Persian and Arabic scripture. Approximately 76% of cleaned source sentences contain harakat. In Arabic, hamza-bearing letters, teh marbuta, and diacritics can encode lexical or grammatical distinctions.

Treating these marks as orthographic noise causes two problems:

1. It corrupts the stored source text by replacing meaningful distinctions with normalized approximations.
2. It increases false-positive deduplication by collapsing genuinely distinct verses into identical normalized keys.

The decision to preserve these features was confirmed with the author.

### Result

The number of retained sentences increased from 13,563 to 13,567.

The resulting split sizes were:

| Split      | Sentences | Ratio |
| ---------- | --------: | ----: |
| Train      |    10,860 | 80.4% |
| Validation |     1,323 |  9.8% |
| Test       |     1,322 |  9.8% |

Cross-boundary deduplication drops decreased from 153 to 62 records:

```text
validation drops = 27
test drops       = 35
```

The leakage audit continued to pass, with zero overlapping normalized keys between the training split and validation/test splits.

### Reproduction

```bash
python -m src.data.preprocess
python -m src.data.split
```

### Limitations and risks

The `metadata.source` field is also normalized, but grouping keys are Latin work names, so work-level grouping is unaffected.

The retrieval index is built over English training targets, and future retrieval stages must continue to avoid leakage from validation or test references.

---

## 2026-06-08 — Work-Level Split with Cross-Boundary Deduplication

### Summary

The original random sentence-level split was replaced with a work-level split plus cross-boundary deduplication.

### What changed

The previous random `train_test_split` procedure was replaced by a deterministic work-level split.

Entire works, books, or documents are assigned to a single split. After this assignment, validation and test records are removed if their normalized source or target text already appears in training.

The splitter writes:

```bash
data/splits/train.jsonl
data/splits/val.jsonl
data/splits/test.jsonl
data/splits/hashes.json
```

### Rationale

The corpus contains Bahá’í scripture translated from Arabic and Persian into English. This material includes substantial formulaic repetition, such as repeated invocations.

A random sentence-level split could place identical or near-identical sentences across train, validation, and test partitions.

This is especially problematic because the retrieval index is built from the English target side of the training set. If a validation or test sentence appears in training, retrieval can find a trivial exact match and inflate evaluation metrics.

The README design required splitting by document or section rather than random row, so the earlier random split was inconsistent with the intended experimental design.

### Method

The revised split procedure has five stages.

First, each sentence is grouped by `metadata.source`, corresponding to the originating work.

Second, whole works are assigned to train, validation, and test using deterministic bin packing. Works are sorted largest-first and placed into the split with the greatest shortfall relative to the target ratio. Ties are resolved deterministically by work name.

Third, cross-boundary deduplication is applied. Training is kept intact. A validation or test record is removed if its normalized Arabic/Persian source input or normalized English target output already appears in training.

Fourth, the script performs a leakage audit. It asserts that there are zero overlapping normalized keys between training and validation/test. The run fails if leakage remains.

Fifth, the script writes a manifest, `hashes.json`, containing split configuration, fractions, seed, normalization rules, pre- and post-deduplication counts, final ratios, work-to-split assignment, and SHA-256 hashes for each output file.

### Result

| Split      | Sentences | Ratio | Works |
| ---------- | --------: | ----: | ----: |
| Train      |    10,850 | 80.9% |    15 |
| Validation |     1,242 |  9.3% |     7 |
| Test       |     1,318 |  9.8% |     6 |

Cross-boundary deduplication removed:

```text
validation records = 121
test records       = 32
```

The leakage audit passed.

Repeated runs produced byte-identical file hashes, confirming deterministic behaviour.

The corpus contains 28 distinct works. The 193 records with unknown provenance are forced into training so that they cannot inflate validation or test performance.

### Reproduction

```bash
python -m src.data.split
```

Optional flags:

```bash
--train_frac
--val_frac
--test_frac
--seed
--group_key
--input_file
--output_dir
```

### Limitations and risks

Split ratios are approximate because entire works must remain intact. Exact 80/10/10 ratios are not always possible.

The whole-work holdout setting creates a harder evaluation regime than a random split. This should be noted when reporting model performance.

The deduplication procedure is normalized-exact rather than fuzzy. It captures differences in casing, punctuation, whitespace, and selected orthographic variants, but it does not detect paraphrases or near-duplicates involving word substitution or larger phrase-level changes.

If evaluation scores remain unexpectedly high, fuzzy near-duplicate detection such as MinHash or Jaccard similarity should be considered.


---

## References

Works cited by name in the entries above, together with the models and metric
implementations the pipeline depends on.

* Artetxe, M., and Schwenk, H. (2019). Margin-based parallel corpus mining with
  multilingual sentence embeddings. *Proceedings of ACL 2019*. — the margin score with
  hub penalization used by `src/retrieval/afsp.py`.
* Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and
  Chen, W. (2021). LoRA: Low-rank adaptation of large language models. — the PEFT arm,
  via the `peft` library.
* Koehn, P. (2004). Statistical significance tests for machine translation evaluation.
  *Proceedings of EMNLP 2004*. — the paired bootstrap of `src/eval/bootstrap.py`.
* Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. (2002). BLEU: a method for
  automatic evaluation of machine translation. *Proceedings of ACL 2002*.
* Popović, M. (2015). chrF: character n-gram F-score for automatic MT evaluation.
  *Proceedings of WMT 2015*.
* Post, M. (2018). A call for clarity in reporting BLEU scores. *Proceedings of
  WMT 2018*. — BLEU and chrF are computed with `sacrebleu`.
* Qwen Team (2024). Qwen2.5 technical report. — `Qwen/Qwen2.5-7B-Instruct`, the frozen
  thesis base.
* Rei, R., et al. (2022). COMET-22: Unbabel-IST 2022 submission for the metrics shared
  task. *Proceedings of WMT 2022*. — the `Unbabel/wmt22-comet-da` checkpoint.
* Wang, L., et al. (2024). Multilingual E5 text embeddings: a technical report. — the
  `intfloat/multilingual-e5-large-instruct` retrieval encoder.

The AFSP method statement and its precedents are cited in `README.md` and
`docs/afsp_strategies.md`; this log records implementation decisions only.
