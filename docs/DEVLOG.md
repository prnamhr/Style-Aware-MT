# Engineering and Decision Log

This log tracks the engineering work behind the Style-Aware Machine Translation project. Entries are ordered from newest to oldest and record the decisions that changed the dataset, pipeline, model runs, or generated artifacts.

Each entry explains what changed, why it changed, how to reproduce it, and what limitations still apply. This is a record of engineering decisions rather than the main results document. Numbers are included when they explain a decision; the thesis results are reported in `README.md` and the files under `results/`.

## Conventions

- Dates use `YYYY-MM-DD`. The date belongs to the work recorded in the entry and comes from the cited git history. If an entry covers more than one day, the range is stated.
- Entries are ordered by date, newest first.
- The standard section headings are `Summary`, `What changed`, `Rationale`, `Verification`, `Reproduction`, and `Limitations and risks`. Some entries add a descriptive subheading when a specific analysis needs one, and early entries may omit sections that did not apply.
- Code references use `path:line`. Line numbers are historical: they refer to the code as it existed on the entry date and are not updated after later edits.
- Commands are kept as exact command-line invocations. A fenced block that lists files or artifact paths is not necessarily shell input.
- When a stage writes data, the entry names the artifact and records the integrity check used for it, such as a hash or manifest.

### Notation

- Φ is the evaluation-time LLM-as-Judge register-fidelity score: the mean over
  segments of a 1-5 rubric rating of a candidate translation against the authorized
reference (`prompts/judge_eval.txt`; see the 2026-07-05 entry). Higher is better. It is distinct from the training-time reward judge used by RLSF.
- `stylo_dist` is the undirected standardized Euclidean distance from a condition's
  mean feature vector to the target-register centroid; `register_fit` is the
directional band-pass distance introduced on 2026-07-18. Lower is better for both.
- `eval_loss` is completion-only token-level cross-entropy on `val.jsonl`.

### Stage and phase labels

Two labelling schemes appear in the titles. Entries of 2026-06-12 use *Phase N*; entries from 2026-07-03 onward use *Stage N*; the 2026-06-08 entries and several later ones use neither. The schemes are historical artifacts of the project and are not one numbering - *Phase 2* (stylometrics) and *Stage 2* are unrelated. Stage 2 has no entry: the demonstration-ordering flag (commit `4e7647d`, 2026-07-04, PR #4 `feat/afsp-stage2-fixes`) is referenced only in passing by the Stage 3 entry.

### Provenance of these entries

The four entries dated 2026-07-18, 2026-07-23, 2026-07-25, and 2026-07-31 were written retrospectively on 2026-07-31, reconstructed from committed artifacts, configs, and git history rather than at the time of each change. The 2026-07-24 PEFT-implementation entry and all entries before 2026-07-18 were written at the time of the change. Where a rationale was not recorded in the repository it is marked as unrecorded rather than inferred.

### Known documentation gaps

Changes visible in the git history that no entry records. They are listed here so the absence is deliberate rather than an oversight; none has a recorded rationale.

- `270b96c` (2026-07-16) re-cut the sweep grid to k ∈ {1, 2, 4, 8, 16} and added the
  zero-shot anchor row. The 2026-07-18 entry describes the grid it replaced, not this
intermediate grid.
- `568ba73` (2026-07-19) restored λ = 0.5 to `DEFAULT_LAMBDAS`, completing the six-value
  λ axis that the 2026-07-23 run used.
- `446eca8` (2026-07-18) removed the lazy `comet` import from `src/eval/comet.py`,
  which the 2026-07-13 entry had relied on. See that entry's limitations.
- The trained adapter's output location moved from `outputs/peft/adapter`
  (2026-07-24 entry) to `models/peft_lora_r{r}_lr{lr}/checkpoint-{step}`
(`configs/peft_sweep.yaml`, `sweep.output_base`). No entry records the move.
- The judge provider switch of 2026-07-20 has no recorded rationale; this is noted in
  the 2026-07-23 entry.

Added by the 2026-08-10 audit:

- Three analysis stages have no entry and produce artifacts that the results rest on:
  `src/eval/stylometrics_ci.py` (`results/stylometrics_ci_val.json` - the paired CIs and
rank distributions behind the `stylo_dist` ladder), `src/eval/metric_agreement.py` (`results/metric_agreement_val.json` - all of RQ4), and `src/eval/peft_register.py` (`results/peft_register_val.json`). Until this audit none of the three was named in `README.md` either, and the README asserted that `stylo_dist` could not be resampled.
- `bd806a7` (2026-08-10) added `src/rlsf/pool.py`, `src/rlsf/omega.py` and
  `tests/test_rlsf_pool.py` - the dev-slice best-of-N pool and the offline ω grid - under a
commit message describing the drift-band change instead. `docs/budget.md` prices both stages; no entry records their implementation.
- `src/eval/human.py` is a human-judgment scaffold committed 2026-07-23. It has never been
  run, holds no artifact, and no entry states whether human evaluation is in scope.

These gaps are open. Nothing in this list has been closed since it was written.

### Audit

A second audit on 2026-08-10 checked `README.md` and `docs/budget.md` against the artifacts rather than this log: split integrity and cross-split leakage, the provenance of the retrieval index, every number in the results table against the file it cites, the bootstrap estimator, and which pipeline stages are recorded anywhere. What it changed is listed under *Known documentation gaps* above and in the README's threats table; it opened no new entry because it recorded no new engineering work.

This log was audited for internal consistency against the committed artifacts, configs, and git history on 2026-08-01. Corrections made in that pass are marked inline as *Correction (2026-08-01)*; they amend the claims of earlier entries but do not restate their history.

## 2026-08-26: The fourteen rows generated on test, and an authorization written after the spend

### Summary

Steps 3, 4 and 5 of `docs/preregistration_test.md` ran. The leakage audit flagged 21 of 1,322 test
segments, below the declared 2.56% trigger, so no quarantined rebuild happened. Twelve local rows
generated at zero cost and two commercial rows cost $10.5462 over 2,644 calls, against a $10.91
projection. `outputs/test_manifest.json` records the fourteen rows at freeze commit `88de192`, and
its digests reproduce against the files on disk.

The step 2 authorization did not exist when the paid rows ran. It is now recorded in
`docs/budget.md` under today's date, and it says there what it says here: the line follows the
spend rather than preceding it.

### The authorization was written after the spend

Step 2 of the order of operations puts a dated authorization in `docs/budget.md` before step 5. No
such line existed when the two paid rows ran. The last commit to that file was `5f597f8` on
2026-08-20, six days earlier, and it stated that the test split was sealed.

The ceiling the run actually enforced was `AUTHORIZED_USD = 15`, a constant in section 10 of
`notebooks/test_generation_colab.ipynb`. It was asserted twice, against the $10.9084 projection
before the pilots and against the pilot-derived revised total before the full passes, and both
assertions held. The cell in that section which names `docs/budget.md` prints its git log and then
a reminder that the authorization must already be a dated line there; it reads nothing and asserts
nothing, so an absent authorization passes it.

What is recorded now is a correction, not the control the pre-registration asked for. The gap is
worth stating precisely because the two lines still unspent are the larger ones: Phi_A and Phi_B
over fourteen conditions are projected at $18.70 to $18.88 and $12.20 to $12.25, roughly three
times what generation cost. Each needs its dated line before it runs.

### Step 3: leakage audit

`results/leakage_test.json` reports 21 of 1,322 test segments flagged, 1.589%, against the 2.56%
trigger: 28 flag pairs over 28 distinct pool rows. The thresholds are the val audit's - cosine
0.95, Jaccard 0.7 over character 4-grams, top 10 neighbours - and the val pass flagged 17 of 1,323,
1.285%. The trigger did not fire. The retrieval conditions were not regenerated on a quarantined
index, `data/splits/pool_quarantine.json` is byte-identical to the val-only list rebuilt on
2026-08-23, and the manifest records `quarantine.applied: false`.

The 28 pool rows reproduce the 28 that the 2026-08-23 audit found against test and that were then
dropped from the quarantine, with `results/leakage_test.json` deleted. The file is written again
here, under `--unseal-test`, as the diagnostic the pre-registration declares.

### Step 4: the twelve local rows

Twelve rows, 1,322 calls each, `cost_usd: 0.0` in every sidecar. Generation took 11,975 seconds,
about 3.3 hours, on an RTX 5090 under torch 2.12.0+cu130, transformers 5.12.1, peft 0.18.0 and
python 3.12.13; `peft_knn` was the slowest at 1,478 s and `zeroshot` the fastest at 587 s. The seal
was opened by deriving `configs/test/*.yaml` from the committed configs rather than editing them,
and the manifest carries a digest for each derived file. Section 9 of the runbook asserts zero cost
across the twelve, no rater key in the environment, the test digest, and a working tree with no
non-test change.

Two sidecars did not survive the session as written. `outputs/zeroshot_test_usage.json` and
`outputs/random_fewshot_test_usage.json` were committed at `a21de13` with `calls: 0`; `f6e694e`
restored the provider-reported counts committed at `88d91d1`. The generation files themselves never
changed - both are untouched since `88d91d1` and both still match their manifest digest. The
runbook records the mechanism in section 10: a repeated pass over a finished output rewrites the
usage sidecar with zeros. The pre-registration's discard rule reads a *nonzero* cost on a local row
as evidence of paid routing, and a zeroed sidecar passes both that rule and the section 9
assertion while carrying no evidence at all. That is why the counts were put back rather than left.

### Step 5: the two commercial rows

Each row ran a 20-segment pilot and then a full pass that resumed over it for 1,302 further calls.

| Row | Pilot rate | Projected rate | Realized rate | Calls | Cost |
|---|---:|---:|---:|---:|---:|
| `commercial_haiku` | $4.650e-4 | $4.774e-4 | $4.707e-4 | 1,322 | $0.6222 |
| `gpt56_sparse_knn` | $8.165e-3 | $7.774e-3 | $7.507e-3 | 1,322 | $9.9240 |
| **Total** |  |  |  | **2,644** | **$10.5462** |

The GPT-5.6 pilot came in at 1.05 times its projection, inside the 1.25x guard, and the full pass
then ran below the projection: the pilot's 20 segments were longer-prompted than the split average.
Realized spend is 0.967 times the projection and 0.70 times the $15 the notebook held.

The kernel was lost before section 11. The two commercial rows therefore carry `seconds: null` and
a `finished` timestamp taken from the output mtime, marked as such under `timing_source`; the
twelve local rows carry measured wall clock. The final seal and the manifest write ran in a fresh
kernel over the artifacts on disk rather than in the session that produced them, which is why the
manifest's freeze commit is `88de192` while the artifacts landed across `88d91d1` to `f6e694e`.

### Verification

Recomputed from the files rather than read out of the manifest:

- All fourteen `output_sha256` values in `outputs/test_manifest.json` match a fresh SHA-256 of the
  corresponding `outputs/<condition>_test.jsonl`.
- `data/splits/test.jsonl` is `3e24e90f`, the digest in `data/splits/hashes.json` and in the
  manifest.
- Each of the fourteen files holds 1,322 rows, in the source order of `test.jsonl`, with no empty
  prediction.
- The twelve local sidecars report 1,322 calls and $0.00 each; the four commercial sidecars sum to
  $10.5462 over 2,644 calls.
- The working tree is clean at `f6e694e`.

No code changed in this pass. The digest binding added at `a9f3caf` has not been exercised yet:
nothing has been scored, so no `_meta.json` carries an `outputs` block for a test condition.

### Reproduction

```bash
python manage.py leakage --config configs/sparse_retrieval.yaml --split test --unseal-test
```

Sections 5, 7 and 10 of `notebooks/test_generation_colab.ipynb` behind `SEAL_OPEN` and `SPEND_OK`,
with the derived configs under `configs/test/` at the digests the manifest records. Rerunning
generation does not reproduce these files: greedy decoding is not byte-stable across sessions, and
a rerun would rewrite the usage sidecars with zeros and break every digest in the manifest.

### Limitations and risks

The authorization order is the defect this entry records and cannot repair. A dated line written
after the fact fixes the record, not the control, and the control only works before the next paid
line - steps 6 and 7.

The two commercial rows have no measured wall clock. Their costs are provider-reported and
unaffected; only the timing is inferred.

The restored counts in the two zeroed sidecars come from the earlier commit, not from a fresh
measurement. Nothing recomputes prompt and completion tokens from the outputs, so those two rows
are provider-reported at one remove.

The 21 flagged segments stay in the primary table. The trigger did not fire, and holding the
retrieval pool identical across the two splits was declared as worth that exposure.

Steps 6 to 9 have not run. Until they do, the fourteen files are generations with no score bound to
them, and any regeneration after scoring begins is money spent twice.

## 2026-08-26: Test-split generation declared before the seal is opened

### Summary

Steps 2-5 of the order of operations in `docs/preregistration_test.md`: the spend authorization,
the leakage audit, and the two generation passes. Written before any of them has run. `outputs/`
holds no `*_test.jsonl`, `results/` holds no `*_test*.json`, and `data/splits/test.jsonl` is
`3e24e90f`, 1,322 segments over 8 works, unchanged in git since `f78ec23`.

Steps 3 and 4 spend nothing and may run first. Step 5 does not start until `docs/budget.md`
carries a dated authorization, and this entry exists so that what the run consists of is fixed
while none of its numbers is available.

### Step 2: authorization

The projection uses the per-call rates measured on the most recent val passes, not the older
planning rates.

| Line | Calls | Rate | Cost |
|---|---:|---:|---:|
| `Phi_A`, `claude-haiku-4-5`, 14 conditions | 18,508 | $1.010e-3 to $1.020e-3 | $18.70 to $18.88 |
| `Phi_B`, `gpt-5.6-terra`, Batch 50%, 14 conditions | 18,508 | $6.589e-4 to $6.619e-4 | $12.20 to $12.25 |
| `gpt56_sparse_knn` generation | 1,322 | $7.774e-3 | $10.28 |
| `commercial_haiku` generation | 1,322 | $4.774e-4 | $0.63 |
| **Total** | **39,660** |  | **$41.81 to $42.04** |

The GPT-5.6 rate is the val pass blended over its pilot and its full run, 1,323 calls for $10.2853;
quoting the full run alone understates it by the pilot's $0.129. The two rater rates are the last
recorded session of each: $1.3365 for 1,323 `Phi_A` calls, $0.8757 for 1,323 `Phi_B` calls.

This is larger than any single authorization the project has recorded. The val rater pass was
authorized at $5.74, and the $25 RLSF cap covers the training reward judge only; neither extends
here. Two reductions are recorded now so that taking one later is a budget decision rather than a
redesign: dropping `gpt56_sparse_knn` removes $10.28 of generation and about $2.20 of rater calls,
and restricting both raters to the six conditions the confirmatory contrasts name removes roughly
$13 more. Neither touches the ten confirmatory tests.

### Step 3: leakage audit

```bash
python manage.py leakage --config configs/sparse_retrieval.yaml --split test --unseal-test
```

Without `--write-quarantine`. That flag writes `leak.quarantine_file`, which is
`data/splits/pool_quarantine.json` - the val-only 22-row list rebuilt on 2026-08-23 - and a test
audit would overwrite it with a list conditioned on the sealed split. The audit writes
`results/leakage_test.json` and nothing else.

The trigger is declared in `docs/preregistration_test.md`: if more than 2.56% of test segments are
flagged as near-duplicates of a pool row, twice the val rate of 1.28% (17 of 1,323), the retrieval
conditions are regenerated against a test-quarantined index and reported as a sensitivity check.
Otherwise no quarantined rebuild happens.

This reverses what the 2026-08-23 entry anticipated. That entry said a test-time quarantine has to
be written at the final pass and the index rebuilt from it. The primary pass instead retrieves from
`data/knn_index`, the unquarantined 10,860-row pool that every val run used, because pruning the
pool for test and not for val changes the method between the two splits, which is a larger defect
than the near-duplicate rate it removes. The audit still runs; its result is reported; the rebuild
is conditional on the trigger rather than automatic.

### Step 4: local generation

Twelve rows on the frozen local base, listed with their configs in the pre-registration's condition
table. The seal is opened by pointing `data.eval_file` at `data/splits/test.jsonl`. `manage.py
infer` carries no test guard - unlike `afsp_sweep`, `afsp_verify`, `peft_sweep` and `leakage`,
which all refuse it - so the opening is a config edit, and recording it here is what makes it
deliberate rather than incidental.

Nothing is re-selected. Decoding is the locked greedy setting, retrieval is k = 8 at
`most_similar_last`, AFSP is the frozen (k, λ, β, σ), sparse is `min_df: 40` with the 500-term list
at rarity digest `8fa5b0b2`, and the adapters are the PEFT checkpoint and the three RLSF
checkpoints selected on the dev slice. The full list is the frozen-settings table in the
pre-registration.

Each row writes `outputs/<condition>_test.jsonl` and a `_usage.json` sidecar. For the twelve local
rows that sidecar must record zero calls and $0.00; a nonzero one means a condition was routed to a
paid provider and the row is discarded rather than reported.

### Step 5: external generation

`commercial_haiku` and `gpt56_sparse_knn`, both paid, each preceded by a `--limit 20` pilot whose
realized per-call rate is checked against the projection above before the full pass runs. The val
GPT-5.6 pass used a 15-row pilot the same way and its cost sidecar is kept separate from the full
run's, which is why the blended rate above is recoverable at all.

Neither row supports a confirmatory test. `commercial_haiku` is an external reference and
`gpt56_sparse_knn` is a generator-family diagnostic; both change the model family and the compute
budget, and the README already records that the GPT-5.6 row is descriptive only.

### Verification

Nothing in this entry has been run. What was checked before writing it:

- `outputs/` and `results/` contain no artifact matching `*test*`; the test split has never been
  generated on.
- `sha256sum data/splits/test.jsonl` reproduces `3e24e90f...`, the digest in
  `data/splits/hashes.json`.
- `pytest tests/` passes at `a9f3caf`, 437 tests.
- The rates in the step 2 table were re-read from the committed usage sidecars, not from
  `docs/budget.md`'s older planning figures.

The score-to-generation binding added earlier today at `a9f3caf` is in place, so the test pass is
the first run under it: every judge and COMET record will carry the SHA-256 of the file it scored.

### Reproduction

```bash
# 3
python manage.py leakage --config configs/sparse_retrieval.yaml --split test --unseal-test

# 4, one row shown; the twelve differ only in --condition and --config
python manage.py infer --condition knn_fewshot --config configs/base_qwen.yaml   # eval_file: test

# 5, pilot then full
python manage.py infer --condition zeroshot --config configs/commercial_haiku_zeroshot.yaml --limit 20
python manage.py infer --condition sparse_knn --config configs/commercial_gpt56_sparse_knn.yaml --limit 20
```

`notebooks/test_generation_colab.ipynb` is the runbook for steps 3, 4 and 5. It gates every cell
that reads the sealed split behind a hand-set `SEAL_OPEN`, derives the test configs into
`configs/test/` rather than editing the committed ones, asserts the frozen settings back out of
each config before generating, and writes `outputs/test_manifest.json` with the SHA-256 of every
generated row - the same digest the raters and COMET will bind to.

The twelve local rows run with no API key in the kernel, asserted, so they cannot spend. Step 5 is
gated separately behind `SPEND_OK` and an `AUTHORIZED_USD` that has to cover the projection. Each
commercial row runs a 20-segment pilot first; its usage sidecar is copied to
`<name>_test_pilot_usage.json` before the full pass, because `manage.py infer` overwrites that file
per invocation and the val GPT-5.6 pass showed the blended rate is otherwise unrecoverable. The
full pass runs only if the realized per-call rate is within 1.25x the projection and the revised
total still fits the authorization. A pilot that reports `cost_usd` of 0 fails rather than passing,
since an unpriced model would make the rate guard inert.

### Limitations and risks

The audit at step 3 is the third recorded read of the test split, after the marker statistics of
2026-08-20 and the quarantine audit of 2026-08-23. It reaches no selection decision: its only
possible consequence is the conditional sensitivity rerun, and that decision rule was fixed before
the audit ran.

The 2.56% trigger is set from the val flagged rate. A test rate just below it leaves the
near-duplicate exposure in place rather than removing it, and that is the accepted cost of holding
the retrieval pool identical across the two splits.

Local generation is not byte-reproducible across sessions even at greedy decoding; the README
records the drift and the `sparse_knn` rows in the val table were invalidated by exactly that. The
binding from `a9f3caf` makes a regeneration after scoring fail loudly instead of silently, but it
does not make generation stable. If a test row is regenerated after it is judged, its rater scores
are spent and have to be bought again.

Steps 6-9 are not covered here. The scoring passes, the ten confirmatory tests and the exploratory
reporting are declared in `docs/preregistration_test.md` and will get their own entry once they
have run.

## 2026-08-26: Scores bound to the generation bytes they scored

### Summary

Nothing linked a score to the file it was computed from. `manage.py judge` and
`manage.py judge_batch` now record the SHA-256 of `outputs/<condition>_<split>.jsonl` in the
segment cache's `_meta.json` and refuse to extend a cache whose recorded digest no longer matches
the file; `manage.py comet` writes the same digest into each condition's record.

### Rationale

The judge segment cache resumes on the source text: `score_condition` compares each cached
`input` against the current sources and stops there. Sources are fixed by the split, so a
condition regenerated between judge runs passes that check with different predictions, and the
cached scores carry over to translations that were never judged. The README already records
cross-session decoding drift under greedy decoding, so this is not hypothetical. The failure is
silent, the raters are paid per call, and the test pass happens once.

### What changed

- `src/eval/_io.py`: `file_digest`, a chunked SHA-256 of a generation file.
- `src/eval/judge.py`: `bind_output(cache_dir, condition, source_path)` records
  `outputs.<condition> = {file, sha256}` in `_meta.json` and raises on a digest change, naming the
cache file to delete. `assert_cache_identity` compares only the keys it is passed, so the new block
does not disturb the (model, tag, template) identity check.
- `src/eval/judge.py`, `src/eval/judge_batch.py`: the digest is bound before scoring and stored as
  `output_sha256` on each condition's result record.
- `src/eval/comet.py`: `output_sha256` on each condition's record.
- `tests/test_judge_provenance.py`: six tests covering the digest, first binding, idempotent
  rebinding, the refusal on a regenerated file, independence across conditions, and that the
identity check still passes once an outputs block exists.

### Verification

`pytest tests/` passes, 437 tests, 6 of them new. `ruff check` and `ruff format --check` pass on
the touched files.

### Limitations and risks

The existing validation artifacts are not backfilled. They were scored before the binding existed,
and writing today's digests into them would assert a link that was never checked; `judge_val.json`
and `comet_val.json` therefore carry no `output_sha256` and the val segment caches carry no
`outputs` block until a condition is re-scored. The binding proves that a cache and a file agree
now, not that the file is the one a paid call actually saw - that guarantee starts with the first
run under this code, which is the test pass.

`manage.py eval` and `manage.py stylometrics` print rather than write per-condition results, so
neither carries a digest. Both recompute from the file on every invocation, which is what makes
the stale-cache hazard specific to the judge.

## 2026-08-25: Register direction derived from the corrected centroid

### Summary

The signed register direction used by AFSP and the PEFT proxy selector is now derived rather than copied across configs. `src/eval/register_direction.py` computes the four coefficients from the current training targets and the corrected `results/stylometrics_centroid.json`, `manage.py register_direction` exposes the derivation, and `results/register_direction.json` is the single artifact consumed by all seven configs that previously carried the four numbers inline.

### Derivation

For each of the 10,860 non-empty training targets, the script computes the four centroid features with the current stylometric extractor, z-scores them against the committed training-target centroid, and takes the Euclidean norm of that per-segment z-vector. Each direction coefficient is then the Pearson correlation between that distance and the corresponding feature z-score. Before fitting, the script rebuilds the centroid statistics from the same targets and refuses to continue if the means or standard deviations differ from the committed centroid. This makes a stale centroid or a later feature-definition change fail loudly instead of silently changing the direction.

The same construction reproduces the former hard-coded coefficients to three decimals when the pre-2026-08-20 case-sensitive archaic-marker regex is substituted: `lex_density 0.355`, `ttr 0.098`, `root_ttr -0.296`, and `marker_rate 0.514`. That closes the provenance gap noted in the 2026-07-18 audit: the values were correlation coefficients from the old centroid geometry, not an unexplained normalized loading vector.

With the corrected case-insensitive archaic marker calculation, the derived direction is:

```text
lex_density  +0.3217233193
ttr          +0.0475045503
root_ttr     -0.2734255957
marker_rate  +0.4141160886
```

### Predeclared comparison rule

The comparison thresholds are constants in `src/eval/register_direction.py` and are written into the result artifact. They are defined on the geometry the band-pass objective actually uses. Because `register_band_distance` divides by the total absolute direction weight, a common scale change cancels; sign changes, vector rotation, and changes in each feature's share of total absolute weight are the relevant quantities.

- **Near-identical:** no sign flips, cosine similarity at least 0.98, and maximum absolute normalized-weight-share shift below 0.05.
- **Very different:** any sign flip, cosine similarity below 0.90, or maximum normalized-weight-share shift at least 0.15. This is the precommitted trigger for regenerating `afsp_full`.
- **Materially different:** anything between those two regions.

### Result

The corrected direction is **near-identical** to the historical one under that rule. Cosine similarity is 0.996200, the angle is 4.996 degrees, there are no sign flips, and the largest normalized-weight-share movement is 0.032640. `marker_rate` remains the largest absolute weight, but its share falls from 0.40697 to 0.39187; `ttr` has the largest share change, falling by 0.03264. The raw coefficient shift is larger, especially for `marker_rate` (0.514 to 0.414), but common-scale movement is not preserved by the normalized band-pass distance and therefore is not used as the rerun gate.

As an additional offline check, the 19 already-generated AFSP sweep outputs were rescored under the corrected direction while holding their translations, chrF values, centroid, and `select_target_sigma = 0.5` fixed. The maximum absolute change in `register_fit` was 0.009390, and the recommendation remained `k = 8`, `lambda_style = 0.75`. This is a re-score of existing outputs, not a rerun of exemplar selection or generation.

Decision: the direction correction does **not** trigger a new `afsp_full` generation. The marker-rate case fix did reach AFSP through the corrected centroid itself, as the existing case-fix sensitivity runs show, but it did not materially rotate or reweight the signed direction. The defect's remaining blast radius through the direction coefficients is therefore small.

### What changed

- Added `src/eval/register_direction.py` and registered `python manage.py register_direction`.
- Added `results/register_direction.json` with the derivation provenance, corrected coefficients, predeclared comparison thresholds, comparison result, and offline AFSP sweep re-score.
- Replaced the inline `style_register_direction` mapping in `base_qwen.yaml`, `afsp_sweep.yaml`, `qwen_smoke.yaml`, `peft_sweep.yaml`, `peft_anchor_e3.yaml`, `peft_smoke.yaml`, and `peft_afsp.yaml` with `style_register_direction_file: results/register_direction.json`.
- Updated the AFSP inference/sweep and metric-agreement loaders to read the generated artifact. Inline mappings remain accepted only for historical configs and unit tests.
- Added `tests/test_register_direction.py` to pin reproducibility of the derived artifact, the one-artifact config wiring, the very-different rerun trigger, and the three-decimal recovery of the legacy vector under the pre-case-fix marker regex.

### Reproduction

```bash
python manage.py register_direction
python manage.py register_direction --no-sweep-impact
```

The first command derives the direction and, when the committed AFSP sweep outputs are present, includes the offline selection-objective re-score in the same artifact. The second derives only the direction and vector comparison.

### Limitations and risks

The derivation identifies the direction through correlation with distance from the training-target centroid. It is a corpus-derived operational definition, not a causal estimate of which features create perceived register. The AFSP sweep impact check rescored already-generated translations; it does not prove that every query would retrieve the same exemplars under the corrected direction. The predeclared gate says a fresh `afsp_full` run is required only for a very-different direction, and that gate was not crossed.

## 2026-08-25: Orphan-condition status and repository cleanup

### Summary

The result inventory was reconciled with the thesis-facing README. `afsp_full_casefix` and `peft_afsp_casefix` are retained as sensitivity-only robustness diagnostics and are now named explicitly in `README.md`; they are not main study conditions. `gpt56_sparse_knn` is retained in the README as an external generator-family diagnostic with its available validation metrics (COMET, chrF, BLEU, `Phi_A`, and full stylometric distance). Its matched-generation companion `gpt56_knn_fewshot` is cut from thesis results because it does not have the complete scoring needed for a controlled retrieval comparison. The GPT-5.6 Sparse-KNN row is therefore descriptive only and is not used to claim a sparse-retrieval effect.

Repository cleanup removed the local Hugging Face cache and the stale `outputs/sparse_knn_val_old.jsonl`. The requested transfer archives were not present in the supplied tree. `.gitignore` now blocks Hugging Face cache directories, transfer archives, and `*_old.jsonl` output residue.

## 2026-08-24: The sparse condition reduced to one exemplar per rare query word

### Summary

Greedy rarity-weighted coverage, the redundancy penalty, the `max_df` ceiling and the surprisal ranking were removed from the `sparse_knn` condition. The condition is now: freeze the 500 rarest training words by document frequency, take the four rarest of them a query carries, retrieve the nearest training example for each, and fill the remaining prompt slots with ordinary kNN. The rebuild and validation selection pass have now been completed with the current `min_df: 40` configuration.

### What changed

- `src/retrieval/rarity.py:83` ranks the vocabulary by document frequency, ascending, breaking ties on total frequency and then on the token string. `char_surprisal`, the `max_df` ceiling, the `rank` switch and the IDF computation are gone, and the payload carries `[term, df, tf]` per term in rank order.
- `src/retrieval/rarity.py:110` `load_irregular` now returns `{term: rank}` from the written order rather than `{term: idf}`. Rank is what the channel sorts by, so the list file is the ranking.
- `src/retrieval/sparse.py:83` `_nearest_per_term` walks the targeted terms rarest first and takes, for each, the pool row carrying it with the highest cosine to the query that no earlier term has taken. `_greedy`, the redundancy penalty and `min_query_terms` are gone.
- The trace records `n_knn` and `served_terms` in place of `coverage`; a targeted term is unserved only when every pool row carrying it is already in the prompt.
- `configs/sparse_retrieval.yaml` and `configs/sparse_knn.yaml` reduce to `min_df: 40`, `freeze_n: 500`, `zwnj: keep` and `sparse.m: 4`. `src/infer/run.py:275` records `m`, `min_df` and `freeze_n`.

### Rationale

Greedy coverage was the reason the previous list looked dead. It maximises rarity weight covered per exemplar, so one exemplar carrying three of a query's rare terms ends the search and the other three slots fall to cosine kNN. The diagnostic recorded 3.021 targeted terms per query and 1.559 filled slots: the channel was working as specified and the specification was wrong. One exemplar per term makes the slot count equal the match count, so the 33.7% of val queries carrying four rare words become full-dose prompts instead of 1.06%.

The `max_df` ceiling is redundant under a df ranking. Taking the 500 rarest terms above a floor caps document frequency wherever the 500th term happens to sit, so the ceiling is an outcome rather than a parameter. The floor is not redundant: it is the only thing standing between the list and the 11,668 terms at df 1, which appear in one training sentence each and which no val query carried. The earlier band sweep recorded below showed that the useful coverage regime begins around df 30-50; the finalized configuration uses `min_df: 40`, leaving 525 eligible terms from which the frozen 500 are selected.

Ranking by df rather than by character surprisal costs little here. Under the finalized `min_df: 40` run, 525 terms are eligible and 500 are frozen, so the list still keeps almost the whole eligible band; df is the quantity the condition is about.

### Verification

`tests/test_rarity.py` covers the df ranking and both tie-breaks, the floor, the exact list size, and the reload path from the written file. `tests/test_sparse.py` covers one exemplar per term at the nearest pool row, no training example used twice, terms served rarest first, a term absent from the pool leaving its slot to the fill, the cap at `m`, and the slot arithmetic 0-4 rare + the rest kNN = 8. `pytest tests/test_rarity.py tests/test_sparse.py` passes, 31 tests. `ruff check` and `ruff format --check` pass. The completed real rebuild is recorded in `results/rarity_train.json`, and the validation selection pass is recorded in `results/sparse_selection_val.json`.

### Result

The gate is passed. `results/rarity_train.json` records 500 frozen terms from 525 eligible terms at `min_df: 40`, with realized df range [40, 410]. `results/sparse_selection_val.json` records 563 full-route queries, 639 partial-route queries, and 121 dense-only queries over 1,323 validation segments: 42.55% fill all four rare slots, above the 30% gate. Mean sparse slots per prompt are 2.646, all targeted terms are served, and mean intra-set cosine is 0.8975 versus 0.9001 for the dense baseline.

### Reproduction

```
python manage.py rarity --config configs/sparse_retrieval.yaml
python manage.py sparse_select --config configs/sparse_retrieval.yaml --split val --index_dir data/knn_index
```

### Limitations and risks

The condition no longer controls exemplar redundancy. The redundancy penalty was what kept the rarity picks from being near-copies of each other; with one exemplar per term, two rare terms that co-occur in the pool can return two similar examples. `results/sparse_selection_val.json` still reports intra-set cosine against the dense baseline, which is now the only check on that.

A word is served by the training example nearest the query among those carrying it, which is a cosine choice, not a register choice. Nothing verifies that the retrieved example uses the rare word in the same sense.

Artifacts built before this change remain historical records of the IDF-ranked or surprisal-ranked variants. The current sparse results are the `min_df: 40` one-exemplar-per-term artifacts named above and reported in `README.md`.

## 2026-08-24: The frozen rarity list was ranked by character surprisal inside a df band

### Summary

The frozen 500-term rarity list was an alphabetical accident, and the sparse channel it fed almost never fired. The pre-fix artifact recorded every selected term at df 2 with one shared IDF value, 9.194. The list was therefore re-specified as the 500 most character-unusual terms inside a document-frequency band. That intermediate rebuild and validation diagnostic were subsequently completed; the final sparse condition above now supersedes this design with df ranking and `min_df: 40`.

### What changed

- `src/retrieval/rarity.py:57` `char_surprisal` scores each term as the mean per-character negative log-probability under a character 4-gram model fit to the pool vocabulary, each type weighted by its token frequency, with `^^^word$` padding and add-0.5 smoothing over the observed alphabet. No external lexicon or frequency list enters the score.
- `src/retrieval/rarity.py:107` `term_stats` now keeps raw counts, so it carries `tf` and `surprisal` alongside `df` and `idf`. The postings matrix stays binary.
- `src/retrieval/rarity.py:140` `frozen_terms` filters to `min_df <= df <= max_df` and takes the `freeze_n` highest-surprisal terms. The term string remains as a tie-break, now a determinism guard rather than the operative key.
- This intermediate version used a bounded df band and surprisal ranking. The current `configs/sparse_retrieval.yaml` and `configs/sparse_knn.yaml` that supersede it use `min_df: 40`, `freeze_n: 500`, and no `max_df` or `rank` setting.
- `src/infer/run.py:280` provenance records `max_df` and `rank` with the other rarity settings.
- The review TSV is now the head of the list in rank order (`{stem}_top{n}.tsv`) with a surprisal column, replacing the seeded sample. `review_sample` is gone: it existed because an IDF sort of a tied tail had no meaningful head.
- `src/retrieval/sparse.py` is unchanged. `targeted_terms` still takes the m highest-IDF matches per query; the df band is what gives it IDF spread to rank with.

### Rationale

`smooth_idf` IDF over a 10,860-document pool is a function of df alone, and the tail is dominated by ties: 11,668 of 22,789 terms sit at df 1 and 3,669 at df 2. Sorting the df >= 2 vocabulary by IDF therefore ranks 3,669 terms equally, and the truncation to 500 was decided entirely by the alphabetical tie-break. What that selected was the head of the Persian collation order, not marked vocabulary.

Rarity as IDF also measures the wrong thing here. A term appearing twice in the pool is usually a segmentation artifact or a proper name, not scriptural register. Character surprisal separates the two directly: a word whose character sequence is improbable under the corpus's own orthography is marked, and one assembled from frequent sequences is not. The df band then keeps the score honest at both ends, dropping one-off noise below `min_df` and the vocabulary every register carries above `max_df`.

### Verification

`tests/test_rarity.py` covers the surprisal score (an off-pattern term outranks the dominant pattern; the ranking follows `tf` rather than type counts; the score is a per-character mean), the closed df band, the exact list size, strictly falling surprisal with nothing eligible passed over, and determinism under document shuffling. `pytest tests/test_rarity.py tests/test_sparse.py` passes, 29 tests. `ruff check` and `ruff format --check` pass. `manage.py rarity` was first exercised end to end on a synthetic 400-document pool outside the repository. The intermediate real rebuild and validation diagnostic were completed later, as recorded in the Result below.

### Result

The intermediate list was rebuilt and the validation diagnostic run after this entry was first written. The `[30, 500]` band admitted 695 of 22,789 pool terms and the frozen 500 realized df [30, 490], IDF 4.10 to 6.86. On validation it produced 3.021 listed terms per query against 0.07 under the original IDF ranking, 88.5% of queries carrying at least one against 6.4%, and 33.7% carrying four or more. These are retained as band-sweep evidence, not as the final sparse configuration.

The go/no-go failed for this intermediate selector. Only 1.06% of queries filled all four rarity slots, against the 30% gate, because greedy coverage stops as soon as the chosen exemplars cover the targeted terms: mean slots filled was 1.559 while mean targeted terms was 3.021. The list was not the binding constraint; the selection rule was. The band sweep in the same session put the alternatives at mean listed terms per query of 0.147 for `[2, inf]`, 0.921 for `[10, 1000]`, 3.021 for `[30, 500]` and 2.892 for `[50, 300]`. That evidence is retained as the reason to keep the final df floor in this range; the current run uses `min_df: 40`, and after the one-exemplar-per-term change `results/sparse_selection_val.json` clears the gate at 42.55% full-route queries.

### Reproduction

```
python manage.py rarity --config configs/sparse_retrieval.yaml
python manage.py sparse_select --config configs/sparse_retrieval.yaml --split val --index_dir data/knn_index
```

### Limitations and risks

The df band is narrow relative to the pool: `results/rarity_train.json` records 2,113 terms at df 10-99 and 173 at df 100+, so `[30, 500]` may admit not much more than the 500 the list keeps, leaving the surprisal rank little to select over. The notebook sweeps the band for this reason.

A character n-gram model fit to the pool scores loanwords and Arabic-origin forms as unusual, which is the intended signal, but it also scores any residual segmentation error as unusual for the same reason. The rank-ordered TSV is the check on that, and it is a manual one.

The intermediate artifacts in this entry remain historical. `README.md` and the current sparse artifacts now describe the finalized `min_df: 40`, df-ranked, one-exemplar-per-term condition.

## 2026-08-23: The pool quarantine was audited against the sealed test split

### Summary

The leakage audit in `notebooks/rarity_retrieval_colab.ipynb` was run as `manage.py leakage --split val test --write-quarantine`. `src/retrieval/leakage.py:161` unions the flagged pool rows over every split it audits, so `data/splits/pool_quarantine.json` held 50 rows: 22 flagged by val, 28 flagged only by test. `build_index --quarantine` then dropped all 50 to write `data/knn_index_clean` (10,810 of 10,860 rows). A pool pruned that way is a function of `data/splits/test.jsonl`, so anything retrieving from that index is conditioned on the sealed split.

The audit also wrote `results/leakage_test.json`, which reports the near-duplicate flags and the max-cosine histogram of the test split against the pool.

### What it touched

Nothing that the README reports. `configs/sparse_knn.yaml` and `configs/sparse_retrieval.yaml` both set `index_dir: data/knn_index`, the unquarantined 10,860-row pool, and `results/sparse_selection_val.json` records that directory in its config block. The only artifacts built on the quarantined index are the four `results/sparse_sweep_val_t{1,2,3,4}.json` route-fraction diagnostics from the `min_query_terms` sweep, which select nothing and appear in no results table. No generation, metric, judge, or bootstrap artifact derives from it.

### What changed

`data/splits/pool_quarantine.json` was rebuilt from the val flags alone: 22 pool rows, 0.20% of the pool, `splits: ["val"]`, against the same `train.jsonl` digest `a70014b8`. The list is reconstructed from `results/leakage_val.json`, which the same audit wrote, so it is identical to what `--split val --write-quarantine` emits. `results/leakage_test.json` was deleted.

Two guards now stand where the protocol was only a convention:

- `src/retrieval/leakage.py:135` refuses `test` in `--split` unless `--unseal-test` is passed, before the config is read or the encoder is loaded.
- `load_quarantine` at `src/retrieval/leakage.py:84` refuses a list whose recorded `splits` include `test`, which `src/retrieval/build_index.py:52` relays as `--unseal-test`. The old 50-row list cannot be used to build an index without that flag.

The notebook's audit cell reads `val` only, and the stale outputs of the audit, index-build, selection, and sweep cells were cleared: they report the 50-row quarantine and the routing computed against it.

### Verification

`pytest tests/` passes, 404 tests, 2 of them new in `tests/test_leakage.py`: a `splits: ["val", "test"]` list raises and yields its rows only under `allow_test=True`, and a val-only list loads. `manage.py leakage --split val test` exits with `refusing to audit the sealed test split` before the encoder loads. On the real files, `load_quarantine` returns 50 rows for the old list under `allow_test=True` and 22 for the new one.

`data/splits/test.jsonl` is unchanged - `3e24e90f`, the digest in `data/splits/hashes.json`, and untouched in git since `f78ec23`. The split was read, not modified.

### Reproduction

The quarantined index and the sweeps it fed have to be rebuilt on a GPU before those four diagnostics are cited again:

```bash
python3 manage.py build_index --config configs/base_qwen.yaml \
    --index_dir data/knn_index_clean \
    --quarantine data/splits/pool_quarantine.json

for thr in 1 2 3 4; do
  python3 manage.py sparse_select --config configs/sparse_retrieval.yaml \
      --split val --index_dir data/knn_index_clean --min_query_terms $thr \
      --out results/sparse_sweep_val_t$thr.json
done
```

### Limitations and risks

The test split has now been read twice on record: this audit, and the marker-capitalization and marker-density statistics of the test targets in the 2026-08-20 entry. Neither reached a selection decision except through the quarantine, which this entry removes. Both reads stay in the log rather than being edited out of it.

Removing the 28 test-flagged rows from the quarantine puts them back in the pool for any future clean-index build. At the final pass, a test-time quarantine has to be written then, with `--unseal-test`, and the index rebuilt from it - not carried over from a list written now.

## 2026-08-20: The marker regex was case-sensitive and the centroid was built through it

Written after the scoring entry below and before the fix is committed, so that the defect and its consequences are on record separately from the change that removes it. The `marker_rate` feature has been miscounted since 2026-06-12, on the target corpus far more than on any system output, and every register number this project has reported was computed through it. The measured consequences are given here; the correction is the next commit.

### Summary

`_MARKERS` matched the archaic pronoun class case-sensitively. The authorized reference capitalizes the reverential pronouns - `Thou`, `Thee`, `Thy`, `Thine` - as a matter of convention, and the systems mostly do not. So the regex deleted 52.1% of the target's markers and only 10-27% of each system's, and the centroid built from those targets sat at a marker density well below the corpus's real one. Under that centroid every condition in the project appeared to *overshoot* the target's marker rate. Ten of twelve in fact undershoot it.

The effect is not a uniform undercount that a standardized distance would absorb. It is a class-selective deletion applied asymmetrically to the two sides of the comparison, and it inverts the sign of the register trajectory that the 2026-08-19 ladder entry rests on.

### The defect

`src/eval/stylometrics.py:25`, unchanged since `e73d2ce`:

```python
_MARKERS = re.compile(
    r"\b(thou|thee|thy|thine|art|hast|hath|dost|doth|shalt|wilt|unto|ye)\b"
    r"|\bO\b",
)
```

No `re.IGNORECASE`. `_words()` lowercases for every other feature - `lex_density`, `ttr`, `root_ttr` all run over `_WORD_RE` output - but `marker_rate` at `src/eval/stylometrics.py:139` calls `_MARKERS.findall(text)` on the raw string. The vocative `O` branch has to stay capital-only or every ordinary "o" counts; the pronoun branch did not, and inherited the same case-sensitivity by sharing the pattern.

### What the gold looks like

Over the 10,860 train targets that build the centroid, counted case-insensitively:

| token | capitalized | lowercase | % capitalized |
|---|---:|---:|---:|
| `thy` | 4,738 | 377 | 92.6% |
| `thee` | 1,729 | 388 | 81.7% |
| `thine` | 463 | 100 | 82.2% |
| `thou` | 1,625 | 576 | 73.8% |
| `wilt` | 10 | 58 | 14.7% |
| `doth` | 6 | 36 | 14.3% |
| `ye` | 42 | 637 | 6.2% |
| `unto` | 54 | 1,183 | 4.4% |
| `dost` | 2 | 43 | 4.4% |
| `hast` | 7 | 529 | 1.3% |
| `hath` | 22 | 2,161 | 1.0% |
| `art` | 2 | 530 | 0.4% |
| `shalt` | 0 | 14 | 0.0% |

15,332 pronoun-class hits, of which 8,700 are capitalized: 56.7%. With the 1,371 vocative `O` hits the target corpus holds 16,703 markers and the case-sensitive regex found 8,003 of them. 8,555 of the 8,700 missed hits - 98.3% - are the four reverential pronouns, which the register capitalizes for the divine referent and the archaic verbs do not.

### The differential miss

The same count over each condition's val predictions, against the val references:

| source | markers | missed | miss rate |
|---|---:|---:|---:|
| train targets (centroid corpus) | 16,703 | 8,700 | 52.1% |
| val targets | 1,227 | 178 | 14.5% |
| `knn_fewshot` | 1,629 | 432 | 26.5% |
| `afsp_margin` | 1,658 | 385 | 23.2% |
| `peft_knn` | 1,335 | 304 | 22.8% |
| `zeroshot` | 2,373 | 520 | 21.9% |
| `afsp_full` | 1,583 | 292 | 18.4% |
| `random_fewshot` | 1,827 | 325 | 17.8% |
| `peft_afsp` | 1,261 | 215 | 17.0% |
| `peft` | 1,402 | 189 | 13.5% |
| `rlsf_w3_6.0` | 1,448 | 193 | 13.3% |
| `rlsf_w3_0.0` | 1,410 | 184 | 13.0% |
| `rlsf_w3_2.0` | 1,501 | 190 | 12.7% |
| `commercial_haiku` | 1,642 | 168 | 10.2% |

The gap between the target's 52.1% and the systems' 10-27% is the whole of the artifact. A measure that undercounted both sides equally would shift the centroid and the conditions together and largely cancel in a z-scored distance. This one moved the reference far more than the things being compared to it.

The val targets miss only 14.5%, against the train targets' 52.1%. The split is work-level, so the centroid corpus and the val references are disjoint sets of works and the capitalization convention travels with the work. Train is led by `Prayers-and-Meditations` and `Epistle-to-Son-of-the-Wolf`, where the reverential pronouns address the divine directly, and capitalizes 56.7% of them; val is led by `Gems-of-Divine-Mysteries` and `Will-and-Testament-Abdul-baha` at 15.4%, and test sits at 33.2%. Raw marker density travels with it too - 1.41 pronoun-class hits per segment in train against 0.87 in val and 0.57 in test. Nothing in the pipeline compares the centroid corpus's marker statistics against the split it is used to score, so neither the case artifact nor the convention gap beneath it had anywhere to surface.

### What it does to the centroid

`results/stylometrics_centroid.json` and `results/stylometrics_centroid_split.json`, rebuilt over the same 10,860 segments:

| | before | after | change |
|---|---:|---:|---:|
| `marker_rate` mean | 0.032684 | 0.057349 | +75.5% |
| `marker_rate` std | 0.056741 | 0.078181 | +37.8% |

The other five features are bit-identical, as they must be: nothing else reads the raw string.

### What it does to the ladder

`stylo_dist` over the twelve conditions, before and after, and the `marker_rate` z that drives the change:

| condition | dist before | rank | dist after | rank | z before | z after |
|---|---:|---:|---:|---:|---:|---:|
| `peft_afsp` | 0.2701 | 1 | 0.3244 | 7 | +0.137 | −0.187 |
| `peft` | 0.2886 | 2 | 0.2890 | 4 | +0.136 | −0.137 |
| `rlsf_w3_0.0` | 0.2960 | 3 | 0.2857 | 3 | +0.150 | −0.128 |
| `rlsf_w3_6.0` | 0.2990 | 4 | 0.2704 | 1 | +0.169 | −0.110 |
| `peft_knn` | 0.3152 | 5 | 0.3589 | 9 | +0.036 | −0.175 |
| `rlsf_w3_2.0` | 0.3279 | 6 | 0.2793 | 2 | +0.195 | −0.092 |
| `afsp_full` | 0.3698 | 7 | 0.3032 | 5 | +0.214 | −0.032 |
| `afsp_margin` | 0.3910 | 8 | 0.3394 | 8 | +0.194 | −0.009 |
| `knn_fewshot` | 0.4005 | 9 | 0.3659 | 10 | +0.164 | −0.018 |
| `random_fewshot` | 0.4736 | 10 | 0.3162 | 6 | +0.366 | +0.097 |
| `commercial_haiku` | 0.5557 | 11 | 0.4873 | 12 | +0.271 | −0.048 |
| `zeroshot` | 0.6518 | 12 | 0.4481 | 11 | +0.547 | +0.275 |

Ten of twelve `marker_rate` z-scores change sign. Before the fix every condition without exception sat above the target's marker density, which read as uniform over-archaizing; after it, only `zeroshot` and `random_fewshot` do, and the rest fall short of the register they are being scored against. The two readings are opposite descriptions of the same generations.

Separation collapses with the ordering. Adjacent-rank gaps that clear zero go from 3 of 11 to 1 of 11, and the single survivor is `knn_fewshot` − `zeroshot`. The ladder committed at `900a891` is a ranking whose ordering was already mostly unsupported; under the fixed instrument it is a ranking of a different order that is even less supported.

### What it does to the trajectory

The sign flip. Held-out register distance per doubling of optimizer step, `results/heldout_traj_val.json` recomputed:

| arm | dist_heldout before | after | z marker_rate before | after |
|---|---:|---:|---:|---:|
| `w3_0.0` | +0.0044 | +0.0058 | −0.0001 | −0.0010 |
| `w3_2.0` | +0.0399* | −0.0205* | +0.0442* | +0.0328* |
| `w3_6.0` | +0.0584* | −0.0259* | +0.0656* | +0.0514* |

`*` = 95% paired interval clear of zero. COMET, chrF and BLEU are bit-identical before and after, which is the check that the fix touches the register axis and nothing else.

`marker_rate` still rises with training in both paid arms, still separates, and still ranks in ω₃ order. What changed is where zero is. Under the old centroid the arms began the ladder at z = +0.166 and +0.151 at step 100, already above the target, so the rise read as movement away from it. Under the fixed centroid they begin at −0.111 and −0.122, below the target, and the same rise carries them toward it for most of the ladder - `w3_2.0` reaches +0.008 and `w3_6.0` +0.076 only at step 1200. Held-out distance therefore falls with training in both paid arms instead of climbing, and `ordered_in_omega` for `dist_heldout` goes from `true` to `false`. Matched-step ω₃ ordering goes from 3 of 5 steps to 0 of 5.

That inverts the reading of the 2026-08-19 ladder entry. Its conclusion was that register drift and measured adequacy came apart, with the paid arms drifting away from an independent target while adequacy held - the Goodhart shape, and the gate that authorized the stacked arm below. Under the fixed instrument the paid arms move *toward* the held-out target as training proceeds, and the asymmetry the entry called "the main asymmetry in the trajectory" does not survive. What survives is the narrower claim that marker density inflates with ω₃ and that adequacy does not pay for it. The claim that inflation is drift away from the register does not.

The declaration below is affected in both directions. Under the fixed instrument P1 would have been confirmed - `peft_afsp` − `peft` on `dist_heldout` becomes +0.0889 [+0.0549, +0.1246], p = 0.0, a separated rise where the committed instrument gave +0.0202 at p = 0.434. P3's failure survives on `stylo_dist`, at −0.0341 [−0.0651, −0.0047], p = 0.027 against the committed −0.0442 at p = 0.004, but on `dist_heldout` it reverts to a null at −0.0251 [−0.0561, +0.0055], p = 0.112. The entry below scores those predictions on the instrument they were written under, which is the only instrument under which they were ever predictions; this entry records what a rerun gives, and the two should not be merged.

### A known property, not an undiscovered bug

`tests/test_stylometrics.py:79-81`, at `900a891`, on `test_marker_rate_normalized_per_word`:

```python
    # "verily" is a function word for lex density but NOT an archaic marker; "thou"
    # and "thee" are. Two markers over four words = 0.5. (The shared marker regex is
    # case-sensitive, matching quick.py, so the words are given lowercase here.)
```

The case-sensitivity was known, deliberate, documented, and consistent across the two call sites - the test even adjusts its own fixture to accommodate it. The test suite passed on it for two months. What was never done is the step from the property to its consequence: nobody asked what a case-sensitive marker count does to a centroid estimated from a corpus whose markers are majority-capitalized. The property was checked; the thing the property was load-bearing for was not.

That distinction is the reason this entry exists separately. It is not a case of an untested code path. It is a case of a tested code path whose downstream effect on the measurement instrument nobody derived, and no test in the suite was positioned to catch it, because every test compared the code against the specification and none compared the specification against the corpus.

### Blast radius

`marker_rate` is not in `REWARD_FEATURES` (`lex_density`, `sent_len_mean`, `sent_len_var`), and the stylometric reward term is off by default, so the RLSF training reward never read the defective count. The three arms were trained against the judge, not against this. For RLSF the defect is measurement only.

AFSP is not. `src/retrieval/afsp.py` reranks exemplars with `register_band_distance` over `CENTROID_FEATURES`, which includes `marker_rate`, against `results/stylometrics_centroid.json` (`configs/base_qwen.yaml:38`, `configs/peft_afsp.yaml:41`). The two rerank conditions - `afsp_full` and `peft_afsp` - selected their exemplars against a target sitting at 0.0327 rather than 0.0573. Rescoring them with the fixed centroid measures generations that were chosen under the old one. `afsp_margin` is not among them: `src/infer/run.py:216` hands it a null centroid and forces `lambda_style` to 0, and `AFSPRetriever.select` gates the style term on both, so the band distance is never computed on that rung and its selection is margin-only. The sweep already showed as much - `afsp_margin` and `afsp_k8_l0` are byte-identical on all 1,323 segments (2026-08-01). The retrieval index itself is unaffected; `src/retrieval/build_index.py` reads no stylometric feature.

`src/eval/quick.py:16` imports `_MARKERS` directly, so the `marker_rate` and `ref_marker_rate` columns of every `manage.py eval` table move too, as do the same columns in `src/infer/sweep.py` and `src/rlsf/select.py`. The λ and k sweeps (`results/afsp_sweep_val.json`, `results/peft_sweep_val.json`, `results/sweep_curves_val.json`) were selected on distances computed through the old centroid.

### Verification

The capitalization counts and the miss table were computed directly over `data/splits/train.jsonl`, `data/splits/val.jsonl` and `outputs/*_val.jsonl`, matching the pronoun class case-insensitively and testing the first character of each hit. The two totals quoted in the summary - 56.7% of pronoun-class hits and 52.1% of all markers - are the same count expressed with and without the vocative `O`, which is capital-only under both regexes.

The before and after reports were produced by the same commands over the same generations, differing only in the working-tree state of `src/eval/stylometrics.py` and the two centroid files. The adequacy columns of the trajectory report reproduce bit-identically across the two, which is what establishes that the difference is confined to the register features.

The rebuilt centroid preserves `n_segments` at 10,860 and reproduces the five untouched features exactly.

### Reproduction

The after-fix reports are not committed. They were written to a scratch path so that no committed artifact was overwritten before the fix itself lands:

```bash
python manage.py stylometrics --build-centroid
python manage.py stylometrics_ci --split val \
  --conditions zeroshot random_fewshot knn_fewshot afsp_margin afsp_full peft \
               peft_knn peft_afsp rlsf_w3_0.0 rlsf_w3_2.0 rlsf_w3_6.0 commercial_haiku \
  --results_path <scratch>/stylometrics_ci_ladder_val_fixed.json
python manage.py heldout_decomp --split val --trajectory --no-figure \
  --results_path <scratch>/heldout_traj_val_fixed.json
python manage.py heldout_decomp --split val --no-figure \
  --conditions peft peft_knn peft_afsp --reference peft \
  --results_path <scratch>/heldout_decomp_peft_afsp_val_fixed.json
python manage.py heldout_decomp --split val --no-figure \
  --conditions peft_knn peft_afsp --reference peft_knn \
  --results_path <scratch>/heldout_decomp_peft_afsp_vs_knn_val_fixed.json
```

0 paid calls and $0.00. All of it is local CPU over already-generated outputs; the trajectory report dominates the runtime and the rest is seconds.

### Limitations and risks

Every committed register number in `results/` and in `README.md` was computed through the defective count and none has been reissued. Until they are, the two states coexist in the repository and the reader cannot tell them apart from the artifacts alone. The reissue is a separate change and is not started here.

The two rerank conditions cannot be corrected by rescoring. Their exemplar selection ran against the old centroid, so a fixed-instrument number for `afsp_full` or `peft_afsp` measures a generation the fixed instrument would not have produced. Establishing what AFSP does under the corrected target requires regenerating both, which is GPU time that has not been booked.

The λ, k and β operating points were selected on distances computed through the old centroid. Whether they remain the selected points under the fixed one is unknown and untested, and re-selecting would add a second selection layer to a val table the README already records as selection-optimistic.

The corrected trajectory reading is a rerun, not a pre-registered prediction. `docs/preregistration_rlsf.md` prediction 4 concerns held-out register distance and was scored as confirmed against the old instrument; it now scores the other way. Nothing in this entry decides how that should be reported, and the addendum has not been amended.

The fixed regex changes `ref_marker_rate` as well as `marker_rate`, so the descriptive tables in the sweep artifacts move on both sides. None of the sweep artifacts has been regenerated.

Whether 0.0573 is the right target is a separate question from whether 0.0327 was wrong. The corrected count is the count the marker list specifies; it is not evidence that the marker list itself is the right operationalization of the register.

## 2026-08-20: The stacked arm, scored against its declaration

Covers 2026-08-19 and 2026-08-20: generation of the two stacked rungs (`72a3c1a`), objective scoring (`57339a3`), the two judge passes (`ec099f6`), and the joint twelve-condition stylometrics ladder (`900a891`). The entry below fixed four predictions before any segment of the arm existed. This entry scores them and records nothing else. Every number is read from the artifact named beside it, as committed at `900a891`.

### Summary

One prediction of four holds, and only under the version of its criterion that was written down rather than the one the instrument turned out to support. Nothing in the arm separates from `peft`, which is where the declaration put both of its reading axes. What does separate is the contrast against `peft_knn` - the plain-retrieval control that P3 named as the single outcome able to change the AFSP verdict, and predicted would show nothing. So the four-quadrant scheme returns no quadrant while the pass is not uninformative, because the scheme was referenced to the wrong condition.

### What was generated

Both rungs decoded the whole of `data/splits/val.jsonl` under the settings the declaration froze: adapter `models/peft_lora_r32_lr2e-4/checkpoint-1358`, k = 8, `most_similar_last`, and for `peft_afsp` λ = 0.75, β = 0.3, σ = 1.0 against the band-pass objective. The `provenance` block introduced for this arm records all of it in `outputs/peft_afsp_val_usage.json` and `outputs/peft_knn_val_usage.json`; nothing was re-selected. 1,323 segments per rung, 0 paid calls - local Qwen2.5-7B-Instruct on a rented card.

### What changed

Three code changes across the four commits, none of them to a metric. `src/eval/stylometrics_ci.py` gained `LADDER_CONDITIONS`, the twelve rungs in reporting order, so that ranks, rank distributions and all 66 pairwise intervals are drawn on one shared resample instead of being assembled from separate calls. `src/eval/judge.py:145` truncates a resumed cache to the number of sources before checking alignment, which is what lets a partially judged condition resume rather than raise. `src/eval/judge_ci.py:281` stops the summary printer raising when the adjacency table is empty.

### The four predictions

Against `peft`, 95% paired bootstrap, n = 1,323.

| | Declared | Observed | Verdict |
|---|---|---|---|
| P1 | `dist_heldout` rises above 0.1707 with the interval clear of zero | 0.1908, delta +0.0202 [−0.0289, +0.0713], p = 0.434 | fails |
| P2 | \|ΔΦ\| stays inside the ≈0.058 detection floor | +0.0514 [+0.0030, +0.1013], p = 0.038 | holds on the declared floor, fails on the realized interval |
| P3 | `peft_afsp` − `peft_knn` separates on no measured quantity | −0.0442 [−0.0771, −0.0127], p = 0.004 on `stylo_dist` | fails |
| P4 | chrF and BLEU hold or fall for both rungs | chrF +0.90, BLEU +0.76, both separated | fails |

P1 fails on separation, and its mechanism fails outright. The direction is right and the magnitude lands inside the declared band - 0.1908 sits between `peft`'s 0.1707 and `afsp_full`'s 0.2990 - but the interval covers zero at p = 0.434, so by the declaration's own rule the rise is not readable. The mechanism was named as two features moving together: `marker_rate` z rising from `peft`'s +0.136 toward `afsp_full`'s +0.214, and `root_ttr` falling from −0.100 toward −0.178. `root_ttr` did exactly that, to −0.148 (delta −0.0475 [−0.0704, −0.0249], p = 0.0002). `marker_rate` went the other way, to +0.051 (delta −0.0850 [−0.1240, −0.0474], p = 0.0), and that move is separated while the composite it feeds is not. The declared floor is also wrong. P1 put it at "about 0.02", citing `dist_heldout` half-widths of 0.019-0.024 in `results/heldout_decomp_afsp_vs_knn_val.json`; that file's `dist_heldout` half-widths are 0.030 and 0.038, the peft-referenced ones in `results/heldout_decomp_prompting_val.json` run 0.049-0.065, and the realized half-width for this contrast is 0.050. P1 was declared against a resolution the instrument does not have, and would have failed at +0.0202 whichever way the arm moved.

P2 holds or fails depending on which half of it is read. ΔΦ = +0.0514 is inside the ≈0.058 floor the prediction named, and under the quadrant rule that floor is what decides whether Φ counts as rising, so the declared criterion is satisfied. The realized half-width is 0.049, not 0.058, and the paired interval [+0.0030, +0.1013] clears zero at p = 0.038. The second rater does not reproduce the separation: `gpt-5.6-terra` gives +0.0234 [−0.0242, +0.0725], p = 0.352. The two raters rank the three conditions identically (`peft_knn` > `peft_afsp` > `peft` on both), with pooled QWK 0.520 [0.500, 0.539] and Spearman 0.683 [0.664, 0.702] over 3,901 shared segments, so the disagreement is about resolution, not about ordering. Read as written, P2 holds. Read against the interval the data produced, a 0.05 Φ gain over `peft` is now detectable and both stacked rungs show one.

P3 fails, and it is the one failure the declaration said could change the verdict. `peft_afsp` − `peft_knn` separates on both register instruments, in AFSP's favour:

| Quantity | `peft_afsp` − `peft_knn` | p |
|---|---:|---:|
| `stylo_dist` | −0.0442 [−0.0771, −0.0127] | 0.004 |
| `dist_heldout` | −0.0387 [−0.0714, −0.0067] | 0.019 |
| chrF | −0.063 [−0.575, +0.465] | 0.831 |
| BLEU | −0.066 [−0.603, +0.476] | 0.825 |
| COMET | +0.0017 [−0.0013, +0.0047] | 0.261 |
| Φ | −0.0068 [−0.0514, +0.0378] | 0.784 |
| Φ (gpt) | −0.0275 [−0.0723, +0.0189] | 0.241 |

Negative is closer to the target centroid. On the frozen base the same contrast was +0.0142 [−0.0233, +0.0527], not separable and pointing the other way; on the fine-tuned base it separates and points toward the target. Holm over the two register contrasts leaves both standing (0.004 < 0.025; 0.019 < 0.05). Holm over all seven quantities carried for this contrast leaves `stylo_dist` standing and drops `dist_heldout` (0.004 < 0.0071; 0.019 > 0.0083), so the claim that survives correction is the four-feature centroid distance, and the three-feature held-out variant supports it without independently establishing it.

P4 fails cleanly and in the opposite direction. Both rungs gained on both surface metrics and both gains separate: `peft_knn` chrF +0.96 [+0.40, +1.53] p = 0.001 and BLEU +0.82 [+0.26, +1.38] p = 0.005; `peft_afsp` chrF +0.90 [+0.33, +1.47] p = 0.001 and BLEU +0.76 [+0.20, +1.31] p = 0.006. COMET, which the declaration left out of the pass and which was run anyway, agrees for the reranked rung: +0.0047 [+0.0011, +0.0081] p = 0.010, against +0.0029 [−0.0008, +0.0066] p = 0.121 for the control. The argument P4 rested on - the adapter was trained completion-only on the zero-shot template, so any few-shot prompt is off-distribution - produces no measurable adequacy cost on three metrics of two kinds.

### The quadrant that was not entered

Both axes were referenced to `peft`. `dist_heldout` does not separate (p = 0.434) and ΔΦ = +0.0514 does not clear the +0.058 the declaration set for "Φ rising", so no quadrant is entered and the *uninformative outcome* clause applies as written: this is reported as a null result of stacking, not as evidence that the two adaptation families compose.

That clause was drafted for the case where nothing moves. Something moved. The separation is on `peft_afsp` − `peft_knn`, which the quadrant scheme has no axis for, because the declaration treated the control as a diagnostic to be consulted only inside Q4 rather than as a contrast in its own right. The declared reading and the informative contrast are not the same comparison, and that is a defect in the declaration rather than in the arm.

### What the declaration deferred and the pass bought anyway

P2 was written as unscored: "the objective axis is measured first so that a later rubric reading cannot be selected on." Both raters were bought in the same pass, so that ordering did not hold. The objective artifacts were committed first (`57339a3`, 2026-08-19) and the judge artifacts second (`ec099f6`, 2026-08-20), which preserves the audit trail but not the argument, since the judge was commissioned knowing the objective result. P2's outcome should be read with that in mind. COMET was likewise declared out of scope and run.

The declaration priced the judge at 2 × 1,323 = 2,646 calls, ≈$1.75, derived from the 2026-08-05 batch rate of $0.00066 per call. That derivation held for the rater it came from: the `gpt-5.6-terra` batch pass was 2,646 calls at $1.7346. It did not cover the primary rater, which was not in the projection. Both were bought.

Paid usage for 2026-08-20: `claude-haiku-4-5` 2,621 calls, $2.6640 (cumulative 3,969 → 6,590 calls, $4.0546 → $6.7186); `gpt-5.6-terra` via batch 2,646 calls, $1.7346 (cumulative 13,230 → 15,876, $8.7220 → $10.4566). Total 5,267 calls and $4.3986, against a declared ≈$1.75. Generation, COMET, chrF, BLEU, the stylometric ladder and every bootstrap were local and free.

`results/judge_val_usage.json` records only the last session - 1,298 calls for `peft_knn` - because the sidecar is overwritten per invocation and `peft_afsp` was judged in a separate call. The pass total is recoverable only from the `cumulative` block. Additive per-session records would avoid that; this is noted rather than fixed.

### Verification

No metric changed and no committed artifact was rewritten in the course of writing this entry. `python -m pytest tests/` at `900a891` is 367 passed, unchanged from the declaration below.

Every figure above was re-read from the committed JSON rather than from a notebook print. The Φ means behind the P2 numbers are `peft` 2.7438, `peft_knn` 2.8020, `peft_afsp` 2.7952 for the primary rater and 3.6150, 3.6633, 3.6383 for the second. `peft`'s cell in `results/heldout_decomp_peft_afsp_val.json` reproduces its cell in `results/heldout_decomp_prompting_val.json` and in the committed `results/heldout_decomp_val.json`: `dist_heldout` 0.1707 and every per-feature z, bit-identical across all three.

The twelve-condition ladder (`900a891`) is where the P3 `stylo_dist` contrast is read from. It reports 3 of 11 adjacent-rank gaps separated, so the ladder's ordering below the level of those three gaps is not a finding and is not used as one here.

### Reproduction

Scoring only; the generation commands are in the declaration below.

```bash
python manage.py eval --conditions peft peft_knn peft_afsp --split val
python manage.py bootstrap --metric chrf --split val --adjacent \
  --conditions peft peft_knn peft_afsp --baseline peft \
  --out results/bootstrap_chrf_peft_afsp_val.json
python manage.py bootstrap --metric bleu --split val --adjacent \
  --conditions peft peft_knn peft_afsp --baseline peft \
  --out results/bootstrap_bleu_peft_afsp_val.json
python manage.py heldout_decomp --split val --no-figure \
  --conditions peft peft_knn peft_afsp --reference peft \
  --results_path results/heldout_decomp_peft_afsp_val.json
python manage.py heldout_decomp --split val --no-figure \
  --conditions peft_knn peft_afsp --reference peft_knn \
  --results_path results/heldout_decomp_peft_afsp_vs_knn_val.json
python manage.py stylometrics_ci --split val \
  --conditions zeroshot random_fewshot knn_fewshot afsp_margin afsp_full peft \
               peft_knn peft_afsp rlsf_w3_0.0 rlsf_w3_2.0 rlsf_w3_6.0 commercial_haiku \
  --results_path results/stylometrics_ci_ladder_val.json
```

COMET runs under `.venv-comet` into `results/comet_val.json`. The two judge passes and their bootstraps are cells 30-45 of `notebooks/peft_afsp_scoring.ipynb`; both are gated on an explicit `SPEND_OK` flag and both are re-runnable at $0 over the stored segment scores.

### Limitations and risks

Val only, n = 1,323, one split. `data/splits/test.jsonl` was not read.

The register instruments are the two this project uses everywhere: `stylo_dist` over the four-feature centroid and `dist_heldout` over the three held-out features. They share `marker_rate` and the corpus centroid in `results/stylometrics_centroid_split.json`. This scoring is against those instruments as committed at `900a891`; it is not independent of them.

The frozen (k, λ, β) were selected on val for the frozen base and transferred unchanged. P3's separation is a separation at those settings, and the README's threats table already records the val table as selection-optimistic.

Multiplicity is stated per claim above and not applied across the four predictions jointly. Scoring four declared predictions is not the same multiple-comparison problem as searching for one, but the correction status of each individual claim is what is recorded, and P3 is the only one whose survival under correction was checked.

P2 was measured after the objective axis was known. It is scored here because it was declared, not because the ordering it specified was honoured.

## 2026-08-19: PEFT+AFSP declared before generation

Written after the ladder entry below and before any segment of the combined arm exists. It states what the arm is predicted to do and what each outcome would mean, so that the reading is fixed while the numbers are still unavailable. The 2026-08-19 addendum to `docs/preregistration_rlsf.md` had to disclose that T1-T4 were descriptions of numbers already in hand; this entry is written under the opposite condition, and the disclosure section says exactly what was readable when it was written.

### Summary

The gate is the checkpoint ladder of the entry below, and it passed. Held-out register distance climbs with optimizer step in both paid arms and does not move in the control, while COMET, chrF and BLEU are flat or rising everywhere. Register drift and measured adequacy came apart along that ladder.

A reward produced that. Whether the decoupling belongs to the channel - GRPO against a rubric - or to pushing register through any channel is not answerable from the arms, because prompting and fine-tuning have never been combined in this project. `peft` runs the zero-shot prompt and carries register in the adapter weights; `afsp_full` runs the frozen base with retrieved exemplars. Stacking them is the untried cell, and this entry declares it before it is generated.

Two conditions, not one. The adapter was trained completion-only on the zero-shot template, so any few-shot prompt fed to it is off-distribution. Without a plain-retrieval control a movement in the stacked AFSP condition cannot be told apart from the effect of exemplars being present at all.

### The gate, and the evidence it passed on

`results/heldout_traj_val.json`, and the table in the entry below. Held-out distance grows +0.0399 and +0.0584 per doubling of optimizer step in `w3_2.0` and `w3_6.0` with intervals clear of zero, and +0.0044 in the control with an interval that is not. `z_marker_rate` tracks it at +0.0442 and +0.0656. Every significant adequacy slope is positive; the two that are not significant are −0.0002 on COMET and −0.02 on BLEU in `w3_6.0`, both straddling zero. `ordered_in_omega` is `true` for both register quantities and `false` for all three adequacy ones. That asymmetry is the passing condition, and it is what authorises the arm below.

### The two conditions

| Condition | Weights | Prompt | Selection |
|---|---|---|---|
| `peft` (reference) | LoRA adapter | zero-shot | - |
| `peft_knn` (control) | LoRA adapter | k = 8 exemplars | plain top-k cosine |
| `peft_afsp` | LoRA adapter | k = 8 exemplars | AFSP, λ = 0.75, β = 0.3, σ = 1.0 |

`configs/peft_afsp.yaml` serves both stacked rungs; the condition is the CLI flag. Every value in it is copied unchanged from `configs/peft_qwen.yaml` and `configs/base_qwen.yaml`: the frozen adapter `models/peft_lora_r32_lr2e-4/checkpoint-1358`, the locked decoding (`temperature 0.0`, `top_p 1.0`, `seed 42`, `max_tokens 1024`, bf16, unquantized), and the AFSP settings frozen on 2026-07-23. Nothing is re-selected. Choosing (k, λ, β) again on the fine-tuned base would add a second layer of val selection to a table the README's threats section already records as selection-optimistic.

The register glossary applies to both stacked rungs, as it does to every other few-shot rung. It is a controlled augmentation in the `prompt:` block, so it does not confound the selection comparison.

### What is predicted

Directional, with the mechanism, and scored as written whichever way they move.

P1 - held-out register distance worsens. `peft_afsp`'s `dist_heldout` rises above `peft`'s 0.1707 and lands between it and `afsp_full`'s 0.2990, with the paired interval clear of zero. The mechanism is the same one the RLSF arms moved along: `marker_rate` z rises from `peft`'s +0.136 toward `afsp_full`'s +0.214, past a target the initialization already sits above, so the rise is movement away rather than toward. `root_ttr` falls from −0.100 toward −0.178 at the same time. The floor a rise must clear is about 0.02 - the `dist_heldout` half-widths against `peft` in `results/heldout_decomp_val.json` are 0.019-0.024 at n = 1,323.

P2 - judge Φ does not separate. |ΔΦ| against `peft` stays inside the detection floor of ≈0.058 named by the 2026-08-01 entry. `afsp_full` − `peft` on Φ was +0.047 [−0.012, 0.105] and did not separate; nothing about stacking makes that gap larger. Φ also tracks COMET (ρ = 0.453 [0.434, 0.472]) far better than it tracks either register proxy (−0.045 and +0.007 at segment level, `results/metric_agreement_val.json`), so P1 moving does not oblige Φ to move with it. Not measured in this pass; see *Limitations*.

P3 - the adaptive layer does not separate from its control. `peft_afsp` − `peft_knn` fails to separate on any measured quantity. On the frozen base that contrast is +0.0142 [−0.0233, +0.0527], p = 0.47 (`results/heldout_decomp_afsp_vs_knn_val.json`) - not separable, and pointing the wrong way - and the 2026-08-01 bootstrap found the same null on COMET, Φ, chrF and BLEU. A separation here would mean the register rerank does something on a fine-tuned base that it does not do on the base. That is the one outcome in this pass that could change the AFSP verdict, and it is predicted not to happen.

P4 - adequacy, secondary. chrF and BLEU hold or fall against `peft` for both stacked rungs. `peft` already leads `afsp_full` by chrF +1.06 and BLEU +2.13, and the adapter has never seen a few-shot prompt in training. This prediction is secondary to the argument and is written down anyway, because chrF and BLEU *are* measured in this pass and an undeclared adequacy number would be read after the fact.

### The four quadrants

Axes: `dist_heldout` against `peft`, and Φ against `peft`. A quadrant is entered only when the relevant paired interval clears zero - improvement means the interval sits below zero, and Φ rising means a gain past +0.058.

Q1 - distance improves, Φ rises. Both measures agree. Retrieval supplies register the adapter did not encode, the combination is the best system on val, and stacking is worth recommending. It is the only quadrant in which it would be. It also contradicts the 2026-08-01 reading directly, because there the adaptive layer moved nothing on the frozen base.

Q2 - distance improves, Φ falls. The two register measures disagree. Since ρ(Φ, `band_dist`) is indistinguishable from zero at segment level, Φ is not measuring register there, and the likelier mechanism is an adequacy cost from off-distribution prompting rather than a genuine disagreement about register. chrF and BLEU decide between those two readings, and both are measured in this pass, so the diagnosis costs nothing.

Q3 - distance worsens, Φ rises. The shape the ladder produced, reached by prompting instead of by reward: the rubric score rises while the measure the rubric does not see moves away from the target. This would put the decoupling outside GRPO and make it a property of pushing register at all. It is the strongest result available from this arm, and the only one for which buying the rater is clearly worth the money.

Q4 - distance worsens, Φ does not rise. Stacking loses on the objective axis and gains nothing on the rubric. `peft_knn` decides the mechanism: if the control moves too, the cause is exemplars at all, which is prompt-format mismatch; if only `peft_afsp` moves, the cause is AFSP's selection.

Predicted: Q4, with Q3 second. Stated so it can be scored.

### What would be uninformative

If neither stacked rung separates from `peft` on any measured quantity, no quadrant is entered. The pass is then reported as a null result of stacking and not as evidence that the two adaptation families compose. This is written down now so it cannot afterwards be read as confirmation, on the same argument as the "uninformative outcome" clause in `docs/preregistration_rlsf.md`.

### What changed

`src/infer/run.py` gained the two rungs. `CONDITIONS` now holds eight, and the membership the dispatch used to spell out as literals is named: `ADAPTER_CONDITIONS`, `RETRIEVAL_CONDITIONS`, `RERANK_CONDITIONS`. The adapter check moved out of the zero-shot branch to the top of the dispatch, so a missing `generator.adapter_path` raises before the retrieval index or the embedding model is loaded. `peft_knn` routes through `index.retrieve()` beside `knn_fewshot`; `peft_afsp` routes through `_select_afsp(rerank=True)` beside `afsp_full`.

The usage sidecar gained a `provenance` block: `adapter_path` when one is set, and k, ordering, index directory and the AFSP parameters for retrieval conditions. The 2026-08-19 entry below records the same gap on `outputs/rlsf_traj/manifest.json` - a run artefact that does not say what the run was made at. Keys are additive and the existing sidecars are not rewritten, as with `per_call_usd` on 2026-08-15.

`src/eval/heldout_decomp.py`: `omega_of()` raised `KeyError` for every condition that is neither an RLSF arm nor a trajectory tag, so `build()` could not read the prompting ladder at all. `OMEGA` now lists the prompting and adapter-stacked rungs at 0.0, which is the entry `peft` already had and for the same reason - they optimize no reward. `adjacent_in_omega` is now empty unless the arms actually differ in ω. Without that guard a pass over conditions that all sit at zero would emit pairs labelled as an ω contrast that are nothing of the kind; the `peft_afsp` − `peft_knn` contrast is real and is read from its own reference-swapped report instead.

`configs/peft_afsp.yaml` and `notebooks/peft_afsp_gpu.ipynb` are new.

### Two artifacts that back the numbers above

Neither existed before this entry, and P1 and P3 quote both.

`results/heldout_decomp_prompting_val.json` - held-out distance over the six prompting and adapter conditions, referenced to `peft`: `peft` 0.1707, `knn_fewshot` 0.2849, `afsp_margin` 0.2936, `afsp_full` 0.2990, `random_fewshot` 0.4200, `zeroshot` 0.5837. Every rung sits further from the target than the adapter, all five intervals clear of zero.

`results/heldout_decomp_afsp_vs_knn_val.json` - the same quantity referenced to `knn_fewshot`, which is the base-model analogue of the contrast P3 predicts: `afsp_margin` +0.0091 [−0.0200, +0.0390] and `afsp_full` +0.0142 [−0.0233, +0.0527], neither separable.

### Reproduction

Declared here; run after this entry is committed.

```bash
# generation, on a rented card (notebooks/peft_afsp_gpu.ipynb)
python manage.py infer --condition peft_knn  --config configs/peft_afsp.yaml
python manage.py infer --condition peft_afsp --config configs/peft_afsp.yaml

# objective scoring, local
python manage.py eval --conditions peft peft_knn peft_afsp --split val
python manage.py bootstrap --metric chrf --split val --adjacent \
  --conditions peft peft_knn peft_afsp --baseline peft \
  --out results/bootstrap_chrf_peft_afsp_val.json
python manage.py bootstrap --metric bleu --split val --adjacent \
  --conditions peft peft_knn peft_afsp --baseline peft \
  --out results/bootstrap_bleu_peft_afsp_val.json
python manage.py stylometrics_ci --split val --conditions peft peft_knn peft_afsp \
  --results_path results/stylometrics_ci_peft_afsp_val.json
python manage.py heldout_decomp --split val --no-figure \
  --conditions peft peft_knn peft_afsp --reference peft \
  --results_path results/heldout_decomp_peft_afsp_val.json
python manage.py heldout_decomp --split val --no-figure \
  --conditions peft_knn peft_afsp --reference peft_knn \
  --results_path results/heldout_decomp_peft_afsp_vs_knn_val.json
```

The two artifacts this entry quotes were written by the same command:

```bash
python manage.py heldout_decomp --split val --no-figure \
  --conditions zeroshot random_fewshot knn_fewshot afsp_margin afsp_full peft \
  --reference peft --results_path results/heldout_decomp_prompting_val.json
python manage.py heldout_decomp --split val --no-figure \
  --conditions knn_fewshot afsp_margin afsp_full --reference knn_fewshot \
  --results_path results/heldout_decomp_afsp_vs_knn_val.json
```

`--no-figure` throughout: `figure()` plots against ω₃, which is zero for every condition here.

### Verification

`python -m pytest tests/` - 367 passed, against 364 at the ladder entry below. The three new tests are in `tests/test_heldout_decomp.py`: one that the unrewarded rungs carry a weight of zero rather than raising, and a pair pinning the adjacency guard in both directions - empty when the arms share ω, and still producing the two RLSF pairs when they do not. `test_an_unnamed_condition_has_no_judge_weight` lost `knn_fewshot` to the OMEGA table and was given `commercial_haiku` in its place, which is the condition that still carries no weight.

`ruff check src`, `ruff format --check src` and `compileall` pass. An offline check confirms the adapter guard raises for all three adapter conditions before any model or index is loaded, that `ADAPTER_CONDITIONS` and `RETRIEVAL_CONDITIONS` are subsets of `CONDITIONS` with `RERANK` inside `RETRIEVAL`, and that `_provenance` records λ = 0.75 for `peft_afsp` and no AFSP block for `peft_knn`.

`peft` in `results/heldout_decomp_prompting_val.json` reproduces its cell in the committed `results/heldout_decomp_val.json` bit-identically - `dist_heldout`, its interval, and every per-feature z. The bootstrap index depends only on n, the seed and `n_resamples`, so adding five conditions could not perturb it; the check is that it did not.

0 paid calls and $0.00. Nothing in this entry ran a model. The two artifacts are local CPU over already-generated val outputs, about 20 s each.

### Limitations and risks

The adapter is off-distribution under any few-shot prompt. That is what the arm tests, and it also means a null could be an artifact of prompt format rather than a statement about AFSP. `peft_knn` is what separates those two readings, and it is the reason the pass generates two conditions instead of one.

The frozen (k, λ, β) were selected on val for the frozen base. Transferring them unchanged avoids a second selection layer, but does not establish that they are the right operating point for a fine-tuned policy. A null under P3 is a null at these settings.

Multiplicity. Three contrasts across chrF, BLEU and the register distance. Holm-Bonferroni is applied per claim and the correction status is stated, as in the README's threats table.

P2 is unscored after this pass. The judge is deferred by design: the objective axis is measured first so that a later rubric reading cannot be selected on. Buying it costs 2 × 1,323 = 2,646 calls; at the rate recorded for the 2026-08-05 batch pass ($6.1173 / 9,261 = $0.00066 per call) that derives to ≈$1.75. That figure is derived from a measured rate, not itself measured, and the per-run confirmation rule in `docs/budget.md` governs whether it is spent.

COMET is not in this pass. It runs locally at $0 and would sharpen the Q2 diagnosis, since Φ tracks COMET more closely than it tracks register. It was left out because the pass was scoped to chrF, BLEU, stylometrics and the held-out decomposition; adding it later is one command and changes no artifact written here.

The test split stays sealed. Nothing in this arm reads `data/splits/test.jsonl`, and the generation notebook asserts it.

### Disclosure: what was visible when this was written

- No segment of `peft_knn` or `peft_afsp` exists. Neither condition has been generated, on
  any split, at any k, by any model. These are predictions and not descriptions, which is the
respect in which this entry differs from T1-T4 in `docs/preregistration_rlsf.md`.
- The held-out distances for the prompting ladder were computed while this arm was being planned
  and were read before P1 and P3 were written. They are what those predictions are anchored on,
and they are committed as the two artifacts named above rather than quoted from a terminal.
- `results/heldout_traj_val.json`, `results/heldout_decomp_val.json`,
  `results/stylometrics_ci_val.json`, `results/metric_agreement_val.json`, the 2026-08-01
bootstrap tables and the README val table were all read.
- The stacked rungs' hyperparameters were not chosen against anything. They are the frozen
  values, copied.

## 2026-08-19: Checkpoint trajectory for register drift and adequacy

Covers 2026-08-18 and 2026-08-19: generation of the ladder on a rented GPU box, and its scoring locally. Neither day had an entry before this one.

### Summary

The 2026-08-15 entry closed on a gap it could not fill: Goodhart needs an independent target moving the wrong way, and the step logs hold only the reward's own view. Held-out register distance is that independent target, and it is a val measurement. This ladder is the measurement - five saved checkpoints per arm, regenerated on the full val split under the arms' own decoding, so distance and marker inflation can be read against optimizer step rather than against the reward that produced them.

Both drift. Held-out distance grows +0.0399 and +0.0584 per doubling of training in the two paid arms and does not move in the control, and `marker_rate` z tracks it almost exactly. Adequacy does not pay for it: COMET, chrF and BLEU are flat or rising everywhere, with a single exception that is itself indistinguishable from zero. That pairing is what the run was generated to test, and until today it rested on COMET alone, in a notebook print.

### What changed

The ladder was generated (2026-08-18, `notebooks/rlsf_trajectory_gpu.ipynb` at `dc6f778`). Fifteen adapters - three arms × steps 100, 200, 400, 800, 1200 - decoded the whole of `data/splits/val.jsonl`, 15 × 1,323 = 19,845 generations. Greedy throughout and identical to the arms' own decoding: `temperature 0.0`, `top_p 1.0`, `seed 42`, `max_tokens 1024`, `bfloat16`, `sdpa`, unquantized. Output at `outputs/rlsf_traj/`, archived at `rlsf_traj_val.tar.gz`. First checkpoint finished `2026-08-18T12:18:57+00:00`, last `2026-08-18T19:29:52+00:00`: 7 h 11 m wall against a 16 h booked budget, 1759.2-1906.8 s per checkpoint.

Adapter hashes. `outputs/rlsf_traj/manifest.json` carries one entry per checkpoint keyed `{cell}_step{N}`, each with `adapter`, `sha256`, `arm`, `output`, `seconds` and `finished`. The digest is sha256 of `adapter_model.safetensors`, taken by the generation notebook's cell 14, which also asserts `r == 32 and lora_alpha == 64`, checks `base_model_name_or_path` against the frozen base, and refuses two checkpoints whose weights hash the same. The scoring notebook re-verifies the other end (`notebooks/rlsf_trajectory_scoring.ipynb` cell 4): every generated row's `adapter` field must equal its manifest entry, and all fifteen digests must be distinct - `15 files x 1323 aligned segments, 15 distinct adapters`. The adapters are not in the repo; `models/` is gitignored and they live in the private HF repo `prnamhr/style-aware-mt-models` under `models/rlsf_grpo_{cell}/checkpoint-{step}`.

chrF and BLEU became trajectory quantities (2026-08-19, `src/eval/heldout_decomp.py`). `surface_segments()` scores every ladder point and the reference per segment with `segment_scores()` (`src/eval/quick.py:26`), the same sentence-level entry point `src/eval/bootstrap.py` already uses, and `trajectory()` resamples them on the identical paired index the register quantities and COMET use. `quantities` in `results/heldout_traj_val.json` now reads `dist_heldout, z_marker_rate, comet, chrf, bleu`, and each acquires a per-checkpoint interval, a delta against `peft`, a per-doubling slope, an endpoint delta and an ω-adjacent comparison from machinery that was already a loop over that tuple.

Corpus chrF and BLEU are carried alongside, as `chrf_corpus` and `bleu_corpus`, from `quick.score()`. These are the numbers `manage.py rlsf_select` ranks checkpoints on and the ones every other table in this project prints. They are not the bootstrapped quantity and are not what the slopes are computed from; see the limitations.

Two figures, `docs/figures/traj_dist_heldout.png` and `docs/figures/traj_marker_rate_z.png`, one arm per line. `trajectory_figure()`'s per-axis body moved into `_traj_panel()` so the standalone and combined figures draw the same panel rather than two drifting copies of it. The combined `heldout_traj_by_step.png` is unchanged at three panels; it was not grown to five.

### The two axes

Per doubling of optimizer step, 95% paired interval, `*` where the interval clears zero:

| | dist_heldout | z marker_rate | COMET | chrF | BLEU |
|---|---:|---:|---:|---:|---:|
| `w3_0.0` | +0.0044 | −0.0001 | +0.0016* | +0.29* | +0.25* |
| `w3_2.0` | +0.0399* | +0.0442* | +0.0009* | +0.34* | +0.19* |
| `w3_6.0` | +0.0584* | +0.0656* | −0.0002 | +0.15* | −0.02 |

Read the first two columns together. They are close to the same number in each arm, they are zero in the control and separated from zero in both paid arms, and the three rates rank in ω₃ order on both. Neither quantity was part of the reward. That is the independent target the 2026-08-15 entry said the step logs could not supply, and it moves the way the pre-registration's prediction 4 said it would.

Read the last three against them. None of the adequacy measures shows a clear decline. Every significant adequacy slope is positive; the two that are not significant, COMET and BLEU in `w3_6.0`, are −0.0002 and −0.02 with intervals straddling zero. The register drift in `w3_2.0` and `w3_6.0` was not bought out of measured adequacy, and it is now three metrics of two different kinds saying so rather than one.

The ω ordering does hold on adequacy as a *gain*, which is the same shape the 2026-08-15 entry found across the arms. `w3_6.0` grows more slowly than `w3_2.0` on chrF by 0.19 per doubling [−0.28, −0.10] and on BLEU by 0.21 [−0.31, −0.11], both separated from zero; the `w3_2.0` against `w3_0.0` comparison is not, on either. So the high arm's cost appears as improvement forgone, which is what the training logs said at 300 rollouts, and not as loss.

`ordered_in_omega` is `true` for both register quantities and `false` for all three adequacy ones. That is the main asymmetry in the trajectory.

### Where the ladder does not support a reading

Matched-step held-out distance ranks in ω₃ order at 3 of 5 steps, and the two that fail are 100 and 200, where the arms have barely separated. The scoring notebook's noise floor cell measures what a single ladder point can carry: comparing each arm's dev-selected checkpoint against the same arm's reported adapter, the generations agree on only 68.9-70.8% of segments and held-out distance differs by up to 0.0136. That is a bound on any step-to-step move, not on the slope over the whole ladder, and it is why the entry above quotes slopes.

### Verification

`python -m pytest tests/` - 364 passed. Five new tests in `tests/test_heldout_decomp.py`, mirroring the COMET block beside them: the two-directory read, refusal on sources that disagree, refusal on references that disagree, pairing of the surface draws on the shared index, and one that pins the segment mean and the corpus score as *different* numbers rather than asserting they agree.

Regenerating the report left `dist_heldout`, `z_marker_rate` and `comet` bit-identical to the committed file, additions only. The shared bootstrap index is drawn before any surface metric is computed, so adding two quantities could not perturb the RNG stream; the check is that it did not.

`chrf_corpus` reproduces the scoring notebook's committed chrF table exactly - `w3_0.0` 41.68, 41.98, 42.24, 42.69, 42.90; `w3_2.0` 42.04, 42.15, 42.43, 43.10, 43.24; `w3_6.0` 42.13, 42.25, 42.61, 42.74, 42.77; `peft` 41.58 - which is what establishes that the corpus path was not silently replaced by the segment mean.

0 paid calls and $0.00 across both days. The ladder is local generation on a rented card; COMET runs locally under `.venv-comet`; chrF and BLEU are local sacrebleu 2.6.0. The scoring notebook's seal cell asserts no `_usage.json` exists under `outputs/rlsf_traj/` and no test-split file was touched. The whole trajectory pass, both surface metrics included, is 17 s of CPU over the sixteen conditions.

### The GPU, and what the manifest does not record

Generation ran on an RTX 4090, 24564 MiB, driver 595.71.05, `sm_89`, `torch 2.12.0+cu130` against CUDA 13.0, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The notebook asserts CUDA ≥ 12.8 and bf16 support before loading anything.

The arms were trained on a different card. `outputs/rlsf/steps_w3_*_manifest.json` records `versions.device` as an RTX 5090 for all three. Nothing here suggests that matters for greedy decoding, and nothing here rules it out either, because no run has ever decoded the same adapter on both. What is recorded is that the ladder and the reported arms were decoded on the *same* card as each other, which is the comparison the scoring rests on.

`manifest.json` carries the `generator` block but not the device. The training manifests carry `versions` including the device; the generation manifest does not, so the card for this run exists only in a notebook's stored output. That is a gap, and it is listed as one rather than backfilled.

### The orphan dependency cascade

On the 4090 box, during `pip install -r requirements.txt`. `torch==2.12.0` broke the pinned-torch ABI that `torchvision`, `torchaudio` and `torchcodec` ship against; the three were uninstalled by hand and generation proceeded. The pipeline is text-only and imports none of them.

This is the third time the same failure has been recorded - `notebooks/peft_full_run_colab.ipynb:215` and `notebooks/afsp_run_colab.ipynb:274` both carry the uninstall for the PEFT and AFSP Colab runs, with the same one-line reason. What is new is `torchcodec`, which appears nowhere in this repository and was pulled in by the box's base image.

`requirements.txt` pins none of the three, so the fix is not reproducible from the file: a fresh install on any image that ships them hits the same cascade and the operator has to know to remove them. Either pin all three to the matching torch build or make the uninstall an explicit documented step in `requirements.txt`. Until one of those happens this is a known hazard, not a solved problem, and it is recorded here as a hazard.

*[accurate information needed]* - the exact removed versions and the pip transcript were not committed and the box is gone. The versions are deliberately left blank rather than reconstructed.

### T1-T4

`docs/preregistration_rlsf.md` gains a dated addendum declaring four trajectory hypotheses as post-hoc exploratory. The ladder was never pre-registered; that document commits to four predictions about the arms at their final adapters and to four val figures, and says nothing about optimizer step. T1 and T2 are the register columns above, T3 is the adequacy columns, and T4 is the two together. The addendum states that every quantity they concern had already been read when they were written, which is the disclosure that makes them descriptions rather than predictions. They carry none of the weight of the four predictions they sit beneath.

### Reproduction

```
python manage.py eval --conditions rlsf_w3_0.0_step100 ... --split val --out_dir outputs/rlsf_traj
.venv-comet/bin/python manage.py comet --conditions ... --split val --out_dir outputs/rlsf_traj \
    --results_path results/comet_traj_val.json --batch_size 16
python manage.py heldout_decomp --trajectory --split val --n_resamples 10000 --seed 42
python -m pytest tests/
```

Generation itself is `notebooks/rlsf_trajectory_gpu.ipynb` and needs the fifteen adapters from the private HF repo and a card with ≥ 24 GiB. Scoring is CPU and reproduces exactly from the committed `outputs/rlsf_traj/` files; the full pass is `notebooks/rlsf_trajectory_scoring.ipynb` top to bottom, whose cells 4, 5, 8 and 26 are assertion cells.

### Limitations and risks

- Segment-mean chrF is not corpus chrF, and the slopes are on the first. The bootstrapped
  quantity is the mean of sentence scores; the corpus score is a different estimator over the same
text and sits 1.3 to 1.8 points higher at every ladder point (`w3_2.0` at step 1200: 41.64 against 43.24). Both are in the file, on purpose. Reading a `chrf` field as the number `rlsf_select` ranked on is the error this shape invites, and one test exists solely to stop a future reader "fixing" the disagreement.
- Sentence BLEU under `effective_order` is a different and noisier estimator than corpus BLEU,
  especially on short segments. It is here because it is the one that can be paired, not because it
is the better BLEU.
- "Adequacy does not fall" is an absence. It is a failure to detect coupling at n = 1,323 over
  five checkpoints, not a demonstration that none exists. Three metrics agreeing on a null is
stronger than one and is still a null.
- Five checkpoints, saved rather than chosen. `save_every_rollouts: 25` set the ladder, the
  slope is a least-squares fit through five points in log-2 of the step, and consecutive
checkpoints on one training run are not independent. The per-doubling figures are descriptions of a trajectory, not tests.
- T1-T4 are post-hoc, and every number they concern had been read first.
- Trained on a 5090, decoded on a 4090, and the generation manifest records neither.
- The dependency cascade is not reproducible from `requirements.txt`, and its exact versions
  are unrecorded.
- `condition` reads `"peft"` on every row of every ladder file. It is inherited from the eval
  config; provenance is carried by `arm`, `step` and `adapter`, and those are what both notebooks
assert against the manifest. It looks like a bug and is not one.

## 2026-08-15: Three RLSF arms completed; judge non-determinism observed

### Summary

All three pre-registered arms trained to their full 300 rollouts. None halted on the register-drift rule. Φ rises monotonically in ω₃, which is the reward working rather than a result; COMET-Kiwi does not fall in any arm, which is the half of the pre-registered Goodhart pair that did not appear. What the arms show instead is a cost: the adequacy gain shrinks as ω₃ grows, and only the high-ω arm separates from the metric-only control.

`w3_6.0` was run twice. The first attempt died at rollout 208 to a kernel kill and its log is kept. Comparing it against the restart turned out to be the most informative thing in the run, because the two attempts share a seed and diverge anyway.

### What changed

`w3_0.0` ran 300 rollouts under `--skip_judge` at 0 paid calls; `w3_2.0` and `w3_6.0` ran 300 each at 9,600 judge calls, $0.7423 and $0.7431. Step logs, manifests and usage sidecars are at `outputs/rlsf/steps_w3_{0.0,2.0,6.0}.jsonl` and their `_manifest.json` / `_usage.json` siblings; adapters are at `models/rlsf_grpo_w3_*` and stay out of the repo. The aborted first attempt at `w3_6.0` is at `outputs/rlsf/aborted/steps_w3_6.0_kernelkill_208.jsonl`.

`src/rlsf/train.py:673` now writes `per_call_usd` into the usage sidecar. `src/rlsf/smoke.py:301` and `src/rlsf/pool.py:398` already did, so the three writers disagreed on the shape of the file they all produce. The two arm sidecars written before this change do not carry the field; they are run artefacts and have not been rewritten, and the rate is `cost_usd / calls`.

### The four figures, by quartile of the 300 rollouts

| | w3_0.0 | w3_2.0 | w3_6.0 |
|---|---:|---:|---:|
| Φ, q1 → q4 | 1.0000 → 1.0000 | 2.9129 → 3.2842 | 2.9521 → 3.3750 |
| COMET-Kiwi, q1 → q4 | 0.6665 → 0.6844 | 0.6644 → 0.6796 | 0.6627 → 0.6699 |
| BLEU, q1 → q4 | 31.66 → 36.34 | 30.45 → 34.96 | 31.33 → 33.61 |
| `marker_rate` z, q1 → q4 | −0.0119 → 0.0491 | −0.0196 → 0.1384 | 0.0100 → 0.1718 |

Φ is flat at 1.0 in `w3_0.0` by construction: `--skip_judge` holds it there, so that row carries no information and is printed only to show the hold worked.

Φ in full, since the endpoints above hide the shape:

| Φ, mean over the block | q1 | q2 | q3 | q4 | q4 − q1 | own sd |
|---|---:|---:|---:|---:|---:|---:|
| `w3_2.0` | 2.9129 | 3.0783 | 3.1529 | 3.2842 | +0.3712 | 0.3967 |
| `w3_6.0` | 2.9521 | 3.0867 | 3.2404 | 3.3750 | +0.4229 | 0.4132 |

Φ rises in both paid arms, monotonically across all four blocks, and by roughly one of each arm's own standard deviations. Four increasing blocks out of four is a stronger statement than the endpoint difference on its own, and it is the clearest movement anywhere in this run. The high arm also sits above the mid arm at every quartile, so the ordering in ω₃ holds throughout - but the gap between the two paid arms never clears its own standard error, reaching at most t = 1.52 at q3. Φ is monotone in ω₃ as an ordering and not separable from noise as a magnitude.

Neither statement is a result. Φ is `prompts/judge_train.txt` scored by the reward judge, which is the quantity the reward maximizes, so a Φ that failed to rise would indicate a broken loop rather than an absent effect. It is reported here because the pre-registration asks for it and because it establishes that the judge term did what it was weighted to do; the limitations below say why it cannot carry a Goodhart claim by itself.

Read across the rows rather than along them. Kiwi rises everywhere, by +0.0179, +0.0152 and +0.0072 - the gain falls as ω₃ rises, and the same ordering holds for BLEU (+4.68, +4.51, +2.27). Over the last 75 rollouts `w3_6.0` sits 0.0145 below the control on Kiwi and `w3_2.0` sits 0.0048 below. So the judge term is paid for out of adequacy improvement forgone, not adequacy lost.

`marker_rate` z moves +0.158 in `w3_2.0` and +0.162 in `w3_6.0` while ω₃² goes from 2/3 to 36/38. The pre-registration's 2026-08-13 addendum set up a disjunction here - either the high arm goes further along the same axis, or the reward has saturated and ω is not the governing axis. Neither branch holds cleanly: style saturates between the two paid arms while adequacy keeps paying, so ω still governs one of the two and no longer governs the other.

### The restart, and what it says about the judge

The kernel kill left an unplanned replicate: two attempts at `w3_6.0` under seed 42, the same config and the same frozen initialization. At rollout 0 both produce the same completions - BLEU and Kiwi agree to 1e-9 - and the judge scores them 2.5625 and 2.46875. `configs/rlsf.yaml` sets `judge.temperature: 0.0` and `judge.seed: 42`; `gpt-4o-mini` honours neither in a way that reproduces. That single disagreement enters the reward, changes the gradient, and the runs diverge from rollout 1 onward: 1 of 208 overlapping rollouts share a BLEU value.

The aggregate results over those 208 rollouts are very similar:

| | aborted | restarted | difference | sd of the restart |
|---|---:|---:|---:|---:|
| Φ | 3.0455 | 3.0775 | +0.0320 | 0.4056 |
| COMET-Kiwi | 0.6651 | 0.6647 | −0.0004 | 0.0284 |
| BLEU | 32.36 | 32.12 | −0.2448 | 7.2254 |
| `marker_rate` z | 0.0129 | 0.0268 | +0.0139 | 0.3114 |

Every difference is a small fraction of the run's own step-to-step spread. The trajectory is not reproducible; the arm-level reading is. That is the useful form of the claim, and it is stronger evidence than a re-run at the same seed would have been, because the two attempts were not seeded to agree at rollout 1 and still land in the same place.

### Verification

19,200 paid verdicts across the two arms, 0 unmeasured. The 2026-08-08 entry made unparseable verdicts a measured quantity rather than an assumed zero; here it is genuinely zero, so the asymmetry it was introduced to detect does not arise in this run.

Degenerate group fractions were 0.032, 0.025 and 0.023, so gradients were carried throughout. Relative adapter movement from initialization was 9.305e-3, 9.337e-3 and 9.396e-3 - the three arms travelled almost exactly the same distance, which is what a shared learning rate, a shared constant schedule and a shared step count should produce, and it means the differences above are not a difference in how far each arm moved.

Peak VRAM was 20.3 of 31.4 GiB on an RTX 5090 in all three, against the 29 GiB bound the wiring check asserts. Wall clock was 4.71 h for `w3_0.0` and 3.08 h and 3.10 h for the paid arms. The free arm taking the longest is the opposite of the expected ordering, since the paid arms add a serial judge block per rollout; no cause is recorded and the manifests carry nothing that distinguishes the sessions.

### Reproduction

```
python manage.py rlsf_train --cell w3_0.0 --steps 300 --skip_judge
python manage.py rlsf_train --cell w3_2.0 --steps 300 --yes
python manage.py rlsf_train --cell w3_6.0 --steps 300 --yes
python manage.py rlsf_select --cell w3_2.0
```

The judge is not reproducible, so re-running either paid arm reproduces the aggregates above and not the step sequence. `w3_0.0` is free and should reproduce exactly.

### Limitations and risks

- Φ rising is not a result. Φ here is `prompts/judge_train.txt` scored by the reward judge,
  which is the quantity being optimized. Goodhart requires an independent target to move the
wrong way, and both independent targets - held-out register distance over `ttr`, `root_ttr` and `marker_rate`, and the evaluation rubric under a different rater - are val measurements that do not exist yet. The pair the pre-registration predicts cannot be scored from these logs.
- Within-arm movements are small against step noise. Expressed in units of each arm's own
  step-to-step sd, the q4 − q1 movements are 0.62/0.50/0.25 for Kiwi, 0.61/0.59/0.31 for BLEU and
0.94/1.02 for Φ. The between-arm contrast over the final 75 rollouts is the stronger reading. A Welch t on that contrast gives −3.19 for `w3_6.0` against the control, but consecutive rollout means are autocorrelated, so it is a description and not a test - the same caveat the 2026-08-13 telemetry entry attaches to the Φ / `marker_rate` correlation.
- No arm halted, and that is weak evidence. The 2026-08-13 addendum measured the rule as
  firing against a +0.35 step about half the time at 8 prompts per rollout. Three runs that did
not trip are consistent with drift the rule cannot see at this rollout width.
- The aborted run's spend is reconstructed, not measured. The usage sidecar is written after
  `trainer.train()` returns, so a kernel kill loses it. 208 rollouts × 32 calls = 6,656 calls,
~$0.51 at the realised rate, inferred from `n_samples` in the surviving step log. It is recorded in `docs/budget.md` as reconstructed and is the one figure there that is not provider-reported.
- Judge non-determinism is unquantified beyond this one pair. Two attempts give one estimate
  of how far a trajectory moves under it, at one ω. Nothing here measures the verdict-level
disagreement rate, which would need the same completions scored twice on purpose.

## 2026-08-13: Learning-rate schedule, micro-batch, and zero-weight judge guard

### Summary

The 50-rollout smoke was run to produce four numbers. It produced three. Reading them found a fifth thing: the manifest recorded `learning_rate` but not the schedule, so it did not determine the run it was written to determine. The schedule is now declared, chosen, and shared by the three arms; the run's wall clock is recorded; the micro-batch stays where it was measured; and a cell that weights the judge at zero refuses to buy verdicts. No arm has run, so the pre-registration's pre-run settings table is amended rather than superseded.

### What changed

The schedule. `grpo_args` set `learning_rate` and left `lr_scheduler_type` to TRL, which takes HF's default of linear decay to zero. `outputs/rlsf/smoke50_steps.jsonl` records what that was: 9.9e-07 at rollout 1, 3e-08 at rollout 49. `configs/rlsf.yaml` now declares `lr_scheduler_type: constant` and `warmup_ratio: 0.0`, `src/rlsf/config.py:232-233` reads both into the `GRPOConfig`, and `run_manifest` needs no change - `src/rlsf/train.py:337` copies the whole `rlsf.train` block, which is why the settings belong in the YAML rather than in a code default.

`assert_warmup_is_reachable` (`src/rlsf/config.py:193-203`) refuses a non-zero warmup under `constant`. `transformers` builds that schedule as `get_constant_schedule`, which takes no warmup argument, so the ramp would be discarded in silence and the manifest would again record something the run did not do. Same shape and same reason as `assert_single_normalization` beside it.

`outcome.elapsed_sec` (`src/rlsf/train.py:654-661`), `time.perf_counter()` around `trainer.train()` and nothing else. The model load is paid once; the rollout rate is what extrapolates from two rollouts to 300, and to the two arms after the first.

`train.per_device_train_batch_size` stays at 1. It was briefly raised to 2 on the headroom in the smoke's 22.04 of 31.36 GiB, then withdrawn. TRL's `dapo` loss normalizes by `num_items_in_batch` - a token count over the whole generation batch, rescaled by `current_gradient_accumulation_steps / steps_per_generation`, which `grpo_args` already holds at 1 (`trl/trainer/grpo_trainer.py:3139-3144`). The micro-batch therefore does not enter the loss: the accumulated gradient is the same average over the same 8 groups at either value. That argument cuts both ways. If 2 changes nothing about the gradient, it buys throughput and nothing else, and it costs an unmeasured amount of memory against a 31.36 GiB card. 22.04 GiB is the only number this project has, and it was read at 1.

The notebook's two-rollout probe would not have licensed 2 either. Peak is set inside rollout 0 for the forward/backward and the generation batch, but two rollouts never reach the first checkpoint write at `save_every_rollouts: 25`, and they see two draws from the prompt-length distribution. Earning 2 means a 25-rollout smoke at 2, not a re-reading of the 2-rollout probe. Until that exists, `grpo_args` derives grad accum 32 and the arms run at the measured setting.

The judge refusal (`src/rlsf/train.py:551-560`). `RewardConfig.weights` keeps a zero-weight component, so `required_components` for `w3_0.0` still names `judge` and the reward path still asks for a score. Nothing but the operator remembering `--skip_judge` stood between RL-Metric and 9,600 verdicts bought to be multiplied by zero. `main()` now refuses the run, ahead of the `--yes` gate so the objection is to the weight rather than to the spend.

`notebooks/rlsf_train_colab.ipynb`: the pre-flight asserts the schedule and the micro-batch, and section 3's existing two-rollout wiring check is now also the memory probe - peak VRAM is set inside the first rollout, so no separate run buys the number. It refuses the arms above 29 GiB.

### Rationale

`constant` is the schedule a fixed rollout budget asks for. Under linear decay the last hundred of 300 rollouts would step at rates too small to move an adapter, which makes "300 rollouts" a count of generations rather than of learning. The smoke shows the shape of that: `adapter_delta.rel` went from 0.00205 at rollout 39 to 0.00207 at rollout 49, a move of 2e-05 against the 0.0021 the run had already accumulated. Those ten rollouts cost the same GPU-time as the first ten and bought a hundredth as much movement.

No warmup, and not because warmup is wrong. The smoke's opening rollouts ran within 1% of the peak rate without instability, which is the evidence that none is needed here, and a ramp would spend part of a fixed budget below the rate the rest of it uses. The setting is written to the config anyway, at zero, so the manifest says so.

The refusal is a refusal and not an auto-skip. `--yes` and `--overwrite` are already shaped that way: the run that happens is the run that was typed. A flag silently added on the operator's behalf is a run whose command line no longer describes it.

### Verification

`python -m pytest tests/` - 336 passed. Six new tests: the schedule reaches the trainer, a warmup under `constant` is refused, the micro-batch times the grad accum is one rollout at any micro-batch size, and `w3_0.0` requires a judge score it weights at zero. No existing test asserted a micro-batch of 1 or a grad accum of 32, so the geometry change broke nothing.

The refusal, which costs nothing and loads no model:

```
$ python manage.py rlsf_train --cell w3_0.0 --steps 300
w3_0.0 weights the judge at 0, so no verdict it buys can enter the reward: 9600 calls
would be paid for and multiplied by zero. Pass --skip_judge; at omega_3 = 0 that is the
arm as declared.
```

`python manage.py rlsf_train --dry_run` was run on CPU with the 0.5B stand-in: it forces the micro-batch back to 1 and is a signature check, not a memory one. 0 paid calls throughout.

### Reproduction

```
python -m pytest tests/
python manage.py rlsf_train --dry_run
python manage.py rlsf_train --cell w3_0.0 --steps 300              # expect the refusal
python manage.py rlsf_train --cell w3_0.0 --steps 300 --skip_judge
python manage.py rlsf_train --cell w3_2.0 --steps 300 --yes
python manage.py rlsf_train --cell w3_6.0 --steps 300 --yes
```

### Limitations and risks

- The micro-batch is unverified at 2. The claim that it holds under 29 GiB rests on the
  smoke's 22.04 and on the doubled micro-batch being the only term that grows. The local card is
an 8 GiB RTX 4060 Laptop, so nothing here measures it; the notebook's section 3 does, before the first arm launches. If it fails, the setting and the pre-registration row go back to 1 together.
- A constant rate does not anneal. The final adapter is wherever the last rollout left the
  policy rather than a settled point. `save_every_rollouts: 25` and `rlsf_select` already existed;
this makes them load-bearing rather than convenient.
- The refusal covers `w_judge == 0`. A cell weighting Kiwi at zero would still start the COMET
  worker; no pre-registered grid cell does, and no exploratory cell is trained.

## 2026-08-13: Third RLSF arm, final run geometry, and telemetry

### Summary

The GRPO loop was complete and pre-registered but had never taken a step. Preparing it to run added a third ω cell, halved the rollout, lowered β, and added four logging channels without which a null result cannot be distinguished from an optimizer that never moved. The pre-registration is amended by dated addendum, not rewritten; `docs/budget.md` carries the re-priced volumes.

### What changed

`configs/rlsf.yaml`:

- `weight_grid.cells` gains `w3_6.0` = (1, 1, 6), unit ω (0.1622, 0.1622, 0.9733). Three arms
  are trained: `w3_0.0`, `w3_2.0`, `w3_6.0`.
- `reference.beta` 0.05 → 0.01; `rollout.prompts_per_step` 16 → 8;
  `train.per_device_train_batch_size` 4 → 1 (grad accum derives to 32);
`generator.max_tokens` 256 → 192; `train.save_every_rollouts` 10 → 25.

`src/rlsf/train.py`:

- `library_versions()` records `torch`, CUDA, the device name and the installed
  `transformers`/`trl`/`peft`/`accelerate`/`bitsandbytes` into the run manifest. A GRPO step is
defined by TRL and PEFT as much as by the config.
- `load_policy(require_ref=True)` raises when β is non-zero and no `ref` adapter registered,
  and prints the adapter list and the active adapter. The existing assertions only fired inside
the branch that loads a reference; with β set and no init adapter the run anchored KL to the bare base silently. The dry run passes `require_ref=False` - it has no init adapter by design.
- `make_optim_callback` collects TRL's `kl`, `grad_norm`, `learning_rate`, `loss`, `entropy`,
  `clip_ratio/region_mean` and `completions/mean_length`; `LoopState.take_optim` averages the μ
passes and stamps them with the rollout that produced them. They arrive *after* that rollout's reward call, so the step-log line carries `optim.rollout = step - 1`. Naming it is preferable to writing every KL one step forward of the rollout it came from.
- `trainable_snapshot` / `adapter_delta` measure ‖θ − θ₀‖₂ and its ratio to ‖θ₀‖₂ over the
  trainable `default` adapter, per rollout and once more into the manifest's `outcome`.

`src/rlsf/select.py`, new, dispatched as `manage.py rlsf_select`: scores every saved checkpoint of an arm on the dev slice and ranks on the chrF adequacy band then held-out register distance. `configs/rlsf_eval_w3_*.yaml`, new: the val inference route for a trained adapter, through the `peft` condition with `--out-name`.

### Rationale

Two arms can show that a rubric term moves the policy. They cannot separate how far it moves from what that costs, because with two points every trade-off is a line. `w3_6.0` is the third point, at ω₃² = 36/38 where the metric terms hold 1/19 of the reward's variance and stop being a guardrail in any useful sense.

β is the change that alters what is tested rather than what it costs. The KL anchor here is a frozen adapter, not the base model, so 0.05 was constraining movement away from a policy the arm starts at; the pre-registration's own "uninformative outcome" clause - neither arm moves, nothing is tested - was the likelier reading of a null result than any statement about the reward. The addendum records that RL-Metric's no-movement prediction is now made under a weaker anchor and is correspondingly easier to falsify.

The adapter-delta channel exists for the same clause. A flat style score is evidence about the reward only if the adapter was updating; without the measurement the two are indistinguishable in the log, and the temptation is to read the more interesting of them.

Checkpoint selection carries no judge deliberately. The training rubric is what the two paid arms optimize, so ranking checkpoints on it is circular, and the evaluation raters are spent once, on the reported val pass.

### Verification

`python -m pytest tests/` - 325 passed. Four assertions moved with the geometry and are recorded here as changed expectations, not as fixes: the generation batch is 32 rather than 64, the step-capped worst case is 51,200 calls and $3.78 at the group-size ceiling rather than 102,400 and $7.55, and the declared cells now separate by 4.44× in advantage magnitude rather than 1.68× before `unit_omega` removes it. New tests pin the manifest's version block, the optimizer block's rollout attribution, and that `adapter_delta` reads zero at the initialization.

`manage.py drift_oc` was re-run because the stop rule's band is set from the per-step standard error, which the rollout halving raises by √2. At 16 prompts and 500 steps it reproduces the 2026-08-09 operating point exactly (band 0.660; 1.3 % / 45.4 % / 88.8 % against null / +0.23 / +0.35), which is what makes the comparison readable: at 8 prompts the band widens to 0.933 and power against a +0.35 step falls to 64.7 % at 500 steps, 49.7 % at the 300 the run uses.

### Reproduction

```
python manage.py drift_oc --steps 300
python manage.py rlsf_train --dry_run
python manage.py rlsf_train --cell w3_0.0 --steps 50 --skip_judge \
  --out outputs/rlsf/smoke50_steps.jsonl --adapter_out models/rlsf_smoke50 --overwrite
python manage.py rlsf_train --cell w3_0.0 --steps 300 --skip_judge
python manage.py rlsf_train --cell w3_2.0 --steps 300 --yes
python manage.py rlsf_train --cell w3_6.0 --steps 300 --yes
python manage.py rlsf_select --cell w3_2.0
```

### Limitations and risks

- The stop rule lost power and was not retuned. A run that does not halt is now much weaker
  evidence that it did not drift. Retuning `k_sigma` to recover the power target after choosing
the geometry that lost it is the edit the pre-registration exists to prevent, so the cost is recorded rather than removed. `marker_rate` z is in the step log and the val table regardless.
- `optim` is empty on the first line of every step log: no optimizer pass has run at rollout 0.
  Consumers reading `optim.kl` off line 1 will find nothing there, by construction.
- `adapter_delta` copies 80.7 M parameters to host memory each rollout. The cost is small
  against a rollout but it is not free, and the float32 baseline holds ~323 MB of RAM.
- Whether PEFT writes a `ref/` subdirectory beside each checkpoint's root adapter is unverified
  until the smoke runs; if it does, each arm's 12 checkpoints cost twice the expected disk.

## 2026-08-12: COMET worker device and stderr deadlock fixes

### Summary

Three fixes to the reward path, none of which changes a number, all of which decide whether a GRPO run finishes: commits `e72b2fb`, `5403993` and `536bc2f`. Recorded late - the run they prepare had not started when they were made.

### What changed

`e72b2fb` - the worker's device is the config's to choose. `reward.kiwi.gpus` reaches `src/rlsf/_kiwi_worker.py`, which sets both Lightning knobs rather than one:

```python
accelerator = "gpu" if gpus else "cpu"
devices = list(range(gpus)) if gpus else "auto"
```

Setting `gpus: 0` alone was not enough - `comet` forces `accelerator="cpu"` at zero devices, and leaving `devices` unset let Lightning auto-detect the card GRPO was training on. The committed value is 0: the encoder allocates on CPU rather than competing for the 32 GB the 7B policy and its rollout logits already fill. `src/rlsf/train.py` correspondingly applies `reserve_vram` only when `kiwi.gpus` is truthy, so `torch.cuda.set_per_process_memory_fraction` does not cap the trainer at 85 % of a card it is not sharing.

`5403993` - the worker's stderr is a file, not a pipe.

```python
# Not a pipe: nothing drains the worker's stderr between requests, so once Lightning
# has written the 64 KB pipe buffer full the worker blocks inside predict and the
# training loop waits on a score that never comes.
self._log = self.stderr_log.open("wb")
```

`KiwiScorer.score` blocks on `readline` from the worker's stdout. With stderr as a pipe that nobody reads between requests, Lightning's progress output fills the 64 KB buffer, the worker blocks writing to it, and both processes wait on each other. The failure has no error and no traceback: a rollout simply never returns, which on rented GPU time is the most expensive shape a bug can take. `logs/kiwi_worker.err` is the destination (git-ignored), and `_stderr()` tails its last 2,000 bytes into any `KiwiError` so the log is still what the exception quotes.

`536bc2f` - two config values that would have been discovered by running. `generator.max_tokens` 1024 → 256, roughly 4× less GRPO activation for a completion budget the dev references never approach; and `save_strategy`/`save_steps` wired from `train.save_every_rollouts`, without which a drift halt would have ended the run with no checkpoint before the drifted final step.

### Limitations and risks

The deadlock fix is verified by construction rather than by reproduction: the pipe-full condition needs a long enough Lightning run to fill 64 KB, and no test forces it. What is pinned is the placement (`tests/test_rlsf_train.py`), not the drain.

## 2026-08-12: RLSF training arm selection added

### Summary

`manage.py rlsf_train` had no way to name a weight-grid cell. It read `rlsf.reward` and trained ω = (1, 1, 1)/√3, which is neither pre-registered arm. `--cell` selects one of the four `weight_grid.cells` by name, and defaults its step log and adapter directory off that name so the arms cannot overwrite each other. The step log now carries the drift verdict the run acted on, and a manifest is written beside it. `notebooks/rlsf_train_colab.ipynb` (new) runs both arms in the pre-registered order.

### What changed

- `src/rlsf/train.py:165` - `arm_reward_config` resolves `--cell` through
  `grid_reward_configs`, so a named arm arrives at unit ‖ω‖ like every other config path.
Only the pre-registered cells are reachable; an exploratory cell raises.
- `src/rlsf/train.py:177` - `arm_path` tags the default step log and adapter directory with
  the cell name: `outputs/rlsf/steps_w3_0.0.jsonl`, `models/rlsf_grpo_w3_2.0`. An explicit
`--out` or `--adapter_out` is used verbatim.
- `src/rlsf/train.py:188` - `run_manifest`, written before the first rollout and rewritten
  with the outcome at the end: ω, what was held flat, the initialization, the drift rule,
and the step a halt happened at.
- `src/rlsf/train.py:228` - the drift verdict is read before the step line is written and
  serialized into it under `drift`.
- `src/rlsf/train.py:396` - "this is a wiring check, not a training result" is no longer
  printed when the component held flat is one the cell weights at zero.
- `src/rlsf/train.py:402` - a non-empty step log for the same arm is refused without
  `--overwrite`.
- `src/rlsf/train.py:448` - the resolved cell reaches both the reward function and
  `_stylo_centroid`, which previously took the base `reward:` block.
- `notebooks/rlsf_train_colab.ipynb` - new.
- `tests/test_rlsf_train.py` - the two arms' ω against the pre-registration table, the
  refusal of an exploratory cell, the per-arm paths, the flat-judge equivalence at ω₃ = 0,
the drift verdict in the step line, and the manifest.

### Rationale

The gap mattered in a way the weight names hide. ω = (1, 1, 1)/√3 differs from RL-Metric's (0.7071, 0.7071, 0) in more than the judge term: the two metric weights are 1.22× apart, and under a weighted sum of z-scores that is an effective step size, which is the confound `unit_omega` exists to prevent. Trained from the config as it stood, the ablation arm would have taken a smaller step than the arm it is the control for, and lr 1e-6 would have meant something different in each.

Serializing the drift verdict rather than replaying it later is the same commitment the pre-registration makes about the rule: a verdict recomputed after the fact is read against whatever rule is on disk then. The rule is pinned by `tests/test_rlsf_config.py`, so a replay would agree today; the point is that the log states what halted the run without depending on that.

### Verification

`python -m pytest tests/ -q` - 320 passed. `python manage.py rlsf_train --dry_run --cell w3_0.0` prints `omega w3_0.0 = bleu 0.7071, kiwi 0.7071, judge 0.0000` and reaches the optimizer on CPU.

### Reproduction

```
python manage.py rlsf_train --cell w3_0.0 --steps 500 --skip_judge
python manage.py rlsf_train --cell w3_2.0 --steps 500 --yes
```

### Limitations and risks

- There is no resume. A dropped session restarts the arm from its initialization, and the
  partial log has to be moved aside or overwritten deliberately.
- `--cell` defaults to none, which still trains the `reward:` block. Nothing refuses an
  unnamed run; the arm is recorded in the manifest rather than enforced at the CLI.
- The manifest's `halted_at_step` is the rollout index the reward function last logged. It
  is the same number as the last `step` in the log, not an independent record of it.
- The verdict's band fields are nan until the baseline and window are complete, so the
  first 25 lines of every run carry bare `NaN`. Python reads it; `jq` does not. The step
log already emitted nan for an unmeasured component, so this widens an existing quirk rather than introducing one.

## 2026-08-12: RLSF preregistration and dev-pool Goodhart diagnostic

### Summary

`docs/preregistration_rlsf.md` (new) fixes the RLSF comparison before it runs: two arms rather than four, RL-Metric (`w3_0.0`) against RLSF-Judge (`w3_2.0`), with the judge arm's outcome predicted in advance - training-time Φ and `marker_rate` rising together, COMET-Kiwi flat, distance over the held-out stylometric features worsening. The drift stop stays at b20 w5 k4.0 for both arms, and the test split stays sealed until Week 3.

`outputs/rlsf/pool_omega.json` was then regenerated under `87659be`, as the entry below requires. It costs nothing: the ω grid re-ranks component scores the pool already paid for, so this run made 0 paid calls. Selection now ranks at the training group size G = 4 and picks `w3_0.0` at a register distance of 0.5581.

The reading confirms the predicted pattern at selection time, before any GRPO step. Across the four cells, as ω₃ goes 0 → 0.5 → 1 → 2, the judge score of the picked samples rises monotonically and every register measure outside the reward degrades monotonically with it. Best-of-N over a frozen policy is not training, so this is not the result the arms exist to produce. It is the same effect at one step of selection instead of five hundred of gradient descent, which makes it a prediction confirmed at the weakest place it could have been.

### The pattern, at the training group size

Picked samples per cell, G = 4, 993 picks over 499 dev segments. `heldout` is distance to the target register over `ttr`, `root_ttr`, `marker_rate` - features no cell here is fitted on. `marker dz` is each pick's `marker_rate` z minus its own group's mean, paired within the segment.

| cell | ω₃ | judge | kiwi | bleu | stylo dist | heldout | marker dz |
|---|---:|---:|---:|---:|---:|---:|---:|
| `w3_0.0` | 0.0 | 3.37 | 0.697 | 40.58 | 0.5581 | 0.5554 | +0.008 ± 0.016 |
| `w3_0.5` | 0.5 | 3.59 | 0.695 | 40.57 | 0.5995 | 0.5956 | +0.055 ± 0.016 |
| `w3_1.0` | 1.0 | 3.76 | 0.692 | 39.66 | 0.6195 | 0.6139 | +0.074 ± 0.016 |
| `w3_2.0` | 2.0 | 3.96 | 0.684 | 37.53 | 0.6645 | 0.6577 | +0.130 ± 0.017 |

Four columns move together and none of them is a coincidence of one cell: the judge score buys +0.59 across the grid, and `marker_rate` follows it up by +0.12 z, register distance by +0.106, held-out distance by +0.102, BLEU down by 3.05 and Kiwi down by 0.013. The exploratory `judge_only` cell sits at the end of the same axis - judge 4.30, marker dz +0.250, the worst held-out distance of any cell read.

The held-out reading is the one with an interval on it. Paired against a uniform draw from the same pool, 2,000 resamples, 95 % CI, negative meaning closer to the target register:

| cell | held-out − random anchor | 95 % CI | p |
|---|---:|---|---:|
| `w3_0.0` | +0.0057 | [−0.0600, +0.0691] | 0.857 |
| `w3_0.5` | +0.0492 | [−0.0187, +0.1104] | 0.147 |
| `w3_1.0` | +0.1050 | [+0.0437, +0.1648] | 0.001 |
| `w3_2.0` | +0.1583 | [+0.0963, +0.2154] | 0.000 |

The metric-only cell is indistinguishable from picking at random on features it does not optimize, which is what a reward with no register term should look like. Both judge-weighted cells are significantly *worse* than random on those features while being much better on the judge's own rubric. No cell in the grid beats the random anchor on the held-out features.

### Why this is the judge signal and not the summand

Three checks separate the explanation from its alternatives.

*Not the form of the term.* The gated cells remove the judge from the sum entirely and use it as a feasibility veto instead. On the groups both cells pick, `gate_j3`, `gate_j4` and `gate_j5` sit +0.106, +0.103 and +0.136 worse in register distance than `w3_0.0`. Making the judge a filter rather than a summand does not escape the pattern.

*Not selection pressure as such.* `chrf_only` and `stylo_only` are the only cells whose picks move `marker_rate` *down* (−0.012 and −0.054) and the only two whose held-out distance beats the anchor at all, neither significantly (p = 0.115, 0.147). Reranking hard on a non-judge component does not inflate markers.

*Not a degenerate-reward artifact.* Every grid cell separates 98-99 % of groups at G = 4, and per-component degeneracy at N = 8 is 1 % for `bleu`, `kiwi`, `chrf` and `stylo` against 4 % for `judge`. The cells are ranking on real spread, and the judge is the *least* discriminative component of the five while being the one whose weight moves every other measure.

### What this does not establish

The `marker_rate` shift of `w3_2.0` at G = 4 is +0.130 ± 0.017, below the 0.23 `min_delta` floor of the drift rule, so `over_threshold` is false for the judge arm as well as for the selected one. Single-step best-of-N does not clear the band the stop rule needs. Whether five hundred steps of GRPO accumulate past it is exactly the open question, and nothing here answers it: best-of-N reranks samples from a frozen policy, while training moves the distribution those samples are drawn from and compounds each step's shift into the next.

The direction is confirmed. The magnitude at training time is not, and a run that drifts by less than 0.23 would leave the stop rule silent without contradicting a single prediction in the pre-registration.

These are dev-slice figures. They select ω and are never reported as a result; val remains the reported split.

### What changed

- `docs/preregistration_rlsf.md` (new) - the arms, the predictions, the stop rule, the seal,
  and a disclosure of what its author had already seen. Its addendum records that selection
ranks `w3_2.0` last, and that RLSF-Judge is run anyway.
- `outputs/rlsf/pool_omega.json` - regenerated. It now carries `seed`, `subgroup.picks` and
  `selection.ranked_at: 4`, none of which the 2026-08-11 file had.

### Verification

- `python manage.py rlsf_omega` reports `0` unmeasured samples and 3,912 feasible of 3,992 in
  every cell; the 80 infeasible are length-band violations, identical across cells because
feasibility does not depend on the weights. No judge verdict in the pool failed to parse, so the feasibility asymmetry the pre-registration flags for `--skip_judge` is zero on this pool.
- The judge's own distribution is not saturated: 241 / 1,105 / 926 / 848 / 872 samples at
  1-5, mean 3.252, sd 1.241. The rubric discriminates; the rise to 3.96 is selection, not a
ceiling effect.
- `outputs/rlsf/pool.jsonl` was not rewritten. Its judge block remains the 3,992 calls of
  2026-08-11 at $0.3013, recorded in `outputs/rlsf/pool_manifest.json`.

### Reproduction

```
python manage.py rlsf_omega
```

### Limitations and risks

The four cells are not four independent observations. They rank the same 3,992 completions under different weightings, so the monotone columns above are one pool read four ways and their errors are not independent of each other. The paired bootstrap against the random anchor is the only reading here with a calibrated interval.

`docs/budget.md` prices the pool at 3,992 calls / $0.29 in *What remains to be spent*, but its *Spend to date* table still ends at 2026-08-08 and does not carry the pool's actual $0.3013. The forecast was accurate; the record is incomplete, and this entry does not fix it.

## 2026-08-12: Effect of `87659be` on the omega artifacts

### Summary

`87659be` (2026-08-11 18:36 +0300) changed the reward floor and the ω pick and selection logic. Both `outputs/rlsf/smoke_steps.jsonl` (2026-08-09) and `outputs/rlsf/pool_omega.json` (2026-08-11 16:14) predate it, so neither was computed under the current code. Read one artifact at a time, that provenance implies less than it appears to. The floor change is numerically inert on every path that writes an artifact; the pick and ranking changes are not, and they fall on `pool_omega.json` alone.

The floor is now `_FLOOR_MARGIN · ||ω||` (`src/rlsf/reward.py:315`) rather than a constant. Every production caller builds its `RewardConfig` through `reward_config` (`src/rlsf/config.py:97`) or `_cells` (`src/rlsf/config.py:116`), and both end in `unit_omega()`. At `||ω|| = 1` the scaled margin is 1.0, which is the constant it replaced. The fourteen weight vectors committed in `pool_omega.json` all measure to unit norm within 1e-4, so no reward value in either artifact would move. The change bites only a hand-constructed config, which is why the test that pins it (`tests/test_rlsf_smoke.py:117`) asserts `-√3` for an unnormalized (1, 1, 1).

`smoke_steps.jsonl` therefore needs no rerun on account of `87659be`. It records reward mean and standard deviation, per-component means and degeneracy - no picks - and every one of those figures is what the current code would produce. Rerunning it would spend judge calls to reproduce the same numbers under a different sampling draw.

`pool_omega.json` does need regenerating, for the pick and ranking changes rather than the floor. Its `subgroup` blocks carry no `picks` key and its `selection` block carries no `ranked_at`: the file still selects on the reading at N = 8, whereas selection now ranks on the reading at the training group size G = 4. Groups with flat reward also now have their pick drawn at the run seed instead of taken in sampling order - 24 of 998 subgroups in `w3_0.0` alone. The regeneration re-ranks cached component scores held in `pool.jsonl` and costs no API calls.

### The pool's completions are valid; only the ranking over them was ever wrong

`outputs/rlsf/pool.jsonl` has not been rewritten since `51e3dd8`, whose message reads *pool run with bug*, and the two later *rerun rlsf pool* commits touched only the notebook and `pool_omega.json`. Recording this explicitly, because every ω cell ranks over that file.

The bug was fixed by `29c540a`, which touched `notebooks/rlsf_pool_colab.ipynb` and `notebooks/rlsf_smoke_colab.ipynb` and no source file. A reporting cell read `concurrency` from the usage sidecar, which does not carry that key; the fix reads it from the config. The `KeyError` was raised after sampling and judging had finished and written their artifacts, and it affected the printed rate summary, not the completions or their scores.

### Verification

- `git log 51e3dd8..HEAD -- src/` does not list `src/rlsf/pool.py`. The sampling path is
  byte-identical to the one that wrote the pool.
- `sha256sum outputs/rlsf/pool.jsonl` matches `hashes.pool.jsonl` in
  `outputs/rlsf/pool_manifest.json`, and `data/splits/rlsf_dev.jsonl` matches the input hash
the same manifest records. The pool's input and output are both the ones it was built from.
- The `||ω|| = 1` claim was checked on the file rather than argued from the config: the
  `weights` block of each cell and exploratory cell in `pool_omega.json` has L2 norm 1.000.

### Limitations and risks

The inertness of the floor rescaling is a property of `unit_omega()` being applied on both config paths, not of the floor itself. A caller that builds a `RewardConfig` directly, or a future path that drops the normalization, gets a different floor and a different reward scale, and any artifact it writes is not comparable to the ones described here.

Re-ranking `pool_omega.json` under the current code may name a different winning cell, since it ranks at G = 4 on a reading the old file never wrote. The previous selection of `w3_0.0` should not be carried forward into the regenerated file's reading.

## 2026-08-09: Drift-stop band calibrated from a simulated false-alarm rate

### Summary

The register-drift rule was priced as a single comparison: a pooled error of 0.24, a 3σ band of 0.72, against a val ladder whose entire `marker_rate` spread is 0.41. Read that way the band is wider than the effect it exists to catch. But the rule is not read once. The monitor returns a verdict on every rollout, so a 500-step run re-tests one baseline draw about 495 times, and what the rule does over a run is not what its per-check threshold says. Simulated at the smoke's measured error, the committed rule halts 20.4 % of drift-free runs. Lowering `k_sigma` to 1.5 raises that to 81.5 %. The fix is the anchor, not the band: `baseline_steps` 5 → 20, `window` 3 → 5, `k_sigma` 3.0 → 4.0, `min_delta` unchanged. Against the drift path this arm is most likely to produce - a slow climb rather than a step - the new rule is better than the old one on both axes at once: 90.6 % caught against 88.1 %, at 1.3 % false alarms against 20.4 %. Set before any GPU spend, and pinned by a test so it is not revisited after one.

### What changed

- `configs/rlsf.yaml:70` - the `stop:` block: `baseline_steps: 20`, `window: 5`,
  `k_sigma: 4.0`. `min_delta` stays at 0.23.
- `src/rlsf/stop.py:18` - the `DriftRule` defaults follow the config, so a caller with no
  `stop:` block does not silently get the 20 % rule.
- `src/rlsf/drift_oc.py` (new) and `manage.py` - `drift_oc`, the simulation every number
  below comes from.
- `tests/test_rlsf_config.py` - the committed operating point is pinned to these four values.

### Rationale

Why the static reading is incomplete. The threshold is `max(min_delta, k_sigma · se)` against `window_mean − baseline`, and at a per-step error of 0.330 that is 0.66 to 0.72. The val ladder runs from 0.136 (peft) to 0.547 (zeroshot), so the band does exceed the full condition spread. What that reading omits is repetition. The baseline is drawn once from the opening steps and never re-estimated, and every later rollout is compared against it: the band is sized for one independent comparison and then spent on several hundred correlated ones. A baseline that draws low biases all of them in the same direction, and at 5 steps its own error is 0.148 - half the band it anchors.

The simulation. Per-prompt sd 1.319, backed out of the smoke's clustered `z_se` of 0.295 over 20 prompts. A training step is 16 prompts, not 20, so a step's error is 0.330 and the smoke's own figure understates it. 500 rollouts, 1,000 runs, seed 42. *Null* is no drift at all; *step* is a sustained shift beginning at rollout 50; *ramp* is a linear climb from rollout 50 to the stated size at rollout 500. Cells are halt rate at median halt step.

| rule | band | null | step +0.23 | step +0.35 | ramp +0.50 |
|---|---:|---:|---:|---:|---:|
| b5 w3 k3.0 (previous) | 0.723 | 20.4 % @175 | 68.5 % @105 | 90.2 % @74 | 88.1 % @283 |
| b5 w3 k1.5 | 0.361 | 81.5 % @53 | 98.4 % @54 | 99.8 % @52 | 100.0 % @78 |
| b20 w3 k4.0 | 0.817 | 2.3 % @238 | 53.1 % @192 | 90.1 % @115 | 90.2 % @359 |
| b20 w5 k4.0 (set) | 0.660 | 1.3 % @336 | 45.4 % @194 | 88.8 % @112 | 90.6 % @371 |
| b30 w5 k4.0 | 0.637 | 1.5 % @228 | 55.2 % @187 | 95.2 % @105 | 95.1 % @359 |

Why not lower the band. At `k_sigma: 1.5` the rule halts four drift-free runs in five, at a median rollout 53 - three after the simulated drift begins and long before any of it has accrued. A rule that halts almost every run says nothing about the run it halted, and its 90 %-plus detection columns are mostly the same false alarms landing on top of a real shift. Making `min_delta` the primary arm is the same move further: the threshold would sit at 0.23 against a per-step error of 0.330.

What 4.0 is not. It is not a claim of evidence at 4σ. The band is the free parameter that sets the false-alarm rate over the whole run, and 4.0 is where that rate is 1.3 % at the measured error. Quoted as a per-check significance level it would be a misreading of what was fitted.

What the rule gives up. Against a +0.23 step - the `min_delta` floor, the peft-to-random_fewshot gap - it fires 45 % of the time, at a median 144 rollouts after the shift. That is accepted rather than fixed. The trained adapter is evaluated on the full val ladder, where `marker_rate` z carries a bootstrap half-width near 0.06, so a drift that size is measured in the reported table whether or not the monitor fired. The monitor exists to stop a run that has already left the regime, not to be the only detector.

The runner-up. `baseline_steps: 30` at the same window and band: 1.5 % false alarms, 55.2 % at +0.23, 95.1 % on the ramp. Better on every power column for 0.2 pp more false alarms. It was not taken because 30 rollouts is 6 % of the planned run absorbed into an anchor that is supposed to stand for the regime the policy started in, and the ramp figure it improves is already above 90 %.

### Verification

`pytest tests/` passes, 236 tests. The pinning test fails on any edit to the four values. Nothing about the rule's logic changed, so the 16 degenerate-case tests in `tests/test_rlsf_stop.py` cover it unchanged.

### Reproduction

```
python manage.py drift_oc
python manage.py drift_oc --runs 5000 --sigma_prompt 1.4
```

### Limitations and risks

- The null model draws each rollout independently. A real policy's `marker_rate` wanders
  between steps, and autocorrelation inflates false alarms, so 1.3 % is a floor rather than
an estimate of what the run will do.
- The per-prompt sd is one 20-prompt draw from the smoke. If a training step's spread is
  wider every rate in the table moves, and `--sigma_prompt` exists to re-price it against the
first real steps. Re-pricing after a run has seen drift is what the pinning test refuses.
- The rule watches `marker_rate` alone. A policy that games the register reward through a
  feature the centroid does not carry is invisible to it at any band.
- The band still uses a normal quantile over roughly 16 clusters per step, carried over from
  the 2026-08-08 entry. A t quantile would be wider; the simulation prices the rule as
implemented, not as it would be with the correction.

## 2026-08-08: GRPO training loop and reward normalization

### Summary

RLSF has a training loop. `src/rlsf/train.py` drives `trl.GRPOTrainer` over the frozen PEFT checkpoint with the reward from `src/rlsf/reward.py`, and `manage.py rlsf_train` runs it. Two decisions in the wiring are recorded here because neither is visible at run time. The reward is normalized once rather than twice: `compute_rewards` already z-scores each component within its group, so the trainer's own group scaling is turned off. And the `train:` block now carries GRPO field names, because `ppo_epochs` and `clip_range` have no GRPOTrainer counterpart and named an update this arm does not run. A CPU dry run on `Qwen2.5-0.5B-Instruct` checked the wiring before any GPU was rented.

### What changed

- `src/rlsf/train.py` - new. Builds the prompt dataset, loads the policy as a trainable
  `PeftModel`, and hands `GRPOTrainer` a single reward function that calls
`compute_rewards`, appends a `StepLog` line, and reads the register-drift rule. A `TrainerCallback` ends the run when that rule fires. `JudgeBudget` checks the declared call and dollar caps before each judge block.
- `src/rlsf/config.py:126` - `rollout_batch`, `optimizer_steps`,
  `assert_single_normalization`, and `grpo_args`, which builds the `GRPOConfig` from the
`rollout:`, `train:` and `reference:` blocks.
- `configs/rlsf.yaml` - `train.ppo_epochs` → `train.num_iterations`, `train.clip_range` →
  `train.epsilon`, `reference.kl_coef` → `reference.beta`. Added `train.scale_rewards`,
`train.loss_type` and `train.per_device_train_batch_size`, plus `output.adapter_dir`. `train.gradient_accumulation_steps` is gone: it is derived from the rollout size.
- `manage.py` - `rlsf_train`.
- `tests/test_rlsf_train.py` - new.

### Rationale

Why the reward is normalized once. `compute_rewards` z-scores each component within its group and then applies the ω weights, so what leaves it is already a combined z-score. GRPOTrainer group-normalizes whatever it is handed. Wired together with both active, the advantage is divided by a second standard deviation that has nothing to do with the first. Nothing crashes. The run trains, the logs look ordinary, and the ω grid stops measuring what it claims to.

The size of that is worth stating, because it is not a rounding effect. Over 200 synthetic groups of four - component scores drawn from a fixed seed, not measured - the four grid cells separate by 1.68× when the combined z-score is passed through unscaled, ordered by the norm of their weight vector. Rescaled, they collapse:

| ω cell | ‖ω‖ | advantage sd, `scale_rewards: none` | advantage sd, `scale_rewards: group` |
|---|---:|---:|---:|
| `w3_0.0` | 1.414 | 1.371 | 1.000 |
| `w3_0.5` | 1.500 | 1.433 | 1.000 |
| `w3_1.0` | 1.732 | 1.622 | 1.000 |
| `w3_2.0` | 2.449 | 2.302 | 1.000 |

The second normalization projects ω onto the unit sphere. Direction survives - the cells still rank samples differently - but magnitude does not, so every cell reaches the optimizer at the same advantage scale and the ablation cell `w3_0.0` takes the same size of step as `w3_2.0`. RQ3 varies ω₃ in order to ask how hard the register term can push. That question has no answer under a second normalization.

Why `scale_rewards: none` and not the raw ω-weighted sum. Both are defensible and both normalize once. Handing GRPOTrainer the raw sum of `ω₁·kiwi + ω₂·BLEU + ω₃·Φ` and letting it normalize would mean the ω weights act on three quantities on unrelated scales - BLEU on 0-100, Kiwi on roughly 0-1, Φ on 1-5 - where ω₃ = 2.0 buys less movement than ω₂ = 1.0 does. Per-component z-scoring first is what makes the weights commensurable, and it is the semantics the grid is written against. The trainer still subtracts the group mean, which is close to a no-op on a mean-zero reward and is not one when the length band has floored a sample; that is the intended behaviour, since a floored sample should sit below its group. `src/rlsf/config.py:assert_single_normalization` refuses any other value of `scale_rewards`, so re-enabling the second normalization is an error rather than a silent change of meaning.

Why the caps still count rollouts. `docs/budget.md` prices a step as 16 prompts × G = 64 judge calls. GRPO reuses each rollout μ times, so with `num_iterations: 4` the optimizer takes four steps per rollout and TRL's `max_steps` is 4× the number the budget priced. `caps.max_steps` keeps its budgeted meaning - 600 rollouts - and `optimizer_steps` does the conversion at the boundary. Handing `caps.max_steps` straight to `GRPOConfig` would have run a quarter of the authorized rollouts while appearing to run all of them.

Why the reference is a copied adapter and not the base model. `reference.adapter_path` names the frozen PEFT checkpoint, and the KL term is meant to hold the policy near that, not near `Qwen2.5-7B-Instruct`. TRL takes two different paths here: given a `peft_config` it wraps a fresh adapter and computes reference log-probs with adapters disabled, which is the base model; given a model that is already a `PeftModel`, with `beta` non-zero and no `target_parameters` in the LoRA config, it copies the loaded adapter into a frozen second adapter named `ref` (`trl/trainer/grpo_trainer.py:465`). The frozen checkpoint's `adapter_config.json` sets `target_parameters: null`, so `load_policy` loads the adapter itself and lets the trainer take the second path.

The same function loads that adapter with `is_trainable=True`. The saved config carries `inference_mode: true`, which freezes every LoRA parameter on load; without the override the run would complete, log rewards, and update nothing.

Why the judge budget is checked before each block. The 2026-08-08 caps entry left enforcement to the training loop, noting the caps were declared but unenforced at run time. `JudgeBudget.reserve` now refuses a block that would take the run past `max_judge_calls`, and refuses to start another once measured spend has reached `max_judge_spend_usd`. It reads the provider-reported figure from `judge.usage`, not an estimate, and it raises rather than degrading quietly, because a cap that lets the run continue is a warning.

### Verification

`tests/test_rlsf_train.py` covers the normalization contract, the rollout-to-optimizer-step conversion, and the budget refusals. The ω-collapse table above is a test rather than a one-off script, but note what it does and does not pin: it computes `compute_rewards` for real and then reproduces TRL's advantage formula by hand (`trl/trainer/grpo_trainer.py:2705`), because that arithmetic is inline in `_generate_and_score_completions` and cannot be called without a trainer. It will catch a change on the reward side. It will not notice if TRL changes how it forms advantages.

The dry run is `manage.py rlsf_train --dry_run`: `Qwen2.5-0.5B-Instruct` on CPU, a fresh LoRA of the frozen checkpoint's shape, two prompts per rollout, two rollouts, judge and Kiwi held flat. It exercises the same branches the GPU run takes - the copied `ref` adapter, μ > 1 and its buffered rollout reuse, the `StepLog` append, the drift monitor, `save_model` - at zero cost. It completed its eight optimizer steps in 1 h 52 m:

```text
plan: 2 rollouts x 2 prompts x G=4 = 8 completions/rollout, 8 optimizer steps at mu=4
  rollout 0: reward -0.000 sd 1.000, 8/8 feasible, 0/2 degenerate, 0 judge calls ($0.0000)
  rollout 1: reward +0.000 sd 1.000, 8/8 feasible, 0/2 degenerate, 0 judge calls ($0.0000)
{'loss': '0.191', 'grad_norm': '7.946', 'kl': '0.001694', 'clip_ratio/low_mean': '0', ...}
```

Four readings in that carry information beyond "it ran".

`reward_sd 1.000` is the normalization decision, visible. With Kiwi and the judge held flat their z-scores are zero, so the combined reward is BLEU's z-score alone and its within-group standard deviation is one by construction. That is the value GRPOTrainer receives, and it is exactly what a second group normalization would have divided by.

`kl` is non-zero, so the frozen `ref` adapter exists and is being scored against. Had TRL fallen through to the adapters-disabled path the KL would have been measured against `Qwen2.5-0.5B-Instruct` itself rather than against the run's initialization.

`grad_norm 7.946` is the `is_trainable=True` fix. Without it the LoRA parameters load frozen under the saved `inference_mode: true`, and the run reports rewards and a loss while updating nothing.

`clip_ratio/low_mean` reached 0.01136 on step 7. The importance ratio does depart from one within a rollout, so `epsilon` binds rather than being inert - the consequence of μ = 4 that would not appear at TRL's default of 1.

Two rollouts produced two `StepLog` lines, which is the intended cardinality: the reward runs once per rollout, not once per optimizer step, so the judge is paid four times less than the optimizer-step count suggests. `save_model` wrote the trained `default` adapter at the root of `output.adapter_dir` with the frozen `ref` beside it in a subdirectory; `PeftModel.from_pretrained` on that directory loads `default` alone, so `src/infer/local_client.py` reads the trained adapter and ignores the copy.

### Reproduction

```bash
python manage.py rlsf_train --dry_run
python manage.py rlsf_train --steps 500 --yes
```

The dry run needs no GPU, no `.venv-comet` and no API key, and makes no paid call. It is slow: 1 h 52 m for eight optimizer steps, about 875 s each. CPU backward passes over a 151,936-token vocabulary dominate, which is also why the dry run drops `per_device_train_batch_size` to 1 - at 4 the fp32 logits push a 16 GB box into swap, and the first attempt spent 12 min a step thrashing.

### Limitations and risks

- The dry run proves signatures, not the run. It samples two prompts per rollout at a
  128-token completion budget against the 16 and 1024 of the configured arm, so it says
nothing about memory on a 4090, about throughput, or about whether the length band behaves on full-length completions. Nothing has yet run the loop on the locked 7B base.
- The three-component reward has been combined only on synthetic scores. The dry run
  holds Kiwi and the judge flat, so only BLEU drove it; the ω table is drawn from a fixed
seed. What has never run is `compute_rewards` inside the training loop with all three components live, which is the first thing the GPU run does.
- `num_iterations: 4` is carried over from `ppo_epochs: 4`, not chosen. It was the PPO
  epoch count and is now μ. At μ > 1 the importance ratio departs from one within a rollout
and `epsilon` starts to bind, which is the regime the clip exists for, but no reading supports 4 over 1 and 1 is TRL's default. It also multiplies optimizer steps per rollout by four, which is the wall-clock cost of the choice.
- The frozen `ref` adapter doubles LoRA parameter memory and adds a forward pass per
  micro-step. Neither is measured on the rental GPU.
- Kiwi blocks the loop the way the judge does. COMET runs in a separate interpreter, so
  each rollout waits on a subprocess round trip as well as on the judge's 8.5 s. The judge
block is timed; the Kiwi block is not.
- Nothing stops a run whose groups have gone flat. `degenerate_frac` is logged per
  rollout and read by the smoke's verdict, but the training loop only halts on the
register-drift rule. A policy that converges onto the reward until every group is flat would keep spending judge calls on zero-gradient steps.
- The saved adapter carries a 70 MB copy of itself. `save_model` writes the frozen `ref`
  adapter into a subdirectory beside the trained one. Loading is unaffected - checked - but
the artifact is twice the size it needs to be, and a reader who opens `ref/` will find a second `adapter_config.json` that is not the trained one.
- The dry run's own margin is thin. The register-drift rule cannot fire in two rollouts,
  `z_se` for `marker_rate` came out 0.0 on the second because both groups scored identically,
and the length band passed 8/8 on 128-token completions. None of that predicts behaviour over 500 rollouts of full-length output.

## 2026-08-08: RLSF smoke run completed and spend caps set

### Summary

The reward-path smoke ran end to end on an RTX 4090 and passed all four checks. It also produced the two numbers the arm had been blocked on: the judge costs $7.375e-5 per call and returns 7.63× parallelism at `judge.concurrency: 8`. Both had been estimates. With a measured rate the RLSF caps in `configs/rlsf.yaml` are no longer guesses, so they are set, and `src/rlsf/config.py:assert_caps_declared` stops refusing to load. `degenerate_frac` now goes into `steps.jsonl` rather than to stdout.

### What changed

- `configs/rlsf.yaml:73` - caps declared: `max_steps: 600`, `max_grid_steps: 200`,
  `max_judge_calls: 340000`, `max_judge_spend_usd: 25.0`.
- `docs/budget.md` - the *Declared cap* section states the $25 figure; a new
  *RLSF arm - caps declared (2026-08-08)* section derives it. The 2026-08-07 envelope is
kept and marked superseded. Both smoke passes are in the spend table.
- `src/rlsf/reward.py:238` - `reward_degeneracy` moves here from `smoke.py`, and
  `compute_rewards` calls it. `StepLog` gains `n_groups`, `degenerate_groups`,
`degenerate_frac` and `min_group_sd`; all four are written per step.
- `src/rlsf/smoke.py` - reads those fields off the log instead of recomputing them, so the
  verdict and the persisted record cannot disagree.

### Rationale

Why the caps could be set now. `docs/budget.md` had said the cap must be transcribed verbatim from `docs/proposal.pdf` because an approximation of a pre-registered figure is not a cap. The proposal states no numeric spend cap, so there was nothing to transcribe and the rule could not be satisfied as written. What it was protecting is that a cap be a recorded commitment rather than a number picked at the moment of spending. Deriving it from a measured rate and dating it meets that; carrying on with `null` caps did not.

Why $25 against a $2.65 plan. The plan is 500 PPO steps ($2.36) plus a 3,992-call dev-slice best-of-N pool ($0.29), with the ω grid re-scored offline over that pool at no extra cost. The step caps bind first: 800 capped steps are $3.78 at group size 4 and $7.55 at the authorized ceiling of 8. The dollar and call caps sit ~3.3× above that on purpose. They are not a second opinion on the step count - they catch the *rate* moving. A longer rubric or a completion-length regression that tripled the per-call price would satisfy every step cap and triple the bill, and nothing else in the loop would see it.

Why the grid stopped costing PPO steps. The 2026-08-07 envelope priced the four ω cells as 50 PPO steps each. Ranking them over one fixed pool of sampled completions re-weights component scores that were already paid for, so the same comparison costs nothing beyond the pool. `max_grid_steps: 200` stays declared because training the grid online is still a reachable design, and a cap that has to be added later is a cap that was not declared.

Why `degenerate_frac` belongs in `steps.jsonl`. It is the reading that says whether a step trained anything: a group whose combined reward is flat across its feasible samples has zero advantages and contributes no gradient. The smoke printed it and dropped it, which is tolerable for a one-step pilot and useless over 500 steps, where the question is whether the fraction climbs as the policy converges on the reward. Moving the function into `reward.py` and having `compute_rewards` call it means the smoke's verdict and the persisted number are the same computation rather than two that agree today.

### Verification

`pytest tests/` passes, 216 tests, two net new. The smoke's paid pass on 2026-08-08:

```
  [PASS] kiwi handshake: worker ready
  [PASS] reward variance: 5% of groups have no reward spread across their feasible samples
  [PASS] steplog written: n_samples=80
  [PASS] length band: 75/80 feasible (94%), ratio mean 0.95, 0 unmeasured
```

80 calls, 417 prompt and 18 completion tokens each, $0.0059, 10.63 s wall clock, mean call 1.014 s, 7.63× achieved parallelism (`outputs/rlsf/smoke_usage.json`). Component degeneracy was 1/20 groups for BLEU, 1/20 for Kiwi, 2/20 for the judge, and 1/20 for the combined reward - the judge going flat in a group twice while the combined reward went flat once is the case the per-component report exists to show.

The cap tests were inverted with the config: `tests/test_rlsf_config.py` previously asserted the committed config *refuses* to load, and now asserts it loads and that nulling any one cap still refuses.

### Reproduction

```bash
python manage.py rlsf_smoke --config configs/rlsf.yaml --segments 4 --skip_judge \
    --out outputs/rlsf/smoke_free.jsonl
python manage.py rlsf_smoke --config configs/rlsf.yaml --segments 20 --group_size 4 --yes
```

The paid pass is the second line and costs $0.0059. `notebooks/rlsf_smoke_colab.ipynb` is the runbook; both passes above are its sections 3 and 4.

### Limitations and risks

- The measured rate is one pass of 80 calls on 20 dev segments. Prompt length tracks
  segment length, and the dev slice is four works. A training run over `rlsf_train.jsonl`
can draw longer segments and pay more per call. The caps have room for roughly 3× of that; a larger move needs a new entry in `docs/budget.md`.
- 7.63× is one reading under no sustained load. A 500-step run holds the connection for
  hours, and provider rate limiting would show up as a fall in `achieved_parallelism` and a
rise in GPU idle, not as a higher bill. The per-step wall clock is the thing to watch.
- Declaring the caps makes the config loadable, which is the point and also the risk.
  Nothing now stops `manage.py rlsf` at load time except the caps themselves. The PPO loop
is not written, so the caps are not yet enforced anywhere at run time - enforcing them against `judge.usage` is work the training loop still owes.
- The smoke notebook asserted the caps were null as its guard against being used once
  training was authorised. That assertion is now inverted to check the pilot ceiling
instead, which is a weaker guard: it no longer distinguishes the pilot from a training run by the config alone.

## 2026-08-08: Judge concurrency timing added

### Summary

`judge.concurrency: 8` was chosen on the argument that judge calls are latency-bound and independent. Nothing measured what the fan-out returned. `outputs/rlsf/smoke_usage.json` carries calls, tokens and cost from the 2026-08-07 pass and no wall clock, so the claim that the loop does not hold the rented GPU idle rests on an untested 8×. The smoke now times the block. Instrumented only: no paid pass has run since, so the file still has no latency fields.

### What changed

- `src/rlsf/reward.py:77` - `JudgeTiming`, a thread-safe accumulator alongside `Usage`.
  It holds the block's wall clock, the summed per-call time, and the call count.
- `src/rlsf/reward.py:106` - `judge_scores` takes an optional `timing` and fills it in.
  Default `None` leaves the scoring path as it was.
- `src/rlsf/smoke.py:326` - `smoke_usage.json` gains `concurrency`, `wall_s`, `call_s`,
  `mean_call_s`, `calls_per_s` and `achieved_parallelism`; the run prints the same.
- `docs/budget.md:91` - the latency gap is stated where the rental estimate is made.

### Rationale

Wall time alone would not settle the question. A block that takes 40 s for 80 calls is fast or slow depending on what one call costs, and the ratio that answers "8× or 2×" is the summed call time over the wall clock. Timing each call as well as the block yields both from one pass, so no serial control run is needed and nothing extra is paid for.

`achieved_parallelism` is the number to read against `judge.concurrency`. At 8 workers, 8 means the fan-out is clean; 2 means six workers are blocked on something - a provider rate limit, a connection pool, or the client serialising. It also converts directly into the figure the budget needs: GPU idle per step is `wall_s`, not `call_s`.

The accumulator locks its counters for the reason `Usage` does. Threads share one object, and a lost increment understates the latency the same way it would understate the bill.

### Verification

`pytest tests/` passes, 214 tests, 2 of them new. A stub judge with fixed per-call delays reads above 2× fanned out over 8 workers and 1× serial, which pins that the number tracks obtained fan-out rather than workers requested.

### Limitations and risks

- Nothing is measured yet. The next paid smoke produces the first reading; until then the
  rental estimate stays an estimate and `docs/budget.md` says so.
- The smoke runs one step. Per-step latency across a training run may differ once the
  provider sees sustained load, so one reading is a floor on the cost, not a mean.
- Timing wraps `client.complete`, so retries and backoff inside the client are counted as
  call time. That is the right total for the GPU-idle question and the wrong one for
reading provider latency alone.

## 2026-08-08: Unparseable judge outputs treated as unmeasured

### Summary

`judge_scores` scored a judge response it could not parse as 1.0, the bottom of the 1-5 rubric. A formatting failure therefore entered the group as the strongest available negative signal, and PPO would move weights away from the sample. Unparseable verdicts now score nan and `compute_rewards` marks the sample infeasible, which is the path length violations already take. Implemented only: no training run has used it.

### What changed

- `src/rlsf/reward.py:73` - `judge_scores` returns nan where `parse_score` returns `None`.
  Its `default` parameter is gone; there is no score to fall back to.
- `src/rlsf/reward.py:276` - `compute_rewards` intersects the length band with a
  finite-score mask over every component, so any unmeasured term drops the sample through
the configured `on_violation` path rather than through its own.
- `src/rlsf/reward.py:224` - `StepLog.n_unmeasured`, written on every step, separating
  samples dropped for a missing score from samples dropped for length.
- `src/rlsf/reward.py:159` - `_measured_mean`, so one unparsed verdict does not make the
  step's logged raw judge mean nan.
- `src/rlsf/smoke.py:119` - the length-band check reports the unmeasured count alongside
  the feasible fraction.

### Rationale

The evaluation path has never done this. `src/eval/judge.py:parse_score` returns `None` and every consumer drops the segment: `judge_batch` records a null, `judge_agreement` and `judge_ci` carry NaN, `paired_bootstrap` drops the pair rather than imputing it (2026-07-31 entry). The reward path was the one place a parse failure was converted into a number, and it converted it into the worst number on the scale.

The rate is not negligible. On `results/judge_gpt_val.json` - Φ_B, the cross-family confirmation judge - coverage runs from 0.9803 to 0.9902 across the seven conditions: 129 unparseable of 9,261 segments, 1.4%. That is on well-formed evaluation output at temperature 0. RLSF samples at T = 1.0 from a policy being updated, where degenerate and truncated generations are exactly the ones a judge answers oddly about, so 1.4% is a floor rather than an estimate. At 64 judge calls a step the weight grid alone is 12,800 calls; with a final run of comparable length the affected count is on the order of 300 samples, each of them a fabricated minimum.

Under `on_violation: floor` the sample still receives `worst - 1.0`, which is lower than the 1.0 it used to get. The difference is that the value is now attributed: it comes from the feasibility rule the config chose, it is counted in `n_unmeasured`, and switching to `drop` removes the sample from the update entirely. A 1.0 written into the judge component was indistinguishable from a verdict the judge actually returned, and it entered the group's mean and standard deviation, moving the advantages of the three samples beside it.

### Verification

`pytest tests/` passes, 204 tests, 3 of them new in `tests/test_rlsf_reward.py`: an unreadable verdict scores nan rather than 1; a nan component marks the sample infeasible and leaves its group-mates' rewards at exactly ±1, which pins that the failure entered neither the mean nor the standard deviation; and under `drop` the sample's reward is nan while the logged raw judge mean is still the mean of what parsed. The existing length-violation test now also asserts `n_unmeasured == 0`, so the two drop reasons stay distinct.

### Limitations and risks

- The 1.4% figure is measured on Φ_B's evaluation run against `prompts/judge_eval.txt`,
  not on `prompts/judge_train.txt` at T = 1.0. The training rubric's own rate is unmeasured
until a run logs `n_unmeasured`.
- No retry is attempted. A verdict lost to a transient formatting slip costs a paid call
  and yields no measurement; whether one re-ask is worth the latency inside the loop is
unanswered.
- Nothing yet bounds the rate. A judge failing on a third of a step would quietly shrink
  every group without tripping a check, since the smoke's length-band threshold is 50%.

## 2026-08-08: Register-drift stop rule moved to a run-specific baseline

### Summary

The drift stop for RLSF training compared a step's `marker_rate` z against the val-set constant for zero-shot, 0.547. Read that way it fires at step 0 of the smoke, which measured 0.577 before a single PPO update. The rule is now stated against the run's own opening steps and requires several consecutive steps outside the band, and each step now logs the standard error the band is sized from. Implemented only: no training run has used it.

### What changed

- `src/rlsf/reward.py:179` - `z_step_stats` returns the per-feature mean z and its
  standard error, clustered on the prompt. `z_deviations` is now a wrapper over it, so the
logged z is unchanged (0.577 recomputed from `outputs/rlsf/smoke_hyps.jsonl`).
- `src/rlsf/reward.py:222` - `StepLog.z_se`, written on every step.
- `src/rlsf/stop.py` (new) - `DriftRule`, `DriftMonitor`, `DriftVerdict`. The monitor takes
  one StepLog per step and returns a verdict; the training loop stops on a tripped one.
- `configs/rlsf.yaml:65` - the `stop:` block: `baseline_steps: 5`, `window: 3`,
  `k_sigma: 3.0`, `min_delta: 0.23`.
- `src/rlsf/config.py:90` - `drift_rule` builds the rule from that block.
- `src/rlsf/smoke.py:282` - the smoke prints the step's z with its error and says that one
  step is a draw of the baseline, not the baseline.

### Rationale

The two numbers being compared were not the same measurement. Zero-shot's 0.547 is 1,323 val segments decoded greedily; the smoke's 0.577 is 80 samples at T = 1.0 from the PEFT-initialized policy. The smoke figure also sits inside zero-shot's own bootstrap interval, [0.476, 0.622], so the excursion it reported was not separable from the constant it was measured against. A threshold carried over from the conditions table measures the sampling regime, not the policy.

The baseline is now the run's own first steps, in the same regime as everything it will be compared with. Step 0 alone is a legal baseline and a poor one: its clustered error on the smoke is 0.295, so a single step cannot distinguish a half-sigma drift from a quiet one. Five steps at an lr of 1e-6 is roughly 80 prompts of accumulated update, which buys a baseline error of 0.13 at a drift cost small enough to accept.

Two conditions must hold together. Every step in the window has to sit at least `min_delta` above the baseline, so one spiking batch cannot carry the window mean over on its own; and the window mean's excess has to clear `k_sigma` pooled standard errors, so a drift the run cannot resolve does not halt it. The floor is there because statistical separation is not the same as a reason to stop: with enough steps a long run resolves excursions too small to care about.

The floor is sized against the val ladder rather than picked. Per-condition `marker_rate` z runs from 0.136 (peft) to 0.547 (zeroshot), so 0.41 separates the condition closest to the reference register from the furthest. `min_delta: 0.23` is the peft-to-random_fewshot gap: the policy is PEFT-initialized, so a drift that size has given back the marker-rate margin that separates it from random few-shot prompting. The 0.5 it replaces was wider than the whole ladder and would have caught only gross marker-stuffing. One caveat on the anchor: the ladder is greedy decoding over 1,323 val segments and a training step is 80 samples at T = 1.0, so a between-condition difference transfers to a within-run drift as a scale, not as an exact quantity.

Which arm the floor binds in depends on the error. At the smoke's clustered per-step se of 0.295 the pooled error is 0.215 and the 3σ band is 0.65, so the statistical arm sets the threshold; `min_delta` would take over only below a per-step se of 0.23, which needs a larger batch than the smoke's 80 samples. The sustained arm uses the floor whatever the error - every window step must sit above `baseline + min_delta` - and that is the arm this value moves today.

The error is clustered on the prompt because a group's four samples answer one source. Pooling all 80 as independent reads 0.163 where the clustered figure is 0.295 - a band 1.8× too narrow, and the rule would fire on prompt-draw noise.

### Verification

`pytest tests/` passes, 201 tests, 16 of them new in `tests/test_rlsf_stop.py`. The case that prompted the change is pinned: a run held flat at 0.577 never trips, though every step of it sits above 0.547. Also pinned: a single spike does not trip, an excursion inside the noise does not trip, a clean excursion below `min_delta` does not trip, downward drift does not trip, and the baseline does not absorb the drift it is measuring.

Replaying the committed `smoke_hyps.jsonl` through `compute_rewards` reproduces `z.marker_rate = 0.577` and records `z_se.marker_rate = 0.295`. At that error the rule's band is 0.65, and a synthetic +1.2 excursion held for three steps trips it.

### Limitations and risks

*Superseded (2026-08-09).* `baseline_steps`, `window` and `k_sigma` were set here from a per-check σ level. That reading treats the rule as one comparison when it is read at every rollout; the values and the reasoning for them are in the 2026-08-09 entry. `min_delta` and the two-arm structure stand as written.

- No training loop calls the monitor yet; `src/rlsf/smoke.py` reports the baseline and
  nothing consumes a verdict.
- The band uses a normal quantile over roughly 16 clusters per step. With that many groups
  a t quantile would be wider, so the rule fires slightly more readily than 3σ implies.
- Steps are pooled as independent draws. If the training loop reuses one prompt set across
  steps, the between-step variance is smaller than the within-step estimate and the band is
conservative.
- The rule watches `marker_rate` alone, one-sided. Register loss and drift in the other
  three centroid features are visible in the step log but do not stop a run.
- `baseline_steps: 5` absorbs whatever drift occurs in the first five steps. At a higher
  learning rate that assumption fails and the baseline should be shortened.

## 2026-08-07: Training judge client and RLSF smoke stage implemented

### Summary

The reward path can now build its own judge client from the config, and `manage.py rlsf_smoke` (new) exercises the whole path on a few dev-slice groups without a PPO update. Implemented and statically verified only: the smoke has not been run, no judge call has been made, no model loaded, and nothing under `outputs/` written. What did run is `--help` and the two refusal paths, both of which exit before any model load or API call.

### What changed

- `src/rlsf/config.py:make_judge_client` - builds the judge from the `judge:` block via
  `src.infer.run.make_client`, and raises if the configured model is one of the two
evaluation raters. The check precedes the lazy torch import so it is unit-testable.
- `src/rlsf/reward.py:judge_scores` - sends `src.eval.judge._JUDGE_SYSTEM` as the system
  message instead of an empty string, so the training and evaluation judges differ in
rubric and nothing else. Reusing the existing constant avoids a second un-hashed prompt artefact alongside the frozen template.
- `src/rlsf/smoke.py` (new, `manage.py rlsf_smoke`) - samples G completions per segment
  from the policy, scores them on BLEU, COMET-Kiwi and the training judge, computes
rewards, writes one `StepLog` line, and reports four checks.
- `src/rlsf/smoke.py` - writes `outputs/rlsf/smoke_usage.json` with the measured per-call
  rate, which is what replaces the token-count estimate in `docs/budget.md`.
- `notebooks/rlsf_smoke_colab.ipynb` (new) - the runbook: setup, pre-flight assertions,
  a free `--skip_judge` pass, the paid pass, and what each failing check means.
- `configs/rlsf.yaml` - `rollout.max_new_tokens: 512` removed. `complete_many` takes only
  `n`, `temperature` and `top_p`; generated length comes from the client's `max_tokens`
at construction, so the field was never read. Beyond being dead, 512 would have been wrong to wire through: every condition config generates at `max_tokens: 1024`, and halving it for RLSF alone makes the arm non-comparable on length against the conditions it is measured against - which matters more here than elsewhere, because length is what the feasibility band gates on. `generator.max_tokens` governs.
- `tests/test_rlsf_smoke.py` (new) - 14 cases; `tests/test_rlsf_config.py` +1. 172 in the
  suite.

### Rationale

The four checks are the failure modes that would otherwise surface only after a long paid run had already produced nothing useful:

1. Kiwi handshake. The worker is a subprocess in a second interpreter with a 600 s
handshake timeout, so it fails slowly and at a distance from its cause.
2. Reward variance. This is the one that silently wastes money. `group_normalize`
returns all zeros when a group's scores are identical or fewer than two samples are feasible, so the samples carry no advantage and the step contributes no gradient. Training would proceed, log plausible-looking rewards, and learn nothing.

Three properties make the check mean what it says. It is computed on the combined reward, not the worst component: a single flat component is survivable when the others still spread the sum, and judging the worst component would fail runs that are fine - `--skip_judge`, which holds the judge flat by construction, most obviously. It is computed over feasible entries only: under `on_violation: floor` a violator takes `worst - 1.0`, which manufactures spread in a group whose feasible samples all scored identically, so including it returns a false pass on precisely the case being tested (`tests/test_rlsf_smoke.py`, verified through `compute_rewards` rather than against a hand-built array - the real path produces rewards `[0, 0, 0, -1]` there, pooled sd 0.43). And it is per group, never pooled: two internally flat groups at different levels have a healthy pooled sd and no usable signal.

The per-component report remains as printed diagnostics, for reading which term went flat once the combined check has already failed. It is not a verdict.
3. StepLog writes. The step log is the only record of a run's reward trajectory.
4. Length band. `on_violation: floor` assigns violators the group's worst reward
minus a margin. If the band rejects most of the batch, the reward is dominated by the floor rather than by the components, and the thresholds - not the weights - are what the policy is learning.

Thresholds are stated rather than inferred: a quarter of groups degenerate fails, and more than half the samples must clear the length band.

The pilot loads the config with `require_caps=False`. The training caps are null and gate PPO; this pilot is what measures the per-call rate needed to declare them, so gating it on them would be circular. It is bounded instead by its own ceiling from the config's `pilot:` block, checked before any call, plus a `--yes` confirmation.

### Verification

```bash
.venv/bin/python -m ruff check src tests manage.py    # All checks passed
.venv/bin/python -m pytest tests/ -q                  # 172 passed
.venv/bin/python manage.py rlsf_smoke --help
.venv/bin/python manage.py rlsf_smoke --segments 20   # refused: 80 calls over the 50 ceiling
.venv/bin/python manage.py rlsf_smoke --segments 12   # refused: no --yes
```

The unit tests cover the checks themselves against cases the smoke cannot be relied on to produce: a flat group, a group with one feasible sample, two flat groups at different levels, and an empty batch. Both refusal paths exit before `_load_rows` and `make_client`.

### Limitations and risks

- Unrun. Nothing is known about whether the rubric discriminates within a group, only
  that the code that would measure it exists and its arithmetic is tested.
- It cannot run on this machine. The policy is `Qwen2.5-7B-Instruct` in bf16, which
  the 8 GB development GPU cannot hold, and `.venv-comet` does not exist locally, so the
Kiwi handshake is untestable here. Colab, per the standing compute constraint.
- `pilot.judge_calls` was raised from 50 to 80 so the declared ceiling matches the
  smoke's brief of 20 segments at G = 4. The alternative - trimming the run to 12 segments
to fit 50, or overriding with `--max_judge_calls` at the command line - would have let a budget number silently reshape a diagnostic. A ceiling moved on purpose is preferable to one that quietly changes what is being tested. 80 calls is ~$0.01.
- `--skip_judge` holds the judge component flat, which makes its group variance
  degenerate by construction. That check is meaningless in that mode and the run's
variance verdict should be read as covering BLEU and Kiwi only.
- A passing smoke says the wiring works, not that the reward is sound. It cannot tell
  whether the judge is discriminating register or returning noise with a healthy spread.

## 2026-08-07: RLSF dev slice created from `train.jsonl`

### Summary

`manage.py rlsf_dev --target 500 --seed 42` was run. It carved 499 segments across four works out of `train.jsonl` as the RLSF weight-selection slice, leaving 10,361 for PPO updates. This is a data operation only: no model was loaded, no judge call made, no reward computed. The reported splits are untouched and no figure changes.

### What changed

Three new files under `data/splits/`:

| File | Segments | sha256 |
|---|---:|---|
| `rlsf_dev.jsonl` | 499 | `45b45e40b03ea3c3…` |
| `rlsf_train.jsonl` | 10,361 | `3c2dbc43d11311c1…` |
| `rlsf_dev_manifest.json` | - | selection record |

The dev works are `Lawh-i-Burhan` (117), `Lawh-i-Dunya` (183), `Lawh-i-Siyyid-i-Mihdiy-i-Dahaji` (77) and `Suriy-i-Vafa` (122). Selection is whole-work and deterministic: the subset minimising `|total − target|`, ties toward fewer works then lexicographic. At target 500 the best achievable is 499. The 193 unlabelled segments are forced to the remainder rather than being eligible for the slice.

`configs/rlsf.yaml` and the README command block were amended, both of which said these files did not exist.

### Rationale

Reward weights are a selection decision, so they need a partition that is neither the one used for updates nor the one used for reporting. Validation is the reported split for every other condition and using it here would make the RLSF row's figures selection- contaminated relative to the rest of the table. Carving from train instead keeps validation clean.

The slice is whole-work for the same reason the top-level split is: partitioning by row would leak an author's stylistic habits across the boundary, and register is the thing being optimized.

### Verification

```bash
.venv/bin/python manage.py rlsf_dev --target 500 --seed 42 --dry_run   # plan
.venv/bin/python manage.py rlsf_dev --target 500 --seed 42
```

Checked after the write:

- `train.jsonl`, `val.jsonl` and `test.jsonl` still hash to the digests in
  `data/splits/hashes.json`. The frozen split was read, not modified.
- `rlsf_dev + rlsf_train` equals `train.jsonl` exactly as a multiset over
  `(input, output)`, and record order is preserved within each part.
- Input overlap between the two parts is 0, and no dev work appears in the remainder.
- All three digests in `rlsf_dev_manifest.json` match the files on disk.

### Limitations and risks

- The slice is not unseen by the model. `models/peft_lora_r32_lr2e-4/checkpoint-1358`,
  which RLSF initializes from, was trained on all of `train.jsonl` including these four
works. Dev-slice figures select reward weights; they do not measure generalization and are never reported as a result.
- `results/stylometrics_centroid.json` was built over all 10,860 train targets, these
  works included, so the `z` field in the step log is measured against a centroid the
slice contributed to. It is a training diagnostic, not evidence.
- 499 segments is small for ranking four grid cells. What separation it can resolve is
  unquantified, and no interval has been computed on it. If the cells land close, the
slice cannot break the tie and saying so is the correct outcome rather than picking the leader.
- Removing these four works changes what PPO trains on relative to what PEFT trained on.
  The two arms are no longer matched on training data, which is a difference between
conditions that the RQ1 comparison has to carry.

## 2026-08-07: RLSF config, reward judge, and deferred spend caps

### Summary

`configs/rlsf.yaml` (new) specifies the RLSF arm: policy init, rollout sampling, reward weights and the RQ3 weight grid, spend caps, and the training-time judge block. `src/rlsf/config.py` (new) loads and validates it. The config is not runnable as committed: every spend cap is `null` and the loader raises until they are set, because budget rule 1 forbids starting a paid run before its volume is recorded and the declared cap is still untranscribed from `docs/proposal.pdf`. Implemented and statically verified only: no PPO step, no judge call, no artefact written. No reported figure changes. RLSF remains an empty row in the results table.

### What changed

- `configs/rlsf.yaml` (new) - the condition config. Policy initializes from the frozen
  PEFT cell `models/peft_lora_r32_lr2e-4/checkpoint-1358`, which is also the KL reference.
- `src/rlsf/config.py` (new) - `load_config` (validating; `require_caps=False` inspects
  without spending), `assert_caps_declared`, `assert_group_size_within_ceiling`,
`reward_config`, `grid_reward_configs`, `worst_case_judge_calls`, `priced_worst_case`.
- `docs/budget.md` - the RLSF arm's authorized envelope, the section's own outstanding
  requirement since 2026-08-05.
- `tests/test_rlsf_config.py` (new) - 18 cases (157 in the suite).

Reward-block field names match `src.rlsf.reward.RewardConfig` exactly, so `reward_config` is a filtered splat rather than a translation layer; the `kiwi:` sub-block is ignored by that filter and read by the scorer.

### Rationale

The reward judge is `gpt-4o-mini`, model-distinct from both evaluation raters. Φ_A is `claude-haiku-4-5` and Φ_B is `gpt-5.6-terra`. Training against either spends that rater on the single condition that most needs two independent ones - RLSF is the only arm optimized against a judge, so it is the arm whose Φ most needs a rater it was not trained against. A third model keeps both. It is also the only candidate honouring `temperature: 0` *and* `seed: 42`: Haiku exposes no seed and Terra treats it as best-effort. That matters more at training time than at evaluation time, because under group normalization a rater that randomly flips a 3 to a 4 inverts a sample's advantage sign - rater noise becomes gradient noise, not just measurement noise.

A local Qwen judge was excluded outright rather than on cost: the policy is `Qwen2.5-7B-Instruct`, so a Qwen judge is self-preference bias by construction - the same objection that makes `commercial_haiku`'s Φ inadmissible as anything but a labelled external reference.

The caps are null on purpose. RLSF spends one judge call per sampled completion, so an uncapped run has no upper bound and a step-count typo is a spend event. Writing a plausible number here would have substituted an estimate for a pre-registered commitment; the cap in `docs/proposal.pdf` is the commitment, and it still cannot be read in this environment. So the keys exist, hold `null`, and the loader raises with the rule-1 reasoning in the message. The envelope is nonetheless priced in `docs/budget.md`, because recording the arithmetic and authorising the spend are separate acts and only the first can happen today.

The ceiling is priced at group size 8 while the config operates at 4. Judge calls scale one-for-one with group size, so raising it after authorisation would be a widening under budget rule 3. Pricing the envelope at 8 makes that headroom part of the original authorisation instead. `assert_group_size_within_ceiling` enforces the boundary.

Rollout sampling is deliberately not the locked greedy decoding. `temperature: 1.0`, `top_p: 0.95`. At temperature 0 every sample in a group is identical, the group's reward sd is 0, and `group_normalize` (`reward.py:115`) returns all zeros - the reward would be identically flat and no update informative. The locked greedy setting still governs the reported inference pass that scores this arm; the matched-decoding requirement is a property of evaluation, not of rollout.

The weight grid varies ω₃ against a fixed adequacy pair, per the README's RLSF section, with a `w_judge: 0.0` ablation cell so the style term's contribution is identifiable rather than inferred. Grid cells are short proxy runs on the dev slice under the §7 selection protocol - sweep, verify, freeze - and a cell is never itself a reported result.

### Verification

No PPO step, no judge call, no network, no GPU.

```bash
.venv/bin/python -m ruff check src tests manage.py    # All checks passed
.venv/bin/python -m pytest tests/ -q                  # 157 passed
```

The new cases assert the properties that would otherwise be prose: the committed config refuses to load; every undeclared cap is named, not just the first; a non-positive cap is rejected; a group size above the ceiling raises citing rule 3; worst-case volume defaults to the ceiling rather than the operative group size (44,800 against 22,400); the grid varies only ω₃ and inherits the base feasibility band; `template_file` is the training rubric and not the evaluation one; and the reward judge's model appears in neither `configs/judge_eval.yaml` nor `configs/judge_eval_gpt.yaml`.

### Reproduction

```bash
.venv/bin/python -m pytest tests/test_rlsf_config.py -q
python -c "from src.rlsf.config import load_config; load_config()"   # must raise
```

### Limitations and risks

- Nothing here has been run, and no trainer exists. `src/rlsf/` holds the reward, the
  COMET-Kiwi worker, and now the config loader. There is no PPO loop, no `rlsf` entry in
`manage.py`, and no `infer` support for the condition. This config specifies an arm that cannot yet execute.
- The data files do not exist. `data/splits/rlsf_train.jsonl` and `rlsf_dev.jsonl` are
  produced by `manage.py rlsf_dev`, which has not been run. Its manifest records that the
dev slice is held out from PPO updates *only* - the PEFT checkpoint RLSF initializes from was trained on all of `train.jsonl`, this slice included, so the slice is not unseen by the model and dev-slice figures select weights rather than measure them.
- The cost figures are estimates over assumed token counts. The $5.11-$6.45 ceiling
  spans 600-800 prompt tokens; at the 400-token assumption it is ~$2.96. The 50-call pilot
measures the real rate. Either way the arm is under a dollar of spend to date, so dollars are not the binding constraint - GPU hours and in-loop judge latency are.
- Family adjacency, logged. `gpt-4o-mini` shares a provider family with Φ_B
  (`gpt-5.6-terra`). This is weaker than model-identical contamination but not nothing:
Φ_B is family-adjacent rather than fully clean for the RLSF row specifically. Added to the threats register.
- Group size 4 is statistically thin. `group_normalize` returns all zeros when fewer
  than 2 samples in a group are length-feasible (`reward.py:110`), so at G = 4 a group
with three length-band violations contributes no gradient signal at all. How often that happens is unmeasured; the step log's `n_feasible` field is what will show it, and the ceiling permits raising to 8 without re-authorisation if it does.
- `learning_rate: 1e-6`, `kl_coef: 0.05`, `clip_range: 0.2`, `ppo_epochs: 4` are
  conventional defaults, not tuned values. They have no support from this project's
data and should not be described as selected.

## 2026-08-07: Training-time judge rubric frozen with a recorded digest

### Summary

`prompts/judge_train.txt` - the Φ term of the RLSF reward, referenced by `src/rlsf/reward.py:13` since the first RLSF stage but never written - now exists, is frozen, and its sha256 is recorded in a new freeze manifest `prompts/hashes.json` before any reward call reads it. `src/rlsf/reward.py:load_train_template` verifies the file against that record on every load and refuses a mismatch. Implemented and statically verified only: no judge call has been made, no RLSF training has run, and nothing under `results/` or `outputs/` was written. No reported figure changes, and the rubric's behaviour as a reward signal is entirely unmeasured.

### What changed

- `prompts/judge_train.txt` (new) - the training-time rubric. Same construct as the
  evaluation rubric (register fidelity, 1-5, scored against the authorized reference)
and the same `source` / `reference` / `prediction` fields `build_prompt` fills, but independently worded, with its own band anchors and its own output contract. sha256 `8eaa11ff341c86ca…`, digest `8eaa11ff341c86ca`.
- `prompts/hashes.json` (new) - freeze record for both rubrics, named and shaped after
  `data/splits/hashes.json`. Records role, freeze date, full sha256, and the 16-character
`digest` that `src/eval/judge.py:74 template_digest` computes and writes into each judge artefact as `template_sha256`.
- `src/rlsf/reward.py:96 frozen_digest` (new) - reads a rubric's recorded digest; raises
  if the manifest is absent or the rubric has no entry in it.
- `src/rlsf/reward.py:123 load_train_template` - now takes `hashes_path`, hashes the file
  it read, and raises `ValueError` if it differs from the freeze record. The
missing-file `FileNotFoundError` and its circularity message are unchanged.
- `tests/test_rlsf_reward.py` - 6 new cases (43 in the file, 139 in the suite).

The frozen eval rubric is untouched: `prompts/judge_eval.txt` still hashes to `ffd6dad41acb0512…`, the digest rater B's 2026-08-05 artefacts carry, so recording it here is a transcription of the current file, not a change to it.

### Rationale

Why the digest is recorded at freeze time rather than at run time. The 2026-08-05 agreement pass reports `template_verified: false` because rater A's artefacts predate per-condition digest recording: that both raters read the same rubric rests on the configs, not on the artefacts, and is closable only by re-spending on Φ_A. That gap opened because the rubric was frozen and the digest was recorded at two different times, with runs in between. Recording the digest as part of the freeze, before the first consumer exists, removes the window in which that can recur for the RLSF reward.

Why the loader verifies rather than logs. A recorded hash that nothing checks is the same class of evidence as a config assertion. Verification on load means an edited rubric fails loudly at the start of training instead of producing rewards that are silently incomparable with those already collected.

Why a separate rubric at all. §Judge circularity mitigation in `README.md` and the 2026-07-05 entry commit the study to two fixed templates, training-time and evaluation-time. Scoring the reward with the same rubric that scores the reported Φ would make the Φ result circular - the policy would be optimized directly against its own evaluation instrument. `load_train_template`'s error message has carried that reasoning since the first RLSF stage; this entry supplies the template it was pointing at.

Three rubric decisions are load-bearing and revisable only until the first RLSF run consumes them:

1. Degenerate output is floored at 1. Empty, truncated, self-repeating, non-English,
and source-script output score 1 regardless of any elevated wording, as does archaic vocabulary applied ungrammatically. The evaluation rubric has no such clause because it never sees policy samples mid-training; a reward that does not floor these is a reward that pays for scattering `Thou` and `hath` into modern prose.
2. A ≤15-word justification precedes the verdict. It costs completion tokens on every
sample. The alternative - a bare `Score: N` - is cheaper, but rater B compressed to a 0.10 range across six conditions under `reasoning_effort: none`, and a judge component with no within-group variance is silently zeroed by `group_normalize` (sd < 1e-9 → all zeros), which would drop the style term from the reward without any error surfacing.
3. Register only; adequacy is explicitly out of scope. Adequacy enters the reward
through the BLEU/chrF and COMET-Kiwi terms, and the length band gates feasibility. Double-counting it in the judge term would confound ω₃ with ω₁ and ω₂ and make the RQ3 weight sweep uninterpretable.

### Verification

No judge calls, no network, no GPU. Nothing was run against a model.

```bash
.venv/bin/python -m ruff check src tests manage.py    # All checks passed
.venv/bin/python -m pytest tests/ -q                  # 139 passed
sha256sum prompts/judge_eval.txt prompts/judge_train.txt
```

The new cases cover: the committed rubric loads and verifies against the committed manifest; it fills through `build_prompt` and is brace-safe against stray braces in inserted text; the two rubrics' digests differ (the circularity control, asserted as a test rather than as prose); an edited rubric raises; a rubric absent from the manifest raises; a missing manifest raises.

`prompts/judge_eval.txt` hashing to `ffd6dad41acb0512…` was checked against the digest recorded in rater B's artefacts, confirming the eval rubric has not drifted since the 2026-08-05 pass.

### Reproduction

```bash
sha256sum prompts/judge_eval.txt prompts/judge_train.txt   # must match prompts/hashes.json
.venv/bin/python -m pytest tests/test_rlsf_reward.py -q
```

### Limitations and risks

- This does not close the `template_verified: false` gap retrospectively. That gap is
  a property of Φ_A's existing artefacts and remains open in the threats register;
only re-spending on Φ_A closes it. What this entry closes is the possibility of the same gap opening for the RLSF reward.
- The rubric is unvalidated. No call has been made against it. Whether it
  discriminates register at all, whether it discriminates it *within* a PPO group of
samples from one policy - a much narrower spread than the seven conditions the eval rubric separates - and whether it is gameable by the policy are all unmeasured. A `--limit N` pilot on existing outputs is the cheapest first check and has not been run.
- The freeze is revisable now and expensive later. Once reward values collected under
  this rubric exist, changing it invalidates them. Any revision to the three decisions
above should happen before the first run, not after.
- Cost is unestimated. The rubric is ~330 tokens of prompt plus source, reference, and
  candidate, and one call per sample. Neither the PPO step cap, batch cap, nor judge spend
cap is fixed in `docs/budget.md` yet, so the total is still undeclared - that precondition is unchanged by this entry.
- The training judge model is unselected. No `configs/rlsf.yaml` exists; which model
  reads this rubric, and whether it is the same family as the evaluation judge, is an open
decision with its own circularity implications.

## 2026-08-05: Second cross-family judge run and rater-dependent Phi contrasts

### Summary

The cross-family confirmation pass was executed. `gpt-5.6-terra` scored all seven conditions on val through the OpenAI Batch API - 9,261 calls, 9,132 segments scored, $6.12 - against the same frozen `prompts/judge_eval.txt` the primary rater reads. The result is the one the threats register has been waiting on, and it is not reassuring: the two raters order the systems similarly at the top but four of the five pre-specified primary contrasts on Φ do not replicate under the second rater, two of them with the sign reversed. The central negative result *does* replicate - AFSP-full still fails to separate from `knn_fewshot` under both raters. No reported figure in `README.md` changes, because Φ_A is unchanged; what changes is how much weight Φ_A alone can carry.

### What changed

Relative to the implementation entry below (same date), three things moved before the run could happen:

- Model identity resolved. `configs/judge_eval_gpt.yaml` names `gpt-5.6-terra`, not the
  unconfirmed `gpt-5.6` placeholder, with real `pricing: [2.00, 12.00]` replacing the
placeholder `[5.00, 30.00]` (commit `7e5ee52`). `reasoning_effort: none` and `max_tokens: 256` bound the output side, which the implementation entry flagged as un-estimable.
- Batch transport added (commit `bb5243e`) - `src/infer/openai_batch.py` (submit /
  poll / collect, `openai_batch.py:78`, `:106`, `:135`) and `src/eval/judge_batch.py`
(`manage.py judge_batch`), which shards a condition into request files under `results/judge_<tag>_<split>_batch/`, records job ids in `results/judge_<tag>_<split>_batch_state.json`, and appends into the same segment cache the synchronous path writes (`judge_batch.py:59` `resume_point`, `:71` `append_results`). This halved the pass: the same 9,236-call session priced synchronously is $12.20.
- `src/eval/judge_ci.py` added (`manage.py judge_ci`) - per-condition Φ with bootstrap
  interval, score histogram, and a bootstrap rank distribution per rater, on shared
resample draws (`judge_ci.py:74`). Written for both raters: `results/judge_ci_val.json` (A) and `results/judge_ci_gpt_val.json` (B).

Artefacts written: `results/judge_gpt_val.json`, `results/judge_gpt_val_segments/` (7 conditions + `_meta.json` sentinel), `results/judge_gpt_val_usage.json`, `results/judge_ci_val.json`, `results/judge_ci_gpt_val.json`, `results/judge_agreement_gpt_val.json`. `results/judge_val.json` and its segment cache were not touched, which is what `assert_results_identity` and `assert_cache_identity` exist to guarantee.

Repository hygiene in the same commit range: the 9,261 lines of batch *request payloads* under `results/judge_gpt_val_batch/` are untracked and gitignored - they are regenerable from `outputs/` plus the frozen rubric, whereas the job ids, the scores and the usage record are tracked. The accidental nested clone at `notebooks/Style-Aware-MT` was committed as a gitlink (`24e81c5`, pointing at the pre-fork origin) and has been removed from the index with `git rm --cached`; the working directory is untouched.

### Cost

| Item | Figure |
|---|---|
| Calls (cumulative, incl. 25-call pilot) | 9,261 |
| Prompt / completion tokens | 3,796,497 / 386,804 |
| Batch discount | 0.5 |
| Cost | $6.1173 ($6.0996 for the 9,236-call full session) |
| Full-price equivalent | $12.24 |

Recorded in `docs/budget.md`, written today. That file was the outstanding precondition this entry's predecessor flagged; the declared cap itself still has to be transcribed from `docs/proposal.pdf`, which cannot be read in the current environment.

### Coverage

9,132 of 9,261 segments returned a parseable rating - 98.6 %, between 1,297 and 1,310 per condition against 1,323 offered. The 129 unscored segments are a property of rater B only; Φ_A coverage is unchanged (1,323 everywhere, 1,322 for `knn_fewshot`). Every agreement and contrast figure below is computed on the intersection of both raters and both conditions, so each contrast carries its own n (1,275-1,295) and none is padded or imputed.

`template_verified` is false in the report: rater A's results predate per-condition digest recording, so while B's `ffd6dad41acb0512` is recorded, *that both raters read the same rubric is asserted from the config, not verified from the artefacts*.

### Rater agreement

Pooled over the six study conditions (n = 7,823):

| Quantity | Value |
|---|---|
| Quadratic-weighted κ | 0.384 [0.370, 0.398] |
| Spearman ρ | 0.592 [0.577, 0.607] |
| Exact agreement | 27.8 % |
| Adjacent (±1) agreement | 77.2 % |
| Severity offset (A − B) | −0.950 [−0.968, −0.932] |

Two readings follow. First, the raters are not interchangeable in absolute terms: B sits almost a full rubric point above A on the same segments (Φ_A 2.55-2.79 against Φ_B 3.61-3.71), so a table averaging the two, or quoting one as a check on the other's level, would be uninterpretable. A constant offset cannot manufacture or destroy a contrast, which is why the deliverable is contrast replication rather than κ. Second, κ ≈ 0.38 is fair agreement at best - the two raters place the same segment in the same rubric band under 28 % of the time. Per-condition κ rises monotonically with how far up the ladder the condition sits, 0.302 (`zeroshot`) → 0.518 (`peft`), so agreement is worst exactly where the outputs are worst.

### System ordering

`afsp_full` is the highest-scoring study condition under both raters (modal rank 2 of 7 including the reference, p = 0.838 under A and 0.874 under B), which is the one ordering claim that survives the swap intact. Below it the orderings diverge: A ranks `afsp_margin` > `knn_fewshot` > `peft` > `random_fewshot` > `zeroshot`, while B ranks `knn_fewshot` > `afsp_margin` > `zeroshot` > `random_fewshot` > `peft`. PEFT falls from 5th to last. Spearman over the six study conditions is ρ = 0.714, p = 0.111 - with six systems the permutation floor is p = 0.0028, so this neither separates nor could it plausibly; n = 6 is simply too small to test ordering agreement, and the coefficient is reported as descriptive.

### Contrast replication

The deliverable. Each of the nine pre-specified contrasts re-run under both raters on one common segment set, Holm correction applied within each rater's family of nine.

| Contrast | Judge A (haiku) | Judge B (gpt) | Verdict |
|---|---|---|---|
| random few-shot − zero-shot | +0.089, p = .0008 ✓Holm | −0.014, p = .574 | rater-dependent, sign flips |
| kNN few-shot − zero-shot | +0.193, p < .001 ✓Holm | +0.027, p = .281 | rater-dependent |
| AFSP-margin − zero-shot | +0.212, p < .001 ✓Holm | +0.018, p = .487 | rater-dependent |
| AFSP-full − zero-shot | +0.244, p < .001 ✓Holm | +0.055, p = .030 ✗Holm | same sign, both separate uncorrected |
| PEFT − zero-shot | +0.202, p < .001 ✓Holm | −0.027, p = .403 | rater-dependent, sign flips |
| AFSP-full − kNN few-shot | +0.046, p = .056 | +0.031, p = .166 | neither separates |
| AFSP-margin − kNN few-shot | +0.013, p = .518 | −0.010, p = .596 | neither separates |
| AFSP-full − AFSP-margin | +0.029, p = .218 | +0.041, p = .059 | neither separates |
| PEFT − AFSP-full | −0.043, p = .169 | −0.097, p = .0018 ✓Holm | rater-dependent |

Three consequences, in order of how much they matter:

1. The "every rung of the ladder separates on Φ" reading is rater-dependent. Under A,
all five primary contrasts against the zero-shot floor separate and survive Holm. Under B, one of the five separates uncorrected and none survives Holm. Two - random few-shot and PEFT - reverse sign. Rater B still recovers the ladder's *direction* for the retrieval rungs (all three retrieval contrasts positive), but at a tenth the magnitude and with intervals spanning zero. The ladder's separation on COMET, chrF and BLEU is untouched by any of this; only the Φ column is affected.
2. The central negative result replicates. All three AFSP-internal secondary
contrasts fail to separate under *both* raters, `afsp_full − knn_fewshot` included (A p = .056, B p = .166). The finding that the adaptive layer is not demonstrated on validation therefore does not depend on rater identity - the one place where a negative result being robust is genuinely useful.
3. PEFT − AFSP-full is rater-dependent against PEFT. Under B, PEFT scores 0.097 below
AFSP-full on Φ and this survives Holm. This does not license "AFSP beats PEFT on register": it is one rater of two, against the other rater finding nothing (p = .169), and PEFT's significant lead on COMET, chrF and BLEU is unaffected. It is a lead, and it points the opposite way from PEFT's `stylo_dist` advantage (0.289, the best of any condition) - which is itself an RQ4 disagreement worth following.

### Self-preference of the primary rater

The sharpest reason for this pass was that rater A and the `commercial_haiku` condition are the same model family. The severity offset makes the effect visible: A sits 0.87-1.09 below B on the six study conditions, but only 0.641 [0.605, 0.678] below B on `commercial_haiku` - that is, A is 0.23-0.45 rubric points *less severe* on its own family's output, relative to B, than it is on everything else. Exact agreement on that condition is 42.7 % against 27.8 % pooled, and adjacent agreement 91.4 %, so the two raters converge there while diverging elsewhere, consistent with the shared-family reading.

But the reference's lead is not an artefact of self-judging. Rater B, a different family, also places `commercial_haiku` first by a wide margin: Φ_B = 3.981 [3.939, 4.022] against `afsp_full` at 3.707 [3.661, 3.753]. The correct statement is that the +0.590 Φ_A margin over PEFT is inflated by self-preference, not that it is manufactured by it. This does not rehabilitate that figure as a register measure - `commercial_haiku` remains a diagnostic external reference, its `stylo_dist` of 0.556 still points the other way, and its Φ is still not admissible as a register finding.

### Verification

- `ruff check src tests manage.py` clean; `pytest tests/ -q` 96 passed (19 added since
  the implementation entry's 77, chiefly `tests/test_judge_batch.py`).
- The 25-call pilot ran first and wrote only the segment cache, per `--limit` semantics;
  its cost is the $0.0177 difference between the session and cumulative figures.
- `results/judge_val.json` compared before and after: unchanged. The tagged paths kept
  Φ_A intact, which was the design requirement.
- Bootstrap parameters identical to every prior inferential run - 10,000 resamples,
  α = 0.05, seed 42, paired at segment level - so the A-side figures in the replication
table reproduce `results/bootstrap_judge_val.json` where the segment sets coincide.
- Rank distributions in both `judge_ci` reports are computed on shared resample draws, so
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

The batch request payloads are gitignored; `judge_batch` regenerates them from `outputs/` and the frozen rubric. Re-running the judge stage costs money again - the segment caches under `results/judge_{val,gpt_val}_segments/` are what make the analysis stages free.

### Limitations and risks

- This bounds rater dependence; it does not validate Φ. Neither rater is ground truth.
  κ = 0.384 with a −0.95 offset is equally consistent with two differently-biased raters as
with one correct and one noisy. What the pass establishes is which conclusions survive changing the rater, and the answer is: the AFSP-versus-baseline null, and `afsp_full` at the top of the ladder. Not the primary contrasts against the floor.
- Φ_B is not seed-controlled either. `seed: 42` is set in the config but documented as
  best-effort by the provider, so a re-run will not reproduce byte-for-byte. The pass
removes single-family dependence; it does not make Φ reproducible, and the standing caution that Φ differences of order 0.05 are within measurement noise still holds - note that seven of the nine contrasts under B are smaller than 0.05 in absolute value.
- Rubric identity is asserted, not verified, for rater A. Until Φ_A is re-run under a
  build that records `template_sha256`, the claim that both raters read the same rubric
rests on the configs rather than on the artefacts. This is cheap to close only by re-spending on Φ_A, so it stays open.
- 129 segments (1.4 %) have no Φ_B. They are not random with respect to content -
  a rating is missing because the response did not parse, which correlates with the
output being unusual. The effect on the reported figures is bounded by 1.4 % of the sample but is not zero, and no imputation was performed.
- The ordering-agreement test is underpowered by construction. At n = 6 systems the
  Spearman permutation floor is p = 0.0028; ρ = 0.714 could not have separated whatever
the data. Do not report the orderings as "statistically consistent" - report the rankings and let them speak.
- Rater B's compression is unexplained. B's ratings span 3.61-3.71 across the six study
  conditions, a range of 0.10, against A's 0.25. Whether that is genuine insensitivity to
register, a ceiling effect from `reasoning_effort: none`, or the rubric reading differently to a different family is not established, and it is the obvious next diagnostic: a small pilot at a higher reasoning effort would distinguish the ceiling explanation from the others.

## 2026-08-05: Second cross-family judge implemented, not yet run

### Summary

The evaluation-time judge pipeline can now score the same outputs with a second rater from a different provider family, and a new `judge_agreement` stage measures how far a Φ-based conclusion depends on rater identity. Implemented and statically verified only: no judge call has been made, no OpenAI artifact exists, and nothing under `results/` or `outputs/` was written. No reported figure changes.

### What changed

- `src/eval/judge.py` - `--tag` routes a second judge's artifacts to
  `results/judge_<tag>_<split>.json` and `results/judge_<tag>_<split>_segments/`,
defaulting to the config's `tag:`. Untagged behaviour is unchanged, so the existing `judge_val.json` and its segment cache keep their paths.
- `src/eval/judge.py:104` `assert_results_identity` - refuses to write a results file
  already scored by a different judge model unless `--allow_model_overwrite` is passed.
- `src/eval/judge.py:79` `assert_cache_identity` - binds a segment cache to one
  `(model, tag, rubric digest)` via a `_meta.json` sentinel and refuses a mismatch.
- `src/eval/judge.py` - records `judge_tag` and `template_sha256` per condition; writes
  `results/judge_<stem>_usage.json` with per-session and cumulative token/cost figures;
adds `--limit N` for a diagnostic pilot, which writes the segment cache but deliberately not the results file.
- `src/eval/judge_agreement.py` (new, `manage.py judge_agreement`) - quadratic-weighted
  κ, Spearman ρ, exact/adjacent rates, the paired severity offset, system-ordering
comparison, and re-runs of the nine pre-specified contrasts under each rater.
- `configs/judge_eval_gpt.yaml` (new) - the second rater. `template_file` is the same
  frozen `prompts/judge_eval.txt`.
- `src/infer/openai_client.py:35`, `src/infer/run.py:72` - optional `pricing: [in, out]`
  from config, so a model absent from the built-in table reports real cost instead of
silently costing zero.
- `notebooks/judge_second_pass_colab.ipynb` (new) - the runbook.
- `tests/test_judge_agreement.py` (new) - 22 cases.

### Rationale

Single-judge dependence has been the open construct-validity threat since the judge was introduced, and the threats register carries it as *cross-family pass unrun*. It is sharper than a generic reliability concern here: the primary rater `claude-haiku-4-5` is the same model family as the `commercial_haiku` condition it scores, so the one system scoring highest on Φ is scored by its own family.

Three design points are load-bearing:

1. Same frozen rubric. Both raters read `prompts/judge_eval.txt`. Re-tuning the
prompt for the second rater would confound rater identity with rubric identity and void the comparison; the digest is recorded per condition and `judge_agreement` raises if the two disagree.
2. Contrast replication is the deliverable, not κ. Two raters can agree poorly
segment-by-segment and still order the systems identically. `contrast_replication` therefore re-runs each pre-specified contrast under both raters on one common segment set - the intersection over both conditions and both raters - and labels each contrast `both separate`, `neither separates`, or `RATER-DEPENDENT`. Holm correction is applied within each rater's family of nine.
3. The primary Φ must be unclobberable. Running the second config without a tag
would have overwritten `results/judge_val.json` in place, and pointing it at the existing segment cache would have found the cache complete, made zero calls, and returned judge A's scores labelled as judge B's. Both paths now raise.

### Verification

Static only, as nothing was run:

- `ruff check src tests manage.py` clean; `pytest tests/ -q` 77 passed (22 new).
- `quadratic_weighted_kappa` agrees with `sklearn.metrics.cohen_kappa_score(weights=
  'quadratic')` to 12 decimal places over five random 200-point samples. The bootstrap
is vectorised through `bincount` because a per-resample Python loop at the pooled n ≈ 9,000 would be ~10⁸ iterations.
- End-to-end plumbing check at full scale in a scratch directory, with a synthetic
  second rater derived by perturbing judge A's scores: 9,260 paired segments, report
renders, `knn_fewshot`'s n = 1,322 propagates correctly, 11 s at 2,000 resamples. Those numbers are plumbing evidence and carry no empirical status whatsoever.
- Notebook pre-flight and cost-estimate cells executed against the real repository;
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

- Unrun, so it establishes nothing yet. A first execution would establish the
  agreement coefficients, the severity offset, and - the only part bearing on the
thesis - which pre-specified contrasts survive the rater swap. Until then single-judge dependence remains a standing limitation and the threats register entry stays open.
- Cost is the gating risk, and it is not yet known. The full pass is 7 × 1,323 =
  9,261 calls over ~15.9 M prompt characters (≈4.0-5.3 M input tokens). The output side
is dominated by hidden reasoning tokens and cannot be estimated from the prompt, so the notebook requires a 25-call pilot and an explicit confirmation before the full run. `docs/budget.md` is still not written; the cap belongs there first.
- Model identity unconfirmed. `configs/judge_eval_gpt.yaml` names `gpt-5.6` with
  placeholder `pricing: [5.00, 30.00]`. Both must be checked against the provider's
current model list before spending. A wrong price only misreports cost; a wrong model id fails the run.
- Φ_B is not seed-controlled either. The config sets `seed: 42`, but the provider
  documents it as best-effort, so the second rater does not resolve the
non-reproducibility of Φ - it only removes the single-family dependence.
- Agreement does not validate Φ. A high κ is equally consistent with two
  similarly-biased raters. The result bounds rater dependence; it does not establish
that either rater measures register correctly.
- The severity offset must not be averaged away. If the two raters differ in mean
  severity, Φ_A and Φ_B are not interchangeable in a table of absolute values. A
constant offset cannot create or destroy a contrast, but a combined "mean Φ" column would be uninterpretable.

## 2026-08-04: Commercial zero-shot reference baseline added to the README

### Summary

`commercial_haiku` has existed as an artifact since the 2026-07-31 six-condition run and its paired-bootstrap comparisons were already in `results/bootstrap_{comet,judge,chrf,bleu}_val.json`, but no figure from it appeared in `README.md`. It is now reported, in a section of its own, explicitly outside the study table. No new inference or scoring was run and no existing figure changed.

### What changed

- `README.md:252-303` - new subsection *External reference baseline - commercial
  zero-shot (not a condition of the study)*: system-level table, the three paired
bootstrap contrasts against `zeroshot`, `peft`, and `afsp_full` with 95 % intervals, and three constraints on how it may be read.
- `README.md:206-207` - the study-conditions table now states that it lists study
  conditions only and points to the new subsection.
- `README.md:176-182` - the judge *Open issue* callout now records that the judge and
  this baseline are the same model.

### Reported figures

System level, val, n = 1,323: COMET 0.7185, chrF 45.24, BLEU 18.06, Φ 3.3333 (coverage 1.00), lexical density 0.3928, TTR 0.8241, `stylo_dist` 0.5557. Cost $0.6316 over 1,323 calls (`outputs/commercial_haiku_val_usage.json`).

Paired bootstrap, 10,000 resamples, α = 0.05, seed 42, segment level. Against `peft`: COMET +0.0199 [0.0153, 0.0245], Φ +0.5896 [0.5298, 0.6478], chrF +3.76 [3.00, 4.53], BLEU +1.26 [0.52, 2.03] (p = .0008; the other eleven at p < .001). Against `zeroshot` and `afsp_full` the direction and significance are the same.

### Rationale

Protocol requires the evidence class of a cited number to be named in the same sentence and diagnostic runs to stay out of the study tables. `commercial_haiku` is neither a study condition nor purely diagnostic: it is a labelled external reference baseline, and `src/eval/metric_agreement.py:43` already treats it that way by reporting `study_only` and `with_reference` scopes separately. Omitting it from the README entirely was the wrong resolution - it hid the one system in the artifact set that scores highest on the primary style metric - so it is reported under its own heading with its class in the heading itself.

### Verification

Figures recomputed from artifacts rather than copied from prose: `src.eval.quick.score` for chrF/BLEU, `src.eval.stylometrics.aggregate` + `distance_to_centroid` for the stylometric columns, `results/comet_val.json` and `results/judge_val.json` for COMET and Φ, and the `comparisons` arrays in the four bootstrap files for the intervals. Nothing under `results/` or `outputs/` was written.

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

- The Φ figure is self-judged. The generator is `claude-haiku-4-5`
  (`configs/commercial_haiku_zeroshot.yaml`) and so is the evaluation-time judge
(`configs/judge_eval.yaml`). Its Φ 3.333 - the highest of any system, +0.590 over PEFT
  - is a model rating its own output, and self-preference bias is a sufficient
  alternative explanation. This does not touch the six study conditions, none of which
the judge generated, but it makes this one Φ inadmissible as a register claim. The README says so at the point of use.
- The two style measures invert on this system. Highest Φ of any system, yet
  `stylo_dist` 0.5557 is second-worst overall, behind every few-shot rung and well behind
PEFT (0.2886). Signed z-deviations: lexical density −0.378 and TTR −0.276, the largest magnitudes in the condition set. Admitting it to the RQ4 correlation moves ρ(Φ, `stylo_dist`) from −0.657 (n = 6) to −0.250 (n = 7).
- Not a matched comparison. Different base model, no adaptation budget, closed
  weights, and a price per call. It cannot enter the single-factor ladder and does not
bear on RQ1. The adequacy margins are a scope statement about the 7 B open base.
- Multiplicity. These three contrasts sit inside the same 14-comparisons-per-metric
  family as the rest of the bootstrap run. Their intervals are wide of zero by margins
that Holm-Bonferroni would not reach, but the correction status is stated nowhere in the new subsection beyond the shared estimator note; if any of these figures is ever used to support a claim rather than to bound a baseline, the correction must be applied explicitly.

## 2026-08-01: AFSP lambda dose-response analysis

### Summary

The bootstrap entry below left the adaptive layer unshown at one λ. This entry reads the whole knob instead: all 18 `k × λ` cells already in `outputs/sweep`, scored on the four centroid features and on COMET, with λ treated as a dose. The motivating claim was that a monotone fall in `stylo_dist` across six λ values would be stronger evidence than the single pairwise test at λ = 0.75 that failed (Φ, p = .078).

`stylo_dist` shows no dose-response, and its per-k trends contradict each other. Pooled over all 18 cells, Spearman ρ = −0.10 (p = .68). Per k: ρ = +0.77 (p = .072) at k = 4 - distance *grows* with λ - ρ = −0.20 (p = .70) at k = 8, and ρ = −0.94 (p = .005) at k = 16. A knob whose sign flips with k is not a dose-response. The headline metric does not answer RQ3 affirmatively.

The signed per-feature view does, and it explains why the norm hides it. λ moves three of the four features monotonically, in the same direction, at every k:

| Feature | z at λ=0 (k=16) | ρ vs λ, k=4 | k=8 | k=16 | Reading |
|---|---|---|---|---|---|
| `lex_density` | −0.275 | +0.89 (p=.019) | +0.77 (p=.072) | +0.83 (p=.042) | undershoots → λ helps |
| `ttr` | −0.188 | +0.83 (p=.042) | +0.71 (p=.111) | +0.94 (p=.005) | undershoots → λ helps |
| `marker_rate` | +0.153 | +0.71 (p=.111) | +0.77 (p=.072) | +0.89 (p=.019) | overshoots → λ hurts |
| `root_ttr` | −0.177 | −0.09 (p=.87) | −0.03 (p=.96) | −0.43 (p=.40) | inert |

λ does not move outputs *toward* the target register. It turns register intensity *up*: denser, more varied, more archaic-marked prose, monotonically in λ. Two of those movements close a gap and one widens one that is already open - every condition in the project overshoots `marker_rate`, and λ pushes it further out, the opposite of the predicted correction. `stylo_dist` is the norm of the four, so the opposing components partly cancel, which is what makes it flat pooled and sign-flipping per k: at k = 4 the `marker_rate` overshoot is largest (+0.265 at λ = 0) and dominates, so distance grows; at k = 16 it starts smallest (+0.153) and has headroom, so the two gains dominate and distance falls. The k-dependence is a consequence of the starting overshoot, not of the rerank behaving differently.

No style-adequacy trade-off at λ = 1. COMET spans 0.6746-0.6886 across all 18 cells, and within each k the λ trend is null (ρ = −0.49, +0.37, +0.03; all p > .32). k, not λ, sets adequacy: k = 16 is above k = 8 above k = 4 at every λ. Pure register rerank with semantic similarity switched off (λ = 1) costs 0.0023 COMET at k = 4 and 0.0000 at k = 16 relative to λ = 0. The anticipated trade-off curve is a null; the figure is kept because the null is the finding.

Φ at the three judged cells moves with `stylo_dist`, on two usable points. k = 8 λ = 0.75 → λ = 1: Φ 2.797 → 2.763 while `stylo_dist` worsens 0.371 → 0.411 - same direction, but the judge CIs (±0.035) overlap and the third judged cell is at a λ already occupied. This is consistent with agreement, not evidence of it.

Everything is inside the bootstrap CIs. Cell-level 95% CIs on `stylo_dist` are ≈±0.045 (e.g. k = 16 λ = 1: 0.3808 [0.3378, 0.4344]); every cell overlaps every other cell. The trends above are rank correlations over point estimates, which is what makes the consistency across k - three independent k series agreeing on the sign for three features - the load-bearing evidence rather than any single cell.

### What changed

- `src/eval/sweep_curves.py` (new), registered as `python manage.py sweep_curves`.
  Computes per-cell aggregates, signed z-deviations, segment bootstrap CIs, and
Spearman trend tests; writes `results/sweep_curves_val.json` and four figures.
- `results/sweep/comet_val.json` (new): COMET for all 18 cells, scored locally on the
  RTX 4060. `afsp_k16_l0.75` reproduces the 2026-07-23 verify value 0.6871 exactly.
- `docs/figures/` (new): `lambda_stylo_dist.png`, `lambda_z_deviations.png`,
  `lambda_style_adequacy.png`, `lambda_judge_overlay.png`.
- `matplotlib` added to the environment; it was not previously a dependency.

### Rationale

`stylo_dist` was the only stylometric number the sweep reported per cell, and it is an undirected norm - it cannot distinguish a condition approaching the centroid from one trading an improvement on one feature against a regression on another. The λ sweep is exactly the case where that matters, because the rerank turns out to move features in opposite senses relative to target. The signed decomposition is four numbers the sweep already had the data for and did not report.

Spearman over six λ points per k is a deliberately weak test (its p floor at n = 6 is .0028) and is not used here to establish significance. It summarizes direction; the argument rests on three k series independently agreeing.

### Reproduction

```
python manage.py comet --conditions afsp_k4_l0 afsp_k4_l0.1 afsp_k4_l0.25 afsp_k4_l0.5 \
  afsp_k4_l0.75 afsp_k4_l1 afsp_k8_l0 afsp_k8_l0.1 afsp_k8_l0.25 afsp_k8_l0.5 \
  afsp_k8_l0.75 afsp_k8_l1 afsp_k16_l0 afsp_k16_l0.1 afsp_k16_l0.25 afsp_k16_l0.5 \
  afsp_k16_l0.75 afsp_k16_l1 \
  --split val --out_dir outputs/sweep --results_dir results/sweep --batch_size 8
python manage.py sweep_curves
```

The COMET pass loads `Unbabel/wmt22-comet-da` once and reuses it across all 18 cells; it fits in the 8 GB card at `--batch_size 8`. `sweep_curves` is CPU-only. The bootstrap is seeded (`--seed 42`, 2,000 resamples), so the CIs are reproducible.

### Limitations and risks

The frozen `afsp_full` outputs are not byte-identical to their sweep cell `afsp_k8_l0.75`: 5 of 1,323 predictions differ, worth 0.0011 in `stylo_dist` (0.3698 vs 0.3709). `afsp_margin` and `afsp_k8_l0` are identical on all 1,323. The frozen condition was therefore re-run rather than copied, and decoding is not fully deterministic across runs. The scale of that drift is ≈2% of the bootstrap CI half-width, so it does not affect any conclusion here, but the frozen artifact and the sweep cell are two samples, not one.

This is validation-split evidence about the *mechanism* of the rerank. It does not license a claim that AFSP beats its baseline - the bootstrap below still stands, and no test-split number exists. In particular, the `marker_rate` finding argues that the current register objective is mis-targeted for this corpus rather than that λ should be raised; acting on it means revisiting the band-pass target of 2026-07-18, not retuning λ.

Six λ values at three k is 18 points with no replication per cell. The per-k Spearman values cannot separate a real k × λ interaction from three noisy draws; the shared direction across k is the only claim the design supports.

## 2026-08-01: Paired bootstrap; AFSP does not separate from its baseline on validation

### Summary

The paired bootstrap that every entry since 2026-07-05 has deferred was run on all six conditions and all four metrics. It is the gate the 2026-07-31 entry named: until it ran, no ordering in the results table could be called a difference. Two of the three observations that entry recorded do not survive it.

The contribution does not separate from its own baseline. On no metric does `afsp_full` beat `knn_fewshot` at α = 0.05:

| Comparison | COMET | Φ | chrF | BLEU |
|---|---|---|---|---|
| `afsp_full` − `knn_fewshot` | +0.001 (p = .43) | +0.043 (p = .078) | +0.20 (p = .41) | +0.45 (p = .065) |
| `afsp_margin` − `knn_fewshot` | −0.002 (p = .25) | +0.016 (p = .42) | −0.07 (p = .70) | −0.13 (p = .49) |
| `afsp_full` − `afsp_margin` | +0.003 (p = .077) | +0.028 (p = .23) | +0.27 (p = .23) | +0.58 (p = .010) |
| `afsp_full` − `peft` | −0.013 (p < .001) | +0.047 (p = .12) | −1.06 (p = .003) | −2.13 (p < .001) |

The PEFT/AFSP "disagreement" was an artifact of reading point estimates. The 2026-07-31 entry framed it as the objective and subjective register measures ranking the two adaptation families oppositely, and called it the RQ4 question. The COMET half is significant; the Φ half is not (+0.047, 95% CI [−0.012, 0.105], p = 0.12). One measure separates the families and the other fails to - which is a null result on the Φ side, not a disagreement between measures. RQ4 must be answered by the agreement analysis the README specifies (pairwise correlation across metrics), not by this pair.

The prompting ladder's floor is solid. `zeroshot` → `random_fewshot` → `knn_fewshot` is significant on all four metrics (Φ +0.087 [0.037, 0.137] and +0.115 [0.066, 0.163]; COMET +0.016 and +0.020, both p < .001). Retrieval works; the adaptive layer over it is what remains unshown.

Detection floor for RLSF. The Φ CI half-width against `peft` at n = 1,323 is ≈0.058 and the COMET half-width ≈0.005. Since RLSF initializes from the PEFT adapter, those are the margins an RL gain must clear to be reportable. For scale, the whole prompting ladder moves Φ by 0.245, so RLSF needs roughly a quarter of the ladder's total effect to register. Judge test-retest spread (≈0.002, measured in the 2026-07-31 entry) is an order of magnitude below the sampling interval, so sampling noise, not judge noise, is the binding constraint.

### What changed

`src/eval/bootstrap.py` gained three things needed to make the run reproducible rather than a terminal session:

- `--out [PATH]` writes the comparison table as JSON - bare `--out` uses
  `results/bootstrap_<metric>_<split>.json`. Each file records the metric, split,
conditions, baseline, `n_resamples`, `alpha`, `seed`, and per-comparison `n`, `mean_a`, `mean_b`, `diff`, `ci_low`, `ci_high`, `p_value`, `significant`. Before this the command printed and exited, which is why no `results/bootstrap_*` artifact existed despite the command having shipped on 2026-07-05.
- `--pairs A:B` adds arbitrary comparisons to the baseline and `--adjacent` sets.
  The comparison this project exists to make - `afsp_full` against `knn_fewshot` - is
neither a comparison against the ladder floor nor an adjacent-rung pair, so it was unreachable from the CLI as shipped. `_parse_extra_pairs` rejects unknown conditions and self-comparisons.
- `_pairs` now deduplicates, so a pair reachable two ways is resampled once.

No scoring code changed; `paired_bootstrap` is untouched.

### Verification

`ruff check src`, `ruff format --check src`, and `python -m compileall -q src` pass. The four artifacts were written in one pass per metric from the same command; the `comet`/`judge` runs read the persisted per-segment vectors and the `chrf`/`bleu` runs recompute from the inference files. Alignment is enforced by the existing `_assert_aligned` guard, which compares the stored `sources` lists index for index - it passed for all six conditions, confirming the paired assumption holds.

`n = 1,322` rather than 1,323 on judge pairs involving `knn_fewshot` is the single unparseable judge segment recorded in the 2026-07-31 entry; `paired_bootstrap` drops the pair rather than imputing it.

### Reproduction

```bash
CONDS="zeroshot random_fewshot knn_fewshot afsp_margin afsp_full peft"
EXTRA="afsp_full:knn_fewshot afsp_margin:peft afsp_full:peft knn_fewshot:peft random_fewshot:peft"
for m in comet judge chrf bleu; do
    python manage.py bootstrap --metric "$m" --conditions $CONDS --split val \
        --baseline zeroshot --adjacent --pairs $EXTRA --out
done
```

Deterministic at `seed 42`: the resample indices come from `np.random.default_rng(seed)`, so re-running reproduces every interval exactly.

### Limitations and risks

The p-values are uncorrected. The canonical run makes 14 comparisons per metric over four metrics, 56 tests in total, at α = 0.05 each. The lone significant AFSP result - `afsp_full` − `afsp_margin` on BLEU, p = .010 - would not survive Holm-Bonferroni over that family, and BLEU is the weakest of the four metrics. It is a lead worth powering up, not a finding. A pre-registered comparison family should be declared before the test-split run so the correction is decided in advance rather than after seeing which comparisons came close.

Two near misses (Φ p = .078 and BLEU p = .065 for `afsp_full` − `knn_fewshot`) mean the null result is a failure to detect, not a demonstration of no effect. At this effect size the val split is underpowered; the honest options are to report the intervals as they stand, or to increase power by a route that does not touch the sealed test split - repeated judge passes to average down judge variance, or a larger exemplar-selection effect.

The chrF and BLEU intervals resample sentence-level scores, whereas the results table reports corpus-level chrF and BLEU. The two are not the same statistic, so a CI here does not bound the corpus figure in the table. COMET and Φ have no such gap: both are means over per-segment scores in both places.

`stylo_dist` is not bootstrappable through this CLI. It is a distance between a condition's *mean* feature vector and the centroid, not a mean of per-segment scores, so the monotone ladder in the results table (0.652 → 0.370) is still descriptive only. Reporting it as an ordering requires either a per-segment formulation or a bootstrap over the feature-vector statistic itself.

The judge remains a single non-reproducible model. These intervals quantify sampling uncertainty at fixed judge output; they do not include the judge's own re-sampling variance, so the true uncertainty on any Φ difference is wider than the interval printed here.

## 2026-07-31: PEFT frozen; six-condition validation evaluation completed

### Summary

The PEFT condition was trained, tuned, confirmed, and generated on the full val split, which completes the val evaluation for all six implemented conditions
- the five prompting rungs plus PEFT. For the first time the project has COMET
and LLM-as-Judge Φ numbers for every condition on the same 1,323-segment split, so the README results table is no longer empty.

The frozen PEFT configuration is r = 32, α = 64, lr = 2e-4, 2 epochs, LoRA on all linear layers, adapter `models/peft_lora_r32_lr2e-4/checkpoint-1358` (80,740,352 trainable parameters across 392 adapter tensors, ≈1.06% of the 7.62 B base). It was selected by `src.peft.sweep` and confirmed by `src.peft.verify` (2026-07-25 entry), and is frozen into `configs/peft_qwen.yaml` as `generator.adapter_path`.

These are val results. The test split remains sealed - `data/splits/test.jsonl` (1,322 segments) has not been generated on by any condition.

| Condition | COMET | chrF | BLEU | Judge Φ | lex_density | TTR | stylo_dist |
|---|---|---|---|---|---|---|---|
| `zeroshot` | 0.6480 | 36.42 | 10.27 | 2.546 | 0.4025 | 0.8434 | 0.6518 |
| `random_fewshot` | 0.6644 | 37.52 | 11.64 | 2.633 | 0.4103 | 0.8444 | 0.4736 |
| `knn_fewshot` | 0.6839 | 39.82 | 13.99 | 2.748 | 0.4034 | 0.8367 | 0.4005 |
| `afsp_margin` | 0.6824 | 39.68 | 13.69 | 2.763 | 0.4060 | 0.8387 | 0.3910 |
| `afsp_full` | 0.6853 | 39.99 | 14.52 | 2.791 | 0.4105 | 0.8421 | 0.3698 |
| `peft` | 0.6986 | 41.58 | 16.90 | 2.744 | 0.4088 | 0.8515 | 0.2886 |

Three observations, all descriptive and none yet significance-tested:

1. The prompting ladder is monotone on COMET and chrF from `zeroshot` to
`knn_fewshot`, and `stylo_dist` falls monotonically across the whole ladder (0.652 → 0.370) - each rung sits closer to the target-register centroid than the one below it.
2. PEFT and AFSP disagree about which is more in-register. *(Not supported by
the bootstrap of 2026-08-01: the Φ half of this comparison is non-significant, p = 0.12.)* PEFT wins every surface and adequacy metric (COMET +0.013, chrF +1.6, BLEU +2.4 over `afsp_full`) and has the smallest stylometric distance (0.289 vs 0.370), but `afsp_full` scores highest on judge Φ (2.791 vs 2.744). The objective and subjective register measures rank the two adaptation families oppositely. This is exactly the RQ4 agreement/disagreement question and should be reported as such, not resolved by picking the flattering metric.
3. `afsp_margin` does not beat `knn_fewshot` on COMET (0.6824 vs 0.6839) -
the margin/hub rung is flat-to-slightly-negative on adequacy - while the register rerank rung (`afsp_full`) adds both adequacy and Φ over it. On this split the gain attributable to AFSP comes from the register rerank, not from margin/hub penalization. *(Not supported by the bootstrap of 2026-08-01: neither AFSP rung separates from the rung below it, and `afsp_full` does not separate from `knn_fewshot` on any metric.)*

### What changed

No source changes. This entry records generation and scoring artifacts:

- `outputs/peft_val.jsonl` (1,323 rows) + `outputs/peft_val_usage.json` - the
  frozen adapter generated on full val under the locked decoding contract
(greedy, `temperature 0.0`, `seed 42`, `max_new_tokens 1024`, bf16, not quantized).
- `results/comet_val.json` - per-segment `Unbabel/wmt22-comet-da` for all six
  conditions, `n = 1,323` each, with the shared `sources` list persisted so the
paired bootstrap can verify segment alignment.
- `results/judge_val.json` + `results/judge_val_segments/*.jsonl` - evaluation-time
  judge Φ for all six conditions. Coverage is 1.0 everywhere except
`knn_fewshot` (0.9992 - one segment unparseable and dropped).
- `results/peft_sweep_val.json`, `results/peft_verify_val.json` - the selection
  and confirmation records (see the 2026-07-25 entry).
- `configs/peft_qwen.yaml` - `adapter_path`, `r: 32`, `alpha: 64`,
  `learning_rate: 0.0002` frozen to the confirmed cell.

The run was executed on Colab across three phases recorded in `notebooks/peft_full_run_colab.ipynb` (train/sweep, verify, final generation); the 8 GB development GPU cannot hold the bf16 base, unchanged from Stage 0.

### Verification

The surface and stylometric numbers in the table above were recomputed locally on 2026-07-31 from the committed inference files. chrF matches the sweep/verify records cell for cell (`peft` 41.58 = the `peft_r32_lr2e-4_e2` sweep cell; `afsp_full` 39.99 = the frozen `afsp_k8_l0.75` cell), confirming that the shipped `peft`/`afsp_full` outputs were generated from the frozen configs and not from some other cell:

```bash
python manage.py eval          --conditions zeroshot random_fewshot knn_fewshot afsp_margin afsp_full peft --split val
python manage.py stylometrics  --conditions zeroshot random_fewshot knn_fewshot afsp_margin afsp_full peft --split val
```

COMET and Φ were read from the committed JSON, not re-run. The adapter parameter count was read from the safetensors header of `models/peft_lora_r32_lr2e-4/checkpoint-1358/adapter_model.safetensors`; its `adapter_config.json` confirms `r: 32`, `lora_alpha: 64`, and all seven linear projections (`q,k,v,o,gate,up,down`) as targets.

Correction (2026-08-01): `stylo_dist` does not match cell for cell, because two of the four comparable runs are not byte-identical to their swept cell. Comparing `outputs/sweep/*.jsonl` against the shipped ladder files segment by segment:

| Shipped run | Swept cell | Differing segments | `stylo_dist` (cell → shipped) |
|---|---|---:|---|
| `afsp_margin` | `afsp_k8_l0` | 0 / 1,323 | 0.3910 → 0.3910 |
| `peft` | `peft_r32_lr2e-4_e2` | 0 / 1,323 | 0.2886 → 0.2886 |
| `zeroshot` | `afsp_zeroshot` | 2 / 1,323 | 0.6515 → 0.6518 |
| `afsp_full` | `afsp_k8_l0.75` | 5 / 1,323 | 0.3709 → 0.3698 |

The seven differing segments are whole-sentence divergences, not token-level jitter. The split tracks the usage records exactly: the two runs that differ are the two that were resumed across sessions (`zeroshot` 1,303 calls, `afsp_full` 1,318, against `n = 1,323`), and the two that match byte for byte each record the full 1,323 calls. That is consistent with divergence across generation sessions rather than within one, so the locked greedy contract of the 2026-07-03 Stage 0 entry buys byte-for-byte reproducibility within a session but has not held across sessions on this hardware. The reported figures are the shipped ladder files, which are the ones COMET and the judge scored; the discrepancy affects `stylo_dist` in the fourth decimal and changes no ordering, but the decoding contract should be restated to claim only within-session determinism.

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

No significance testing has been run. `manage.py bootstrap` exists and the per-segment COMET/Φ vectors it needs are persisted, but no `results/bootstrap_*` artifact exists. Every ordering above - including the 0.047 Φ gap between `afsp_full` and `peft` and the 0.0015 COMET gap between `knn_fewshot` and `afsp_margin` - is currently an unqualified point estimate. None of them may be described as a difference until the paired bootstrap is run.

*Superseded (2026-08-01): the bootstrap has now been run; see that entry. Both gaps named here are non-significant, and observations 2 and 3 of this entry's summary do not survive it. The observations are left standing as written, because they record what the point estimates showed on 2026-07-31 and why the bootstrap was the next step - not as claims that may be cited.*

The judge is `claude-haiku-4-5` at `temperature 0.0`, a single model from a single family with no seed control (see the 2026-07-23 entry). Φ differences on the order of 0.05 are plausibly within judge noise, and the cross-family confirmation pass the methodology calls for has not been run. Φ is also the *primary* stylistic metric for the thesis, which makes this the weakest link in the current results.

The size of that judge noise is now measurable, and it is not negligible. Two conditions were scored twice by the same judge on the same 1,323 segments - once during confirmation and once in the final evaluation pass - and the two passes disagree: `afsp_full` scores 2.797 in `results/afsp_verify_val.json` against 2.791 in `results/judge_val.json`, and `peft` scores 2.741 in `results/peft_verify_val.json` against 2.744 in `results/judge_val.json`. For `afsp_full` the shipped file also differs from the confirmed cell in 5 segments, so that pair is not a pure re-scoring; the `peft` pair is, and it moves by 0.002 on identical inputs. Test-retest spread of that order sits inside the 0.047 Φ gap this entry reports between `afsp_full` and `peft`, and the freeze decisions of the 2026-07-23 and 2026-07-25 entries turned on Φ gaps of 0.019 and 0.009. Repeated judge passes, not only a cross-family pass, are needed before any Φ ordering is reported.

Cost and latency are not instrumented. `outputs/*_usage.json` records token counts only (`cost_usd` is 0 for local weights), and the `calls` field is below `n = 1,323` for the resumed runs (1,184-1,318), so it counts generation calls actually issued, not segments - it cannot be used as a per-condition cost measure. The README reports trainable parameters but leaves inference latency unmeasured, which is one of the three components its cost row promises.

RLSF remains unimplemented (`src/rlsf/` holds only an empty `__init__.py`). RQ1 compares four conditions - the prompting baseline, AFSP, PEFT, and RLSF - so it is a three-condition comparison today.

## 2026-07-25: PEFT sweep and verification tools

*Scope: the sweep and verify drivers were written on 2026-07-24 and 2026-07-25 (`de81c23`, `d813146`, `deb4298`, `1de6059`, `d2452be`, `c0efb67`); the smoke runs that validated them ran on 2026-07-25 and 2026-07-26, and the sweep and verify runs whose results this entry reports ran on 2026-07-27 to 2026-07-30. The entry is dated to the driver work and reports the run it selected.*

### Summary

The 2026-07-24 entry left the LoRA hyperparameters as placeholder defaults with an explicit warning that they had to be dev-tuned before the reported run. That tuning was implemented as a two-stage driver that mirrors the AFSP `sweep → verify → freeze` pattern, so both adaptation arms are selected under the same rule on the same axis. Without this, PEFT would have been selected on `eval_loss` and AFSP on register fidelity, and a PEFT-vs-AFSP comparison would have confounded the adaptation method with the selection criterion.

The consequential design decision is that `eval_loss` is not the selector. It is demoted to a free pre-filter that decides which checkpoints are worth generating with; the reported adapter is picked on chrF adequacy band + register fidelity, then confirmed on COMET + Φ.

### What changed

`src/peft/sweep.py` (new, `manage.py peft_sweep`). Trains the grid in `configs/peft_sweep.yaml` - four (r, lr) cells: (16, 2e-4) as the labelled anchor, (8, 2e-4), (32, 2e-4), (16, 1e-4), all with α = 2r - for the full three-epoch budget. `enumerate_candidates` then treats every saved epoch checkpoint as its own candidate, so a candidate is an (r, lr, epoch) triple, not a cell. `prune_by_eval_loss(candidates, keep)` keeps the `keep` lowest-`eval_loss` checkpoints per cell (default 2, `--epochs-keep -1` disables) purely to bound generation cost, and logs every pruned candidate so the cap is never silent. Surviving candidates are generated on val and scored on the AFSP-matched proxy axis (chrF via `src.eval.quick`, `register_fit` via `src.infer.afsp_sweep._register_fit_fn`), then ranked into `results/peft_sweep_val.json`.

Making epoch a tuned axis required regime changes in `configs/peft_sweep.yaml`: `save_total_limit: null` (keep every epoch checkpoint) and `load_best_model_at_end: false` (do not collapse the run to the `eval_loss`-best epoch). A per-cell `epoch_checkpoints.json` manifest maps epoch → checkpoint path.

`src/peft/verify.py` (new, `manage.py peft_verify`). The analogue of `src/infer/afsp_verify.py`: regenerates the top-N proxy candidates on full val, scores them with COMET and - only when `--judge-config` is supplied - the judge, and applies the identical freeze rule (keep cells within `--comet-adequacy` of the best COMET, pick the highest Φ among them). Writes `results/peft_verify_val.json` with per-segment COMET and the shared source list.

`src/peft/train.py`: LoRA `target_modules` moved from `[q_proj, v_proj]` to `all-linear`, matching the README's stated design and `peft_sweep.yaml`. A `enable_input_require_grads()` call was added on the bf16 path - with LoRA plus gradient checkpointing and a frozen base, the checkpointed inputs carry no `requires_grad`, so backward produced no gradients and training silently did nothing until this was fixed (`1de6059`). Further config-plumbing fixes landed in `d2452be` / `c0efb67`.

New configs: `configs/peft_sweep.yaml` (grid + `register:` block carrying `select_target_sigma: 0.5` and the register direction), `configs/peft_smoke.yaml` (loop validation), `configs/peft_anchor_e3.yaml` (three-epoch anchor cell).

### Why `eval_loss` is not the selector - and what the sweep showed

Completion-only MLE loss falls as the adapter fits target tokens; the thesis objective is register fidelity in generated text. On this grid the two ordered oppositely:

| Candidate | eval_loss | chrF | register_fit (lower = better) |
|---|---|---|---|
| `peft_r16_lr2e-4_e1` (anchor) | 1.5312 (best) | 41.96 | 1.0829 (worst) |
| `peft_r8_lr2e-4_e1` | 1.5336 | 41.40 | 1.0776 |
| `peft_r16_lr1e-4_e1` | 1.5379 | 41.10 | 1.1015 |
| `peft_r32_lr2e-4_e1` | 1.5398 | 42.18 | 1.0736 |
| `peft_r16_lr1e-4_e2` | 1.5553 | 41.66 | 1.0449 |
| `peft_r8_lr2e-4_e2` | 1.5583 | 41.57 | 1.0616 |
| `peft_r16_lr2e-4_e2` | 1.5601 | 41.73 | 1.0386 |
| `peft_r32_lr2e-4_e2` | 1.5733 (worst) | 41.58 | 1.0265 (best) |

Every epoch-2 checkpoint has a worse `eval_loss` and a better `register_fit` than its epoch-1 counterpart, and the selected candidate is the single worst cell by `eval_loss`. Selecting on `eval_loss`, as the placeholder config implied, would have frozen the *least* in-register adapter of the eight. All eight sit inside a 1.1-point chrF band (41.10-42.18), so adequacy does not discriminate between them and the register axis decides.

`src.peft.verify` then confirmed the proxy pick on the reported metrics (`results/peft_verify_val.json`, `proxy_pick_held: true`):

| Candidate | COMET | Φ |
|---|---|---|
| `peft_r32_lr2e-4_e2` (frozen) | 0.6986 | 2.741 |
| `peft_r16_lr2e-4_e2` | 0.7016 | 2.732 |
| `peft_r16_lr1e-4_e2` | 0.6978 | 2.697 |

The three sit within 0.004 COMET - inside the 0.01 adequacy band - so the freeze fell to Φ, which kept the proxy pick.

### Verification

Two Colab smoke runs preceded the real sweep: a first sweep smoke on 2026-07-25 (32-segment cell, retained under `archive/peft-smoke/`) and a multi-epoch smoke on 2026-07-26 (`notebooks/peft_multiepoch_smoke_colab.ipynb`). Per project findings discipline these are loop validation and produce no thesis numbers. The real sweep and verify runs are recorded in `notebooks/peft_full_run_colab.ipynb`.

### Reproduction

```bash
python manage.py peft_sweep  --config configs/peft_sweep.yaml --dry-run     # grid plan, trains nothing
python manage.py peft_sweep  --config configs/peft_sweep.yaml               # train + generate + rank
python manage.py peft_sweep  --config configs/peft_sweep.yaml --score-only  # re-rank, no GPU
python manage.py peft_verify --config configs/peft_sweep.yaml --top 3 \
    --judge-config configs/judge_eval.yaml
```

### Limitations and risks

The grid is small and not a cross-product: four (r, lr) cells with α tied to 2r, so α/r is never varied independently and the (8, 1e-4) and (32, 1e-4) corners are unexplored. Epoch 3 checkpoints were trained and remain on disk (`checkpoint-2037`) but were pruned from selection by the default `--epochs-keep 2`; since `register_fit` improved monotonically from epoch 1 to epoch 2 in every cell, a third epoch might have improved it further and was never generated with. That pruning is an `eval_loss`-driven cost cap on an axis the entry above argues `eval_loss` should not govern - it is logged, but it is the one place `eval_loss` still constrains the outcome.

Each cell was trained once at `seed: 42`; there is no seed replication, so `register_fit` gaps of ~0.01 between adjacent cells are not distinguishable from training noise. The COMET spread across the three verified candidates (0.004) is likewise well inside the paired-bootstrap noise floor that has not yet been computed, so "r = 32 at 2 epochs is the best adapter" is a selection decision, not a demonstrated result.

## 2026-07-24: PEFT LoRA condition implemented, not yet run

*Date corrected on 2026-08-01: this entry was filed under 2026-07-23; the commit it records, `fcb0834` "feat: implement PEFT (LoRA) fine-tuning condition", is dated 2026-07-24, and the entry has been moved to its position in the ordering.*

### Summary

The PEFT condition - the third of the four conditions to be built, leaving only RLSF unimplemented - was implemented. `src/peft/` had held only an empty `__init__.py`; it now contains `src/peft/train.py`, a LoRA supervised fine-tuning entry point wired into `manage.py` as the `peft` command and paired with a new inference condition `peft` in `src/infer/run.py`. This entry records the implementation and its static verification only. No training was run: the frozen thesis base is `Qwen2.5-7B-Instruct`, which does not fit on the 8 GB development GPU (RTX 4060 Laptop) in bf16, so training - like the AFSP sweep - is deferred to a larger-memory environment (Colab) to keep the base unquantized. Nothing here is a thesis finding; the PEFT results row stays empty.

### What changed

`src/peft/train.py` implements LoRA SFT on the locked base. LoRA adapters are applied to the query/value projections (`target_modules: [q_proj, v_proj]`, `src/peft/train.py:142`), base weights frozen, trained with token-level MLE on `data/splits/train.jsonl`. The training prompt is byte-identical to the `peft` inference condition: the locked style instruction as system, the zero-shot translate directive as user (`_ZEROSHOT_USER`, `src/peft/train.py:34`, replicated from `run.build_zeroshot_user` to avoid pulling the commercial-API client imports into training), and the reference translation as the assistant turn. Loss is completion-only - the prompt prefix is tokenized separately and its tokens masked to `-100` (`build_example`, `src/peft/train.py:43`) - so only the target tokens are supervised. Model selection is by `eval_loss` on `val.jsonl` (`load_best_model_at_end`, `metric_for_best_model="eval_loss"`); the best adapter
+ tokenizer and a `train_metrics.json` are saved to `outputs/peft/adapter`
(git-ignored). QLoRA is the memory-pressure fallback (`load_in_4bit` → `BitsAndBytesConfig` + `prepare_model_for_kbit_training`). The saved adapter is also the intended RLSF initialization.

Inference wiring: `LocalChatClient` gained an optional `adapter_path` (`src/infer/local_client.py`); when set, the LoRA adapter is layered onto the frozen base via `PeftModel.from_pretrained` (lazy import). `make_client` passes `generator.adapter_path` through, and `run.py` adds `peft` to `CONDITIONS` - it reuses the zero-shot prompt (no exemplars; the register lives in the weights) and raises if `adapter_path` is unset. Decoding stays locked (greedy, seed 42, max_new_tokens 1024). Config: `configs/peft_qwen.yaml`. Dependency `peft==0.18.0` added to `requirements.txt`.

### Reproduction

python manage.py peft --config configs/peft_qwen.yaml # train (Colab / A100) python manage.py infer --condition peft --config configs/peft_qwen.yaml # generate on val

### Verification (static only - no model load, no network)

`ruff check src`, `ruff format --check src`, `python -m compileall -q src` all pass. The completion-masking arithmetic in `build_example` was checked offline against a stub tokenizer with synthetic data: only the assistant/target tokens are supervised, the prompt is fully masked, and truncation below the prompt length degrades to an all-masked example without crashing.

### Limitations and risks

LoRA rank, learning rate, and epoch/step count in `configs/peft_qwen.yaml` are placeholder defaults (r=16, lr=2e-4, 3 epochs) - they must be dev-tuned on `eval_loss` before the frozen PEFT run, exactly as `k`/`λ` were swept for AFSP. *(The 2026-07-25 entry carried out that tuning and rejected the `eval_loss` criterion proposed here: on this grid `eval_loss` ranks candidates in the opposite order to register fidelity.)* The completion mask relies on the prompt-only chat-template render being a strict token prefix of the full render; this holds for the Qwen2.5 template (clean `<|im_start|>assistant\n` boundary) but should be re-checked if the base or template changes.

## 2026-07-23: AFSP validation sweep, verification, and freeze (k = 8, lambda = 0.75)

*Scope: the sweep, the confirmation, the judge-provider change, and the ladder generation span 2026-07-19 to 2026-07-23. The entry is dated to the last of them (`93906ab`, `7230065`, `4812465`, `9fbc249`).*

### Summary

The AFSP sweep and confirmation described as *not run* in the 2026-07-06 and 2026-07-16 entries were executed on Colab, and the resulting (k, λ) was frozen. The proxy sweep ranked the grid, `afsp_verify` confirmed the top three cells on full val under COMET and the judge, the proxy pick held, and k = 8, λ_style = 0.75 was frozen into `configs/base_qwen.yaml`. All five prompting rungs were then generated on the full val split. The evaluation-time judge was also changed to a different provider during this period (below), which supersedes a decision recorded in the 2026-07-05 entry.

### The sweep

`results/afsp_sweep_val.json` - 19 rows: the k ∈ {4, 8, 16} × λ ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0} grid (18 cells) plus a `afsp_zeroshot` reference row, each on all 1,323 val segments, β fixed at 0.3.

Selection ran the documented rule - keep cells within `adequacy_margin` 1.0 chrF of the grid best (40.44), then take the best `register_fit` - and recommended `afsp_k8_l0.75` (chrF 39.99, `register_fit` 0.9611, `stylo_dist` 0.3709).

Two things in the table are worth recording:

- The zero-shot row has a better `register_fit` (0.9555) than every AFSP cell
  inside the adequacy band, the best of which is the selected `afsp_k8_l0.75`
at 0.9611. It is excluded only by that band (chrF 36.42 vs 40.44). The register objective alone would have selected no exemplars at all; the adequacy band is what makes the selection meaningful, so its width is load-bearing, not cosmetic. *Correction (2026-08-01): the original entry claimed zero-shot beat every AFSP cell in the grid. It does not. `afsp_k4_l1` (0.9217) and `afsp_k4_l0.75` (0.9273) both score better, and both fall outside the band - which strengthens rather than weakens the point, since the two cells with the best register fidelity in the whole grid are also the two the adequacy band has to reject.*
- k and λ trade off against each other. k = 16 dominates on chrF (up to 40.44)
  but has the worst `register_fit` at low λ (1.069 at λ = 0.1); k = 4 has good
`register_fit` (0.922 at λ = 1.0) but the worst chrF (38.81). k = 8 at λ = 0.75 wins by sitting inside the adequacy band with the best fidelity available there.

### The confirmation

`results/afsp_verify_val.json` regenerated the top three proxy cells on full val and scored them with COMET and the judge (`proxy_pick_held: true`):

| Cell | chrF | COMET | Φ | coverage |
|---|---|---|---|---|
| `afsp_k8_l0.75` (frozen) | 39.99 | 0.6853 | 2.797 | 1.0 |
| `afsp_k8_l1` | 39.76 | 0.6824 | 2.763 | 1.0 |
| `afsp_k16_l0.75` | 40.44 | 0.6871 | 2.778 | 1.0 |

All three lie within 0.005 COMET - inside the 0.01 band - so the freeze fell to Φ and kept the proxy pick, even though `afsp_k16_l0.75` is nominally better on both chrF and COMET.

`configs/base_qwen.yaml` was frozen to `retrieval.k: 8`, `afsp.lambda_style: 0.75`, and the five ladder rungs were generated on full val into `outputs/{zeroshot,random_fewshot,knn_fewshot,afsp_margin,afsp_full}_val.jsonl`.

*Correction (2026-08-01): the original entry attributed the freeze to commit `6ae191a`. That commit is dated 2026-07-17, its diff touches `.gitignore` only, and it predates both the band-pass objective and the grid re-cut of 2026-07-18 - so it cannot be the freeze that this sweep produced, despite its message. The commit that actually wrote `retrieval.k: 8` and `afsp.lambda_style: 0.75` into `configs/base_qwen.yaml` is `220bba2` (2026-07-17, message "fix: add new req"), and no later commit touches either line. The frozen (k, λ) therefore entered the config before the sweep this entry reports, and was never rewritten in response to it. The values coincide with the post-band-pass recommendation, but the config is not evidence of that recommendation having been applied; the sweep confirms the setting rather than having produced it. This should be stated that way in the methodology, or the freeze should be re-applied from `results/afsp_verify_val.json` in a commit that says so.*

### Judge provider change (supersedes the 2026-07-05 decision)

`configs/judge_eval.yaml` was changed repeatedly on 2026-07-20 and settled on `provider: anthropic`, `model: claude-haiku-4-5`, `temperature: 0.0`, `thinking: false`, having passed through Gemini Flash variants from the originally documented OpenAI judge. No rationale for the switch is recorded in the repository; only the config history shows it.

This directly contradicts the Stage 4 entry (2026-07-05), which chose an OpenAI judge specifically because `temperature=0` + `seed` gave a reproducible primary, and stated that "the Anthropic cross-family judge is non-deterministic (no temperature control in the client) and should be treated as a confirmation pass, not a reproducible primary." The Anthropic judge is now the primary. Every Φ in `results/judge_val.json` comes from it.

Half of that objection was addressed in the same window, though no entry says so: commit `b0fb732` (2026-07-20) added temperature plumbing to `src/infer/anthropic_client.py`, which now forwards `temperature` to `messages.create` whenever `thinking` is off - the configuration `configs/judge_eval.yaml` uses. So the Stage 4 statement was accurate when written and is now outdated on temperature. It remains accurate on determinism: the Anthropic client exposes no seed, so the primary judge is temperature-0 but not reproducible, and the 2026-07-31 entry measures the resulting test-retest spread at about 0.002 Φ on identical inputs. The methodology must therefore either re-establish a seeded primary or defend a non-reproducible one with repeated-pass variance reported alongside every Φ; leaving the choice undocumented is not an option, since the switch itself has no recorded rationale.

### 4-bit flag (transient, did not affect reported runs)

On 2026-07-18 `load_in_4bit` was flipped to `true` in `base_qwen.yaml` and `afsp_sweep.yaml`, and a `configs/base_qwen_t4_test.yaml` was added, to fit the base on a T4. This would have violated the frozen-unquantized-base contract of the Stage 0 entry. It was reverted the same day (`446eca8`), the T4 config no longer exists in the tree, and every commit from 2026-07-19 onward carries `load_in_4bit: false`. The sweep, verify, and ladder generations all postdate the revert, and the run notebooks contain no 4-bit override, so the reported AFSP outputs are bf16 on the unquantized base.

### Housekeeping

`4812465` archived the pre-band-pass sweep outputs to `archive/pre-band-pass-fix/` (26 stale cells from the old k ∈ {1, 2, 4, 8, 16} grid plus the notebook that produced them) with a README marking them non-canonical. `9fbc249` restored `ruff` lint/format compliance across `src`.

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

The freeze between `afsp_k8_l0.75` and `afsp_k16_l0.75` rests on a Φ gap of 0.019 from a single judge with no seed control, inside a COMET spread of 0.005. That is almost certainly below the noise floor of both metrics; the pick is defensible as a pre-registered rule mechanically applied, but it is not evidence that k = 8 beats k = 16.

β was held at 0.3 throughout and is still unswept, so β × (k, λ) interactions remain unseen. The sweep's own zero-shot row shows the register objective is satisfiable by degenerate means, which is a standing argument for reporting the adequacy band alongside any register claim.

## 2026-07-18: AFSP register objective changed to a band-pass target

### Summary

The register-fit term that drives AFSP's λ rerank was changed from salience (mean absolute z-score, unbounded) to a band-pass objective: the distance to a target point sitting σ standard deviations along a *signed* register direction. The sweep's ranking axis changed with it, from the undirected `stylo_dist` to the new directional `register_fit`. The k × λ grid was simultaneously re-cut. All sweep outputs produced before this change were invalidated and later archived.

### Rationale

The 2026-07-04 Stage 1 entry adopted salience precisely to fix the opposite error - toward-centroid reranking had been selecting the register-*bland* corpus average - and flagged in its own limitations that salience "is unbounded and length-sensitive in isolation", safe only because the candidate pool bounds it. Two problems remain with it as a *ranking* axis:

- Salience is the mean absolute z-score, so it is direction-blind: an exemplar
  that deviates from the centroid the wrong way scores as well as one that deviates
the right way.
- Being unbounded, "more" is always "better", so a λ sweep pushes monotonically
  toward the extreme rather than toward a target register.

Band-pass fixes both: the target is an explicit point, not an extremum, and the direction is signed.

### What changed

`src/eval/stylometrics.py` gained `register_band_distance(feats, centroid, sigma, direction)`. It z-scores the segment's centroid features, forms the target `sign(d) · σ`, and returns the direction-magnitude-weighted standardized distance to it - `sqrt(Σ wᵢ(zᵢ − targetᵢ)² / (Σw / n))` with `w = |d|`, so features that load weakly on the register direction contribute proportionally less. `distance_to_centroid` and `register_salience` are retained: the former undirected for reporting, both still selectable.

`src/retrieval/afsp.py` gained `STYLE_OBJECTIVES = ("bandpass", "proximity", "salience")` with `bandpass` as the default, plus `style_target_sigma`, `style_register_direction`, and `_resolve_direction` (which accepts a dict keyed by centroid feature name - order-independent and validated against the centroid's own feature list, raising on a missing or mis-sized spec). `_register_fit` dispatches on the objective; `proximity` reproduces the pre-Stage 1 behaviour and `salience` the Stage 1 (2026-07-04) behaviour, so all three are still reachable for ablation.

`src/infer/afsp_sweep.py` now computes a per-cell `register_fit` via `_register_fit_fn` and ranks on it instead of `stylo_dist` (`446eca8`): `recommend` takes `min(register_fit)` within the chrF band, ties broken toward higher chrF then smaller k; `stylo_dist` is retained in the table for reporting only. `src/infer/run.py` passes the same three settings through to the retriever.

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

Read out: the register direction is dominated by archaic-marker rate and lexical density, with `root_ttr` loading negatively - on this corpus the target register is *less* lexically varied per unit length, not more. This is why a direction-blind objective mismeasures it: salience rewards deviation in `root_ttr` in either direction, while the corpus says only one direction is register-bearing.

The grid was re-cut in the same window. `97c1744` cut the default k axis to {4, 8, 16}, dropping k = 1 and k = 2 as too few exemplars, and `6169de9` densified λ at the low end by adding 0.1.

*Correction (2026-08-01): the original entry described the grid being replaced as k ∈ {1, 2, 3, 4} × λ ∈ {0, 0.25, 0.5, 0.75, 1.0}. That was the Stage 5 grid of 2026-07-06. The grid actually in force on 2026-07-18 was k ∈ {1, 2, 4, 8, 16} (`PREVIOUS_KS` in `src/infer/afsp_sweep.py`), set by `270b96c` on 2026-07-16 along with the zero-shot anchor row - a change no entry records. So k = 16 was not added here, and the re-cut narrowed an existing five-value k axis to three rather than widening a four-value one. The 26 cells archived by `4812465` (25 grid cells plus the anchor) are from that intermediate grid, as the housekeeping note of the 2026-07-23 entry correctly states. The λ axis also did not reach its final form here: `843fe38` repaired a malformed `DEFAULT_LAMBDAS` tuple later the same day, leaving five values, and λ = 0.5 was restored only on 2026-07-19 by `568ba73`. The 18-cell grid the 2026-07-23 run used dates from that commit, not from this one.*

### Verification

Not separately verified at the time beyond the static checks; the objective's effect is visible in the sweep table recorded in the 2026-07-23 entry, where `register_fit` and the undirected `stylo_dist` rank cells differently. `afsp_k16_l1` has the second-best `stylo_dist` of the 18 cells at 0.3808 but only the tenth-best `register_fit` at 0.9811 - the clearest single case that the two axes disagree. *(Correction, 2026-08-01: the original entry called this the seventh-best `register_fit`; recounted against `results/afsp_sweep_val.json` it is tenth of 18.)*

### Limitations and risks

The provenance of `style_register_direction` is not recorded anywhere in the repository. The four coefficients are hard-coded in `configs/base_qwen.yaml` and duplicated in `configs/peft_sweep.yaml`; no script derives them and no entry explains how they were obtained. They look like a normalized loading vector, and their signs agree with the correlations reported in the 2026-07-04 Stage 1 entry (distance-marker +0.51, distance-lex +0.36), but that is inference, not a record. Since this vector defines what the project *means* by "the target register", it needs a derivation script or a documented source before it appears in the thesis.

*Correction (2026-08-01): the original entry stated that `style_target_sigma` is 1.0 in `base_qwen.yaml` but 0.5 in `configs/peft_sweep.yaml`, and concluded that the AFSP and PEFT arms target register points at different distances. They do not - the two values are different parameters, not two settings of one. `style_target_sigma` (1.0) is the σ the retriever uses when reranking exemplars during generation; `select_target_sigma` (0.5) is the σ the selection metric uses when scoring a finished cell's `register_fit`. `configs/afsp_sweep.yaml` carries both, at 1.0 and 0.5 respectively, and `configs/peft_sweep.yaml` carries only the latter, also at 0.5. The PEFT arm is therefore selected on exactly the AFSP selection axis, σ included, and the 2026-07-25 entry's "matched axis" claim holds.*

The real σ mismatch is inside the AFSP arm: exemplars are reranked toward a target 1.0 σ along the register direction while cells are scored against a target at 0.5 σ, so the objective the retrieval optimizes is not the objective the sweep ranks on. That is defensible - reranking and evaluation need not share a target - but it is undocumented, and the 0.5/1.0 split appears to be inherited rather than chosen.

The direction vector is duplicated verbatim in six configs - `base_qwen.yaml`, `afsp_sweep.yaml`, `qwen_smoke.yaml`, `peft_sweep.yaml`, `peft_anchor_e3.yaml`, and `peft_smoke.yaml` - with no single source. Editing one will silently desynchronize the arms.

## 2026-07-16: AFSP verification hardening and judge-coverage gate

### Summary

Two correctness fixes to `src/infer/afsp_verify.py` before the confirmation is run for real on the A100, both aimed at making the freeze decision trustworthy rather than merely runnable.

First, a judge-gating fix. Supplying `--judge-config` is a statement of intent: *confirm the pick on both axes - adequacy (COMET) and register fidelity (judge Φ).* But `_freeze_pick` decides "do I have judge scores?" from the data, not from that intent: if the judge pass came back empty (every call failed) or too thin to trust (a handful of segments parsed), `have_judge` silently went `False` and the code fell through to freezing on the best-COMET cell alone. That path is indistinguishable in the output from a deliberate COMET-only run, so a broken or under-covered judge pass could quietly freeze a (k, λ) on adequacy while the run looked like a two-axis confirmation. The fix refuses instead of falling back.

Second, persisting per-segment COMET. `comet.score` already returns the segment-level vector, but the verifier kept only the system score, so the paired bootstrap the thesis uses for significance had nothing aligned to resample. The verifier now persists the per-segment COMET for each confirmed cell, plus the shared source list, and guards that the cells are actually segment-aligned before it does.

Still not run here (needs the Qwen base + COMET, ~15 GB); static checks pass and the sweep is grinding on the A100 as this lands, so `afsp_verify` is correct by the time its results are consumed.

### What changed

`src/infer/afsp_verify.py`:

- After the judge pass, when `--judge-config` was supplied, the run now checks every
  top cell has a usable Φ - `mean is not None` and `coverage >= --min-judge-coverage`
(new flag, default 0.9). Any thin cell raises `SystemExit` with the offending tags, a pointer to the resumable per-segment cache under `results/judge_<split>_segments/`, and the explicit alternative (rerun without `--judge-config` to freeze on COMET deliberately). `_freeze_pick` is unchanged; its COMET-only branch is now reached only when no judge was requested, which is the one case it is correct for.
- The COMET loop keeps `res["segments"]` alongside `res["system"]`. Because a paired
  bootstrap requires segment i to mean the same source across cells, the loop asserts
every top cell loads an identical source list (same val rows, same order) and raises `SystemExit` on any mismatch rather than pairing misaligned segments.
- `results/afsp_verify_<split>.json` gains a top-level `sources` (persisted once) and
  a per-cell `comet_segments`, so the bootstrap can re-verify alignment and resample
without re-running COMET.

### Verification

No generation, no COMET load. Static checks pass:

```bash
python -m py_compile src/infer/afsp_verify.py
ruff check src/infer/afsp_verify.py
```

Reasoned through the gate: with `--judge-config` and a full-coverage judge pass the run freezes on Φ within the COMET band as before; with a judge pass that fails or falls below `--min-judge-coverage` it now exits non-zero instead of freezing on COMET; without `--judge-config` the COMET-only freeze is untouched. The alignment guard fires before COMET is scored on the second cell, so a misaligned regeneration is caught early rather than silently paired.

### Limitations and risks

The coverage threshold (0.9) is a judgement call - a legitimately hard cell that the judge refuses on many segments will now block the freeze rather than freeze on a thin mean; that is the intended trade (fail loud), but it means a genuinely low-coverage val split needs `--min-judge-coverage` lowered consciously, not by accident. Persisted `comet_segments` assume the COMET checkpoint is fixed across cells (it is - one model load), so the stored vectors are only comparable within a single verify run, not across runs on different checkpoints. The paired bootstrap that consumes these segments is not yet written.

## 2026-07-16: Full-validation confirmation of AFSP sweep candidates

### Summary

The Stage 5 sweep ranks the 20-cell k × λ grid on free, local proxies - a chrF adequacy band, then stylometric distance to the register centroid - and may be run on a reduced selection subset to keep generation compute affordable. Those proxies are crude next to the COMET and LLM-as-Judge metrics the thesis actually reports, and the selection set may be a sample rather than the full split. Either gap could let the proxy-picked (k, λ) be a false winner. Stage 5b closes both with a first-class command, `src/infer/afsp_verify.py` (`manage.py afsp_verify`): it takes the top-N proxy cells, regenerates only those on the full val split, scores them with COMET (free/local) and - only when a judge config is supplied - the paid judge, and reports whether the proxy pick holds up on the reported metrics or a runner-up overtakes it. The freeze decision is made here, on full val under the reported metrics; the sealed test split is never touched.

This makes the methods claim defensible: *hyperparameters were pre-ranked with inexpensive proxies (optionally on a selection subset), and the top candidates were confirmed on full val under COMET and the judge before freezing; all reported results are on the full val and test splits.* That is stronger than a full-val proxy sweep, which spends the compute on more proxy estimates while still selecting on a metric the thesis does not report.

Like Stage 5, this was not run: the confirmation regenerates cells with the Qwen-7B base (bf16, ~15 GB) and loads COMET, neither of which the 8 GB dev box can hold, so a live run is deferred to a Colab-class GPU. Everything that runs without the base model and COMET - the candidate ranking, the freeze rule, the test-split seal, the CLI/dispatch - was verified offline and the CI-equivalent static checks pass.

### What changed

`src/infer/afsp_sweep.py`: the per-cell generation loop was extracted into `generate_cells(cfg, cells, ...)`, which takes an explicit list of (k, λ) cells; `generate_grid` now delegates to it with the full cross-product, so the sweep's behaviour is byte-for-byte unchanged (same shared index/retriever reuse, same atomic-write + cell-level resume, same `test` seal). `afsp_verify` reuses `generate_cells` to regenerate just the top cells on a different (full) split. A new `ranked_cells(rows, adequacy_margin)` exposes the sweep's own selection ordering - in-adequacy-band cells first, register fidelity (lower `stylo_dist`) within each group, ties toward higher chrF then smaller k - so the verifier pulls exactly the candidates `recommend` would have chosen from.

`src/infer/afsp_verify.py` (new). It loads `results/afsp_sweep_<split>.json` (derived from the sweep config's `eval_file`, or `--sweep-result`), ranks the cells with `ranked_cells` at the sweep's own `adequacy_margin`, takes `--top` (default 3), and regenerates them on `--val-file` (default `data/splits/val.jsonl`, with the same `test`-seal refusal). It COMET-scores each cell with one model load reused across cells, and - only if `--judge-config` is given - runs the resumable judge (per-cell segment cache under `results/judge_<split>_segments/`), so the paid metric is opt-in. The freeze rule (`_freeze_pick`) is the real-metric analogue of the sweep's proxy rule: with judge scores present, keep cells within `--comet-adequacy` (default 0.01) of the best COMET and pick the highest register Φ among them; without the judge, take the best COMET cell. It prints a COMET-ordered table marking both the proxy pick and the freeze pick, states whether the pick held, writes `results/afsp_verify_<split>.json`, and prints the exact `base_qwen.yaml` edits to freeze. `manage.py` registers the `afsp_verify` command.

### Verification

No generation and no COMET load (no Qwen base, no GPU). CI-equivalent static checks pass with `ruff==0.14.2`:

```bash
ruff check src/infer/afsp_sweep.py src/infer/afsp_verify.py manage.py
ruff format --check src/infer/afsp_sweep.py src/infer/afsp_verify.py
python -m compileall -q src/infer/afsp_sweep.py src/infer/afsp_verify.py manage.py
```

Offline behavioural checks on synthetic proxy tables: `ranked_cells` leads with the in-band lowest-`stylo_dist` cell and sinks an out-of-band cell despite its better fidelity; `_freeze_pick` returns the best-COMET cell with no judge, switches to the highest-Φ cell when all candidates sit within the COMET band, and re-excludes a cell that falls outside a tightened band before Φ is consulted. `manage.py afsp_verify --help` dispatches and argparses, the command appears in the dispatcher listing, and pointing `--val-file` at `test.jsonl` raises the seal (`refusing to verify on the sealed test split`) before any model is loaded.

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

Then read `results/afsp_verify_val.json`, freeze the reported `retrieval.k` and `afsp.lambda_style` into `configs/base_qwen.yaml`, and only then run the final `afsp_full` condition on `test.jsonl`.

### Limitations and risks

The confirmation has not been run; there are no COMET/judge numbers and no frozen (k, λ) yet. It confirms only the top-N proxy cells, so a strong cell that the proxy ranked below N is never re-scored - widen `--top` if the proxy table is flat near the top. The COMET-band-then-Φ freeze rule is a modelling choice mirroring the sweep's proxy rule; `--comet-adequacy` defaults to 0.01 (COMET-DA system scores are ~0-1) and should be set against the paired-bootstrap noise floor, not eyeballed. The judge is a paid commercial API and, per the project's findings discipline, any judge number is a sanity signal for selection, not a reported thesis result - the reported judge numbers still come from the frozen conditions on val/test. β stays fixed at 0.3, unchanged from the sweep.

## 2026-07-13: COMET dependency isolated in a separate environment

### Summary

`unbabel-comet==2.2.6` was split out of `requirements.txt` into a new `requirements-comet.txt`, installed in its own virtualenv. The Stage-4 log flagged this as a risk; it is a hard, three-way conflict, so `pip install -r requirements.txt` as written never resolves.

### What changed

- `requirements.txt`: removed the `unbabel-comet==2.2.6` line (replaced with a comment
  pointing to the new file). The generation stack now installs cleanly.
- `requirements-comet.txt` (new): `unbabel-comet==2.2.6` plus the two conflict-critical
  transitive pins from the verified resolution (`transformers==4.57.6`, `numpy==1.26.4`),
with a header documenting the local (CPU torch) and Colab (GPU torch) setup.
- `README.md`: Setup adds the `.venv-comet` environment; the Evaluation section notes
  `manage.py comet` must run from it.

### Rationale

COMET 2.2.6's metadata pins `transformers>=4.17,<5.0`, `numpy>=1.20,<2.0`, and `huggingface-hub<1.0`. The generation stack pins `transformers==5.12.1` and `numpy==2.4.1`. All three constraints conflict, so co-resolution is impossible - this is not a version-bump-away problem. COMET only scores the inference `*.jsonl` files post-hoc (`src/eval/comet.py:19` imports `comet` lazily; `src/eval/_io.py` is stdlib-only), so the eval backbone already decouples and a second venv is the clean fix.

### Reproduction and verification

```bash
python -m venv .venv-comet && source .venv-comet/bin/activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-comet.txt
python manage.py comet --help   # imports comet + torch; exits 0
```

Verified 2026-07-13 on Python 3.11: both `requirements.txt` (minus COMET) and `requirements-comet.txt` resolve independently via `pip install --dry-run`; COMET imports and `manage.py comet --help` runs under the isolated stack.

### Limitations and risks

- The COMET stack pulls its own `transformers`/`numpy`/`torch` (~CPU torch locally, GPU
  torch on Colab); keep the two venvs strictly separate - never `pip install` one file
into the other's environment.
- `transformers==4.57.6` / `numpy==1.26.4` are pinned to the resolution verified today;
  if COMET is ever upgraded, re-verify these against its new metadata.

Correction (2026-08-01): the decoupling this entry relies on no longer holds. Commit `446eca8` (2026-07-18) moved `from comet import download_model, load_from_checkpoint` out of `load_model` and up to module scope, so `src/eval/comet.py` now requires the `comet` package merely to be imported. The consequence reaches beyond that module: `src/infer/afsp_verify.py:13` imports `src.eval.comet` at module scope, so `manage.py afsp_verify` - which also needs the Qwen base to regenerate cells - can no longer be loaded from the generation venv at all, and the two-environment split of this entry cannot satisfy it. `src/peft/verify.py:79` imports COMET inside its function and is unaffected. The confirmation runs of 2026-07-23 and 2026-07-25 postdate the change but did not surface it, because they were executed on Colab, where one environment carries both stacks. Either restore the lazy import or document that `afsp_verify` requires a combined environment.

## 2026-07-06: AFSP k x lambda validation sweep

### Summary

The thesis hyperparameter sweep was implemented as `src/infer/afsp_sweep.py` (`manage.py afsp_sweep`) with its own config `configs/afsp_sweep.yaml`. It runs the AFSP condition over the grid k ∈ {1, 2, 3, 4} × λ_style ∈ {0, 0.25, 0.5, 0.75, 1.0} - 20 cells - on the val split, on the frozen local Qwen base, and recommends a single (k, λ) for a human to freeze into `configs/base_qwen.yaml` before `test.jsonl` is ever touched. This is deliberately separate from `src/infer/sweep.py`, the model × shot-count *smoke* sweep that drives commercial APIs and produces no findings.

The sweep was not run: generating 20 × 1,323 val translations needs a GPU that holds Qwen-7B in bf16 (~15 GB), which the development machine (8 GB) cannot provide - the same constraint recorded for the Stage 0 local generator. Generation is therefore deferred to a Colab-class GPU. Everything that runs without the base model - cell tagging, the free/local scoring pass, the selection rule, the test-split seal, the resumable-generation skip logic, and the CLI/dispatch - was verified offline, and the CI-equivalent static checks pass.

### What changed

`src/infer/afsp_sweep.py` (new). `generate_grid` loads one shared source-side index and a single `AFSPRetriever` (with the register centroid loaded) and reuses them across all cells, mutating only `lambda_style` and the selection `k`, so the candidate-hubness and register caches are computed once. Each cell reuses the Stage 3 prompt path exactly - `_load_configured_glossary`, `order_exemplars`, `build_fewshot_user` - so a swept cell is byte-identical in prompt construction to the corresponding `afsp_margin`/`afsp_full` rung (λ = 0 is margin-only; λ > 0 adds the register rerank). Cells are written to `outputs/sweep/afsp_k{k}_l{λ}_{split}.jsonl` in the standard inference schema (plus `k` and `lambda_style` fields), and existing cells are skipped unless `--overwrite`, making a multi-hour GPU sweep resumable across interrupted sessions.

`generate_grid` refuses to run when the eval file's split stem is `test`, so the sweep cannot accidentally consume the sealed split. Selection uses only free, local signals - chrF/BLEU against the gold targets (via `src.eval.quick`) and the standardized stylometric distance to the target-register centroid (via `src.eval.stylometrics`); the expensive COMET / LLM-as-Judge metrics are reserved for the final frozen conditions, not spent on 20 exploratory cells.

`recommend` encodes the thesis objective - register preservation without sacrificing adequacy - as: keep cells whose chrF is within `--adequacy-margin` (default 1.0) of the grid's best chrF, then pick the smallest register distance among them, breaking ties toward higher chrF and then smaller k (cheaper prompt). It falls back to max chrF when no centroid is available. The recommended (k, λ), the full cell table, and the margin are written to `results/afsp_sweep_<split>.json`, and the exact `base_qwen.yaml` edits to freeze are printed. `--score-only` re-scores and re-selects over already-generated cells with no GPU.

`configs/afsp_sweep.yaml` (new) mirrors `base_qwen.yaml`'s locked decoding and retrieval, pins the val split, and documents that `retrieval.k` / `afsp.lambda_style` are placeholders overridden per cell by the grid. `manage.py` registers the `afsp_sweep` command.

### Verification

No generation (no Qwen base loaded, no GPU). CI-equivalent static checks pass with `ruff==0.14.2`:

```bash
ruff check src
ruff format --check src
python -m compileall -q src
python tests/test_stylometrics.py    # 7/7
```

Offline behavioural checks: `cell_tag` produces collision-free, filesystem-safe tags with `%g`-formatted λ (`afsp_k2_l0.25`, `afsp_k4_l1`); `recommend` picks the lowest register distance strictly within the chrF band, widens its choice as the margin grows, breaks ties toward smaller k, falls back to max chrF without a centroid, and returns `None` on an empty grid; `generate_grid` raises on a `test` eval file (seal intact). The `--score-only` CLI was then run end to end over synthetic `outputs/sweep/*.jsonl` cells: it scored the present cells, skipped the absent grid positions with a note, printed the register-fidelity-ordered table with the recommendation marked, and wrote `results/afsp_sweep_val.json`. `manage.py afsp_sweep --help` dispatches and argparses.

### Reproduction

```bash
# On a Colab-class GPU (Qwen-7B bf16). Full grid over val:
python manage.py afsp_sweep --config configs/afsp_sweep.yaml

# Re-score / re-select over already-generated cells anywhere (no GPU):
python manage.py afsp_sweep --config configs/afsp_sweep.yaml --score-only
```

Then read `results/afsp_sweep_val.json`, freeze the recommended `retrieval.k` and `afsp.lambda_style` into `configs/base_qwen.yaml`, and only then run the final `afsp_full` condition on `test.jsonl`.

### Limitations and risks

The sweep has not been run; there are no val numbers and therefore no frozen (k, λ) yet. The selection rule is a defensible default (register-first within an adequacy band) but is a modelling choice; the chrF adequacy proxy and the stylometric register distance are both crude relative to the COMET/judge metrics reserved for the frozen conditions, so the recommendation should be sanity-checked against a COMET/judge pass on the top few cells before freezing. β is held fixed at 0.3 across the sweep - it is not a swept axis here, so an interaction between β and (k, λ) would go unseen. The full grid is 20 × |val| generations; `--overwrite` is off by default and cells are skipped when present, but the wall-clock and the requirement to keep the base unquantized (no `load_in_4bit`) both point to Colab.

## 2026-07-05: Evaluation backbone: COMET, LLM-as-Judge, and paired bootstrap

*Date corrected on 2026-08-01: this entry was filed under 2026-07-04; the commit it records, `0c1f81c`, is dated 2026-07-05. References to "the 2026-07-04 entry" elsewhere in this log that concern the judge, COMET, or the bootstrap refer to this entry and have been updated to 2026-07-05.*

### Summary

The evaluation backbone needed to report any AFSP result was implemented: learned adequacy (COMET), register-fidelity LLM-as-Judge with a frozen evaluation-time template, paired-bootstrap confidence intervals for pairwise condition comparisons, and a reusable per-segment chrF/BLEU entry point so every metric can be computed over any `<condition>_<split>.jsonl`. A shared loader keeps all scorers aligned segment-for-segment, which the bootstrap requires. No COMET checkpoint was downloaded and no judge API calls were made; the pieces that can be checked without a model or network - bootstrap statistics, per-segment chrF/BLEU, judge parsing/aggregation (via a mock client), lazy imports, and CLI wiring - were verified offline, and the CI-equivalent static checks pass.

### What changed

`src/eval/_io.py` (new) centralizes loading: `load_condition(out_dir, condition, split)` returns aligned `(sources, predictions, references)` from the inference JSONL. `quick.py`'s `score` was refactored onto it, and `quick.py` gained `segment_scores(preds, refs, metric)` - sentence-level chrF (default) or BLEU (`effective_order=True`) - so chrF is now a reusable library entry point rather than living only in the smoke sweep (addresses the Stage-4 chrF item).

`src/eval/comet.py` (new, `manage.py comet`) scores each condition with `Unbabel/wmt22-comet-da` against the gold targets and writes per-segment scores to `results/comet_<split>.json`. The `comet` package and the ~2.3 GB checkpoint are imported/downloaded lazily inside `load_model`, so importing the module needs neither; GPU is auto-detected with CPU fallback. `unbabel-comet==2.2.6` was pinned in `requirements.txt`.

`src/eval/judge.py` (new, `manage.py judge`) + `prompts/judge_eval.txt` (new) implement the evaluation-time register-fidelity judge, deliberately separate from the RLSF training-time reward judge. The frozen template rates a candidate's register against the authorized reference on a 1-5 rubric; the judge model is read from a `judge:` block (`configs/judge_eval.yaml`, new - an OpenAI model at temperature 0 + seed for determinism) and driven through the existing `make_client`. `parse_score` prefers an explicit `Score: N` verdict and falls back to the last lone 1-5 integer, rejecting `12`/`3.5` while allowing a sentence-final `3.`; unparseable/refused segments become `None` and are dropped rather than mis-scored. Per-segment scores, mean, and coverage are written to `results/judge_<split>.json`. Cross-family confirmation is a second run with a different-family judge block (commented in the config).

`src/eval/bootstrap.py` (new, `manage.py bootstrap`) implements segment-level paired resampling (Koehn, 2004): `paired_bootstrap(a, b, n_resamples, alpha, seed)` returns the observed mean difference, the two-sided 95% CI, a bootstrap p-value, and a significance flag, dropping NaN pairs first. The CLI works on any metric - chrF/BLEU computed on the fly, COMET/judge read from their JSON - compares each condition against the ladder floor (`zeroshot`), and with `--adjacent` reports each consecutive-rung difference. `manage.py` registers `comet`, `judge`, and `bootstrap`; the README evaluation section documents the full flow.

### Verification

No COMET download, no judge API calls. CI-equivalent static checks pass with `ruff==0.14.2`:

```bash
ruff check src
ruff format --check src
python -m compileall -q src
python tests/test_stylometrics.py    # 7/7
```

Offline behavioural checks (no model, no network): `paired_bootstrap` gives `diff = 0` with a zero-width CI and non-significance on identical inputs, a positive significant difference with CI excluding 0 on clearly-separated inputs, is reproducible under a fixed seed, and drops NaN pairs (n falls accordingly); `segment_scores` ranks a close match above a distant one for both chrF and BLEU and rejects unknown metrics; `parse_score` handles the `Score: N`, bare-integer, `3.`, `3.5`, `12`, refusal, and empty cases; `build_prompt` fills the frozen template and is brace-safe against stray braces in inserted text; `score_condition`
+ `_aggregate` produce the right scores/mean/coverage against a mock client with a
simulated refusal; the `comet` module imports without `comet` installed and `load_model` raises only when called. The `bootstrap` CLI was then run end to end over synthetic fixtures for both the on-the-fly chrF path and the stored-JSON COMET/judge path, and `manage.py` was confirmed to dispatch and argparse the three new commands.

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

`unbabel-comet==2.2.6` was pinned but not installed or resolved in this environment; COMET depends on `pytorch-lightning` and a `transformers` range that may conflict with the pinned `transformers==5.12.1`, so the first real install should confirm the resolution (and may need a separate environment). COMET was not run, so no adequacy numbers exist for any condition.

The judge is an intentional commercial-API evaluator, not smoke, but it has not been run: no register-fidelity scores exist yet, and the rubric/template will need a small human-agreement check before the numbers are trusted. Judge determinism relies on an OpenAI model honouring `temperature=0` + `seed`; the Anthropic cross-family judge is non-deterministic (no temperature control in the client) and should be treated as a confirmation pass, not a reproducible primary.

The paired bootstrap assumes conditions are aligned segment-for-segment (identical eval order and count); the CLI asserts equal counts but cannot detect a reordering, so all conditions must be generated from the same eval file. The two-sided p-value is the standard twice-the-smaller-tail estimate and is granular at small `n_resamples`; the default 10,000 is adequate for α = 0.05 reporting.

## 2026-07-04: Ablation ladder and glossary decoupling

### Summary

The prompting arm was restructured from three conditions (`reference`, `knn_fewshot`, `afsp`) into a five-rung ablation ladder of discrete conditions, and the register glossary was decoupled from the AFSP arm so it no longer confounds the selection-mechanism comparison. Two changes: (1) `reference` was renamed to `zeroshot`, and `random_fewshot`, `afsp_margin`, and `afsp_full` were added, so each rung adds exactly one component over the one before it; (2) the `[Terms]` glossary line, previously emitted only inside `build_afsp_user`, was moved into a single shared few-shot prompt builder and promoted to a controlled `prompt:` flag applied uniformly to every few-shot rung. No inference was run. The change was verified with the CI-equivalent static checks, the stylometrics test, and an offline exercise of the new selection and prompt-assembly paths.

### 1. The ladder

`src/infer/run.py` now defines `CONDITIONS = (zeroshot, random_fewshot, knn_fewshot, afsp_margin, afsp_full)` and dispatches on it:

| Condition | Exemplars | Selection | β | λ | Isolates |
|---|---|---|---|---|---|
| `zeroshot` | none | - | - | - | instruction only |
| `random_fewshot` | k | random (seeded) | - | - | having examples at all |
| `knn_fewshot` | k | cosine top-k | 0 | 0 | relevance-based retrieval |
| `afsp_margin` | k | margin + hub | β>0 | 0 | AFSP margin/hub penalty |
| `afsp_full` | k | margin + register rerank | β>0 | λ>0 | full AFSP method |

`afsp_margin` and `afsp_full` are both served by the existing `AFSPRetriever` via a `_select_afsp(..., rerank)` helper: `afsp_margin` forces `lambda_style = 0` and passes `centroid=None`, so it exercises margin/hub selection only and does not require the target-register centroid to be built; `afsp_full` uses the configured `beta` and `lambda_style` and loads the centroid. This preserves the Stage-0 ablation property that `β = 0, λ = 0` reduces AFSP to `knn_fewshot`, and adds an intermediate rung isolating the margin from the register rerank. `random_fewshot` samples `k` exemplars per source from the training pool via the shared seeded `random.Random`, so it is reproducible across reruns.

`reference` is renamed `zeroshot` because "reference" was ambiguous with the gold reference target used for scoring; "zeroshot" names precisely what the rung is (instruction only, no in-context examples).

### 2. Glossary confound

`_term_line` was called only inside `build_afsp_user`, so the multi-view word pairs rode along with AFSP alone. A win of `afsp` over `knn_fewshot` could therefore have come from the glossary rather than the adaptive selection - a confound. The two prompt builders (`build_knn_fewshot_user`, `build_afsp_user`) were unified into one `build_fewshot_user(source, exemplars, glossary=None)` shared by every few-shot rung; when `glossary` is empty its output is byte-identical to the old baseline builder. The glossary is now loaded once by `_load_configured_glossary` from a controlled `prompt:` block flag (`word_pairs` + `glossary_file`, with a fallback to the legacy `afsp:` location) and applied uniformly across `random_fewshot`, `knn_fewshot`, `afsp_margin`, and `afsp_full`. It is therefore held constant across the selection ladder - no longer an AFSP-only rider - and can be toggled as its own controlled factor. `configs/base_qwen.yaml` moved `word_pairs`/`glossary_file` from the `afsp:` block to the `prompt:` block accordingly.

`src/infer/sweep.py` (smoke harness) was updated to the new builder/label names, and `src/eval/quick.py`'s default/example conditions were relabelled.

### Verification

No inference was run. CI-equivalent static checks pass with the pinned `ruff==0.14.2`:

```bash
ruff check src
ruff format --check src
python -m compileall -q src
python tests/test_stylometrics.py    # 7/7
```

Two pre-existing one-line-docstring format deviations (in `run.py` from the demonstration-ordering commit and in `stylometrics.py` from the Stage-1 commit), which had left this branch's `ruff format --check` red, were also collapsed so the format gate is green.

Offline (no generator, no embedding model, no API): the shared builder was shown to omit `[Terms]` when the glossary is empty and to inject it when non-empty; `_select_random` was confirmed seeded, per-source, and of length `k`; `_load_configured_glossary` was confirmed to read the `prompt:` block, fall back to the legacy `afsp:` location, and default to disabled; and `run("reference", …)` was confirmed to raise, listing the five valid rungs.

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

Holding the glossary constant across the ladder resolves the confound but does not by itself measure the glossary's effect; that requires an explicit contrast (the same condition with `word_pairs` on vs. off), which the flag now makes possible but which has not been run. `random_fewshot` draws from the whole training pool without a topical floor, so at very small `k` its exemplars may be off-topic - that is the intended contrast against `knn_fewshot`, but its variance across seeds should be reported. The ladder has been exercised only offline on the selection and prompt-assembly paths; no translations were generated, so no register results exist for any rung.

## 2026-07-04: AFSP retrieval, margin, and register-fit fixes

### Summary

The AFSP mechanisms implemented on 2026-07-03 were built but diverged from the intended design in ways that would have silently degraded the method. This entry records four corrective changes: (1) the retrieval index is committed to the source side so the margin runs in one comparable space; (2) the margin terms are confirmed consistent with that space; (3) the two heavy-tailed sentence-length features are dropped from the register-fit centroid where they were dead weight; and (4) the register rerank is changed from *toward-centroid* (which surfaced the register-bland corpus average) to a directed register-salience score that prefers register-bearing exemplars. No thesis inference was run. The changes were verified with static checks, the stylometrics test, an offline corpus-geometry analysis, and a handful of real validation queries through `AFSPRetriever.select`.

### 1. Source-side index (the space the margin lives in)

The margin in `src/retrieval/afsp.py` scores a candidate as `cos(x, y) − β·½(hub(x) + hub(y))`, with all three terms read from `self.index.embeddings`. That is only coherent if the index is a single embedding space. The committed `build_index.py` embedded the English target side, which made `cos(x, y)` and the query hubness `hub(x)` cross-lingual (source→target) while the candidate hubness `hub(y)` was monolingual English - three quantities in two different spaces, added together. This is the margin of Artetxe and Schwenk (2019) applied incoherently.

A verification pass showed the on-disk `data/knn_index/embeddings.npy` was in fact already source-side (cosine 1.00 to the source of pair 0, 0.86 to its target), and its `meta.json` carried an `indexed_side: "source"` key that no committed code ever wrote - the artifact had been rebuilt source-side out of band while the code and docs still described a target-side, cross-lingual index.

Decision: commit to the source-side index. Retrieval is now monolingual on the source side - a Persian/Arabic query is matched against the embedded training sources, and each match maps back to its aligned English target. This makes the margin's similarity, query hubness, and candidate hubness one comparable source-side space, and it preserves the ablation property that `β = 0, λ = 0` reduces AFSP exactly to the `knn_fewshot` baseline (both become source-side top-k). Target-side register is scored separately by the style rerank from the exemplar's English *text*, which needs no embedding, so nothing register-related is lost by not embedding the targets. The alternative - keeping a cross-lingual variant - was rejected: it would require a second (source) embedding index anyway to compute a coherent candidate hubness, and it conflates semantic similarity with translation-pair alignment quality.

`src/retrieval/build_index.py` now embeds `pairs[i]["input"]` and records `indexed_side: "source"`. `src/retrieval/embed.py`'s query instruction was rewritten from "Retrieve English translations …" (cross-lingual) to a source-to-source instruction; on pair 0 this raised the self-similarity of the query embedding against its own stored row from 0.929 to 0.967 and produced a cleaner top-3 neighbourhood, confirming the old instruction was mismatched to the source index. `retrieve.py`, `afsp.py`, the README, and `configs/*.yaml` retrieval comments were updated to describe source-side retrieval.

### 2. Margin formula vs. the committed space

With the index committed to the source side, the existing margin is correct as written: `sims`, `qhub` (`_topk_mean(sims)`), and `chub` (`_candidate_hubness`) are all computed from the source embeddings, so `qhub` and `chub` are the two halves of one symmetric same-space margin. No formula change was needed; the fix was to guarantee the space, not to rewrite the algebra. The cached `hubness_top{m}.npy` is source-space (built from the source embeddings) and stays valid; per the cache-key caveat, it must be cleared only if the index is rebuilt.

### 3. Inert sentence-length features in the register centroid

In the target-register centroid the length features had centroid std ≈ 16.6 (`sent_len_mean`) and ≈ 58.4 (`sent_len_var`) against means of 24.7 and 6.7. Under the z-scoring in `distance_to_centroid` such a large denominator makes their standardized contribution vanish, so they were dead weight in any `lambda_style` sweep, and they measure segment structure rather than register.

`stylometrics.py` now defines `CENTROID_FEATURES = [lex_density, ttr, root_ttr, marker_rate]` and builds the centroid and the register-fit distance over that subset. The two length features remain in `FEATURE_NAMES`, so the reporting table and the H2 across-segment variance analysis still see them; only the register-fit distance drops them. `distance_to_centroid` already iterated `centroid["features"]` and needed no change; the `--build-centroid` printout was fixed to iterate the centroid's own feature list rather than the full `FEATURE_NAMES`. The centroid was rebuilt (n = 10,860, four features).

### 4. Register rerank: salience, not toward-centroid

`_style_fit` previously returned `−distance_to_centroid`, i.e. it preferred exemplars closest to the centroid. Because the centroid is the corpus *mean*, this selects the register-bland average and demotes strongly-styled exemplars. An offline analysis over the 10,860 training targets confirmed it empirically:

- closest-5%-to-centroid mean marker rate 0.024 and closest-25% 0.019, both
  below the corpus mean of 0.033;
- the farthest-5% mean marker rate 0.136, and the top-5% highest-marker
  (most register-bearing) exemplars sit at the 93rd distance percentile;
- `corr(distance, marker_rate) = +0.51`, `corr(distance, lex_density) = +0.36`.

So minimizing distance to the centroid actively pushes the most register-bearing exemplars to the bottom - the opposite of the "target-distribution-priority" and anti-neutral-register intent in `docs/afsp_strategies.md`.

The rerank now uses `register_salience`: the mean z-score of an exemplar's register features relative to the centroid, in the register-positive direction (every `CENTROID_FEATURES` entry - marker rate, lexical density, ttr, root_ttr - marks stronger register when higher). `_style_fit` returns this salience directly (no negation); larger = more strongly in-register. `distance_to_centroid` is retained, undirected, for reporting how far a system's output distribution sits from the target centre.

On a handful of validation queries through `AFSPRetriever.select`, raising `lambda_style` from 0.0 to 0.5 shifted the mean marker rate of the selected exemplars from 0.0197 to 0.0270 - toward stronger register, as intended. Unbounded salience over the *whole* corpus tops out on degenerate two-word fragments ("O Lord.", marker rate 0.5), but these never enter the rerank: it operates only within the `pool_mult·k` source-similarity pool, whose members are topically relevant and of realistic length, so the pool constraint bounds the effect.

### Verification

No inference was run. CI-equivalent static checks pass on the touched sources:

```bash
ruff check src
ruff format --check src
python -m compileall -q src
python tests/test_stylometrics.py    # 7/7 (features() and FEATURE_NAMES unchanged)
```

Offline (no generator, no API): the source/target identity of `embeddings.npy` was confirmed against the embedding model; the source-to-source query instruction was shown to improve self-similarity and neighbourhood quality; the toward-centroid bias was quantified over all training targets; and `AFSPRetriever.select` was run on validation queries with `lambda_style ∈ {0, 0.5}` to confirm the salience rerank shifts selection toward register-bearing exemplars.

### Reproduction

```bash
python manage.py build_index --config configs/base_qwen.yaml   # now source-side
python manage.py stylometrics --build-centroid                 # four register features
python manage.py infer --condition afsp --config configs/base_qwen.yaml
```

If `data/knn_index/` already holds a target-side `embeddings.npy` from before this change, delete the directory (including `hubness_top*.npy`) and rebuild so the index and its hubness cache are source-side.

### Limitations and risks

The register-salience direction assumes each `CENTROID_FEATURES` entry is monotonically register-positive on this corpus, which held empirically here (distance-marker and distance-lex correlations are positive) but is a corpus-specific assumption to revisit if the feature set changes. Salience is unbounded and length-sensitive in isolation; it is only safe because it reranks within the source-similarity pool - a much larger `pool_mult` would begin to admit short high-marker fragments and should be swept with that in mind. `lambda_style` and `beta` still need a dev-split sweep; these fixes make that sweep meaningful (non-inert features, coherent margin, correctly-signed register term) but do not themselves select values.

## 2026-07-03: Local Qwen2.5-7B generator

### Summary

The local generator client was implemented, completing the last model-agnostic gap in the inference pipeline. Until now the canonical base configuration (`configs/base_qwen.yaml`) declared `provider: local` as a placeholder, and no open-source generator existed: the pipeline could be exercised only through the commercial-API smoke clients, which produce no thesis findings. `LocalChatClient` now runs `Qwen/Qwen2.5-7B-Instruct` through HuggingFace `transformers`, exposing the same `complete(system, user)` interface as the OpenAI and Anthropic clients, so the four thesis conditions (reference, PEFT, knn_fewshot/AFSP, RLSF) can be generated on the locked open-source base rather than a commercial API.

The decoding contract for every thesis run was fixed and documented as part of this stage. The client and its wiring were verified end to end on the `Qwen2.5-0.5B-Instruct` sibling, which shares the Qwen2.5 chat template and tokenizer family. The full 5-segment reference smoke on the 7B base was not run locally: the development GPU (RTX 4060 Laptop, 8 GB VRAM) cannot hold the 7B base in `bfloat16` (approximately 15 GB), so that run is deferred to a larger-memory environment (Colab) in order to keep the frozen base unquantized. This entry records the implementation and the decoding decision; it reports no translation quality results, and a 5-segment slice would in any case be loop-validation, not a finding.

### What changed

`src/infer/local_client.py` was added, providing `LocalChatClient` (`src/infer/local_client.py:26`). It is a dataclass mirroring the existing clients: a `complete(system, user) -> str` method (`src/infer/local_client.py:77`) and a shared `Usage` accumulator. `torch` and `transformers` are imported lazily inside `__post_init__` (`src/infer/local_client.py:43`) and `complete`, so importing the module - for example from `make_client` - does not require the heavy dependencies or a GPU.

Generation applies the model's chat template to a system/user message pair (`src/infer/local_client.py:84`) with `add_generation_prompt=True`, generates with `torch.no_grad`, and decodes only the newly generated tokens by slicing off the prompt prefix. Prompt and completion token counts are recorded through `Usage`; because local weights carry no per-token price, the pricing table is empty and `cost_usd` remains zero while token counts still accumulate for throughput and reproducibility bookkeeping.

In `src/infer/run.py`, a `local` branch was added to `make_client` (`src/infer/run.py:41`) that constructs `LocalChatClient` from the `generator` block, reading `temperature`, `top_p`, `seed`, `dtype`, `device_map`, and `load_in_4bit` with defaults. The error message for an unrecognized provider now lists `local` (`src/infer/run.py:54`).

`configs/base_qwen.yaml` was updated: the `provider: local` placeholder comment and the "client not implemented" note were removed, and the locked decoding settings were made explicit and documented in the `generator` block.

`configs/qwen_smoke.yaml` was added as a loop-validation configuration. It mirrors the base model and locked decoding but sets `data.limit: 5` and runs the `reference` condition; it is labelled, like the commercial-API smoke configs, as producing no thesis findings, the distinction being that it exercises the real thesis base rather than a commercial API.

`requirements.txt` gained `transformers==5.12.1`, `accelerate==1.14.0` (required for `device_map: auto`), and `bitsandbytes==0.49.2` (required only for the 4-bit fallback).

### Decoding decision

Decoding is fixed identically across all four conditions so that cross-condition differences are attributable to adaptation rather than to sampling. The locked settings, recorded in `configs/base_qwen.yaml`, are:

```text
temperature   = 0.0     greedy decoding, fully deterministic (do_sample=False)
top_p         = 1.0     inert under greedy; used only if temperature is raised
seed          = 42      fixes any residual kernel nondeterminism / sampling RNG
max_new_tokens = 1024   covers a paragraph-length translation
```

Greedy decoding was chosen as the reproducible default: it removes sampling variance entirely, so a run can be reproduced byte-for-byte. `temperature` and `top_p` remain configurable for later sampling experiments, in which case `seed` governs the RNG through `transformers.set_seed`.

`dtype`, `device_map`, and `load_in_4bit` are treated as hardware-placement settings, not part of the decoding contract. The frozen thesis base is full-precision `bfloat16`; `load_in_4bit` (NF4, approximately 5 GB) exists only as a fallback for GPUs with 8 GB of VRAM or less and is off by default, because enabling it would quantize the base model and therefore change the frozen-base definition that underpins cross-condition attribution.

### Verification

The touched sources pass the CI-equivalent static checks:

```bash
ruff check src/infer configs
python -m py_compile src/infer/local_client.py src/infer/run.py
```

Wiring was validated without loading weights by parsing `configs/base_qwen.yaml` and `configs/qwen_smoke.yaml` through `make_client`, confirming `provider: local` dispatch, the propagated decoding settings, and the `complete(self, system, user)` signature.

The generation path was then exercised end to end on `Qwen/Qwen2.5-0.5B-Instruct`, the smallest member of the same model family and therefore sharing the Qwen2.5 chat template and tokenizer. Using the project style instruction and `build_reference_user`, the client produced a correct English translation of an Arabic test source, recorded non-zero prompt and completion token counts through `Usage`, and - under the locked greedy configuration - produced byte-for-byte identical output across two calls, confirming determinism. The 0.5B run validates the client code path (chat-template application, generation, prompt-prefix slicing, token accounting, and decoding); it does not characterize 7B behaviour.

The 7B smoke on the locked base was not run. `Qwen/Qwen2.5-7B-Instruct` in `bfloat16` requires roughly 15 GB of weights and does not fit the development GPU's 8 GB of VRAM.

### Reproduction

The 5-segment loop-validation smoke, intended for a GPU with at least 16 GB of VRAM (for example a Colab T4/L4/A100) so the base loads in full `bfloat16`:

```bash
pip install -r requirements.txt
python -m src.infer.run --condition reference --config configs/qwen_smoke.yaml
```

The full development run uses the canonical configuration, which reads the entire validation split (`data.limit: null`):

```bash
python -m src.infer.run --condition reference --config configs/base_qwen.yaml
```

`configs/base_qwen.yaml` sets `device_map: auto`, which requires `accelerate`. On a GPU with 8 GB of VRAM or less, set `load_in_4bit: true` (requiring `bitsandbytes`); this fits the model at the cost of quantizing the base, which must then be documented as the base-model definition. The Qwen weights are fetched from the HuggingFace Hub on first run; no `HF_TOKEN` is required for these public weights.

### Limitations and risks

The end-to-end verification used the 0.5B sibling, not the 7B thesis base. It confirms the client is correct as code but not that the 7B base loads and generates within a given memory budget; that is confirmed only when the smoke runs on adequate hardware.

The development GPU cannot run the locked base in full precision. Until the smoke is run on a larger-memory environment, `configs/base_qwen.yaml` is implemented but not yet exercised on its own base model. Enabling the `load_in_4bit` fallback to run locally would change the frozen base to a quantized model and must not be treated as equivalent to the `bfloat16` base.

A 5-segment run validates the loop only and is not an evaluation; no translation quality numbers from `configs/qwen_smoke.yaml` may be reported as thesis findings.

The `accelerate` and `bitsandbytes` pins were selected as current releases compatible with `transformers==5.12.1` but have not been installed and exercised together in this environment; the first full `pip install -r requirements.txt` on the target machine should confirm the resolution.

## 2026-07-03: AFSP implementation

### Summary

The AFSP method was implemented as an adaptive selection layer over the existing `knn_fewshot` retrieval index, and an `afsp` condition was added to the provider-agnostic inference pipeline. AFSP had until now been specified only in `docs/afsp_strategies.md`, with an empty results row; the corresponding code now exists.

This entry records the implementation only. The AFSP condition was not run: no inference was performed, no API calls were made, and no results were produced. The AFSP row in the README remains empty until the method is run on the development split and, subsequently, the sealed test split.

### What changed

`src/retrieval/afsp.py` was added, providing `AFSPRetriever` and `load_centroid`. In `src/infer/run.py`, the `afsp` condition was added together with its prompt builder `build_afsp_user`, the glossary loader `load_glossary`, and the helper `_term_line`; `afsp` was registered as a choice in the `run.py` argument parser. `eval/quick.py` and `eval/stylometrics.py` accept conditions as arbitrary strings and therefore require no change.

The implementation follows the four mechanisms of `docs/afsp_strategies.md`, operating over the shared index of English training targets.

1. Margin-based scoring with hub penalization. The candidate score is
`cos(x, y) − β · ½(hub(x) + hub(y))`, where `hub` is the mean cosine similarity of a row to its `knn_hubness` nearest neighbours, self-similarity excluded. This is the distance-form margin of Artetxe and Schwenk (2019) with a tunable `β`. Candidate hubness is precomputed in row batches and cached to `data/knn_index/hubness_top{m}.npy`.

2. Target-distribution-priority selection. A candidate pool of size `pool_mult·k`
is drawn by margin score, and each candidate's English target is then scored by its standardized distance to the target-register centroid (`results/stylometrics_centroid.json`), reusing `stylometrics.features` and `distance_to_centroid`. The final ranking combines margin similarity and register fit, weighted by `lambda_style`.

3. Demonstration ordering. `select` returns the exemplars with the
highest-scoring one last, adjacent to the test source. `build_afsp_user` preserves this order and does not reverse it, in contrast to the baseline builder.

4. Multi-view word-level pairs. A curated seed glossary
`prompts/register_glossary.tsv` (`source_term<TAB>target_term`) is substring-matched against each exemplar and the query source, and the matches are inserted as a `[Terms]` line to emphasize register-bearing terminology.

A new `afsp:` block in `configs/base_qwen.yaml` exposes `beta`, `knn_hubness`, `pool_mult`, `lambda_style`, `centroid_file`, `word_pairs`, and `glossary_file`.

By construction, `beta = 0` together with `lambda_style = 0` reduces AFSP to the `knn_fewshot` baseline. The two parameters therefore isolate the contribution of the adaptive components from that of naive retrieval, and support a direct ablation.

### Verification

No inference was run. The touched sources pass the CI-equivalent static checks:

```bash
ruff check src
ruff format --check src
python -m compileall -q src
```

A separate offline check, loading no embedding model and making no API calls, exercised the numeric helpers and prompt assembly against synthetic arrays and a small in-memory index. It confirmed min-max normalization and top-k mean, the batched candidate-hubness computation (which ranked a near-duplicate cluster as hubs), the target-register fit (which ranked an archaic exemplar above a plain one using the real stylometric centroid), glossary parsing, and the ordering and `[Terms]` injection of the assembled prompt.

### Reproduction

Not yet run. AFSP requires the retrieval index and the target-register centroid to be built beforehand:

```bash
python manage.py build_index --config configs/base_qwen.yaml
python manage.py stylometrics --build-centroid
python manage.py infer --condition afsp --config configs/base_qwen.yaml
python manage.py eval --conditions knn_fewshot afsp --split val
python manage.py stylometrics --conditions knn_fewshot afsp --split val
```

`configs/base_qwen.yaml` specifies `provider: local`, whose client is not yet implemented, so an end-to-end run requires either that client or a commercial-API generator block. Development tuning should sweep `beta`, `lambda_style`, and `k` before any test-set run.

### Limitations and risks

The register glossary is a hand-curated seed list rather than a statistical word alignment. Matching is plain substring matching over undiacritized forms and may miss diacritized forms or over-match short terms; a subsequent revision should align and extend the list against the corpus. The mechanism is inactive when the file is absent or `word_pairs` is false.

Candidate hubness is an `O(N²)` precomputation over the index. It is batched and cached, and is inexpensive at roughly 10,860 rows, but would need revisiting at a larger scale. The cache key encodes only `knn_hubness`, so an index rebuilt at the same path requires its `hubness_top*.npy` files to be cleared.

The target-register centroid must exist before the AFSP condition runs; `load_centroid` raises an explicit error when it is missing rather than degrading silently.

AFSP has not been evaluated against reference translations. Its effect on register and adequacy relative to the `knn_fewshot` baseline is unknown until the condition is run.

## 2026-07-03: Model x shot-count smoke sweep

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

- prompt assembly via `build_reference_user` and `build_knn_fewshot_user` from `src/infer/run.py`;
- provider dispatch via `make_client`, so OpenAI and Anthropic models are driven through one interface;
- scoring via `sacrebleu` BLEU/chrF and the shared `_MARKERS` regex from `src/eval/stylometrics.py`, keeping metric definitions identical to the quick scorer.

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

The leaderboard sorts by chrF purely as a display convenience. It is not a principled "best model" criterion - register fidelity in particular is better judged from the human-eval artifacts than from chrF or the crude marker-rate proxy.

The retrieval index is reloaded once per shot count when prompts are built, a minor inefficiency that is negligible at this corpus size but would matter if the sweep grew large.

Shot counts are bounded by the base model context window; the registry and defaults assume `n ≤ 8`.

The human-eval TSV score columns are intentionally blank and must be filled by a human; no automatic judge is applied at this stage.

## 2026-06-12: Stylometrics module

### Summary

A new stylometric evaluation module was added to measure whether model outputs match the target scriptural register. This module provides per-segment style features, builds a target-register centroid from the training targets, and reports the distance of system outputs from that centroid.

### What changed

Added `src/eval/stylometrics.py`, which implements the objective style-measurement layer promised in the README. This module also supports two later stages of the project: style-based reranking and the H2 variance hypothesis.

The module computes the following per-segment features:

- `lex_density`
- `ttr`
- `root_ttr`
- `sent_len_mean`
- `sent_len_var`
- `marker_rate`

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

## 2026-06-12: Repository hygiene, test sealing, and baseline relabeling

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

## 2026-06-12: Qwen2.5-7B tokenizer fertility check

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

- corpus-level fertility
- mean per-sentence fertility
- median fertility
- p90 fertility
- maximum fertility

The script also prints a threshold-based verdict:

```text
< 2.5      lock
2.5-4.0    borderline
4.0+       catastrophic
```

### Rationale

Before committing to an open-source 7-8B local model, the project needed to verify that the model’s tokenizer handles Arabic/Persian-script source text efficiently.

Fertility is used as a proxy for tokenizer efficiency:

```text
fertility = subword tokens / whitespace words
```

Very high fertility indicates byte-fallback behaviour, which inflates sequence length and can degrade model quality. The decision rule was set in advance: a value near or below 2.5 would be acceptable, while a value around 4-5 or higher would be unacceptable.

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

This is slightly above the ideal 2.5 target but far below the catastrophic 4-5+ range. The distribution is also tight, with the median close to the corpus mean and p90 at 3.25.

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

## 2026-06-11: API-backed smoke pipeline

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

- BLEU
- chrF
- archaic marker rate
- reference marker rate

### Rationale

The supervisor requested validating the approach against commercial APIs before building the local 7-8B pipeline.

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

The smoke run used `n = 25` and `k = 4`, on the first 25 segments of `data/splits/test.jsonl` - at this date the smoke configs still pointed at the test split, which is the defect the 2026-06-12 Phase 1 entry corrected by sealing it. The outputs were written as `*_test.jsonl` and are now quarantined under `outputs/smoke_api_quarantine/` with `.smoke.` in their names.

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

The commands below are written against the post-sealing layout of the 2026-06-12 Phase 1 entry, in which the smoke configs read `data/splits/val.jsonl` and the inference script derives the split tag from the eval file. Re-running them therefore reproduces the pipeline but not the table above, which was produced on the test split before sealing and must not be regenerated there.

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

## 2026-06-08: `manage.py` command dispatcher

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

## 2026-06-08: Arabic diacritics and hamza-bearing characters preserved

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

## 2026-06-08: Work-level split with cross-boundary deduplication

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

## References

Works cited by name in the entries above, together with the models and metric implementations the pipeline depends on.

- Artetxe, M., and Schwenk, H. (2019). Margin-based parallel corpus mining with
  multilingual sentence embeddings. *Proceedings of ACL 2019*. - the margin score with
hub penalization used by `src/retrieval/afsp.py`.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and
  Chen, W. (2021). LoRA: Low-rank adaptation of large language models. - the PEFT arm,
via the `peft` library.
- Koehn, P. (2004). Statistical significance tests for machine translation evaluation.
  *Proceedings of EMNLP 2004*. - the paired bootstrap of `src/eval/bootstrap.py`.
- Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. (2002). BLEU: a method for
  automatic evaluation of machine translation. *Proceedings of ACL 2002*.
- Popović, M. (2015). chrF: character n-gram F-score for automatic MT evaluation.
  *Proceedings of WMT 2015*.
- Post, M. (2018). A call for clarity in reporting BLEU scores. *Proceedings of
  WMT 2018*. - BLEU and chrF are computed with `sacrebleu`.
- Qwen Team (2024). Qwen2.5 technical report. - `Qwen/Qwen2.5-7B-Instruct`, the frozen
  thesis base.
- Rei, R., et al. (2022). COMET-22: Unbabel-IST 2022 submission for the metrics shared
  task. *Proceedings of WMT 2022*. - the `Unbabel/wmt22-comet-da` checkpoint.
- Wang, L., et al. (2024). Multilingual E5 text embeddings: a technical report. - the
  `intfloat/multilingual-e5-large-instruct` retrieval encoder.

The AFSP method statement and its precedents are cited in `README.md` and `docs/afsp_strategies.md`; this log records implementation decisions only.
