# Engineering and Decision Log

This document records the main pipeline stages, architectural decisions, implementation changes, and rationale for the Style-Aware Machine Translation project. Entries are ordered in reverse chronological order, with the most recent decision first.

Each entry documents four points: what changed, why the change was made, how the result can be reproduced, and what limitations or risks should be monitored. This log serves as the authoritative record for explaining why the dataset, pipeline, and generated artifacts have their current form.

## Conventions

* Dates are written in absolute form: `YYYY-MM-DD`.
* Code references should use the format `path:line`.
* Commands should be recorded as exact command-line invocations.
* When a pipeline stage writes data, the entry should identify the artifact paths and describe how integrity was verified, for example through hashes or manifests.

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

# Add the paid judge pass (top cells only) to decide within the COMET band:
python manage.py afsp_verify --config configs/afsp_sweep.yaml --top 3 \
    --judge-config configs/afsp_sweep.yaml   # any YAML carrying a `judge:` block
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

### Why

COMET 2.2.6's metadata pins `transformers>=4.17,<5.0`, `numpy>=1.20,<2.0`, and
`huggingface-hub<1.0`. The generation stack pins `transformers==5.12.1` and
`numpy==2.4.1`. All three constraints conflict, so co-resolution is impossible — this
is not a version-bump-away problem. COMET only scores the inference `*.jsonl` files
post-hoc (`src/eval/comet.py:19` imports `comet` lazily; `src/eval/_io.py` is
stdlib-only), so the eval backbone already decouples and a second venv is the clean fix.

### How to reproduce / verify

```bash
python -m venv .venv-comet && source .venv-comet/bin/activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-comet.txt
python manage.py comet --help   # imports comet + torch; exits 0
```

Verified 2026-07-13 on Python 3.11: both `requirements.txt` (minus COMET) and
`requirements-comet.txt` resolve independently via `pip install --dry-run`; COMET
imports and `manage.py comet --help` runs under the isolated stack.

### Risks / to monitor

- The COMET stack pulls its own `transformers`/`numpy`/`torch` (~CPU torch locally, GPU
  torch on Colab); keep the two venvs strictly separate — never `pip install` one file
  into the other's environment.
- `transformers==4.57.6` / `numpy==1.26.4` are pinned to the resolution verified today;
  if COMET is ever upgraded, re-verify these against its new metadata.

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

## 2026-07-04 — Stage 4: Evaluation Backbone (COMET, LLM-as-Judge, paired bootstrap)

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

`src/eval/_io.py` (new) centralises loading: `load_condition(out_dir, condition,
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
paired resampling (Koehn 2004): `paired_bootstrap(a, b, n_resamples, alpha, seed)`
returns the observed mean difference, the two-sided 95 % CI, a bootstrap p-value,
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
three quantities in two different spaces, added together. This is the
Artetxe–Schwenk (2019) margin applied incoherently.

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
rebuilt (`n = 10860`, four features).

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
`load_in_4bit` with defaults. The error message for an unrecognised provider now
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
slicing, token accounting, and decoding); it does not characterise 7B behaviour.

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

1. Margin-based scoring with hub penalisation. The candidate score is
   `cos(x, y) − β · ½(hub(x) + hub(y))`, where `hub` is the mean cosine
   similarity of a row to its `knn_hubness` nearest neighbours, self-similarity
   excluded. This is the distance-form margin of Artetxe and Schwenk (2019) with
   a tunable `β`. Candidate hubness is precomputed in row batches and cached to
   `data/knn_index/hubness_top{m}.npy`.

2. Target-distribution-priority selection. A candidate pool of size `pool_mult·k`
   is drawn by margin score, and each candidate's English target is then scored
   by its standardised distance to the target-register centroid
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
   are inserted as a `[Terms]` line to emphasise register-bearing terminology.

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
small in-memory index. It confirmed min-max normalisation and top-k mean, the
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
alignment. Matching is plain substring matching over undiacritised forms and may
miss diacritised forms or over-match short terms; a subsequent revision should
align and extend the list against the corpus. The mechanism is inactive when the
file is absent or `word_pairs` is false.

Candidate hubness is an `O(N²)` precomputation over the index. It is batched and
cached, and is inexpensive at roughly 10.8k rows, but would need revisiting at a
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

The `stylometrics` command was registered in `manage.py`, and `tests/test_stylometrics.py` was added as a pytest-free assertion script. The tests include a hand-labeled lexical-density check.

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

This included a hand-labeled excerpt from training target #0, where the expected values were:

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

The `_MARKERS` regex is case-sensitive. As inherited from the quick scorer, sentence-initial `Thou` is not matched, while capital `O` is matched only as the vocative form. This was left unchanged to preserve existing marker-rate behavior.

Sentence splitting uses a simple regex. Single-sentence segments receive `sent_len_var = 0`.

---

## 2026-06-12 — Phase 1: Repository Hygiene, Test Sealing, and Baseline Relabeling

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

The local generator client is not yet implemented, so `provider: local` is currently a placeholder for the upcoming local-inference phase. The two commercial-API configs were clearly relabeled as smoke-only configs and should not be treated as thesis conditions or findings.

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

This prevents validation or development runs from being mislabeled as test results. The real `test.jsonl` is reserved for the final evaluation run only.

Commercial-API smoke outputs were quarantined under:

```bash
outputs/smoke_api_quarantine/
```

They were renamed to reflect their actual status. For example:

```bash
afsp_test.jsonl → knn_fewshot_test.smoke.jsonl
```

A warning banner was also added to the earlier development-log entry explaining that the BLEU/chrF numbers are smoke checks only, not thesis findings.

The implemented retrieval method was relabeled from `afsp` to `knn_fewshot`, because it is a plain top-k cosine nearest-neighbor baseline rather than the full adaptive AFSP method.

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

Relabeling the current retrieval implementation as `knn_fewshot` keeps the experimental comparison honest. The future AFSP contribution should be evaluated against this baseline, not confused with it.

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

Very high fertility indicates byte-fallback behavior, which inflates sequence length and can degrade model quality. The decision rule was set in advance: a value near or below 2.5 would be acceptable, while a value around 4–5 or higher would be unacceptable.

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

Retrieval used brute-force cosine similarity with NumPy rather than FAISS. At approximately 10.8k rows and 1024 dimensions, retrieval is only a matrix multiplication, so FAISS was deferred.

The OpenAI client supports reasoning-style request behavior by using:

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

The smoke run used `n = 25` and `k = 4`.

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

The dispatcher removes the subcommand from `sys.argv` before importing the target module. This preserves each module’s existing `argparse` behavior and allows all flags to pass through unchanged.

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

Repeated runs produced byte-identical file hashes, confirming deterministic behavior.

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
