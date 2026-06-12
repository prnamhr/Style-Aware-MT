# Engineering and Decision Log

This document provides a chronological record of pipeline stages, architectural decisions, implementation changes, and their underlying rationale. Entries are ordered in reverse chronological order, with the most recent decision listed first.

Each entry should document four elements: **what changed, why the change was made, how the result can be reproduced, and what limitations or risks future contributors should monitor**. This log is the authoritative record for explaining why the dataset, pipeline, and derived artifacts have their current form. Any decision that may not be immediately obvious to a future reader should be documented here.

> **Conventions**
>
> * Dates are recorded in absolute form: `YYYY-MM-DD`.
> * Code references should use the form `path:line`.
> * Commands should be recorded as exact command-line invocations.
> * When a pipeline stage writes data, the entry should identify the artifact paths and specify how integrity was verified, such as through hashes or manifests.

---

## 2026-06-12 — Phase 2: stylometrics module (per-segment features + target-register centroid)

### What changed

Added `src/eval/stylometrics.py`, the objective style-measurement layer the README
promises (`README.md:136-137`) and the dependency of two later stages: the style
rerank (which scores candidates by closeness to the target register) and **H2**, the
variance hypothesis, which was previously *unmeasurable* — nothing recorded per-segment
feature variance at all.

- **Per-segment features** (`features(text)`, `FEATURE_NAMES` order via
  `feature_vector(text)` — the form the rerank will consume): `lex_density`, `ttr`,
  `root_ttr`, `sent_len_mean`, `sent_len_var`, `marker_rate`.
- **Target-register centroid** (`build_centroid`) over the 10,860 training English
  targets, saved to `results/stylometrics_centroid.json` with per-feature **mean and
  std** so the rerank can z-score candidate vectors (the "standardized feature vector"
  of `README.md:137`). `std` is floored at `1e-9` to keep z-scoring safe.
- **Reporting** (`aggregate`, `distance_to_centroid`): a new `manage.py stylometrics`
  command prints per-condition feature **mean *and* across-segment std** (the std is
  the H2 signal) plus `stylo_dist`, the standardized Euclidean distance of a condition's
  mean vector to the centroid.
- **Single source of truth for register markers:** the archaic-marker regex `_MARKERS`
  now lives in `stylometrics.py:35`; `src/eval/quick.py:25` imports it instead of
  redefining it, so the smoke scorer and the stylometrics report can never drift. No
  change to `quick.py`'s `_marker_rate` semantics.
- Registered `"stylometrics"` in `manage.py:24`. Added `tests/test_stylometrics.py`, a
  pytest-free assertion script (none is installed) including a **hand-labeled**
  lexical-density check.

### Rationale

**Lexical density is a function-word ratio with a register-aware list, not POS-based.**
A word is content unless it is in `FUNCTION_WORDS` — a standard English function-word
set extended with the archaic thee/thou/hath family (`thou, thee, thy, thine, ye, hath,
doth, hast, art, unto, verily, whilst, wherefore, thereof, therein, shall, …`). This
keeps the project's no-spaCy/no-nltk stance and is deterministic, at the cost of being
approximate. Tokenizing strips punctuation first (`[a-z']+` on lowercased text) so
`"word."` and `"word"` are one token. Both raw `ttr` and length-robust
`root_ttr = types/√tokens` (Guiraud) are reported because scripture segments vary
widely in length and raw TTR is length-confounded.

### Limitations / risks

- **`lex_density` is approximate** — name it "function-word ratio with a register-aware
  list" in the thesis, with a one-paragraph limitation note.
- Because archaic pronouns (`thou/thee/thy`) are counted as **function** words, a
  heavily archaic passage scores *lower* lexical density. `lex_density` and
  `marker_rate` are therefore **complementary, not redundant** signals — do not read a
  lex-density drop as a register drop.
- The shared `_MARKERS` regex is **case-sensitive** (inherited from `quick.py`):
  sentence-initial "Thou" is not matched, and `\bO\b` matches only the capital-O
  vocative. Left as-is to preserve `quick.py`'s established marker rates.
- Regex sentence splitter; single-sentence segments give `sent_len_var = 0`.

### Verification

`ruff check` clean on all touched files. `python tests/test_stylometrics.py` → 7/7
pass, including the hand-labeled excerpt (train target #0, 42 tokens → 18 content / 31
types → `lex_density = 18/42`). Centroid built over n=10,860 with plausible means
(`lex_density 0.4344`, `ttr 0.8540`, `marker_rate 0.0327`); gold targets report
`stylo_dist = 0.0` against their own centroid. Per-condition path exercised against the
quarantined smoke outputs (`reference`/`knn_fewshot`, n=25), and
`manage.py eval` still runs — confirming the `_MARKERS` import refactor did not break
the smoke scorer.

### Reproduction

```bash
python tests/test_stylometrics.py                                  # 7/7 sanity checks
python manage.py stylometrics --build-centroid                     # writes results/stylometrics_centroid.json
python manage.py stylometrics --targets-split train                # H2 reference distribution (per-feature mean + std)
python manage.py stylometrics --conditions reference knn_fewshot --split val   # per-condition report + stylo_dist
```

---

## 2026-06-12 — Phase 1 repo hygiene: re-point to open-source base, seal test, quarantine smoke, relabel baseline

### What changed

Repository hygiene pass before any AFSP code is written, to stop the commercial-API prototype from leaking into the thesis as findings and to fix mislabeled artifacts. Four threads:

1. **Re-pointed to the open-source base.** Added `configs/base_qwen.yaml` as the canonical config naming the locked base `Qwen/Qwen2.5-7B-Instruct` (rationale: the 2026-06-12 fertility entry). The local generator client (`provider: local`) is intentionally **not yet implemented** — this phase is hygiene, not feature work; the config records the base and run settings so the client can be wired later. The two commercial-API configs were re-labeled as **SMOKE ONLY — not a thesis condition, not findings**.
2. **Sealed the test split.** Both smoke configs pointed `data.test_file` at `data/splits/test.jsonl`. The data key was renamed `test_file → eval_file` and re-pointed to `data/splits/val.jsonl`; `src/infer/run.py` now derives a split tag from the eval-file stem and writes `outputs/<condition>_<split>.jsonl` (e.g. `_val`), so dev outputs can never be filed under a `_test` name. `test.jsonl` is reserved for the single final run.
3. **Quarantined the smoke results.** The four commercial-API output files were moved to `outputs/smoke_api_quarantine/` (git-ignored) with a `README.md` stating why they are not findings, and renamed to reflect reality (`afsp_test.jsonl → knn_fewshot_test.smoke.jsonl`). A ⚠️ banner was added to the 2026-06-11 DEVLOG entry over its BLEU/chrF table.
4. **Relabeled cosine retrieval as the `knn_fewshot` baseline.** The implemented retriever is plain top-k cosine NN — a baseline row, not the adaptive AFSP contribution (margin scoring / CAP / target-distribution priority / word-level weighting, per `docs/afsp_strategies.md`). Renamed `src/afsp/ → src/retrieval/`, `data/afsp_index/ → data/knn_index/`, class `AfspIndex → RetrievalIndex`, function `build_afsp_user → build_knn_fewshot_user`, config block `afsp: → retrieval:`, and the inference/eval condition string `afsp → knn_fewshot`. README gained a `knn_fewshot (baseline)` results row and a paragraph separating it from AFSP.

Also: **deleted `notebooks/SplitData.ipynb`** — a misleading random row-level `train_test_split` (Train 10849 / Val 1357 / Test 1357) that contradicts the real work-disjoint split. It implied a leaky split was used; the canonical splitter is `src/data/split.py`.

### Rationale

The commercial-API run was always a loop check, but its numbers and the `afsp` label were one careless copy-paste away from being read as thesis results — especially dangerous because the runs used a different model *and* touched the sealed test set. Sealing test to `val.jsonl` enforces the README's own train/dev/test discipline. Relabeling to `knn_fewshot` keeps the eventual AFSP-vs-baseline comparison honest: any register shift must be attributable to the adaptive machinery, not to naive retrieval the baseline already captures.

### Verification

`src/data/split.py` was re-run and confirmed to reproduce the committed splits **byte-identically** — `sha256` of `train/val/test.jsonl` matches the digests in `data/splits/hashes.json`, and `git status` on `data/splits/` is clean. Counts match the manifest exactly (10860 / 1323 / 1322; cross-boundary dedup drops val=27, test=35; leakage audit PASS). After the rename, `from src.retrieval.retrieve import RetrievalIndex` and `from src.retrieval.build_index import build_index` import cleanly, the `run.py`/`quick.py` CLIs expose `{reference, knn_fewshot}`, and `grep` finds zero residual `afsp`/`AfspIndex` references in `src/`, `configs/`, `manage.py` (outside `docs/afsp_strategies.md`).

### Reproduction

```bash
python -m src.data.split                                              # byte-identical to hashes.json
python -m src.retrieval.build_index --config configs/base_qwen.yaml   # writes data/knn_index/
python -m src.infer.run --condition knn_fewshot --config configs/openai_smoke.yaml   # SMOKE only, runs on val.jsonl
python -m src.eval.quick --conditions reference knn_fewshot --split val
```

### Caveats and watch-outs

`configs/base_qwen.yaml` is not yet runnable end-to-end: `provider: local` has no client until the local-inference phase. The smoke configs remain the only runnable inference path and stay API-backed — keep their SMOKE banners intact and never report their numbers. `data/knn_index/` is git-ignored and must be rebuilt with `build_index`; the rename from `data/afsp_index/` was a disk move only. AFSP itself is still unbuilt — the README/DEVLOG describe it as the planned contribution, and the `AFSP` results row stays empty until it exists.

---

## 2026-06-12 — Tokenizer fertility vetting of Qwen2.5-7B-Instruct as the local base model

### What changed

Added `src/eval/fertility.py` and registered a `fertility` command in `manage.py:22`. The module loads the source side (the `input` field) of one or all splits, tokenizes each sentence with a given HF tokenizer (`add_special_tokens=False`), and reports corpus-level fertility (total subword tokens / total whitespace words) alongside per-sentence mean, median, p90, and max. A threshold verdict is printed (`<2.5` lock, `2.5–4` borderline, `4+` catastrophic).

### Rationale

Before committing to an open-source 7–8B base for the local pipeline, we needed to confirm its tokenizer encodes the Arabic/Persian-script source efficiently. Fertility is the standard proxy: high fertility (4–5+) means byte-fallback shredding that inflates sequence length and degrades quality. The source side is what the base model must encode, so it is what we measure. The decision rule was set in advance: under ~2.5 tok/word → lock the model; catastrophic (4–5+) → do not switch.

### Result

`Qwen/Qwen2.5-7B-Instruct` over the full corpus (13,505 sentences, 212,140 source words):

| metric | full corpus (`--split all`) | train split |
| ------ | --------------------------: | ----------: |
| corpus fertility | **2.7000** | 2.6987 |
| mean per-sentence | 2.7383 | 2.7393 |
| median | 2.7000 | 2.6991 |
| p90 | 3.2500 | 3.2500 |
| max | 7.0000 | 7.0000 |

**2.70 subword tokens/source-word.** Marginally above the ~2.5 target but far from the 4–5+ catastrophic zone, with a tight distribution (median = corpus mean, p90 only 3.25). **Decision: Qwen2.5-7B-Instruct is locked as the local base model.** The lone max of 7.0 is a single very short sentence where the ratio is noisy.

### Reproduction

```bash
python manage.py fertility --model Qwen/Qwen2.5-7B-Instruct --split all
python manage.py fertility --model Qwen/Qwen2.5-7B-Instruct --split train
```

Requires `transformers`; the Qwen tokenizer is fetched from the HF Hub on first run (tokenizer files only, no model weights). No `HF_TOKEN` needed for this public tokenizer.

### Caveats and watch-outs

The source is **largely classical Arabic, not only Persian** (e.g. `هو الله العلیّ الأعلی`); the methodology section should describe it as Arabic/Persian-script source. The 2.70 figure reflects Qwen's reasonable Arabic-script vocabulary coverage. Fertility is a tokenizer-efficiency proxy only — it says nothing about the base model's translation or register quality, which still requires the downstream evaluation harness. Word count uses whitespace splitting, which slightly undercounts Persian orthographic words joined by ZWNJ; this inflates fertility marginally and is conservative for the lock decision.

---

## 2026-06-11 — API-backed AFSP smoke pipeline (reference + AFSP, OpenAI + Anthropic)

> **⚠️ SMOKE / PIPELINE-VALIDATION ONLY — NOT THESIS FINDINGS (annotated 2026-06-12).**
> The BLEU/chrF/marker-rate table below was produced by commercial APIs (`gpt-5.5`,
> `claude-opus-4-8`) on `test.jsonl`, before the base model was locked. It is disqualified
> as findings on two counts: (1) the thesis base is the open-source `Qwen2.5-7B-Instruct`,
> not these APIs; (2) the runs touched the sealed test split. The condition called `afsp`
> here is plain cosine k-NN retrieval — i.e. the `knn_fewshot` **baseline**, not the
> adaptive AFSP contribution. As of 2026-06-12 the code is renamed accordingly
> (`src/afsp/` → `src/retrieval/`, condition `afsp` → `knn_fewshot`) and these outputs are
> quarantined under `outputs/smoke_api_quarantine/`. Read the numbers below only as evidence
> that the loop runs, never as results. See the 2026-06-12 Phase-1 hygiene entry.

### What changed

A complete, provider-pluggable inference and scoring loop was added so the AFSP design (retrieval → prompt assembly → generation → output → scoring) could be validated against a commercial API before the open-source 7–8B local pipeline is built. New components:

* `prompts/style_instruction.txt` — shared system prompt describing Shoghi Effendi's register.
* `configs/openai_smoke.yaml` / `configs/anthropic_smoke.yaml` — one config per provider; identical except the `generator` block. OpenAI targets `gpt-5.5` (per supervisor direction; `gpt-4o-mini` remains a one-line swap); Anthropic targets `claude-opus-4-8`. Both carry AFSP `k`, embed model, and a `data.limit` slice.
* `src/afsp/build_index.py` — embeds the English target side of the training split and writes `data/afsp_index/` (`embeddings.npy`, `pairs.jsonl`, `meta.json`).
* `src/afsp/retrieve.py` — `AfspIndex`: brute-force cosine retrieval over the L2-normalized embedding matrix.
* `src/infer/usage.py` — shared token/cost accounting (`Usage`) parameterized by a per-model price table.
* `src/infer/openai_client.py` / `src/infer/anthropic_client.py` — provider-isolated chat wrappers, both exposing `complete(system, user) -> str` and `.usage`, with retries. Keys read from `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` (or `.env`).
* `src/infer/run.py` — `make_client()` dispatches on `generator.provider`; `reference` (zero-shot) and `afsp` (k-shot, exemplars ordered most-similar-last) conditions; writes `outputs/<condition>_test.jsonl` + `_usage.json`.
* `src/eval/quick.py` — BLEU, chrF, and an archaic-register marker rate per condition.
* `manage.py` — registered `build_index`, `infer`, `eval` commands.
* `src/afsp/embed.py` — `load_model` now selects CUDA when available and prints the device.

### Rationale

The supervisor requested validating the approach against a commercial API first. This de-risks the model-agnostic part of the pipeline cheaply and without GPU training: if retrieval + prompting do not shift register on a strong API model, they will not on a local 7B model either. Two providers are wired so register quality can be compared across model families before the local model is chosen.

### Method and dependencies

Retrieval uses NumPy brute-force cosine rather than FAISS: at ~10.8k rows × 1024-dim the similarity is a single matmul, so the FAISS dependency was deferred. Both API clients accommodate reasoning models: the OpenAI client uses `max_completion_tokens`, omits `temperature`/`seed` when unset, and forwards `reasoning_effort` — required because `gpt-5.5` rejects a custom temperature and bills hidden reasoning tokens as output (so `max_tokens` is raised to leave room for them). The Anthropic client likewise omits `temperature`/`top_p`/`seed` — on Claude Opus 4.x / Fable 5 those parameters are removed and return a 400; determinism and output-only formatting are steered through the prompt instead. Dependencies added and pinned: `openai==2.41.1`, `anthropic==0.109.1`, `sacrebleu==2.6.0`, `PyYAML==6.0.3`, `python-dotenv==1.2.2`. `.gitignore` now excludes `.env`, `outputs/`, and `data/afsp_index/`.

### Result (n = 25, k = 4)

Three model families were run on the same test slice. AFSP injects ~4× the prompt tokens of the reference condition (the exemplars).

| Provider / model | Condition | BLEU | chrF | marker_rate | ref_marker_rate |
| ---------------- | --------- | ---: | ---: | ----------: | --------------: |
| OpenAI `gpt-4o-mini` | reference | 20.83 | 47.61 | 0.44 | 0.28 |
| OpenAI `gpt-4o-mini` | afsp      | 19.56 | 47.30 | 0.48 | 0.28 |
| Anthropic `claude-opus-4-8` | reference | 22.30 | 50.00 | 0.48 | 0.28 |
| Anthropic `claude-opus-4-8` | afsp      | 23.72 | 51.12 | 0.32 | 0.28 |
| OpenAI `gpt-5.5` (reasoning, effort=low) | reference | 26.16 | 52.14 | 0.40 | 0.28 |
| OpenAI `gpt-5.5` (reasoning, effort=low) | afsp      | 25.15 | 51.90 | 0.36 | 0.28 |

OpenAI `gpt-4o-mini` spend ≈ $0.0056 total (6,019 / 25,987 prompt tokens for reference / afsp). The pipeline runs end-to-end on all three (the `gpt-5.5` run confirms the reasoning-model request path: `max_completion_tokens`, no `temperature`, `reasoning_effort`). BLEU/chrF differences at n=25 are within noise and not formally interpreted, but two directions are consistent: stronger base models score higher overall (gpt-5.5 > claude-opus-4-8 > gpt-4o-mini on reference BLEU/chrF), and AFSP pulls the archaic-marker rate toward the target distribution on the two stronger models (claude 0.48→0.32, gpt-5.5 0.40→0.36). AFSP improved overlap metrics only on `claude-opus-4-8`; on `gpt-4o-mini` and `gpt-5.5` the overlap effect was flat-to-slightly-negative.

Register finding (model-dependent): the blanket style instruction **over-archaizes relative to the references** — reference-condition marker rate is 0.44–0.48 vs the targets' 0.28 on both models. On `gpt-4o-mini` this produced overt errors: a source addressing a list of US states was rendered "O thou provinces of the Northeast…", a singular sacred pronoun misapplied to a plural, secular address, and k=4 exemplars did not correct it (afsp marker rate rose to 0.48). On `claude-opus-4-8` that failure did not occur (both conditions render the passage plainly), and AFSP **pulled the marker rate down from 0.48 to 0.32 — toward the target distribution** — consistent with the example-driven register hypothesis (RQ2). The over-archaizing is therefore partly a weak-model artifact, and AFSP's corrective effect is visible on the stronger model.

### Reproduction

```bash
python manage.py build_index --config configs/openai_smoke.yaml
python manage.py infer --condition reference --config configs/openai_smoke.yaml
python manage.py infer --condition afsp      --config configs/openai_smoke.yaml
python manage.py eval  --conditions reference afsp
# Anthropic: same commands with --config configs/anthropic_smoke.yaml (index is shared).
```

Requires `OPENAI_API_KEY` (OpenAI config) or `ANTHROPIC_API_KEY` (Anthropic config) in the environment or a project-root `.env`.

### Caveats and watch-outs

The smoke config runs on `data/splits/test.jsonl`. This is acceptable for a one-shot harness check, but **any prompt tuning or k-sweep must switch `data.test_file` to `val.jsonl`** — the test split is reserved for final evaluation only, and tuning against it would invalidate results.

`src/eval/quick.py` is a smoke scorer, not the final harness: COMET, paired-bootstrap CIs, and LLM-as-Judge are still to come. The marker-rate regex is a crude register proxy. Both `gpt-4o-mini` and `claude-opus-4-8` are commercial API stand-ins for validating the pipeline, not the project's open-source base model.

Output filenames are not provider-tagged (`outputs/<condition>_test.jsonl`), so running a second provider overwrites the first provider's predictions. The cross-provider comparison above was captured from the `_usage.json` files and the eval table before overwrite; add a provider/model tag to the output paths before any run that must be retained.

---

## 2026-06-08 — Django-style `manage.py` command dispatcher

### What changed

A repository-level `manage.py` entry point was added as a lightweight command dispatcher. It maps user-facing command names to pipeline modules:

* `preprocess` → `src.data.preprocess`
* `split` → `src.data.split`

The dispatcher removes the subcommand from `sys.argv` before importing the target module. This preserves each module’s existing `argparse` interface and allows all previously supported flags to pass through unchanged.

### Rationale

This change establishes a single, discoverable command interface:

```bash
python manage.py <command>
```

This is preferable to requiring users to remember module paths such as:

```bash
python -m src.data.<module>
```

It also simplifies future extension: adding a new pipeline command requires only one additional entry in the `COMMANDS` dictionary.

### Reproduction

```bash
python manage.py --help            # list available commands
python manage.py preprocess        # equivalent to python -m src.data.preprocess
python manage.py split --seed 7    # forwards flags verbatim
```

### Caveats and watch-outs

This file is not a Django application entry point. It does not introduce Django settings, apps, or project configuration. It is only a dispatch shim.

New commands must be registered in the `COMMANDS` dictionary at `manage.py:15`, and each target module must expose a callable `main()` function.

---

## 2026-06-08 — Preservation of Arabic diacritics and hamza-bearing characters during source normalization

**Stage:** `src/data/preprocess.py` and `src/data/split.py`
**Inputs:** `data/raw/*.tsv`
**Outputs:** regenerated `data/processed/*` and `data/splits/*`

### What changed

Two source-side normalization procedures were revised to avoid destructive transformations of Arabic orthography.

First, in `preprocess.py`, `PERSIAN_STANDARDIZATION_MAP` was reduced from nine mappings to three. The retained mappings are limited to unambiguous Persian–Arabic encoding standardizations and kashida removal:

* `ي → ی`
* `ك → ک`
* removal of tatweel/kashida: `ـ`

The pipeline no longer folds hamza-bearing characters such as `أ`, `إ`, `ؤ`, and `ٱ`; it also no longer normalizes teh marbuta `ة` or the Persian ezafe form `ۀ`.

Second, in `split.py`, the `normalize_key()` function used for cross-boundary deduplication no longer strips Arabic harakat. In addition, `PUNCT_RE` was expanded to preserve the Arabic combining-mark range, because Python’s `\w` character class does not include combining marks and could therefore remove diacritics unintentionally during punctuation stripping. The now-redundant `is_source` parameter was removed.

### Rationale

The source corpus contains mixed Persian and Arabic scripture. Approximately 76% of cleaned source sentences contain harakat. In Arabic, hamza-bearing characters, teh marbuta, and diacritics can encode meaningful lexical or grammatical distinctions. Treating these marks as orthographic noise has two undesirable consequences.

First, it corrupts the stored training signal by replacing meaningful source distinctions with normalized approximations. Second, it increases false-positive deduplication by collapsing genuinely distinct verses into identical normalized keys, thereby over-pruning validation and test data.

This decision was confirmed with the author.

### Result

The number of retained sentences increased from 13,563 to 13,567. The resulting split sizes are:

| Split      | Sentences | Ratio |
| ---------- | --------: | ----: |
| Train      |    10,860 | 80.4% |
| Validation |     1,323 |  9.8% |
| Test       |     1,322 |  9.8% |

Cross-boundary deduplication drops decreased from 153 to 62 records: 27 from validation and 35 from test. The leakage audit continues to pass, with zero overlapping normalized keys between the training split and the validation/test splits.

### Reproduction

```bash
python -m src.data.preprocess       # regenerate data/processed/*
python -m src.data.split            # regenerate data/splits/*
```

### Caveats and watch-outs

The `metadata.source` field is also normalized; however, grouping keys are Latin work names, so work-level grouping is unaffected.

The AFSP retrieval index is built over paragraph-level English text, whereas the test set contains sentence-level English targets. These may share text. Such overlap should be treated as genuine retrieval leakage at AFSP construction time and handled during that stage.

---

## 2026-06-08 — Work-level split with cross-boundary deduplication

**Stage:** `src/data/split.py`
**Command:** `python -m src.data.split`
**Inputs:** `data/processed/sentences_cleaned.jsonl` containing 13,563 records
**Outputs:** `data/splits/{train,val,test}.jsonl` and `data/splits/hashes.json`

### What changed

The previous random sentence-level `train_test_split` procedure was replaced with a work-level split. Under the new approach, entire works, books, or documents are assigned to a single split, after which cross-boundary deduplication is applied.

### Rationale

The corpus consists of Bahá’í scripture translated from Arabic and Persian into English. This material contains substantial formulaic repetition, including recurring invocations and expressions such as “Glorified art Thou, O Lord my God!” A random row-level split can therefore distribute identical or near-identical sentences across training, validation, and test partitions.

This is especially problematic because AFSP constructs its retrieval index from the English target side of the training partition. If a test sentence also appears in training, the retriever may obtain a trivial exact match, thereby inflating retrieval and downstream evaluation metrics. Such results would reflect data leakage rather than genuine generalization.

The README design specification already required splitting by document or section rather than by random row. The previous implementation was therefore inconsistent with the stated experimental design.

### Method

The revised splitting procedure consists of five steps.

1. **Work-level grouping**
   Each sentence is grouped by `metadata.source`, corresponding to the originating work.

2. **Deterministic bin packing**
   Whole works are assigned to train, validation, and test splits using an 80/10/10 target ratio. The `assign_works` procedure places works largest-first into the split with the greatest current shortfall relative to its target count. Ties are resolved deterministically by work name. No work is permitted to span multiple splits.

3. **Cross-boundary deduplication**
   The training split is retained intact. A validation or test record is removed if its normalized Arabic source input or normalized English target output already appears in the training split. The priority order is train → validation → test, ensuring that no normalized key is shared across split boundaries.

   The matching key, `norm_key`, is used only for comparison and is never written to disk. It applies NFKC normalization, case folding, punctuation removal, whitespace collapse, and — in the earlier implementation — source-side Arabic diacritic removal. This procedure is intended to capture cosmetic variants of repeated scriptural language.

4. **Leakage audit**
   Before writing output files, the script asserts that there are zero overlapping normalized keys between the training split and the validation/test splits. The run fails loudly if this condition is violated.

5. **Manifest generation**
   The script writes `hashes.json`, which records the split configuration, fractions, seed, normalization rules, pre- and post-deduplication counts, final ratios, complete `work_to_split` assignment, and SHA-256 hashes for each output file.

### Result

| Split      | Sentences | Ratio | Works |
| ---------- | --------: | ----: | ----: |
| Train      |    10,850 | 80.9% |    15 |
| Validation |     1,242 |  9.3% |     7 |
| Test       |     1,318 |  9.8% |     6 |

Cross-boundary deduplication removed 121 validation records and 32 test records containing near-identical normalized keys. The leakage audit passed. Repeated runs produce byte-identical file hashes, confirming deterministic behavior.

The corpus contains 28 distinct works. The 193 records with unknown provenance, represented by an empty `source` field, are forced into the training split so that they cannot inflate validation or test performance.

### Reproduction

```bash
python -m src.data.split            # defaults: 80/10/10, seed 42, group_key=source
# knobs: --train_frac --val_frac --test_frac --seed --group_key --input_file --output_dir
```

### Caveats and watch-outs

Split ratios are necessarily approximate. Because entire works must remain intact, exact 80/10/10 ratios are not always achievable. This is an expected consequence of work-level partitioning, particularly because the largest work accounts for approximately 16% of the dataset. The slight ratio imbalance is an acceptable trade-off for preventing leakage.

The whole-work holdout setting creates a more difficult evaluation regime. The test split includes works such as `Gems-of-Divine-Mysteries`, `Will-and-Testament-Abdul-baha`, `Four-Valleys`, `Lawh-i-Aqdas`, `Bisharat`, and `Kitab-i-Ahd`, none of which have stylistic representation in the training split. This is the appropriate generalization setting, but it should be noted when reporting model performance, since it is more stringent than a random sentence-level split.

The deduplication procedure is normalized-exact rather than fuzzy. It captures variation in casing, punctuation, whitespace, and selected orthographic features, but it does not detect paraphrases or near-duplicates that differ by word substitution or phrase-level edits. If evaluation scores remain unexpectedly high, fuzzy near-duplicate detection, such as MinHash or Jaccard similarity, should be considered as the next mitigation.
