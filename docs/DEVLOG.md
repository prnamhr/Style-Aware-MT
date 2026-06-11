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

## 2026-06-11 — OpenAI-backed AFSP smoke pipeline (reference + AFSP end-to-end)

### What changed

A complete, model-swappable inference and scoring loop was added so the AFSP design (retrieval → prompt assembly → generation → output → scoring) could be validated against a commercial API before the open-source 7–8B local pipeline is built. New components:

* `prompts/style_instruction.txt` — shared system prompt describing Shoghi Effendi's register.
* `configs/openai_smoke.yaml` — single config for the smoke run; generator is model-swappable (defaults to `gpt-4o-mini`), with AFSP `k`, embed model, and a `data.limit` slice.
* `src/afsp/build_index.py` — embeds the English target side of the training split and writes `data/afsp_index/` (`embeddings.npy`, `pairs.jsonl`, `meta.json`).
* `src/afsp/retrieve.py` — `AfspIndex`: brute-force cosine retrieval over the L2-normalized embedding matrix.
* `src/infer/openai_client.py` — provider-isolated chat wrapper with retries and token/cost accounting; reads `OPENAI_API_KEY` from env or `.env`.
* `src/infer/run.py` — `reference` (zero-shot) and `afsp` (k-shot, exemplars ordered most-similar-last) conditions; writes `outputs/<condition>_test.jsonl` + `_usage.json`.
* `src/eval/quick.py` — BLEU, chrF, and an archaic-register marker rate per condition.
* `manage.py` — registered `build_index`, `infer`, `eval` commands.
* `src/afsp/embed.py` — `load_model` now selects CUDA when available and prints the device.

### Rationale

The supervisor requested validating the approach against the OpenAI API first. This de-risks the model-agnostic part of the pipeline cheaply and without GPU training: if retrieval + prompting do not shift register on a strong API model, they will not on a local 7B model either.

### Method and dependencies

Retrieval uses NumPy brute-force cosine rather than FAISS: at ~10.8k rows × 1024-dim the similarity is a single matmul, so the FAISS dependency was deferred. Dependencies added and pinned: `openai==2.41.1`, `sacrebleu==2.6.0`, `PyYAML==6.0.3`, `python-dotenv==1.2.2`. `.gitignore` now excludes `.env`, `outputs/`, and `data/afsp_index/`.

### Result (n = 25, gpt-4o-mini, k = 4)

| Condition | BLEU | chrF | marker_rate | ref_marker_rate |
| --------- | ---: | ---: | ----------: | --------------: |
| reference | 20.83 | 47.61 | 0.44 | 0.28 |
| afsp      | 19.56 | 47.30 | 0.48 | 0.28 |

Total spend ≈ $0.0056 (reference 6,019 prompt tokens; AFSP 25,987 prompt tokens — the ~4× increase is the injected exemplars). The pipeline runs end-to-end; the BLEU/chrF gap at n=25 is within noise and is not interpreted.

The informative finding is qualitative: **both conditions over-archaize relative to the references** (marker rate 0.44–0.48 vs 0.28). The clearest failure is a source addressing a list of US states, which AFSP rendered "O thou provinces of the Northeast…" — a singular sacred pronoun misapplied to a plural, secular address. The blanket style instruction triggers archaic forms unconditionally, and k=4 exemplars did not correct it.

### Reproduction

```bash
python manage.py build_index --config configs/openai_smoke.yaml
python manage.py infer --condition reference --config configs/openai_smoke.yaml
python manage.py infer --condition afsp      --config configs/openai_smoke.yaml
python manage.py eval  --conditions reference afsp
```

Requires `OPENAI_API_KEY` in the environment or a project-root `.env`.

### Caveats and watch-outs

The smoke config runs on `data/splits/test.jsonl`. This is acceptable for a one-shot harness check, but **any prompt tuning or k-sweep must switch `data.test_file` to `val.jsonl`** — the test split is reserved for final evaluation only, and tuning against it would invalidate results.

`src/eval/quick.py` is a smoke scorer, not the final harness: COMET, paired-bootstrap CIs, and LLM-as-Judge are still to come. The marker-rate regex is a crude register proxy. `gpt-4o-mini` is a prototyping stand-in, not the project's open-source base model.

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
