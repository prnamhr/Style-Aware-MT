# Engineering & Decision Log

A running record of pipeline stages, design decisions, and their rationale.
Newest entries at the top. Each entry should answer: **what changed, why, how to
reproduce, and what to watch out for.** This is the source of truth for "why is
the data the way it is" — if a choice would surprise a future reader, it belongs
here.

> Conventions
> - Dates are absolute (YYYY-MM-DD).
> - Reference code as `path:line` and commands as exact CLI invocations.
> - When a stage writes data, record the artifact paths and how integrity is
>   verified (hashes/manifests).

---

## 2026-06-08 — Work-level split + cross-boundary de-duplication

**Stage:** `src/data/split.py` (`python -m src.data.split`)
**Inputs:** `data/processed/sentences_cleaned.jsonl` (13,563 records)
**Outputs:** `data/splits/{train,val,test}.jsonl` + `data/splits/hashes.json`

### What changed
Replaced the random sentence-level `train_test_split` with a **whole-work
(book/document) split** followed by **cross-boundary de-duplication**.

### Why
The corpus is Bahá'í scripture translation (Arabic/Persian → English). Scripture
is highly repetitive — invocations and formulae ("Glorified art Thou, O Lord my
God!") recur across works. A random row-level split scatters near-identical
sentences across train/val/test. Because AFSP builds its retrieval index over the
**English target side of the training partition** (README §AFSP), a test sentence
that also appears in train gives the retriever a trivial exact match — inflating
retrieval and downstream metrics with leakage rather than real generalization.
The README design spec already mandated "split by document/section, not by random
row" (§Splits); the old code contradicted it.

### How it works
1. **Group** every sentence by `metadata.source` (the work).
2. **Bin-pack** whole works into train/val/test targeting 80/10/10: works are
   placed largest-first into whichever split currently has the largest shortfall
   against its target count (`assign_works`). Deterministic — ties broken by
   name. No work ever spans a boundary.
3. **De-duplicate across the boundary** (`dedup_against_seen`): train is kept
   fully intact; a val/test record is dropped if its *normalized* Arabic input
   **OR** English output already appears in train. Priority order train → val →
   test, so no normalized key is shared across splits.
   - `norm_key` (matching only — never written to disk): NFKC, strip Arabic
     diacritics/tashkil on the source side, casefold, strip punctuation, collapse
     whitespace. Catches cosmetic-only scripture variants.
4. **Leakage audit:** asserts 0 overlapping normalized keys between train and
   val/test before writing. Run fails loudly if violated.
5. **Manifest** `hashes.json`: config (fracs, seed, rules), pre/post-dedup
   counts, final ratios, the full `work_to_split` assignment, and SHA-256 of each
   output file.

### Result
| split | sentences | ratio | works |
|-------|-----------|-------|-------|
| train | 10,850 | 80.9% | 15 |
| val   | 1,242  | 9.3%  | 7  |
| test  | 1,318  | 9.8%  | 6  |

- Cross-boundary dedup dropped 121 val + 32 test near-identical pairs.
- Leakage audit: PASS. Re-runs produce byte-identical hashes (deterministic).
- 28 distinct works; the 193 unknown-provenance records (empty `source`) are
  forced into train so they can never inflate eval.

### Reproduce
```bash
python -m src.data.split            # defaults: 80/10/10, seed 42, group_key=source
# knobs: --train_frac --val_frac --test_frac --seed --group_key --input_file --output_dir
```

### Caveats / watch-outs
- **Ratios are approximate (±~1%)** — unavoidable when keeping lumpy works intact
  (largest work ≈ 16% of data). This is the correct trade.
- **Whole-work holdout = harder eval.** Test works (Gems-of-Divine-Mysteries,
  Will-and-Testament-Abdul-baha, Four-Valleys, Lawh-i-Aqdas, Bisharat,
  Kitab-i-Ahd) have *no* stylistic representation in train. That's the realistic
  generalization setting, but it's a higher bar than a random split — flag it when
  reporting numbers.
- **Dedup is normalized-exact, not fuzzy.** It catches casing/diacritic/punctuation
  variants but not one-word-different paraphrases. If test scores still look
  suspiciously high, fuzzy near-dup (MinHash/Jaccard) is the next lever.
