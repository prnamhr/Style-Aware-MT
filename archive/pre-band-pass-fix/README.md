# Archived — pre-band-pass-fix sweep (NOT canonical)

These artifacts were produced by the AFSP reranker **before** the band-pass fix.
They are kept only for provenance/comparison and must not be used as results.

- `sweep/` — 26 stale `afsp_k*_l*_val.jsonl` from the old reranker. The canonical
  sweep lives in the top-level `outputs/sweep/`.
- `afsp_run_colab_with_before.ipynb` — the runbook that generated the above.

Canonical results come from `outputs/sweep/`, `results/afsp_sweep_val.json`, and
`results/afsp_verify_val.json` on the post-fix run.
