# Adaptive Few-Shot Prompting (AFSP) — Research Strategies

This document consolidates research-based strategies for the AFSP component of the thesis on style-aware machine translation of Bahá'í scriptures. The goal is to move beyond naive few-shot prompting toward an adaptive, retrieval-based pipeline that preserves Shoghi Effendi's English register.

---

## 1. Retrieval and Selection Mechanisms

Standard few-shot prompting relies on random example selection. AFSP requires more principled retrieval.

### Margin-Based Similarity
Rather than pure cosine similarity, use a **margin-based scoring method** when retrieving examples from the parallel datastore. This penalizes hub sentences — sentences with broad but shallow similarity to many candidates — and reduces semantic noise, yielding more targeted and stylistically relevant demonstrations.

### Context-Aware Prompting (CAP)
For document-level input, apply CAP as follows:
1. Select the most relevant sentences within the source document using attention scores.
2. Summarize that subset into a concise context representation.
3. Retrieve parallel examples from the datastore based on similarity to the **summary**, not the raw source.

This maximises the use of the LLM's limited context window by concentrating stylistically and topically relevant content.

---

## 2. Demonstration Ordering

The arrangement of few-shot examples materially affects output quality due to LLM positional sensitivity.

### Spatial Proximity Rule
- Place the **most stylistically representative or "cleanest" examples last** in the prompt, closest to the test sentence.
- Place noisier or less relevant examples at the beginning, where their influence is weakest.

### Target Distribution Priority
Because the model is generating in English, the **quality and register of the target-side examples matters more than source-side similarity**. When selecting demonstrations, prioritise matches to Shoghi Effendi's English lexical and syntactic patterns over surface-level Persian/Arabic similarity.

---

## 3. Multi-View Word-Level Weighting

To prevent the model from defaulting to a neutral register:

- Augment each demonstration with explicit **word-level pairs** alongside the full sentence: `source_term : target_term`.
- This draws the model's attention to specific terminology and register markers (archaic forms, formal lexical choices) characteristic of the authorized translation style.
- Reduces the influence of high-frequency but stylistically irrelevant tokens.

**Prompt template sketch:**

```
[Word pairs]: ای → O Thou | مبارک → blessed | حضرت → His Holiness
[Source]: <Persian sentence>
[Target]: <Shoghi Effendi translation>
```

---

## 4. Sensitivity Analysis (RQ2)

RQ2 asks how AFSP shifts model outputs toward the target register relative to a zero-shot baseline. The following experimental design addresses this directly.

### Shot Count Sweep
Test $K \in \{1, 4, 8\}$ shots systematically. Track the point at which stylistic fidelity plateaus or degrades to identify the optimal $K$ for this corpus.

### Quantifiable Shift Metrics
| Metric | Purpose |
|---|---|
| LLM-as-Judge protocol | Holistic register fidelity rating |
| Lexical density | Proportion of content words — higher in formal/scriptural register |
| Type–token ratio (TTR) | Vocabulary richness; Shoghi Effendi exhibits elevated TTR |
| Cosine distance to reference embeddings | Semantic proximity to authorized translations |

### ICL vs. Instruction-Only Baseline
The literature suggests ICL is **example-driven, not instruction-driven**. Verify this for the Bahá'í corpus by comparing:
- Prompt A: instruction only (`"Translate in a formal, archaic scriptural register."`)
- Prompt B: 5-shot style-bearing exemplars, no explicit instruction

Expect Prompt B to outperform Prompt A on register metrics; document the gap.

---

## 5. Low-Resource Mitigation

If the Persian/Arabic parallel corpus is too small to populate a meaningful datastore:

- Use **allied-task proxy demonstrations**: examples where the source is a different high-resource language (e.g., Arabic Classical → English) but the target register remains consistent with the authorized Bahá'í style.
- The stylistic signal is carried by the target side; source language can vary as long as the target exemplifies the desired register.

---

## Pipeline Summary

```
Input sentence
     │
     ▼
[CAP] Attention-based context selection + summarization
     │
     ▼
[Retrieval] Margin-based similarity search over parallel datastore
     │
     ▼
[Ordering] Proximity ranking — best examples last
     │
     ▼
[Augmentation] Word-level pair injection into prompt template
     │
     ▼
LLM inference → style-aware translation
     │
     ▼
[Evaluation] LLM-as-Judge + stylometric features
```

---

## Open Questions

- What is the optimal margin parameter $\beta$ for hub penalization on this specific domain?
- Does CAP summarization introduce information loss for highly allusive scriptural passages?
- At what corpus size does the allied-task proxy approach stop being beneficial?
