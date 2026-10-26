# SPEC-001 — Multimodal RAG over the Apple iMac Product Environmental Report

## Problem
The report (12 pages, Oct 2024) puts its most important facts outside plain prose:

| Where | What | Why naive text extraction fails |
|---|---|---|
| p.2 waterfall chart | 445 kg baseline → −99 kg reductions → 346 kg; segment breakdown | Labels come out as a word soup ("Materials and process emissions" ×3, no structure); segment values are not printed at all |
| p.10 tables | 346 kg CO2e, scope 2/3, 50/5/45/<1 % lifecycle split, 391 kg for 4-port 512GB | Rule-based table detection splits the zebra-striped table into 1-row fragments |
| p.12 table | Power draw by mode × voltage (100/115/230 V) | Printed in the middle of the endnotes; voltage headers detached from values |
| p.1, p.6 callouts | "76 %" above "recycled content in fiber packaging" | Value and caption are separate blocks |
| Endnotes 1–19 | Configuration tested, test conditions, regional exceptions, baselines | Linked to claims only by tiny superscripts; endnote 10 even contradicts the body (512GB vs 256GB) |

## Design
```
PDF ─┬─ text layer (PyMuPDF dict) ── headings, paragraphs, [^n] markers, endnotes{n: text}
     └─ page PNGs (150 dpi) ── Gemini vision + page text ─→ JSON {tables, figures, callouts, summary}
                                                             └─ numeric verification vs text layer
        ↓
   typed chunks: text · table · figure · callout · footnote · summary   (page, section, footnote_refs)
        ↓
   Gemini embeddings (768-d, NumPy)  +  BM25   → RRF fusion (k=60), top-8
        ↓                                       + explicit "endnote N" pinning
   endnote expansion (linked endnote text attached to each hit)
        ↓
   Gemini generation: numbered sources + up to 2 page images of top visual hits
        ↓
   answer with [n] citations → invalid citations stripped
```

### Key decisions
1. **Two complementary extraction layers.** The text layer is exact but structureless; the vision layer
   understands structure but can hallucinate. Vision gets the page's text layer in its prompt, and every
   number it labels "printed" must appear in the text layer, otherwise it is downgraded to "estimated".
   Chart values inferred from bar geometry are explicitly tagged *approximate*, and the generator is told
   to say so.
2. **Footnotes are first-class.** Superscripts (span flag bit 0 or <75 % of body font size; `CO2` subscripts
   excluded) become inline `[^n]`. Endnotes become their own chunks that list the statements they qualify,
   and every chunk's embedding text includes the endnotes it references. At query time the full text of
   linked endnotes rides along with each hit, so qualifications reach the LLM even if the endnote chunk
   itself didn't rank.
3. **Tables are indexed twice in one chunk**: as a Markdown grid (for the LLM) and as one sentence per row
   ("Sleep — 100V: 0.56W; 115V: 0.57W; 230V: 0.59W") so BM25 and embeddings match cell-level questions.
4. **Layout repair** for infographic pages: line-blocks that overlap horizontally, share font style and
   sit directly below each other are re-joined; short chart-label fragments on pages with a figure are
   dropped from text chunks (the figure chunk covers them); the repeated life-cycle navigation wheel is
   ignored and near-duplicate figures are de-duplicated.
5. **Visual grounding at answer time**: page images of the top-ranked table/figure/callout hits are sent
   with the text context, so the model can check the chart itself.
6. **Caching**: vision results are cached per page keyed by PDF SHA-256 + prompt version; the index is
   rebuilt only when the PDF, prompt version or embedding model changes.

## Evaluation
`data/eval/sample_questions.jsonl` — 30 questions with reference answers, evidence strings and
required facts: 9 table, 4 chart, 2 callout, 2 infographic, 9 footnote, 2 text, 2 unanswerable.
* Retrieval: hit@k / MRR for evidence in retrieved context; ablation over hybrid / dense / BM25 and
  "no vision layer".
* End-to-end: required-fact keyword check + LLM judge vs reference; citation validity; latency.
