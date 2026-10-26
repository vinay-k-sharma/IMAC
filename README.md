# Apple iMac Environmental Report — Multimodal RAG

Question answering over Apple's **iMac Product Environmental Report (October 2024)** that handles the
information plain-text RAG misses: **tables, charts, big-number callouts and footnotes**.
Design rationale: [`specs/SPEC-001-multimodal-rag.md`](specs/SPEC-001-multimodal-rag.md).

```
PDF ─┬─ text layer (PyMuPDF) ─ headings · paragraphs · [^n] footnote markers · 19 endnotes
     └─ page images ─ Gemini vision (+ page text) ─ tables · charts · callouts  ── numbers verified against text layer
          ↓
     typed chunks (text / table / figure / callout / footnote / summary) with page, section, endnote links
          ↓
     Gemini embeddings + BM25 → RRF hybrid retrieval → endnote expansion
          ↓
     Gemini answer, grounded on text + page images, with validated [n] citations
```

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # set GEMINI_API_KEY
```
The report is in `data/raw/` (source: apple.com/environment/pdf/products/desktops/iMac_PER_Oct2024.pdf).

## Use
```bash
.venv/bin/python -m imac_rag.cli ingest            # parse + vision + chunk + index (~40 s, cached afterwards)
.venv/bin/python -m imac_rag.cli ask "What is the sleep power at 115V?"
.venv/bin/python -m imac_rag.cli chat              # multi-turn REPL
.venv/bin/python -m imac_rag.cli serve             # chat UI + API on http://127.0.0.1:8000
.venv/bin/python -m imac_rag.cli eval              # retrieval ablation + end-to-end eval
.venv/bin/python -m imac_rag.cli eval --questions my_questions.jsonl
```

UI tests (Playwright, against a running server; asks 3 questions, so ~3 Gemini requests):
```bash
.venv/bin/pip install playwright pytest && .venv/bin/python -m playwright install chromium
.venv/bin/python -m pytest tests/test_ui.py -v     # screenshots land in tests/screenshots/
```

API: `POST /ask {"question": "...", "history": [...]}` · `POST /ingest?force=true` · `GET /health` · `GET /pages/{n}.png`

## Layout
| Path | Purpose |
|---|---|
| `imac_rag/ingest/pdf_parser.py` | Text layer: reading order, headings, superscript → `[^n]`, endnote parsing, column-aware line merging |
| `imac_rag/ingest/vision.py` | Page image → structured tables / figures / callouts (JSON schema), cached, number-verified |
| `imac_rag/ingest/chunker.py` | Typed chunks; tables as Markdown + row sentences; endnote back-references |
| `imac_rag/ingest/pipeline.py` | Ingestion orchestrator (SHA-256 + prompt-version caching) |
| `imac_rag/index.py`, `retrieve.py` | Dense + BM25 index, RRF fusion, endnote pinning/expansion |
| `imac_rag/generate.py` | Prompting, page-image grounding, citation validation |
| `imac_rag/api.py`, `static/index.html` | FastAPI + single-file chat UI with clickable citations and page previews |
| `evaluation/run.py`, `data/eval/sample_questions.jsonl` | 30-question eval set and harness |

## Evaluation

30 questions with known answers (`data/eval/sample_questions.jsonl`): 9 table, 4 chart, 2 callout,
2 infographic, 9 footnote, 2 text, 2 unanswerable.

**Retrieval** (evidence for the answer present in the top-k retrieved context, 28 answerable questions):

| Configuration | hit@1 | hit@3 | hit@8 | MRR |
|---|---|---|---|---|
| Hybrid (dense + BM25, RRF) | 1.00 | 1.00 | 1.00 | 1.00 |
| Dense only | 1.00 | 1.00 | 1.00 | 1.00 |
| BM25 only | 0.93 | 1.00 | 1.00 | 0.96 |
| Hybrid, **no vision layer** (text only) | 0.71 | 0.75 | 0.75 | 0.73 |

Without the vision layer, every power-table question and every chart question fails retrieval. That's
the measured value of the multimodal extraction.

**End-to-end** (gemini-3.x-flash generation; LLM judge vs reference answer):
30/30 contain the required facts, 30/30 judged correct, 28/28 answerable answers cite sources, 0 invalid
citations, both unanswerable questions correctly declined, median latency ≈ 4.7 s.

> Note: the free Gemini tier allows ~20 requests/day **per model**. The generation chain falls through
> several models (see `config.py`) when one is exhausted, and the eval reports API errors separately
> (unscored) rather than counting them as wrong answers. Re-run single questions with `--ids q17,q18`.

## Eval file format
One JSON object per line:
```json
{"id": "q05", "modality": "table", "question": "...", "reference": "0.57W.",
 "evidence": ["0.57W"], "must_include": [["0.57"]]}
```
`evidence`: strings that must appear in the retrieved context (retrieval hit). `must_include`: groups of
alternatives; every group must appear in the answer. `reference` is used by the LLM judge.
