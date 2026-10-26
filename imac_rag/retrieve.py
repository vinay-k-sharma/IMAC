"""Hybrid retrieval: dense (Gemini embeddings) + BM25, fused with Reciprocal Rank Fusion.

Two document-aware additions on top of plain RRF:
  * explicit endnote lookup - "footnote 7" / "endnote 14" pins that endnote chunk
  * endnote expansion - every retrieved chunk carries the full text of the
    endnotes it references, so qualifications (configuration, test conditions,
    exclusions) reach the LLM even when the endnote chunk itself didn't rank
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import config, llm
from .index import load, tokenize
from .ingest.chunker import Chunk

ENDNOTE_Q = re.compile(r"\b(?:foot|end)?note\s*#?\s*(\d{1,2})\b|\[\^?(\d{1,2})\]", re.I)
CANDIDATES = 30


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None
    bm25_rank: int | None
    linked_endnotes: dict[int, str] = field(default_factory=dict)


def retrieve(query: str, top_k: int = config.TOP_K, mode: str = "hybrid",
             exclude_types: frozenset[str] = frozenset()) -> list[Hit]:
    """mode: hybrid | dense | bm25. exclude_types is used by the ablation (e.g. text-only baseline)."""
    idx = load()
    n = len(idx.chunks)
    allowed = [c.type not in exclude_types for c in idx.chunks]

    dense_order: list[int] = []
    if mode in ("hybrid", "dense"):
        q = llm.embed([query], task_type="RETRIEVAL_QUERY")[0]
        dense_order = [i for i in np.argsort(-(idx.embeddings @ q)) if allowed[i]][:CANDIDATES]
    bm25_order: list[int] = []
    if mode in ("hybrid", "bm25"):
        scores = idx.bm25.get_scores(tokenize(query))
        bm25_order = [i for i in np.argsort(-scores) if scores[i] > 0 and allowed[i]][:CANDIDATES]

    fused: dict[int, float] = {}
    for order in (dense_order, bm25_order):
        for rank, i in enumerate(order):
            fused[i] = fused.get(i, 0.0) + 1.0 / (config.RRF_K + rank + 1)

    # Explicit endnote references in the question go straight to the top.
    asked = {int(a or b) for a, b in ENDNOTE_Q.findall(query)}
    for i, c in enumerate(idx.chunks):
        if allowed[i] and c.type == "footnote" and c.meta.get("endnote") in asked:
            fused[i] = fused.get(i, 0.0) + 1.0

    ranked = sorted(fused, key=fused.get, reverse=True)[: min(top_k, n)]
    endnotes = {c.meta["endnote"]: c.text for c in idx.chunks if c.type == "footnote"}
    hits = []
    for i in ranked:
        c = idx.chunks[i]
        linked = {r: endnotes[r].split("\n\nThis endnote qualifies")[0]
                  for r in c.footnote_refs if r in endnotes and c.type != "footnote"}
        hits.append(Hit(
            c, fused[i],
            dense_order.index(i) + 1 if i in dense_order else None,
            bm25_order.index(i) + 1 if i in bm25_order else None,
            linked,
        ))
    return hits
