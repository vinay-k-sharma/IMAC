"""Answer generation with Gemini: numbered sources, page images for visual grounding,
and post-hoc citation validation."""
from __future__ import annotations

import re
import time

from . import config, llm
from .retrieve import Hit, retrieve

SYSTEM = """You answer questions about the Apple iMac Product Environmental Report (October 2024) using ONLY the numbered sources provided (and the attached page images, which are the same report pages).

Rules:
- Cite every factual claim with the source number in square brackets, e.g. [2] or [1][4]. Only cite numbers that exist in the source list.
- Quote numbers exactly as printed, with units (e.g. "0.57W", "346 kg CO2e").
- Tables and charts: say which table/chart a number comes from. If a value is marked "approximate, estimated from the chart graphic", say it is approximate and that the report does not print it.
- Endnotes matter: if an endnote qualifies the answer (configuration tested, test conditions, exclusions, regional exceptions, baselines), state the qualification. Mention the endnote number.
- If sources disagree (e.g. a body text and an endnote name different configurations), point out the discrepancy rather than silently picking one.
- If the sources do not contain the answer, say "The report does not provide this information." and briefly say what related information it does contain, if any. Never use outside knowledge.
- Be concise: lead with the direct answer, then supporting detail. Use a small Markdown table when comparing several values."""

CITE_RE = re.compile(r"\[(\d{1,2})\]")
VISUAL_TYPES = {"figure", "table", "callout"}


def _format_source(n: int, h: Hit) -> str:
    c = h.chunk
    out = f"[{n}] (page {c.page} · {c.type} · {c.section})\n{c.text}"
    if h.linked_endnotes:
        out += "\nEndnotes referenced by this source:\n" + "\n".join(
            f"  - {t}" for _, t in sorted(h.linked_endnotes.items()))
    return out


def _pages_for_images(hits: list[Hit]) -> list[int]:
    pages: list[int] = []
    for h in hits[:5]:  # only highly ranked visual content earns an image
        if h.chunk.type in VISUAL_TYPES and h.chunk.page not in pages:
            pages.append(h.chunk.page)
    return pages[: config.MAX_PAGE_IMAGES]


def answer(question: str, *, top_k: int = config.TOP_K, use_images: bool = True,
           history: list[dict] | None = None) -> dict:
    t0 = time.time()
    query = question
    if history:  # make follow-ups ("and at 230V?") retrievable on their own
        last_user = [m["content"] for m in history if m.get("role") == "user"][-2:]
        query = " ".join(last_user + [question])
    hits = retrieve(query, top_k=top_k)
    t_retrieve = time.time() - t0

    sources = "\n\n".join(_format_source(i + 1, h) for i, h in enumerate(hits))
    convo = ""
    if history:
        convo = "Conversation so far:\n" + "\n".join(
            f"{m['role']}: {m['content'][:600]}" for m in history[-6:]) + "\n\n"
    prompt = f"{convo}Sources:\n{sources}\n\nQuestion: {question}"

    contents: list = []
    image_pages = _pages_for_images(hits) if use_images else []
    for p in image_pages:
        contents += [f"Image of report page {p}:", llm.image_part((config.PAGES_DIR / f"page_{p:02d}.png").read_bytes())]
    contents.append(prompt)

    text, model = llm.generate(contents, system=SYSTEM, models=config.GEN_MODELS)

    cited = sorted({int(n) for n in CITE_RE.findall(text)})
    invalid = [n for n in cited if not 1 <= n <= len(hits)]
    for n in invalid:  # never show a citation that points nowhere
        text = text.replace(f"[{n}]", "")
    citations = [
        {
            "n": n,
            "chunk_id": hits[n - 1].chunk.id,
            "page": hits[n - 1].chunk.page,
            "type": hits[n - 1].chunk.type,
            "section": hits[n - 1].chunk.section,
            "text": hits[n - 1].chunk.text,
        }
        for n in cited if n not in invalid
    ]
    return {
        "question": question,
        "answer": text.strip(),
        "citations": citations,
        "invalid_citations": invalid,
        "retrieved": [
            {"rank": i + 1, "chunk_id": h.chunk.id, "page": h.chunk.page, "type": h.chunk.type,
             "score": round(h.score, 4), "dense_rank": h.dense_rank, "bm25_rank": h.bm25_rank}
            for i, h in enumerate(hits)
        ],
        "image_pages": image_pages,
        "model": model,
        "latency_s": {"retrieve": round(t_retrieve, 2), "total": round(time.time() - t0, 2)},
    }
