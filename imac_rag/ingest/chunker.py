"""Turn the text layer + vision layer into typed, self-contained chunks.

Chunk types
  text      prose, split on section/sub-heading boundaries (~350 tokens)
  table     one per table: Markdown grid + one linearised sentence per cell, so
            "sleep power at 115V" can match lexically and semantically
  figure    one per chart/diagram: description + every data point, with printed
            vs. estimated values labelled
  callout   big-number statistics on a page, as full sentences
  footnote  one per endnote, carrying where in the report it is referenced
  summary   one per page: what the page communicates (good for broad questions)

Every chunk carries its page, section and the endnotes it references; the
endnote text is folded into the text that gets embedded, so a question phrased
in endnote terms ("tested with which power adapter?") still finds the claim.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .. import config
from .pdf_parser import ParsedDoc

DOC_NAME = "Apple iMac Product Environmental Report (October 2024)"
MARKER_RE = re.compile(r"\[\^(\d+)\]")


@dataclass
class Chunk:
    id: str
    type: str
    page: int
    section: str
    text: str                       # content shown to the LLM and the user
    footnote_refs: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    embed_text: str = ""            # text used for embeddings + BM25 (adds context)

    def to_dict(self) -> dict:
        return asdict(self)


def _tokens(s: str) -> int:
    return int(len(s.split()) * 1.3)


def _refs(*texts: str) -> list[int]:
    return sorted({int(n) for t in texts for n in MARKER_RE.findall(t)})


def _footnote_digest(refs: list[int], endnotes: dict[int, str], limit: int = 300) -> str:
    return "\n".join(f"Endnote {n}: {endnotes[n][:limit]}" for n in refs if n in endnotes)


def _split(paras: list[str]) -> list[str]:
    """Greedy pack paragraphs (or sentences of oversized paragraphs) into ~target-sized chunks with overlap."""
    units: list[str] = []
    for p in paras:
        if _tokens(p) > config.CHUNK_MAX_TOKENS:
            units.extend(re.split(r"(?<=[.!?])\s+(?=[A-Z])", p))
        else:
            units.append(p)
    chunks, cur = [], []
    for u in units:
        if cur and _tokens(" ".join(cur + [u])) > config.CHUNK_TARGET_TOKENS:
            chunks.append("\n".join(cur))
            overlap: list[str] = []
            for prev in reversed(cur):
                if _tokens(" ".join(overlap + [prev])) > config.CHUNK_OVERLAP_TOKENS:
                    break
                overlap.insert(0, prev)
            cur = overlap
        cur.append(u)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _column_names(t: dict) -> list[str]:
    """Vision merges multi-level headers into "Power consumption for iMac - 100V"; drop the repeated title."""
    return [c.replace(f"{t['title']} - ", "") for c in t["columns"]]


def _table_markdown(t: dict) -> str:
    cols = _column_names(t) or [f"col{i+1}" for i in range(max(len(r) for r in t["rows"]))]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(r + [""] * (len(cols) - len(r))) + " |" for r in t["rows"]]
    return "\n".join(lines)


def _table_sentences(t: dict) -> str:
    cols = t["columns"]
    out = []
    for r in t["rows"]:
        if not r:
            continue
        if len(cols) == 2 or len(r) == 2:
            out.append(f"{t['title']}: {r[0]} = {r[-1]}.")
        else:
            names = _column_names(t)
            cells = "; ".join(f"{names[i] if i < len(names) else ''}: {v}" for i, v in enumerate(r[1:], start=1))
            out.append(f"{t['title']} - {r[0]} - {cells}.")
    return "\n".join(out)


def build_chunks(doc: ParsedDoc, vision: dict[int, dict]) -> list[Chunk]:
    chunks: list[Chunk] = []
    en = doc.endnotes

    def add(type_: str, page: int, section: str, text: str, refs: list[int], meta: dict | None = None):
        cid = f"p{page:02d}-{type_}-{sum(1 for c in chunks if c.page == page and c.type == type_) + 1}"
        header = f"{DOC_NAME} | page {page} | {section} | {type_}"
        embed = f"{header}\n{text}"
        if refs and type_ != "footnote":
            embed += "\n" + _footnote_digest(refs, en)
        chunks.append(Chunk(cid, type_, page, section, text, refs, meta or {}, embed))

    # ---- text chunks (endnote pages are represented by footnote chunks instead) ----
    endnote_pages = {p for p, t in doc.page_titles.items() if t == "Endnotes"}
    for page in range(1, doc.num_pages + 1):
        section = doc.page_titles.get(page, "")
        has_figure = bool(vision.get(page, {}).get("figures"))
        groups: list[tuple[str, list[str]]] = [("", [])]
        for b in (b for b in doc.blocks if b.page == page):
            if b.kind == "section" and b.text == section:
                continue
            if b.kind == "subhead":
                groups.append((b.text, []))
                continue
            # Short label fragments on a chart page are chart labels; the figure chunk covers them.
            if has_figure and len(b.text.split()) < 6 and b.kind != "section":
                continue
            if page in endnote_pages and not b.text.startswith("©"):
                continue  # table rows printed among endnotes -> covered by the vision table chunk
            groups[-1][1].append(b.text)
        for sub, paras in groups:
            if not paras:
                continue
            for piece in _split(paras):
                body = f"{sub}\n{piece}" if sub else piece
                add("text", page, section + (f" > {sub}" if sub else ""), body, _refs(piece))

    # ---- vision-derived chunks ----
    seen_figures: list[set[str]] = []
    for page, v in sorted(vision.items()):
        section = doc.page_titles.get(page, "")
        for t in v.get("tables", []):
            if not t.get("rows"):
                continue
            refs = sorted(set(t.get("footnote_refs") or []))
            body = f"Table: {t['title']}\n{_table_markdown(t)}"
            if t.get("notes"):
                body += "\n\nNote: " + re.sub(r"^Note:\s*", "", t["notes"])
            body += f"\n\nRow by row:\n{_table_sentences(t)}"
            if not t.get("verified", True):
                body += f"\n(Unverified against PDF text: {', '.join(t['unverified_numbers'])})"
            add("table", page, section, body, refs, {"title": t["title"], "verified": t.get("verified")})

        for f in v.get("figures", []):
            labels = {dp["label"].lower() for dp in f.get("data_points", [])}
            if labels and any(len(labels & s) / len(labels | s) > 0.6 for s in seen_figures):
                continue  # same graphic repeated on another page
            seen_figures.append(labels)
            refs = sorted(set(f.get("footnote_refs") or []))
            pts = []
            for dp in f.get("data_points", []):
                tag = "printed value" if dp["source"] == "printed" else "approximate, estimated from the chart graphic"
                pts.append(f"- {dp['label']}: {dp['value']} ({tag})")
            body = f"Figure ({f['figure_type']}): {f['title']}\n{f['description']}"
            if pts:
                body += "\nData points:\n" + "\n".join(pts)
            add("figure", page, section, body, refs, {"title": f["title"], "figure_type": f["figure_type"]})

        callouts = v.get("callouts", [])
        if callouts:
            refs = sorted({n for c in callouts for n in (c.get("footnote_refs") or [])})
            body = "Key figures highlighted on this page:\n" + "\n".join(f"- {c['statement']}" for c in callouts)
            add("callout", page, section, body, refs)

        if v.get("page_summary"):
            add("summary", page, section, f"Page {page} overview: {v['page_summary']}", [])

    # ---- footnote chunks ----
    for n, text in sorted(en.items()):
        where = doc.endnote_refs.get(n, [])
        pages = sorted({w["page"] for w in where})
        # Vision may have found markers the text layer missed (e.g. badge footnotes).
        for page, v in vision.items():
            items = v.get("tables", []) + v.get("figures", []) + v.get("callouts", [])
            if any(n in (i.get("footnote_refs") or []) for i in items) and page not in pages:
                pages.append(page)
        ctx = "; ".join(f"page {w['page']}: \"{w['context'][:120]}\"" for w in where[:4])
        body = f"Endnote {n}: {text}"
        if ctx:
            body += f"\n\nThis endnote qualifies the following statements in the report: {ctx}"
        page = 11 if n <= 16 else 12
        add("footnote", page, "Endnotes", body, [n], {"endnote": n, "referenced_on_pages": sorted(pages)})

    return chunks
