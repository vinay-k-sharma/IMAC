"""Text-layer extraction with PyMuPDF.

What this layer is responsible for:
  * body text in reading order, with running headers/footers stripped
  * section headings (from font size) so every chunk knows where it lives
  * footnote markers: superscript digits become inline ``[^n]`` tokens, so the
    link between a claim and its endnote survives chunking
  * the endnotes themselves, parsed into ``{n: text}``

Tables and charts are deliberately NOT handled here; rule-based table detection
fragments this report's zebra-striped tables, so they come from the vision layer
(see vision.py), which is then cross-checked against this text layer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

RUNNING_HEADER = re.compile(r"^iMac\s*\|\s*Product Environmental Report$")
ENDNOTE_START = re.compile(r"^\s*(\d{1,2})\s+(\S.*)$", re.S)
SECTION_MIN_SIZE = 17.0      # page-level titles ("Package and Ship", "Carbon Footprint")
SUBHEAD_MIN_SIZE = 10.5      # bold sub-headings ("Zero Waste to Landfill")
ENDNOTE_CONTINUATION_GAP = 10.0


@dataclass
class Block:
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    kind: str  # "section" | "subhead" | "para"
    footnote_refs: list[int] = field(default_factory=list)


@dataclass
class ParsedDoc:
    path: str
    title: str
    num_pages: int
    blocks: list[Block]
    page_titles: dict[int, str]
    page_text: dict[int, str]          # raw text per page (used to verify vision output)
    endnotes: dict[int, str]
    endnote_refs: dict[int, list[dict]]  # n -> [{"page", "context"}] where the marker appears


def _is_marker(span: dict, body_size: float, prev_text: str) -> bool:
    t = span["text"].strip()
    if not t.isdigit() or len(t) > 2 or not (1 <= int(t) <= 99):
        return False
    if prev_text.rstrip().endswith("CO"):  # CO2 subscript, not a footnote
        return False
    return bool(span["flags"] & 1) or span["size"] < 0.75 * body_size


def _line_text(line: dict, body_size: float) -> tuple[str, list[int]]:
    out, refs = "", []
    for span in line["spans"]:
        if _is_marker(span, body_size, out):
            n = int(span["text"].strip())
            refs.append(n)
            out = out.rstrip() + f"[^{n}] "
        else:
            out += span["text"]
    return out, refs


def _join_lines(lines: list[str]) -> str:
    text = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if text.endswith("-") and not text.endswith(" -") and ln[:1].islower():
            text = text + ln  # "low-\ncarbon" -> "low-carbon": this report doesn't hyphenate words, only compounds
        else:
            text = f"{text} {ln}" if text else ln
    return re.sub(r"[ \t ]+", " ", text).strip()


def _merge_wrapped_lines(items: list[dict]) -> list[dict]:
    """Infographic pages emit one block per visual line, interleaved across columns.
    Re-join blocks that share a left edge and font style and sit directly below each other."""
    groups: list[dict] = []
    for it in sorted(items, key=lambda i: (round(i["bbox"][1]), i["bbox"][0])):
        for g in reversed(groups):
            overlap = min(g["bbox"][2], it["bbox"][2]) - max(g["bbox"][0], it["bbox"][0])
            narrower = min(g["bbox"][2] - g["bbox"][0], it["bbox"][2] - it["bbox"][0])
            same_col = overlap > 0.5 * narrower  # handles centred infographic text too
            same_style = abs(g["size"] - it["size"]) < 0.6 and g["bold"] == it["bold"]
            gap = it["bbox"][1] - g["bbox"][3]
            if same_col and same_style and -1 <= gap <= 0.6 * it["size"]:
                g["text"] = _join_lines([g["text"], it["text"]])
                g["refs"] += it["refs"]
                g["bbox"] = [min(g["bbox"][0], it["bbox"][0]), g["bbox"][1],
                             max(g["bbox"][2], it["bbox"][2]), it["bbox"][3]]
                break
        else:
            groups.append(dict(it))
    return sorted(groups, key=lambda g: (round(g["bbox"][1]), g["bbox"][0]))


def parse_pdf(path: str) -> ParsedDoc:
    doc = pymupdf.open(path)
    blocks: list[Block] = []
    page_titles: dict[int, str] = {}
    page_text: dict[int, str] = {}
    endnotes: dict[int, str] = {}
    endnote_refs: dict[int, list[dict]] = {}

    for pno, page in enumerate(doc, start=1):
        page_text[pno] = page.get_text()
        height = page.rect.height
        raw_blocks = [b for b in page.get_text("dict")["blocks"] if b.get("lines")]
        sizes = [s["size"] for b in raw_blocks for l in b["lines"] for s in l["spans"] if s["text"].strip()]
        body_size = sorted(sizes)[len(sizes) // 2] if sizes else 10.0

        is_endnote_page = any(
            "Endnotes" in "".join(s["text"] for l in b["lines"] for s in l["spans"]) for b in raw_blocks[:3]
        ) or "Endnotes" in page_text[pno][:200]
        current_note: int | None = None
        last_note_bottom = 0.0

        items: list[dict] = []
        for b in sorted(raw_blocks, key=lambda b: (round(b["bbox"][1]), b["bbox"][0])):
            lines, refs = [], []
            for l in b["lines"]:
                t, r = _line_text(l, body_size)
                lines.append(t)
                refs.extend(r)
            text = _join_lines(lines)
            if b["bbox"][1] > height * 0.9 and RUNNING_HEADER.match(re.sub(r"\s*(\[\^)?\d+\]?\s*$", "", text)):
                continue  # running footer "iMac | Product Environmental Report  <page>"
            if not text or (text.isdigit() and b["bbox"][1] > height * 0.9):
                continue
            spans = [s for l in b["lines"] for s in l["spans"] if s["text"].strip()]
            items.append({
                "bbox": list(b["bbox"]), "text": text, "refs": refs,
                "size": max(s["size"] for s in spans),
                "bold": all(s["flags"] & 16 or "Bold" in s["font"] or "Semibold" in s["font"] for s in spans),
                "first": "".join(s["text"] for s in b["lines"][0]["spans"]),
            })

        if not is_endnote_page:
            items = _merge_wrapped_lines(items)

        for it in items:
            b, text, refs, max_size, bold = it, it["text"], it["refs"], it["size"], it["bold"]
            if is_endnote_page and text != "Endnotes":
                # Endnote markers are the first span of the block (tab + number).
                m = ENDNOTE_START.match(it["first"].replace("\t", " "))
                if m and int(m.group(1)) not in endnotes and int(m.group(1)) == (max(endnotes, default=0) + 1):
                    current_note = int(m.group(1))
                    endnotes[current_note] = re.sub(r"^\[\^\d+\]\s*|^\d+\s+", "", text).strip()
                    last_note_bottom = b["bbox"][3]
                    continue
                if current_note and 0 <= b["bbox"][1] - last_note_bottom <= ENDNOTE_CONTINUATION_GAP:
                    endnotes[current_note] += "\n" + text
                    last_note_bottom = b["bbox"][3]
                    continue

            if max_size >= SECTION_MIN_SIZE and len(text) < 60:
                kind = "section"
                page_titles.setdefault(pno, text)
            elif bold and max_size >= SUBHEAD_MIN_SIZE and len(text) < 70:
                kind = "subhead"
            else:
                kind = "para"
            blocks.append(Block(pno, tuple(b["bbox"]), text, kind, refs))
            for n in refs:
                ctx = re.sub(r"\[\^\d+\]", "", text)
                endnote_refs.setdefault(n, []).append({"page": pno, "context": ctx[:160]})

    return ParsedDoc(
        path=path,
        title=doc.metadata.get("title") or "Document",
        num_pages=doc.page_count,
        blocks=blocks,
        page_titles=page_titles,
        page_text=page_text,
        endnotes=endnotes,
        endnote_refs=endnote_refs,
    )


def render_pages(path: str, out_dir, dpi: int) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for pno, page in enumerate(pymupdf.open(path), start=1):
        p = out_dir / f"page_{pno:02d}.png"
        if not p.exists():
            page.get_pixmap(dpi=dpi).save(p)
        paths.append(p)
    return paths
