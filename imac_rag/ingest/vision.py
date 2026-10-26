"""Vision layer: a multimodal LLM reads each rendered page and returns structured
tables, figures (charts / diagrams / infographics) and stat callouts as JSON.

Why: in this report the most important numbers live in places the text layer
mangles - a stacked waterfall chart whose labels come out as a word soup, tables
whose zebra striping splits them into fragments, and big-number callouts whose
captions are separated from their values.

Guarding against hallucination: the page's own text layer is passed in with the
image, and every number the model reports as "printed" is checked against that
text layer afterwards (``verify``). Values the model reads off the geometry of a
chart are labelled "estimated" and surfaced as approximate downstream.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .. import config, llm

STR = {"type": "string"}
INT_LIST = {"type": "array", "items": {"type": "integer"}}

SCHEMA = {
    "type": "object",
    "properties": {
        "page_summary": {**STR, "description": "2-3 sentences: what this page communicates."},
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": STR,
                    "columns": {"type": "array", "items": STR},
                    "rows": {"type": "array", "items": {"type": "array", "items": STR}},
                    "notes": {**STR, "description": "Notes printed under/near the table, verbatim."},
                    "footnote_refs": INT_LIST,
                },
                "required": ["title", "columns", "rows"],
            },
        },
        "figures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": STR,
                    "figure_type": {**STR, "description": "e.g. stacked bar chart, waterfall chart, process diagram, icon grid, infographic"},
                    "description": {**STR, "description": "Detailed description a blind reader could answer questions from: structure, every label, ordering, relative sizes, comparisons, the takeaway."},
                    "data_points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": STR,
                                "value": STR,
                                "source": {"type": "string", "enum": ["printed", "estimated"]},
                            },
                            "required": ["label", "value", "source"],
                        },
                    },
                    "footnote_refs": INT_LIST,
                },
                "required": ["title", "figure_type", "description", "data_points"],
            },
        },
        "callouts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "value": STR,
                    "statement": {**STR, "description": "Full sentence joining the big value with its caption."},
                    "footnote_refs": INT_LIST,
                },
                "required": ["value", "statement"],
            },
        },
    },
    "required": ["page_summary", "tables", "figures", "callouts"],
}

PROMPT = """You are extracting structured data from page {page} of the Apple iMac Product Environmental Report so it can be searched by a retrieval system.

The page's machine-readable text layer is given below (reading order may be scrambled; it is the ground truth for spelling and printed numbers):
<text_layer>
{text}
</text_layer>

Extract from the IMAGE:
1. tables - every table, with clean column headers (include multi-level headers such as voltage columns, merged into each column name) and one row per printed row. Keep units exactly as printed (e.g. "0.57W", "346 kg CO2e", "<1%").
2. figures - every chart, diagram or infographic (ignore decorative photos and logos unless they carry information, e.g. a certification badge). Skip the small circular life-cycle wheel (Design and Source / Make / Package and Ship / Use / Recover) that is used as a section-navigation marker on most pages, UNLESS it is the main content of the page with descriptive text for each stage. For charts list every segment/bar with its label. Mark a value "printed" only if it is literally printed on the page; if you infer a magnitude from bar heights, mark it "estimated" and give an approximate value (e.g. "~150 kg"). For stacked/waterfall bars, also estimate EVERY segment's value from its height relative to the printed totals (e.g. label "Baseline emissions - Electricity for product use", value "~140 kg", source "estimated"). Describe what each column/bar contains and how the parts relate (e.g. which segments are reductions).
3. callouts - large highlighted statistics (e.g. "76%" above "recycled content in fiber packaging") turned into a full sentence.

Small superscript numbers are footnote references - record them in footnote_refs of the element they attach to. Do not invent content that is not on the page. Return empty arrays when there is nothing of that kind."""


NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _numbers(s: str) -> set[str]:
    return {n.lstrip("0") or "0" for n in NUM_RE.findall(s.replace(",", ""))}


def verify(extraction: dict, page_text: str) -> dict:
    """Annotate each table/data point with whether its numbers appear in the PDF text layer."""
    known = _numbers(page_text)
    for t in extraction.get("tables", []):
        cells = " ".join(" ".join(r) for r in t.get("rows", []))
        missing = sorted(_numbers(cells) - known)
        t["verified"] = not missing
        t["unverified_numbers"] = missing
    for f in extraction.get("figures", []):
        for dp in f.get("data_points", []):
            if dp.get("source") == "printed":
                dp["verified"] = _numbers(dp["value"]) <= known
                if not dp["verified"]:
                    dp["source"] = "estimated"  # claimed printed but not in text layer: treat as estimate
    for c in extraction.get("callouts", []):
        c["verified"] = _numbers(c.get("value", "")) <= known
    return extraction


def extract_page(page: int, png_path: Path, page_text: str, pdf_sha: str) -> dict:
    config.VISION_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.VISION_DIR / f"page_{page:02d}.json"
    key = hashlib.sha256(f"{pdf_sha}:{page}:{config.VISION_PROMPT_VERSION}".encode()).hexdigest()
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("_cache_key") == key:
            return cached

    contents = [llm.image_part(png_path.read_bytes()), PROMPT.format(page=page, text=page_text)]
    raw, model = llm.generate(contents, models=config.VISION_MODELS, json_schema=SCHEMA)
    data = verify(json.loads(raw), page_text)
    data.update({"_cache_key": key, "_model": model, "page": page})
    cache.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data
