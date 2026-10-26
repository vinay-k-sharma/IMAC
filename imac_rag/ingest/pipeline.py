"""Ingestion orchestrator: PDF -> text layer + rendered pages -> vision layer -> chunks -> hybrid index."""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import config, index
from .chunker import build_chunks
from .pdf_parser import parse_pdf, render_pages
from .vision import extract_page


def default_pdf() -> Path:
    pdfs = sorted(config.RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {config.RAW_DIR}")
    return pdfs[0]


def run(pdf_path: str | None = None, force: bool = False, log=print) -> dict:
    pdf = Path(pdf_path) if pdf_path else default_pdf()
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if not force and index.is_current(sha, config.VISION_PROMPT_VERSION):
        log(f"Index is up to date for {pdf.name} (sha {sha[:10]}); use --force to rebuild.")
        return index.load().manifest

    log(f"[1/5] Parsing text layer: {pdf.name}")
    doc = parse_pdf(str(pdf))
    log(f"      {doc.num_pages} pages, {len(doc.blocks)} blocks, {len(doc.endnotes)} endnotes")

    log(f"[2/5] Rendering pages at {config.RENDER_DPI} dpi")
    images = render_pages(str(pdf), config.PAGES_DIR, config.RENDER_DPI)

    log("[3/5] Vision extraction (tables, charts, callouts) - cached per page")
    def one(p: int):
        return p, extract_page(p, images[p - 1], doc.page_text[p], sha)
    with ThreadPoolExecutor(max_workers=4) as ex:
        vision = dict(ex.map(one, range(1, doc.num_pages + 1)))
    n_tab = sum(len(v["tables"]) for v in vision.values())
    n_fig = sum(len(v["figures"]) for v in vision.values())
    unverified = [(p, t["title"]) for p, v in vision.items() for t in v["tables"] if not t.get("verified")]
    log(f"      {n_tab} tables, {n_fig} figures; tables failing number check: {unverified or 'none'}")

    log("[4/5] Chunking")
    chunks = build_chunks(doc, vision)

    log(f"[5/5] Embedding {len(chunks)} chunks with {config.EMBED_MODEL} + BM25")
    manifest = index.build(chunks, sha, pdf.name)
    log(f"Done: {manifest['by_type']}")
    return manifest


if __name__ == "__main__":
    run()
