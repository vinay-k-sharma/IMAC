"""Hybrid index: Gemini dense embeddings (NumPy) + BM25, persisted under data/processed/index/."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from . import config, llm
from .ingest.chunker import Chunk

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOP = set("the a an of in on to for and or is are was were be by with as at from that this it its what which how does do did".split())


def tokenize(s: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(s.lower().replace("—", " ")) if t not in STOP]


@dataclass
class Index:
    chunks: list[Chunk]
    embeddings: np.ndarray
    bm25: BM25Okapi
    manifest: dict


def manifest_path():
    return config.INDEX_DIR / "manifest.json"


def is_current(pdf_sha: str, vision_version: str) -> bool:
    p = manifest_path()
    if not p.exists():
        return False
    m = json.loads(p.read_text())
    return (m.get("pdf_sha"), m.get("vision_version"), m.get("embed_model"), m.get("embed_dim")) == (
        pdf_sha, vision_version, config.EMBED_MODEL, config.EMBED_DIM)


def build(chunks: list[Chunk], pdf_sha: str, source: str) -> dict:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    emb = llm.embed([c.embed_text for c in chunks], task_type="RETRIEVAL_DOCUMENT")
    np.save(config.INDEX_DIR / "embeddings.npy", emb)
    with open(config.INDEX_DIR / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    manifest = {
        "source": source,
        "pdf_sha": pdf_sha,
        "vision_version": config.VISION_PROMPT_VERSION,
        "embed_model": config.EMBED_MODEL,
        "embed_dim": config.EMBED_DIM,
        "n_chunks": len(chunks),
        "by_type": {t: sum(c.type == t for c in chunks) for t in sorted({c.type for c in chunks})},
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest_path().write_text(json.dumps(manifest, indent=2))
    load.cache_clear()
    return manifest


@lru_cache(maxsize=1)
def load() -> Index:
    if not manifest_path().exists():
        raise RuntimeError("Index not built. Run: python -m imac_rag.cli ingest")
    chunks = [Chunk(**json.loads(l)) for l in open(config.INDEX_DIR / "chunks.jsonl")]
    emb = np.load(config.INDEX_DIR / "embeddings.npy")
    bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks])
    return Index(chunks, emb, bm25, json.loads(manifest_path().read_text()))
