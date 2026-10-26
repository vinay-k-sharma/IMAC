"""Central configuration. Everything tunable lives here and can be overridden via env vars."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PAGES_DIR = PROCESSED_DIR / "pages"        # rendered page PNGs (vision input + UI previews)
VISION_DIR = PROCESSED_DIR / "vision"      # cached vision-LLM extractions, one JSON per page
INDEX_DIR = PROCESSED_DIR / "index"
EVAL_FILE = ROOT / "data" / "eval" / "sample_questions.jsonl"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_PROVIDER = os.getenv("GPO_RAG_LLM_PROVIDER", "gemini")

# Generation / vision models, tried in order until one succeeds. On the Gemini free tier each
# model has its own daily request quota, so a longer chain means more answers per day.
GEN_MODELS = os.getenv("RAG_GEN_MODELS", "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-3-flash-preview,gemini-3.5-flash-lite,gemini-3.1-flash-lite").split(",")
VISION_MODELS = os.getenv("RAG_VISION_MODELS", "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash").split(",")
JUDGE_MODELS = os.getenv("RAG_JUDGE_MODELS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.6-flash").split(",")

EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("RAG_EMBED_DIM", "768"))

RENDER_DPI = 150

# Chunking (token counts are approximated as words * 1.3)
CHUNK_TARGET_TOKENS = 350
CHUNK_MAX_TOKENS = 450
CHUNK_OVERLAP_TOKENS = 50

# Retrieval
TOP_K = 8
RRF_K = 60
MAX_PAGE_IMAGES = 2  # page images sent to the LLM alongside text for visual grounding

# Bump to invalidate cached vision extractions when the prompt/schema changes.
VISION_PROMPT_VERSION = "v5"
