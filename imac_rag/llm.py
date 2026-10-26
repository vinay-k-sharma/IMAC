"""Thin wrapper around the Gemini SDK: generation with model fallback, retries, and embeddings."""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
from google import genai
from google.genai import types

from . import config


@lru_cache(maxsize=1)
def client() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set (put it in .env)")
    return genai.Client(api_key=config.GEMINI_API_KEY)


def generate(
    contents,
    *,
    models: list[str] | None = None,
    system: str | None = None,
    temperature: float = 0.0,
    json_schema: dict | None = None,
    retries: int = 3,
) -> tuple[str, str]:
    """Return (text, model_used). Tries each model in order, with exponential backoff per model."""
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system,
        response_mime_type="application/json" if json_schema else None,
        response_json_schema=json_schema,
    )
    errors: list[str] = []
    for model in models or config.GEN_MODELS:
        for attempt in range(retries):
            try:
                resp = client().models.generate_content(model=model, contents=contents, config=cfg)
                if resp.text:
                    return resp.text, model
                errors.append(f"{model}: empty response")
            except Exception as e:  # noqa: BLE001 - SDK raises many error types
                msg = str(e)
                errors.append(f"{model}: {msg[:200]}")
                if "NOT_FOUND" in msg or ("RESOURCE_EXHAUSTED" in msg and "quota" in msg.lower()):
                    break  # model unavailable or out of quota: move to the next one
                time.sleep(2 ** attempt)
    raise RuntimeError("All models failed:\n" + "\n".join(errors))


def image_part(png_bytes: bytes) -> types.Part:
    return types.Part.from_bytes(data=png_bytes, mime_type="image/png")


def embed(texts: list[str], task_type: str) -> np.ndarray:
    """Embed texts; returns L2-normalised float32 matrix (n, EMBED_DIM)."""
    out: list[list[float]] = []
    for i in range(0, len(texts), 50):
        batch = texts[i : i + 50]
        for attempt in range(4):
            try:
                resp = client().models.embed_content(
                    model=config.EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=config.EMBED_DIM),
                )
                out.extend(e.values for e in resp.embeddings)
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    mat = np.asarray(out, dtype=np.float32)
    return mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9, None)
