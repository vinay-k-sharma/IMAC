"""FastAPI app: /ask, /ingest, /health, /pages/{n}.png, and the chat UI at /."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import config, index
from .generate import answer

app = FastAPI(title="iMac Environmental Report RAG")


class Turn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    history: list[Turn] = []
    top_k: int = Field(config.TOP_K, ge=1, le=20)
    use_images: bool = True


@app.get("/")
def ui():
    return FileResponse(config.ROOT / "static" / "index.html")


@app.get("/health")
def health():
    try:
        m = index.load().manifest
        return {"status": "ok", "index": m}
    except RuntimeError as e:
        return {"status": "no_index", "detail": str(e)}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return answer(req.question, top_k=req.top_k, use_images=req.use_images,
                      history=[t.model_dump() for t in req.history])
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e


@app.post("/ingest")
def ingest(force: bool = False):
    from .ingest.pipeline import run
    logs: list[str] = []
    manifest = run(force=force, log=logs.append)
    return {"manifest": manifest, "log": logs}


@app.get("/pages/{page}.png")
def page_image(page: int):
    p = config.PAGES_DIR / f"page_{page:02d}.png"
    if not p.exists():
        raise HTTPException(404, "page not found")
    return FileResponse(p, media_type="image/png")
