"""Command line: ingest, ask, chat, eval, serve.

    python -m imac_rag.cli ingest [--force]
    python -m imac_rag.cli ask "What is the sleep power at 115V?" [--show-sources]
    python -m imac_rag.cli chat
    python -m imac_rag.cli eval [--retrieval-only] [--no-judge] [--questions FILE]
    python -m imac_rag.cli serve [--port 8000]
"""
from __future__ import annotations

import json

import typer

app = typer.Typer(add_completion=False, help="Multimodal RAG over the Apple iMac Product Environmental Report.")


def _print_answer(r: dict, show_sources: bool) -> None:
    typer.echo("\n" + r["answer"] + "\n")
    typer.secho("Sources:", bold=True)
    for c in r["citations"]:
        snippet = c["text"].replace("\n", " ")[:110]
        typer.echo(f"  [{c['n']}] p.{c['page']} {c['type']:<8} {c['chunk_id']:<16} {snippet}")
    meta = f"model={r['model']}  latency={r['latency_s']['total']}s"
    if r["image_pages"]:
        meta += f"  page images sent: {r['image_pages']}"
    typer.secho(meta, dim=True)
    if show_sources:
        typer.echo(json.dumps(r["retrieved"], indent=1))


@app.command()
def ingest(pdf: str = typer.Option(None, help="PDF path (default: first PDF in data/raw)"),
           force: bool = typer.Option(False, help="Rebuild even if unchanged")):
    """Parse, extract tables/charts with vision, chunk and index the report."""
    from .ingest.pipeline import run
    run(pdf, force=force)


@app.command()
def ask(question: str, show_sources: bool = typer.Option(False, help="Print the full retrieval ranking"),
        no_images: bool = typer.Option(False, help="Text-only generation (no page images)")):
    """Ask one question."""
    from .generate import answer
    _print_answer(answer(question, use_images=not no_images), show_sources)


@app.command()
def chat():
    """Interactive multi-turn chat."""
    from .generate import answer
    typer.secho("iMac Environmental Report RAG - ask a question (Ctrl-C or 'exit' to quit)", bold=True)
    history: list[dict] = []
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        r = answer(q, history=history)
        _print_answer(r, False)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": r["answer"]}]
        history = history[-12:]


@app.command("eval")
def eval_(questions: str = typer.Option(None, help="JSONL eval file (default data/eval/sample_questions.jsonl)"),
          retrieval_only: bool = typer.Option(False, help="Skip generation"),
          no_judge: bool = typer.Option(False, help="Skip the LLM judge"),
          ids: str = typer.Option(None, help="Comma-separated question ids to run, e.g. q17,q18")):
    """Run retrieval ablation + end-to-end evaluation."""
    from evaluation.run import main
    main(questions, retrieval_only=retrieval_only, use_judge=not no_judge,
         ids=ids.split(",") if ids else None)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Start the API + chat UI."""
    import uvicorn
    uvicorn.run("imac_rag.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
