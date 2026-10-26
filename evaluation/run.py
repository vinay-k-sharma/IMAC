"""Evaluation harness.

Retrieval (no LLM generation, cheap):
  hit@k  - an evidence string for the question appears in the retrieved context
           (chunk text or the endnotes linked to it) within the top k
  MRR    - reciprocal rank of the first chunk containing evidence
  run for several configurations as an ablation (hybrid / dense / bm25, and a
  "no vision layer" baseline that drops table/figure/callout/summary chunks)

End-to-end (generation):
  keyword accuracy - every must_include group has at least one alternative in the answer
  judge accuracy   - an LLM judge compares answer vs reference answer
  citation validity, latency
"""
from __future__ import annotations

import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

from imac_rag import config, llm
from imac_rag.generate import answer
from imac_rag.retrieve import retrieve

VISION_TYPES = frozenset({"table", "figure", "callout", "summary"})
ABLATIONS = {
    "hybrid (full)": dict(mode="hybrid"),
    "dense only": dict(mode="dense"),
    "bm25 only": dict(mode="bm25"),
    "hybrid, no vision layer": dict(mode="hybrid", exclude_types=VISION_TYPES),
}
RESULTS_DIR = config.ROOT / "evaluation" / "results"

JUDGE_PROMPT = """You are grading a RAG system's answer about the Apple iMac Product Environmental Report.

Question: {q}
Reference answer: {ref}
System answer: {ans}

Is the system answer correct and consistent with the reference? Extra correct detail is fine. Approximate chart values are fine if labelled approximate. For questions whose reference says the report does not provide the information, the answer is correct only if it declines rather than inventing facts.
Return JSON: {{"correct": true|false, "reason": "<one sentence>"}}"""


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("–", "-").replace("—", "-").replace("**", ""))


def load_questions(path: Path = config.EVAL_FILE) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def evidence_rank(hits, evidence: list[str]) -> int | None:
    if not evidence:
        return None
    for rank, h in enumerate(hits, start=1):
        ctx = norm(h.chunk.text + " " + " ".join(h.linked_endnotes.values()))
        if any(norm(e) in ctx for e in evidence):
            return rank
    return None


def eval_retrieval(questions: list[dict], ks=(1, 3, 5, 8)) -> dict:
    out = {}
    answerable = [q for q in questions if q.get("evidence")]
    for name, kw in ABLATIONS.items():
        ranks = [evidence_rank(retrieve(q["question"], top_k=max(ks), **kw), q["evidence"]) for q in answerable]
        out[name] = {
            **{f"hit@{k}": round(sum(r is not None and r <= k for r in ranks) / len(ranks), 3) for k in ks},
            "mrr": round(sum(1 / r for r in ranks if r) / len(ranks), 3),
            "misses": [q["id"] for q, r in zip(answerable, ranks) if r is None],
        }
    return out


def keyword_ok(ans: str, groups: list[list[str]]) -> bool:
    a = norm(ans)
    return all(any(norm(alt) in a for alt in g) for g in groups)


def judge(q: dict, ans: str) -> dict:
    try:
        raw, _ = llm.generate(JUDGE_PROMPT.format(q=q["question"], ref=q["reference"], ans=ans),
                              models=config.JUDGE_MODELS,
                              json_schema={"type": "object", "properties": {"correct": {"type": "boolean"}, "reason": {"type": "string"}},
                                           "required": ["correct", "reason"]})
        return json.loads(raw)
    except Exception as e:  # noqa: BLE001
        return {"correct": None, "reason": f"judge failed: {e}"[:200]}


def eval_e2e(questions: list[dict], use_judge: bool = True, log=print) -> dict:
    rows = []
    for q in questions:
        t = time.time()
        error = None
        try:
            r = answer(q["question"])
        except Exception as e:  # noqa: BLE001 - quota/API failures are reported, not scored
            error = str(e)[:300]
            r = {"answer": "", "citations": [], "invalid_citations": [], "retrieved": [],
                 "model": None, "latency_s": {"total": round(time.time() - t, 2)}}
        kw = keyword_ok(r["answer"], q["must_include"]) if not error else None
        j = judge(q, r["answer"]) if use_judge and not error else {"correct": None, "reason": ""}
        cited_pages = sorted({c["page"] for c in r["citations"]})
        rows.append({
            "id": q["id"], "error": error, "modality": q["modality"], "question": q["question"], "reference": q["reference"],
            "answer": r["answer"], "keyword_ok": kw, "judge_ok": j["correct"], "judge_reason": j["reason"],
            "cited_pages": cited_pages, "n_citations": len(r["citations"]),
            "invalid_citations": r["invalid_citations"], "model": r["model"], "latency_s": r["latency_s"]["total"],
        })
        if error:
            log(f"{q['id']} [{q['modality']:<15}] API ERROR (not scored): {error[:120]}")
            continue
        log(f"{q['id']} [{q['modality']:<15}] kw={'Y' if kw else 'N'} judge={ {True: 'Y', False: 'N', None: '?'}[j['correct']] } "
            f"{r['latency_s']['total']:>5.1f}s  {q['question'][:70]}")

    by_mod: dict[str, list] = defaultdict(list)
    for row in rows:
        by_mod[row["modality"].split("+")[0]].append(row)

    def rate(rs, key):
        vals = [r[key] for r in rs if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "keyword_accuracy": rate(rows, "keyword_ok"),
        "judge_accuracy": rate(rows, "judge_ok"),
        "by_modality": {m: {"n": len(rs), "keyword": rate(rs, "keyword_ok"), "judge": rate(rs, "judge_ok")}
                        for m, rs in sorted(by_mod.items())},
        "answers_with_citations": round(sum(r["n_citations"] > 0 for r in rows if r["modality"] != "negative" and not r["error"])
                                        / max(1, sum(r["modality"] != "negative" and not r["error"] for r in rows)), 3),
        "invalid_citations": sum(len(r["invalid_citations"]) for r in rows),
        "api_errors": [r["id"] for r in rows if r["error"]],
        "latency_median_s": round(statistics.median(r["latency_s"] for r in rows), 2),
        "rows": rows,
    }


def main(path: str | None = None, retrieval_only: bool = False, use_judge: bool = True,
         ids: list[str] | None = None, log=print) -> dict:
    questions = load_questions(Path(path) if path else config.EVAL_FILE)
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    log(f"Evaluating {len(questions)} questions\n\n== Retrieval ablation ==")
    report = {"retrieval": eval_retrieval(questions)}
    for name, m in report["retrieval"].items():
        log(f"{name:<26} hit@1={m['hit@1']:.2f} hit@3={m['hit@3']:.2f} hit@5={m['hit@5']:.2f} "
            f"hit@8={m['hit@8']:.2f} MRR={m['mrr']:.2f} misses={m['misses']}")
    if not retrieval_only:
        log("\n== End-to-end ==")
        report["e2e"] = eval_e2e(questions, use_judge=use_judge, log=log)
        e = report["e2e"]
        log(f"\nkeyword accuracy={e['keyword_accuracy']}  judge accuracy={e['judge_accuracy']}  "
            f"answers with citations={e['answers_with_citations']}  invalid citations={e['invalid_citations']}  "
            f"median latency={e['latency_median_s']}s  api errors (unscored)={e['api_errors']}")
        for m, v in e["by_modality"].items():
            log(f"  {m:<12} n={v['n']:<3} keyword={v['keyword']}  judge={v['judge']}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log(f"\nSaved {out}")
    return report


if __name__ == "__main__":
    main()
