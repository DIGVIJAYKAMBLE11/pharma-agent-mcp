"""Evaluation harness: run a question set through the agent and score each step.

Metrics per question (then averaged):
    routing_exact        supervisor chose exactly the expected specialists
    routing_recall       share of expected specialists it chose
    source_recall        share of expected data sources that returned evidence
    citation_precision   share of the answer's citations that were real evidence ids (before the guardrail)
    grounded_points      share of key points with at least one valid citation
    term_hit             share of expected key terms present in the answer text
    latency_ms           end-to-end time (p50 / p95 in the summary)

Results are written to eval/results/<timestamp>.json; every run is also in the SQLite run log.
"""

import asyncio
import json
import statistics
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from .schemas import AskResponse


def load_questions(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# LLMs write non-breaking hyphens/spaces ("DPP‑4", "heart‑failure"). Normalise the answer and the
# expected term the same way: NFKC, then every dash or hyphen becomes a space.
DASHES = dict.fromkeys(map(ord, "-‐‑‒–—−"), " ")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(DASHES)
    return " ".join(text.lower().split())


def answer_text(resp: AskResponse) -> str:
    return normalise(" ".join([resp.answer.answer, *(p.text for p in resp.answer.key_points)]))


def score(item: dict, resp: AskResponse) -> dict:
    chosen = {t.agent for t in resp.plan.tasks}
    expected_agents = set(item["expected_agents"])
    sources = {e.source for e in resp.evidence}
    expected_sources = set(item.get("expected_sources", []))
    g = resp.guardrail
    exact = g.valid_citations - len(g.repaired_citations)  # repaired ids count as model mistakes
    total_citations = g.valid_citations + len(g.invalid_citations)
    points = resp.answer.key_points
    terms = item.get("expected_terms", [])
    text = answer_text(resp)
    return {
        "routing_exact": float(chosen == expected_agents),
        "routing_recall": len(chosen & expected_agents) / len(expected_agents),
        "source_recall": len(sources & expected_sources) / len(expected_sources) if expected_sources else 1.0,
        "citation_precision": exact / total_citations if total_citations else 0.0,
        "grounded_points": (len(points) - g.unsupported_points) / len(points) if points else 0.0,
        "term_hit": sum(normalise(t) in text for t in terms) / len(terms) if terms else 1.0,
        "latency_ms": resp.latency_ms,
    }


def summarise(rows: list[dict]) -> dict:
    ok = [r["scores"] for r in rows if "scores" in r]
    summary: dict = {"questions": len(rows), "answered": len(ok), "errors": len(rows) - len(ok)}
    if not ok:
        return summary
    for key in ["routing_exact", "routing_recall", "source_recall", "citation_precision", "grounded_points", "term_hit"]:
        summary[key] = round(statistics.mean(r[key] for r in ok), 3)
    lat = sorted(r["latency_ms"] for r in ok)
    summary["latency_p50_s"] = round(statistics.median(lat) / 1000, 1)
    summary["latency_p95_s"] = round(lat[min(len(lat) - 1, int(0.95 * len(lat)))] / 1000, 1)
    return summary


async def run_eval(agent, questions: list[dict], delay_seconds: float = 0, log=print) -> dict:
    rows = []
    for i, item in enumerate(questions, 1):
        started = time.perf_counter()
        try:
            resp = await agent.ask(item["question"])
            row = {
                "id": item["id"], "question": item["question"], "run_id": resp.run_id,
                "chosen_agents": sorted(t.agent for t in resp.plan.tasks),
                "expected_agents": item["expected_agents"],
                "evidence_sources": sorted({e.source for e in resp.evidence}),
                "answer": resp.answer.answer, "confidence": resp.answer.confidence,
                "guardrail": resp.guardrail.model_dump(), "scores": score(item, resp),
            }
            log(f"[{i}/{len(questions)}] {item['id']}: routing={row['scores']['routing_exact']:.0f} "
                f"cit_prec={row['scores']['citation_precision']:.2f} {resp.latency_ms / 1000:.1f}s")
        except Exception as exc:  # noqa: BLE001  one failure must not stop the whole evaluation
            row = {"id": item["id"], "question": item["question"], "error": f"{type(exc).__name__}: {exc}"[:500],
                   "seconds": round(time.perf_counter() - started, 1)}
            log(f"[{i}/{len(questions)}] {item['id']}: ERROR {row['error'][:120]}")
        rows.append(row)
        if delay_seconds and i < len(questions):
            await asyncio.sleep(delay_seconds)
    return {"summary": summarise(rows), "rows": rows}


def save_report(report: dict, results_dir: str | Path, cfg: dict) -> Path:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = {"llm": cfg["llm"], "tool_mode": cfg["tools"]["mode"], **report}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # atomic rename: a half-written report never appears
    return path
