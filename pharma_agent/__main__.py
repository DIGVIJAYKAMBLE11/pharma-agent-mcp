"""Command line.

    python -m pharma_agent ask "Does metformin have a boxed warning?"
    python -m pharma_agent ask "..." --mode mcp            # tools via the MCP server (stdio)
    python -m pharma_agent eval --limit 5                   # evaluation on the first 5 questions
"""

import argparse
import asyncio
import logging
import sys

from .config import ROOT, load_config, resolve_path
from .evaluation import load_questions, run_eval, save_report
from .service import PharmaAgent


def build_cfg(args) -> dict:
    overrides = {}
    if args.mode:
        overrides["tools.mode"] = args.mode
    return load_config(overrides=overrides)


async def cmd_ask(args) -> None:
    agent = await PharmaAgent.create(build_cfg(args))
    r = await agent.ask(args.question)
    print(f"\nPlan: {[t.agent for t in r.plan.tasks]}  ({r.plan.reasoning})")
    print(f"\nAnswer ({r.answer.confidence} confidence):\n{r.answer.answer}\n")
    for p in r.answer.key_points:
        print(f"  - {p.text}  {p.citations}")
    print("\nSources:")
    for e in r.evidence:
        print(f"  [{e.id}] {e.title[:90]}\n      {e.url}")
    print(f"\nGuardrail: {r.guardrail.model_dump()}  |  {r.latency_ms / 1000:.1f}s  |  run {r.run_id}")


async def cmd_eval(args) -> None:
    cfg = build_cfg(args)
    questions = load_questions(resolve_path(cfg["eval"]["questions"]))[: args.limit]
    agent = await PharmaAgent.create(cfg)
    delay = cfg["eval"]["delay_seconds"] if args.delay is None else args.delay
    report = await run_eval(agent, questions, delay_seconds=delay)
    path = save_report(report, resolve_path(cfg["eval"]["results_dir"]), cfg)
    print("\nSummary:", report["summary"])
    print("Saved:", path.relative_to(ROOT))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="pharma_agent")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ask", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--mode", choices=["direct", "mcp"])
        if name == "ask":
            p.add_argument("question")
        else:
            p.add_argument("--limit", type=int, default=None)
            p.add_argument("--delay", type=float, default=None)
    args = parser.parse_args()
    asyncio.run(cmd_ask(args) if args.command == "ask" else cmd_eval(args))


if __name__ == "__main__":
    main()
