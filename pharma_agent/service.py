"""The one entry point used by the API, the CLI, the notebooks and the evaluation.

    agent = await PharmaAgent.create()                 # build everything from configs/default.yaml
    response = await agent.ask("Does metformin have a boxed warning?")
    print(response.answer.answer)
"""

import time
import uuid

from .agents import build_graph, build_specialists, unique_evidence
from .config import load_config
from .http import HttpClient
from .llm import LLMRoles, build_llms
from .schemas import AskResponse
from .store import Store
from .wiring import build_http, build_store, load_tools


class PharmaAgent:
    def __init__(self, cfg: dict, store: Store, tools: list, specialists: dict, graph):
        self.cfg = cfg
        self.store = store
        self.tools = tools
        self.specialists = specialists
        self.graph = graph

    @classmethod
    async def create(
        cls,
        cfg: dict | None = None,
        llms: LLMRoles | None = None,
        store: Store | None = None,
        http: HttpClient | None = None,
    ) -> "PharmaAgent":
        cfg = cfg or load_config()
        store = store or build_store(cfg)
        http = http or build_http(cfg, store)
        tools = await load_tools(cfg, http)
        llms = llms or build_llms(cfg["llm"])
        specialists = build_specialists(cfg["agents"], tools, llms.specialist, cfg["agent_limits"]["max_steps"])
        graph = build_graph(llms.supervisor, llms.synthesizer, specialists)
        return cls(cfg, store, tools, specialists, graph)

    async def run_graph(self, question: str) -> dict:
        """The raw final LangGraph state (useful in notebooks to see every step)."""
        return await self.graph.ainvoke({"question": question, "findings": [], "evidence": [], "tool_calls": []})

    async def ask(self, question: str) -> AskResponse:
        started = time.perf_counter()
        state = await self.run_graph(question)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        evidence = unique_evidence(state.get("evidence", []))
        response = AskResponse(
            run_id=uuid.uuid4().hex[:12],
            question=question,
            answer=state["answer"],
            plan=state["plan"],
            findings=state.get("findings", []),
            evidence=evidence,
            guardrail=state["guardrail"],
            latency_ms=latency_ms,
        )
        # ACID: the run and all its evidence rows are saved together or not at all.
        self.store.save_run(
            run_id=response.run_id,
            question=question,
            answer=response.answer.model_dump(),
            plan=response.plan.model_dump(),
            guardrail=response.guardrail.model_dump(),
            latency_ms=latency_ms,
            evidence=evidence,
        )
        return response
