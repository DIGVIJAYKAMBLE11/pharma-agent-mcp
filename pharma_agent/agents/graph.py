"""The LangGraph workflow.

    START -> supervisor --(one Send per task, run in parallel)--> specialist -> synthesize -> guardrail -> END

    supervisor  LLM with structured output (RoutePlan): which specialists, with what instruction
    specialist  runs one ReAct agent; its findings and evidence are appended to the state
    synthesize  LLM with structured output (FinalAnswer) over all findings + evidence
    guardrail   drops citations that are not real evidence ids (see guardrail.py)

Every LLM step has a fallback, so one bad model reply degrades the answer instead of failing it.
"""

import logging
import operator
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..schemas import Evidence, FinalAnswer, Finding, GuardrailReport, RoutePlan, Task
from .guardrail import check_citations
from .prompts import SUPERVISOR, SYNTHESIZER
from .specialist import Specialist

log = logging.getLogger(__name__)
EXCERPT_CHARS = 350


class GraphState(TypedDict, total=False):
    question: str
    plan: RoutePlan
    findings: Annotated[list[Finding], operator.add]   # each specialist appends
    evidence: Annotated[list[Evidence], operator.add]  # each specialist appends
    answer: FinalAnswer
    guardrail: GuardrailReport
    tool_calls: Annotated[list[dict], operator.add]


class SpecialistInput(TypedDict):
    task: Task
    question: str


def unique_evidence(evidence: list[Evidence]) -> list[Evidence]:
    return list({e.id: e for e in evidence}.values())


def build_graph(
    supervisor_llm: BaseChatModel,
    synthesizer_llm: BaseChatModel,
    specialists: dict[str, Specialist],
    structured_method: str = "json_schema",
):
    """`structured_method="json_schema"`: on Groq, function_calling failed ~1 in 6 routing calls
    ("model did not call a tool"); json_schema went 6 for 6 in the same test."""
    catalog = "\n".join(f"- {name}: {s.spec.description}" for name, s in specialists.items())

    async def supervisor(state: GraphState) -> dict:
        question = state["question"]
        try:
            plan = await supervisor_llm.with_structured_output(RoutePlan, method=structured_method).ainvoke(
                [SystemMessage(SUPERVISOR.format(catalog=catalog)), HumanMessage(question)]
            )
            tasks = list({t.agent: t for t in plan.tasks if t.agent in specialists}.values())
            reasoning = plan.reasoning
        except Exception as exc:  # noqa: BLE001
            log.warning("supervisor failed, using every agent: %s", exc)
            tasks, reasoning = [], f"routing failed ({type(exc).__name__}); used every agent"
        if not tasks:
            tasks = [Task(agent=name, instruction=question) for name in specialists]
            reasoning = reasoning or "no valid agent chosen; used every agent"
        return {"plan": RoutePlan(reasoning=reasoning, tasks=tasks)}

    def dispatch(state: GraphState) -> list[Send]:
        return [Send("specialist", {"task": task, "question": state["question"]}) for task in state["plan"].tasks]

    async def specialist(inp: SpecialistInput) -> dict:
        task = inp["task"]
        try:
            text, evidence, calls = await specialists[task.agent].run(task.instruction, inp["question"])
        except Exception as exc:  # noqa: BLE001
            log.warning("%s failed: %s", task.agent, exc)
            text, evidence, calls = f"Agent failed: {type(exc).__name__}: {str(exc)[:200]}", [], []
        finding = Finding(agent=task.agent, instruction=task.instruction, text=text,
                          evidence_ids=[e.id for e in evidence])
        return {"findings": [finding], "evidence": evidence,
                "tool_calls": [{"agent": task.agent, **c} for c in calls]}

    async def synthesize(state: GraphState) -> dict:
        evidence = unique_evidence(state.get("evidence", []))
        findings = state.get("findings", [])
        if not evidence:
            return {"answer": FinalAnswer(
                answer="I could not find evidence in the connected sources to answer this question.",
                key_points=[], confidence="low",
                caveats=["No matching records were returned by the data sources."],
            )}
        prompt = SYNTHESIZER.format(
            question=state["question"],
            findings="\n".join(f"- {f.agent}: {f.text}" for f in findings),
            evidence="\n".join(f"[{e.id}] {e.title}: {e.text[:EXCERPT_CHARS]}" for e in evidence),
        )
        try:
            answer = await synthesizer_llm.with_structured_output(FinalAnswer, method=structured_method).ainvoke([HumanMessage(prompt)])
        except Exception as exc:  # noqa: BLE001
            log.warning("synthesizer failed, returning raw findings: %s", exc)
            answer = FinalAnswer(
                answer=" ".join(f.text for f in findings)[:1200],
                key_points=[], confidence="low",
                caveats=[f"Structured answer failed ({type(exc).__name__}); showing raw specialist findings."],
            )
        return {"answer": answer}

    def guardrail(state: GraphState) -> dict:
        answer, report = check_citations(state["answer"], unique_evidence(state.get("evidence", [])))
        return {"answer": answer, "guardrail": report}

    g = StateGraph(GraphState)
    g.add_node("supervisor", supervisor)
    g.add_node("specialist", specialist)
    g.add_node("synthesize", synthesize)
    g.add_node("guardrail", guardrail)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", dispatch, ["specialist"])
    g.add_edge("specialist", "synthesize")
    g.add_edge("synthesize", "guardrail")
    g.add_edge("guardrail", END)
    return g.compile()
