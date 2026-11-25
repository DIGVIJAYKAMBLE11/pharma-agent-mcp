"""Typed data that moves between layers. Pydantic validates it at every boundary.

Evidence      what a data source returns (one label section, one trial, one article)
RoutePlan     what the supervisor LLM decides (which specialists, with which instruction)
Finding       what one specialist agent reports back
FinalAnswer   the structured answer the user gets, with citations to Evidence ids
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    id: str = Field(description="Stable citation id, e.g. 'pmid:38000000' or 'nct:NCT06399835'")
    source: str = Field(description="Which data source produced it, e.g. 'pubmed'")
    title: str
    text: str
    url: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    agent: str = Field(description="Name of the specialist agent to call")
    instruction: str = Field(description="What this agent should find out, in one sentence")


class RoutePlan(BaseModel):
    reasoning: str = Field(description="One sentence on why these agents were chosen")
    tasks: list[Task] = Field(description="One task per specialist agent that is needed")


class Finding(BaseModel):
    agent: str
    instruction: str
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class KeyPoint(BaseModel):
    text: str = Field(description="One factual statement")
    citations: list[str] = Field(description="Evidence ids that support the statement, copied exactly")


class FinalAnswer(BaseModel):
    answer: str = Field(description="Direct answer to the question in 2-4 sentences")
    key_points: list[KeyPoint] = Field(description="3-6 supporting points, each with citations")
    confidence: Literal["low", "medium", "high"] = Field(description="How well the evidence answers the question")
    caveats: list[str] = Field(default_factory=list, description="Limits of the evidence")


class GuardrailReport(BaseModel):
    evidence_count: int
    valid_citations: int
    invalid_citations: list[str]
    unsupported_points: int


class AskResponse(BaseModel):
    run_id: str
    question: str
    answer: FinalAnswer
    plan: RoutePlan
    findings: list[Finding]
    evidence: list[Evidence]
    guardrail: GuardrailReport
    latency_ms: float
