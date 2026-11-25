"""The full agent on the real Groq models. Kept to a few questions to respect the free tier."""

import pytest
from fastapi.testclient import TestClient

from pharma_agent.api import create_app
from pharma_agent.config import load_config
from pharma_agent.service import PharmaAgent

pytestmark = [pytest.mark.groq, pytest.mark.live]


async def test_label_question_routes_to_label_agent_with_valid_citations(agent):
    r = await agent.ask("Does metformin have a boxed warning? What is it about?")
    assert "label_agent" in {t.agent for t in r.plan.tasks}
    assert any(e.source == "openfda_labels" for e in r.evidence)
    assert "lactic acidosis" in (r.answer.answer + " ".join(p.text for p in r.answer.key_points)).lower()
    assert r.guardrail.invalid_citations == []
    assert r.answer.key_points and all(p.citations for p in r.answer.key_points)
    assert agent.store.get_run(r.run_id)["question"].startswith("Does metformin")


async def test_multi_agent_question_uses_two_specialists(agent):
    r = await agent.ask("What are the boxed warnings for warfarin, and are there recruiting clinical trials of warfarin?")
    chosen = {t.agent for t in r.plan.tasks}
    assert {"label_agent", "trials_agent"} <= chosen
    assert {"openfda_labels", "clinicaltrials"} <= {e.source for e in r.evidence}
    assert r.guardrail.valid_citations > 0


async def test_tools_served_over_mcp_stdio(cfg, store, groq_key):
    mcp_cfg = load_config(overrides={"store.path": ":memory:", "tools.mode": "mcp", "tools.mcp.transport": "stdio"})
    agent = await PharmaAgent.create(mcp_cfg, store=store)
    assert {t.name for t in agent.tools} == {
        "search_drug_labels", "get_adverse_event_counts", "search_clinical_trials", "search_pubmed",
    }
    r = await agent.ask("Find recent PubMed research on SGLT2 inhibitors in heart failure.")
    assert any(e.source == "pubmed" for e in r.evidence)
    assert r.guardrail.invalid_citations == []


def test_api_endpoints(cfg, store, groq_key):
    async def factory():
        return await PharmaAgent.create(cfg, store=store)

    with TestClient(create_app(factory)) as client:
        assert client.get("/health").json()["llm_provider"] == "groq"
        assert len(client.get("/tools").json()) == 4
        assert {a["name"] for a in client.get("/agents").json()} == {"label_agent", "trials_agent", "literature_agent"}
        assert client.post("/ask", json={"question": "hi"}).status_code == 422  # too short
        body = client.post("/ask", json={"question": "What are the contraindications of apixaban?"}).json()
        assert body["guardrail"]["invalid_citations"] == []
        assert client.get(f"/runs/{body['run_id']}").json()["id"] == body["run_id"]
        assert client.get("/runs/does-not-exist").status_code == 404
