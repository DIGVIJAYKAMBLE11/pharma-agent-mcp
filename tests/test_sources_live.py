"""Each data source against the real public API."""

import json

import pytest

from pharma_agent.mcp_server import build_server
from pharma_agent.tools import evidence_to_json, langchain_tools, parse_evidence

pytestmark = pytest.mark.live


def test_label_returns_the_boxed_warning(sources):
    [ev] = sources["search_drug_labels"].search(drug="metformin", section="warnings")
    assert ev.id.startswith("fda-label:") and ev.metadata["has_boxed_warning"]
    assert "lactic acidosis" in ev.text.lower()


def test_unknown_drug_returns_empty(sources):
    assert sources["search_drug_labels"].search(drug="notarealdrugxyz") == []


def test_faers_counts(sources):
    [ev] = sources["get_adverse_event_counts"].search(drug="atorvastatin", top_n=5)
    assert ev.id == "faers:atorvastatin" and len(ev.metadata["top_terms"]) == 5


def test_clinical_trials_filter_by_status(sources):
    found = sources["search_clinical_trials"].search(intervention="semaglutide", status="RECRUITING", limit=3)
    assert 1 <= len(found) <= 3
    assert all(e.id.startswith("nct:NCT") and e.metadata["status"] == "RECRUITING" for e in found)


def test_clinical_trials_needs_a_query(sources):
    with pytest.raises(ValueError):
        sources["search_clinical_trials"].search()


def test_pubmed_returns_abstracts(sources):
    found = sources["search_pubmed"].search(query="SGLT2 inhibitors heart failure", limit=2)
    assert len(found) == 2 and all(e.id.startswith("pmid:") and len(e.text) > 50 for e in found)


def test_tool_json_round_trips_to_evidence(sources):
    found = sources["search_pubmed"].search(query="metformin lactic acidosis", limit=1)
    assert parse_evidence(evidence_to_json(found)) == found
    assert json.loads(evidence_to_json([]))["note"]


async def test_mcp_and_langchain_tools_share_names_and_schemas(cfg, sources):
    """One function, two transports: the LLM sees the same tools either way."""
    direct = {t.name: t.args for t in langchain_tools(list(sources.values()))}
    server = build_server(cfg)
    mcp_tools = {t.name: t.inputSchema["properties"] for t in await server.list_tools()}
    assert direct.keys() == mcp_tools.keys()
    for name in direct:
        assert direct[name].keys() == mcp_tools[name].keys()


async def test_mcp_tool_call_returns_evidence(cfg):
    server = build_server(cfg)
    result = await server.call_tool("search_drug_labels", {"drug": "warfarin", "section": "warnings"})
    blocks = result[0] if isinstance(result, tuple) else result
    [ev] = parse_evidence([{"type": "text", "text": b.text} for b in blocks])
    assert "bleeding" in ev.text.lower()
