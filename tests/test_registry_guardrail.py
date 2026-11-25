"""Registry and citation guardrail: plain logic, no network."""

import pytest

from pharma_agent.agents.guardrail import check_citations
from pharma_agent.agents.prompts import DISCLAIMER
from pharma_agent.registry import SOURCES, Registry
from pharma_agent.schemas import Evidence, FinalAnswer, KeyPoint


def test_builtin_sources_are_registered():
    assert {"openfda_labels", "openfda_events", "clinicaltrials", "pubmed"} <= set(SOURCES.names())


def test_registry_rejects_duplicates_and_explains_unknown_names():
    reg = Registry("thing")
    reg.register("a")(dict)
    with pytest.raises(ValueError):
        reg.register("a")(list)
    reg.register("a", replace=True)(list)
    assert reg.create("a") == []
    with pytest.raises(KeyError, match="Registered: \\['a'\\]"):
        reg.get("b")


def test_guardrail_removes_invented_citations_and_lowers_confidence():
    evidence = [Evidence(id="pmid:1", source="pubmed", title="t", text="x", url="u")]
    answer = FinalAnswer(
        answer="a",
        key_points=[
            KeyPoint(text="supported", citations=["pmid:1"]),
            KeyPoint(text="invented", citations=["pmid:999"]),
        ],
        confidence="high",
    )
    cleaned, report = check_citations(answer, evidence)
    assert report.valid_citations == 1
    assert report.invalid_citations == ["pmid:999"]
    assert report.unsupported_points == 1
    assert cleaned.key_points[1].citations == []
    assert cleaned.confidence == "medium"
    assert DISCLAIMER in cleaned.caveats


def test_guardrail_keeps_a_clean_answer_unchanged():
    evidence = [Evidence(id="nct:NCT1", source="clinicaltrials", title="t", text="x", url="u")]
    answer = FinalAnswer(answer="a", key_points=[KeyPoint(text="p", citations=["nct:NCT1"])], confidence="high")
    cleaned, report = check_citations(answer, evidence)
    assert cleaned.confidence == "high" and report.invalid_citations == []


def test_guardrail_repairs_a_unique_truncated_id_but_not_an_ambiguous_one():
    """A live Groq run cited 'fda-label:0230a3c3' for the real id 'fda-label:0230a3c3:warnings'."""
    evidence = [
        Evidence(id="fda-label:0230a3c3:warnings", source="openfda_labels", title="t", text="x", url="u"),
        Evidence(id="nct:NCT0611", source="clinicaltrials", title="t", text="x", url="u"),
        Evidence(id="nct:NCT0612", source="clinicaltrials", title="t", text="x", url="u"),
    ]
    answer = FinalAnswer(
        answer="a",
        key_points=[
            KeyPoint(text="truncated but unique", citations=["fda-label:0230a3c3"]),
            KeyPoint(text="ambiguous prefix", citations=["nct:NCT061"]),
        ],
        confidence="high",
    )
    cleaned, report = check_citations(answer, evidence)
    assert cleaned.key_points[0].citations == ["fda-label:0230a3c3:warnings"]
    assert report.repaired_citations == ["fda-label:0230a3c3 -> fda-label:0230a3c3:warnings"]
    assert report.invalid_citations == ["nct:NCT061"]
    assert cleaned.confidence == "medium"
