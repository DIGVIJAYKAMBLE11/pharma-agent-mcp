"""The SQLite store keeps its ACID promises."""

import sqlite3

import pytest

from pharma_agent.schemas import Evidence
from pharma_agent.store import Store


def ev(i: str) -> Evidence:
    return Evidence(id=i, source="pubmed", title="t", text="x", url="https://example.org")


def save(store: Store, run_id: str, evidence: list[Evidence]) -> None:
    store.save_run(run_id, "q?", {"answer": "a"}, {"tasks": []}, {"ok": True}, 1.0, evidence)


def test_run_and_evidence_are_saved_together():
    store = Store(":memory:")
    save(store, "r1", [ev("pmid:1"), ev("pmid:2")])
    run = store.get_run("r1")
    assert [e["id"] for e in run["evidence"]] == ["pmid:1", "pmid:2"]


def test_atomic_failure_leaves_nothing_behind():
    """Duplicate evidence ids break the PRIMARY KEY halfway through: the run row must roll back too."""
    store = Store(":memory:")
    with pytest.raises(sqlite3.IntegrityError):
        save(store, "r2", [ev("pmid:1"), ev("pmid:1")])
    assert store.get_run("r2") is None
    assert store.count("runs") == 0 and store.count("run_evidence") == 0


def test_consistency_constraints_reject_bad_rows():
    store = Store(":memory:")
    with pytest.raises(sqlite3.IntegrityError):  # evidence for a run that does not exist
        with store.transaction() as tx:
            tx.execute("INSERT INTO run_evidence VALUES ('missing-run', 'pmid:1', 'pubmed', '{}')")
    save(store, "r3", [])
    with pytest.raises(sqlite3.IntegrityError):  # the same run id twice
        save(store, "r3", [])
    assert store.count("runs") == 1


def test_durable_across_reconnect(tmp_path):
    path = tmp_path / "runs.db"
    first = Store(path)
    save(first, "r4", [ev("nct:NCT1")])
    first.close()
    assert Store(path).get_run("r4")["evidence"][0]["id"] == "nct:NCT1"


def test_cache_round_trip():
    store = Store(":memory:")
    store.cache_put("k", 200, "body")
    assert store.cache_get("k") == (200, "body")
    assert store.cache_get("other") is None
