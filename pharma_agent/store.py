"""SQLite store for the HTTP cache and the run log, with ACID writes.

    Atomic       `transaction()` wraps each write: a run and all its evidence rows are saved
                 together or not at all (any error rolls the whole thing back).
    Consistent   PRIMARY KEY, NOT NULL and FOREIGN KEY constraints reject bad rows.
    Isolated     BEGIN IMMEDIATE takes the write lock up front, so two writers never
                 interleave; WAL mode lets readers keep reading while one writes.
    Durable      synchronous=FULL: COMMIT returns only after the data is on disk.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .schemas import Evidence

SCHEMA = """
CREATE TABLE IF NOT EXISTS http_cache (
    key         TEXT PRIMARY KEY,
    status      INTEGER NOT NULL,
    body        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    question        TEXT NOT NULL,
    answer_json     TEXT NOT NULL,
    plan_json       TEXT NOT NULL,
    guardrail_json  TEXT NOT NULL,
    latency_ms      REAL NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_evidence (
    run_id         TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    evidence_id    TEXT NOT NULL,
    source         TEXT NOT NULL,
    evidence_json  TEXT NOT NULL,
    PRIMARY KEY (run_id, evidence_id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # isolation_level=None: we issue BEGIN/COMMIT ourselves, so nothing is implicit.
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """All statements inside commit together, or roll back together on any error."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    # ---- HTTP cache -------------------------------------------------------------------
    def cache_get(self, key: str) -> tuple[int, str] | None:
        with self._lock:
            row = self._conn.execute("SELECT status, body FROM http_cache WHERE key = ?", (key,)).fetchone()
        return (row["status"], row["body"]) if row else None

    def cache_put(self, key: str, status: int, body: str) -> None:
        with self.transaction() as tx:
            tx.execute(
                "INSERT OR REPLACE INTO http_cache (key, status, body, created_at) VALUES (?, ?, ?, ?)",
                (key, status, body, _now()),
            )

    # ---- Run log ----------------------------------------------------------------------
    def save_run(
        self,
        run_id: str,
        question: str,
        answer: dict,
        plan: dict,
        guardrail: dict,
        latency_ms: float,
        evidence: list[Evidence],
    ) -> None:
        """Atomic: the run row and every evidence row are written in one transaction."""
        with self.transaction() as tx:
            tx.execute(
                "INSERT INTO runs (id, question, answer_json, plan_json, guardrail_json, latency_ms, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, question, json.dumps(answer), json.dumps(plan), json.dumps(guardrail), latency_ms, _now()),
            )
            tx.executemany(
                "INSERT INTO run_evidence (run_id, evidence_id, source, evidence_json) VALUES (?, ?, ?, ?)",
                [(run_id, e.id, e.source, e.model_dump_json()) for e in evidence],
            )

    def get_run(self, run_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            ev = self._conn.execute(
                "SELECT evidence_json FROM run_evidence WHERE run_id = ? ORDER BY rowid", (run_id,)
            ).fetchall()
        return {
            "id": row["id"],
            "question": row["question"],
            "answer": json.loads(row["answer_json"]),
            "plan": json.loads(row["plan_json"]),
            "guardrail": json.loads(row["guardrail_json"]),
            "latency_ms": row["latency_ms"],
            "created_at": row["created_at"],
            "evidence": [json.loads(r["evidence_json"]) for r in ev],
        }

    def list_runs(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT r.id, r.question, r.latency_ms, r.created_at, COUNT(e.evidence_id) AS evidence"
                " FROM runs r LEFT JOIN run_evidence e ON e.run_id = r.id"
                " GROUP BY r.id ORDER BY r.created_at DESC, r.rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self, table: str) -> int:
        if table not in {"http_cache", "runs", "run_evidence"}:
            raise ValueError(table)
        with self._lock:
            return self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
