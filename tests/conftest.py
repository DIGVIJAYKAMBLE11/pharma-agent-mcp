"""Shared fixtures.

Nothing is mocked: LLM tests call real Groq and source tests call the real public APIs.
The agent is built once per test session so the Groq free tier is used sparingly.
"""

import os

import pytest

from pharma_agent.config import load_config
from pharma_agent.service import PharmaAgent
from pharma_agent.sources import build_sources
from pharma_agent.store import Store
from pharma_agent.wiring import build_http


@pytest.fixture(scope="session")
def cfg() -> dict:
    # A throwaway in-memory database per session: tests never touch data/pharma_agent.db.
    return load_config(overrides={"store.path": ":memory:"})


@pytest.fixture(scope="session")
def store() -> Store:
    return Store(":memory:")


@pytest.fixture(scope="session")
def sources(cfg, store):
    return {s.name: s for s in build_sources(cfg["sources"], build_http(cfg, store))}


@pytest.fixture(scope="session")
def groq_key(cfg) -> None:  # cfg first: load_config reads .env
    if not os.getenv("GROQ_API_KEY"):
        pytest.fail("GROQ_API_KEY is not set: add it to .env (local) or as a repository secret (CI).")


@pytest.fixture(scope="session")
async def agent(cfg, store, groq_key) -> PharmaAgent:
    return await PharmaAgent.create(cfg, store=store)
