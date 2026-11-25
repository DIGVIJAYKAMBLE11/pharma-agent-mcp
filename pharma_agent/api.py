"""FastAPI service.

    uvicorn pharma_agent.api:app --port 8000        then open http://localhost:8000/docs

    GET  /health        liveness + which LLM provider and tool mode are active
    GET  /tools         tool names, descriptions and argument schemas
    GET  /agents        specialists and the tools each one can use
    POST /ask           {"question": "..."} -> structured, cited answer
    GET  /runs          recent runs from the SQLite run log
    GET  /runs/{id}     one stored run with its evidence
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .schemas import AskResponse
from .service import PharmaAgent


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500, examples=["Does metformin have a boxed warning?"])


def create_app(agent_factory=None) -> FastAPI:
    """`agent_factory` is an async callable returning a PharmaAgent (tests pass their own)."""
    factory = agent_factory or PharmaAgent.create

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.agent = await factory()
        yield

    app = FastAPI(title="PharmaAgent", version="0.1.0", lifespan=lifespan,
                  description="Multi-agent assistant over openFDA, ClinicalTrials.gov and PubMed.")

    def agent(request: Request) -> PharmaAgent:
        return request.app.state.agent

    @app.get("/health")
    def health(request: Request):
        cfg = agent(request).cfg
        return {"status": "ok", "llm_provider": cfg["llm"]["provider"], "tool_mode": cfg["tools"]["mode"]}

    @app.get("/tools")
    def tools(request: Request):
        return [
            {"name": t.name, "description": t.description, "args": t.args}
            for t in agent(request).tools
        ]

    @app.get("/agents")
    def agents(request: Request):
        return [
            {"name": n, "description": s.spec.description, "tools": s.spec.tools}
            for n, s in agent(request).specialists.items()
        ]

    @app.post("/ask", response_model=AskResponse)
    async def ask(body: AskRequest, request: Request):
        return await agent(request).ask(body.question)

    @app.get("/runs")
    def runs(request: Request, limit: int = 20):
        return agent(request).store.list_runs(limit)

    @app.get("/runs/{run_id}")
    def run(run_id: str, request: Request):
        found = agent(request).store.get_run(run_id)
        if found is None:
            raise HTTPException(status_code=404, detail="run not found")
        return found

    return app


app = create_app()
