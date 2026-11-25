# PharmaAgent: multi-agent GenAI over public drug and clinical-trial data

Ask a pharmaceutical question in plain English. A **LangGraph supervisor** routes it to specialist
**tool-calling agents** that query **openFDA drug labels, FDA adverse-event reports (FAERS),
ClinicalTrials.gov and PubMed**. Their findings become a **structured answer in which every point cites
its source**, and a **guardrail removes any citation a tool did not actually return**.
The tools are also served by an **MCP server**, so any MCP client (Claude Desktop, Cursor, other agents)
can use them.

* LLMs: **Groq free tier only** (`openai/gpt-oss-120b` + `openai/gpt-oss-20b`)
* Data: free public APIs; no keys needed
* Every test runs on the **real** Groq models and the **live** APIs; nothing is mocked

```
                      ┌─ label_agent ───── search_drug_labels, get_adverse_event_counts ─┐
question ─► supervisor┼─ trials_agent ──── search_clinical_trials                        ├─► synthesize ─► guardrail ─► cited answer
           (RoutePlan)└─ literature_agent  search_pubmed                                 ┘   (FinalAnswer)   (drop invented ids)
                                   ▲
                 tools: in-process (direct) or from the MCP server (stdio / streamable HTTP)
```

## Quick start

```bash
py -3.11 -m venv .venv && .venv\Scripts\activate        # Linux/macOS: python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                                  # then paste your free key from https://console.groq.com/keys

python -m pharma_agent ask "Does metformin have a boxed warning, and are there recruiting trials of metformin in cancer?"
python -m pharma_agent ask "What are the most reported adverse events for atorvastatin?" --mode mcp
uvicorn pharma_agent.api:app --port 8000                # then open http://localhost:8000/docs
pytest -q                                               # 22 tests, real Groq + live APIs
python -m pharma_agent eval                             # 16-question evaluation -> eval/results/
```

Docker (API + MCP server as two services): `docker compose up --build`. Kubernetes: `k8s/pharma-agent.yaml`.

## Learn it with the notebooks

| Notebook | What it shows |
|---|---|
| [01_data_sources](notebooks/01_data_sources.ipynb) | The `Evidence` contract, the shared HTTP client, the ACID cache, each source called directly |
| [02_tools_and_mcp](notebooks/02_tools_and_mcp.ipynb) | One function → LangChain tool **and** MCP tool; the MCP server's tools, resource and prompt; MCP over HTTP |
| [03_agents_and_graph](notebooks/03_agents_and_graph.ipynb) | Supervisor, one specialist, the full LangGraph run step by step, the citation guardrail |
| [04_api_and_evaluation](notebooks/04_api_and_evaluation.ipynb) | Every FastAPI endpoint, the SQLite run log, the evaluation metrics and latest report |
| [05_extend_the_system](notebooks/05_extend_the_system.ipynb) | Add a new data source (FDA recalls) and a new agent with **no core code changes** |

## Project layout

```
pharma_agent/
  config.py        YAML + env overrides -> plain dict
  registry.py      name -> plugin maps (SOURCES, LLMS)
  schemas.py       Evidence, RoutePlan, Finding, FinalAnswer, AskResponse (pydantic)
  http.py          shared HTTP client: retries, back-off, cache
  store.py         SQLite: HTTP cache + run log, ACID transactions
  sources/         one file per data source; all implement DataSource.search() -> list[Evidence]
  tools.py         source -> tool function -> LangChain tool (MCP uses the same function)
  mcp_server.py    FastMCP server: tools + resource + prompt; stdio or streamable HTTP
  llm.py           Groq models per role (supervisor / specialist / synthesizer)
  agents/          specialist (ReAct), graph (LangGraph), guardrail, prompts
  wiring.py        the only place config becomes objects (composition root)
  service.py       PharmaAgent.create() / ask(): used by CLI, API, notebooks and eval
  api.py           FastAPI
  evaluation.py    metrics + report
configs/default.yaml   models, sources, agents, limits
eval/questions.jsonl   16 questions with expected agents, sources and key terms
tests/                 ACID store, registry, guardrail, live sources, MCP parity, agent + API on Groq
notebooks/             the five notebooks above
```

## Design principles

**SOLID (code)**
* *Single responsibility*: a source only fetches and normalises; a tool only wraps; an agent only reasons; the API only serves.
* *Open/closed*: add a source with `@SOURCES.register(...)`, an agent with a YAML entry, an LLM provider with `@LLMS.register(...)`. Core files do not change (notebook 05 does this live).
* *Liskov*: every source honours the same `search(...) -> list[Evidence]` contract, so any source can back any tool.
* *Interface segregation*: small interfaces (`DataSource.search`, `Store` methods), no god objects.
* *Dependency inversion*: objects receive their HTTP client, store, tools and LLMs; only `wiring.py` builds them from config.

**ACID (data)**: `store.py` runs every write in an explicit transaction.
* *Atomic*: a run and all its evidence rows are saved together or not at all.
* *Consistent*: primary, foreign and NOT NULL constraints reject bad rows.
* *Isolated*: `BEGIN IMMEDIATE` + WAL; one writer at a time, readers never blocked.
* *Durable*: `synchronous=FULL`. Evaluation reports are written to a temp file and renamed atomically.

`tests/test_store_acid.py` checks each property, e.g. it forces a failure halfway through a save and asserts nothing was written.

## Staying inside the Groq free tier

The free tier allows about 1,000 requests/day and 8,000 tokens per minute **per model**. The project:
* splits roles across two models, so each has its own per-minute budget
* uses `reasoning_effort: low` for gpt-oss models, clips tool outputs and caps each agent at 2 tool rounds
* lets the Groq SDK wait and retry on 429 responses (`max_retries: 6`)
* pauses between evaluation questions (`eval.delay_seconds`)

## Evaluation

`python -m pharma_agent eval` runs `eval/questions.jsonl` on the live models and writes a JSON report to
`eval/results/`. Metrics: routing accuracy, source recall, citation precision (before the guardrail),
grounded key points, expected-term hit rate, and p50/p95 latency. See notebook 04 for the latest numbers.

## What real-model testing caught

The first live run answered "metformin has no boxed warning". It does (lactic acidosis). The label tool
had picked an older generic label without a separate `boxed_warning` field. The tool now prefers labels
that have the requested section's primary field and always puts the boxed warning first; a live test
(`test_label_returns_the_boxed_warning`) guards against the regression.

## Limits

* Informational summaries of public data, **not medical advice**. FAERS counts are reports, not proof of causation.
* openFDA and NCBI rate-limit anonymous use; heavy use needs their free API keys.
* Small evaluation set (16 questions); expected terms are a light correctness check, not full grading.

## Using the MCP server from Claude Desktop or Cursor

```json
{"mcpServers": {"pharma": {"command": "<repo>/.venv/Scripts/python.exe",
                           "args": ["-m", "pharma_agent.mcp_server"], "cwd": "<repo>"}}}
```
