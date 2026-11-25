"""MCP server: exposes every configured data source as a Model Context Protocol tool.

Any MCP client (this project's agents, Claude Desktop, Cursor, ...) can use the tools.

    python -m pharma_agent.mcp_server                                 # stdio (a client spawns it)
    python -m pharma_agent.mcp_server --transport streamable-http     # HTTP on :8765/mcp

It also serves:
    resource  pharma://sources            which sources and tools this server has
    prompt    drug_safety_review(drug)    a ready-made prompt that uses the tools
"""

import argparse
import json

from mcp.server.fastmcp import FastMCP

from .config import load_config
from .sources import build_sources
from .tools import tool_function
from .wiring import build_http, build_store


def build_server(cfg: dict | None = None, http=None) -> FastMCP:
    cfg = cfg or load_config()
    http = http or build_http(cfg, build_store(cfg))
    sources = build_sources(cfg["sources"], http)

    server = FastMCP(
        "pharma-agent",
        instructions="Public drug data: FDA labels, FAERS adverse-event counts, ClinicalTrials.gov, PubMed.",
    )
    for source in sources:
        server.add_tool(tool_function(source), name=source.name, description=source.description)

    @server.resource("pharma://sources")
    def list_sources() -> str:
        """The data sources and tools this server exposes."""
        return json.dumps(
            [{"source": type(s).__name__, "tool": s.name, "description": s.description} for s in sources],
            indent=2,
        )

    @server.prompt()
    def drug_safety_review(drug: str) -> str:
        """A safety review prompt that walks through the label, FAERS and trials."""
        return (
            f"Review the safety of {drug}. 1) Read the boxed warning and warnings from its FDA label. "
            f"2) Get the most reported FAERS reactions. 3) List recruiting trials. Cite every source id."
        )

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="PharmaAgent MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = build_server()
    server.settings.host = args.host
    server.settings.port = args.port
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
