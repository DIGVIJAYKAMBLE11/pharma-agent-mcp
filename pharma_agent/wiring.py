"""The composition root: the only module that turns config into concrete objects.

Everything else receives its dependencies (store, HTTP client, tools, LLMs) as arguments,
which is what makes each piece easy to test and to swap (Dependency Inversion).
"""

import sys

from langchain_core.tools import BaseTool

from .config import ROOT, resolve_path
from .http import HttpClient
from .sources import build_sources
from .store import Store
from .tools import langchain_tools


def build_store(cfg: dict) -> Store:
    return Store(resolve_path(cfg["store"]["path"]))


def build_http(cfg: dict, store: Store | None = None, transport=None) -> HttpClient:
    h = cfg.get("http", {})
    return HttpClient(
        timeout=h.get("timeout", 20),
        retries=h.get("retries", 3),
        backoff=h.get("backoff", 1.0),
        cache=store if h.get("cache", True) else None,
        transport=transport,
    )


def mcp_connection(mcp_cfg: dict) -> dict:
    """Connection settings for langchain-mcp-adapters' MultiServerMCPClient."""
    if mcp_cfg.get("transport", "stdio") == "stdio":
        return {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "pharma_agent.mcp_server", "--transport", "stdio"],
            "cwd": str(ROOT),
        }
    return {"transport": "streamable_http", "url": mcp_cfg["url"]}


async def load_tools(cfg: dict, http: HttpClient) -> list[BaseTool]:
    """Direct mode builds tools in-process; MCP mode asks the MCP server for them."""
    mode = cfg["tools"]["mode"]
    if mode == "direct":
        return langchain_tools(build_sources(cfg["sources"], http))
    if mode == "mcp":
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({"pharma": mcp_connection(cfg["tools"]["mcp"])})
        return await client.get_tools()
    raise ValueError(f"tools.mode must be 'direct' or 'mcp', got {mode!r}")
