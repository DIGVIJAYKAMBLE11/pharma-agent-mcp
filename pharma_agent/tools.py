"""Turn data sources into tools: one function per source, served two ways.

    source.search(...)  ->  tool_function(source)  ->  LangChain StructuredTool   (direct, in-process)
                                                   ->  MCP tool on FastMCP        (mcp_server.py)

Both transports are built from the same function, so the agents see the same name,
description and argument schema whichever way the tools reach them.

A tool returns JSON text: {"results": [Evidence, ...]}. Agents read it, and the graph parses
it back into Evidence so the final answer's citations can be checked.
"""

import inspect
import json
from typing import Any, Callable

from langchain_core.tools import BaseTool, StructuredTool

from .schemas import Evidence
from .sources import DataSource

EMPTY_NOTE = "No matches. Try another spelling, the generic name, or a broader query."


def evidence_to_json(evidence: list[Evidence]) -> str:
    payload: dict[str, Any] = {"results": [e.model_dump() for e in evidence]}
    if not evidence:
        payload["note"] = EMPTY_NOTE
    return json.dumps(payload, ensure_ascii=False)


def tool_function(source: DataSource) -> Callable[..., str]:
    """Wrap `source.search` so it returns JSON text but keeps its typed signature.

    LangChain and FastMCP both build the tool's JSON schema from the function signature,
    so we copy the signature of `search` (minus `self`) and only change the return type.
    """
    method = source.search

    def run(*args, **kwargs) -> str:
        return evidence_to_json(method(*args, **kwargs))

    run.__signature__ = inspect.signature(method).replace(return_annotation=str)
    run.__annotations__ = {**getattr(method, "__annotations__", {}), "return": str}
    run.__name__ = source.name
    run.__qualname__ = source.name
    run.__doc__ = source.description
    return run


def langchain_tools(sources: list[DataSource]) -> list[BaseTool]:
    """Direct mode: the tools run in this Python process."""
    return [
        StructuredTool.from_function(func=tool_function(s), name=s.name, description=s.description)
        for s in sources
    ]


def content_to_text(content: Any) -> str:
    """Tool and model outputs may be a string or a list of content blocks; return plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def parse_evidence(content: Any) -> list[Evidence]:
    """Read Evidence back out of a tool message (direct or MCP). Ignores anything unparseable."""
    text = content_to_text(content)
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    results = payload.get("results", []) if isinstance(payload, dict) else []
    out = []
    for item in results:
        try:
            out.append(Evidence.model_validate(item))
        except Exception:  # noqa: BLE001  a malformed item should not break the run
            continue
    return out
