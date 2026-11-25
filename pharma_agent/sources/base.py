"""The one interface every data source implements.

A source = one external API = one tool the agents can call. To add a source:

    1. subclass DataSource
    2. set `name` (the tool name the LLM sees) and `description` (when to use it)
    3. implement `search(...)` with type-annotated arguments; use
       Annotated[type, Field(description=...)] so the LLM sees what each argument means
    4. return a list of Evidence
    5. decorate the class with @SOURCES.register("<config name>")

Do not use `from __future__ import annotations` in source modules: the tool layer reads
the real annotation objects to build the JSON schema for both LangChain and MCP.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from ..http import HttpClient
from ..schemas import Evidence


class DataSource(ABC):
    name: ClassVar[str]
    description: ClassVar[str]

    def __init__(self, http: HttpClient):
        self.http = http

    @abstractmethod
    def search(self, *args, **kwargs) -> list[Evidence]:
        """Query the source and return normalised Evidence (empty list when nothing matches)."""


def clip(text: str, limit: int) -> str:
    """Shorten text to `limit` characters at a word boundary, keeping LLM prompts small."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."
