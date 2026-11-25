"""Data sources. Importing this package registers every built-in source in SOURCES."""

from ..http import HttpClient
from ..registry import SOURCES
from .base import DataSource, clip
from . import clinicaltrials, openfda, pubmed  # noqa: F401  (imports register the sources)


def build_sources(names: list[str], http: HttpClient) -> list[DataSource]:
    """Create the sources named in the config, all sharing one HTTP client."""
    return [SOURCES.create(name, http=http) for name in names]


__all__ = ["DataSource", "build_sources", "clip", "SOURCES"]
