"""One small HTTP client shared by all data sources: retries, back-off and an optional cache.

Sources receive it in their constructor (dependency injection), so tests can pass a client
built on `httpx.MockTransport` and never touch the network.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .store import Store

log = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}
CACHEABLE_STATUSES = {200, 404}  # openFDA answers 404 for "no match", which is worth caching too


@dataclass
class HttpResult:
    status: int
    text: str
    from_cache: bool = False

    def json(self) -> Any:
        return json.loads(self.text)


class HttpClient:
    def __init__(
        self,
        timeout: float = 20,
        retries: int = 3,
        backoff: float = 1.0,
        cache: Store | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.retries = retries
        self.backoff = backoff
        self.cache = cache
        self._client = httpx.Client(
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
            headers={"User-Agent": "pharma-agent/0.1 (open-source research demo)"},
        )

    @staticmethod
    def cache_key(url: str, params: dict | None) -> str:
        return str(httpx.URL(url, params=sorted((params or {}).items())))

    def get(self, url: str, params: dict | None = None) -> HttpResult:
        key = self.cache_key(url, params)
        if self.cache is not None and (hit := self.cache.cache_get(key)):
            return HttpResult(status=hit[0], text=hit[1], from_cache=True)

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self._client.get(url, params=params)
                if resp.status_code not in RETRY_STATUSES:
                    if self.cache is not None and resp.status_code in CACHEABLE_STATUSES:
                        self.cache.cache_put(key, resp.status_code, resp.text)
                    if resp.status_code >= 400 and resp.status_code != 404:
                        resp.raise_for_status()
                    return HttpResult(status=resp.status_code, text=resp.text)
                last_error = httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request, response=resp)
            except httpx.TransportError as exc:
                last_error = exc
            wait = self.backoff * 2**attempt
            log.warning("GET %s failed (%s), retry %d in %.1fs", url, last_error, attempt + 1, wait)
            time.sleep(wait)
        raise RuntimeError(f"GET {url} failed after {self.retries} attempts: {last_error}")

    def close(self) -> None:
        self._client.close()
