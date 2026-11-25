"""Plugin registries: the one place that maps a name in the config to a class or function.

Adding a plugin never means editing core code (the Open/Closed principle):

    @SOURCES.register("my_source")
    class MySource(DataSource): ...

then list "my_source" under `sources:` in configs/default.yaml.
"""

from typing import Any, Callable


class Registry:
    """A name -> factory map with clear errors."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, *, replace: bool = False):
        """Decorator. Pass replace=True to swap an existing plugin (handy in notebooks)."""

        def decorator(obj):
            if name in self._items and not replace:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"Unknown {self.kind} '{name}'. Registered: {self.names()}") from None

    def create(self, name: str, **kwargs) -> Any:
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


SOURCES = Registry("data source")
LLMS = Registry("LLM provider")
