"""Settings: read a YAML file, then let environment variables override it.

The config is a plain dict so it is easy to print, copy and tweak in a notebook:

    cfg = load_config(overrides={"llm.provider": "fake"})
"""

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

# Environment variable -> dotted key in the config dict.
ENV_OVERRIDES = {
    "PHARMA_LLM_PROVIDER": "llm.provider",
    "PHARMA_LLM_MODEL": "llm.model",
    "PHARMA_TOOL_MODE": "tools.mode",
    "PHARMA_MCP_TRANSPORT": "tools.mcp.transport",
    "PHARMA_MCP_URL": "tools.mcp.url",
    "PHARMA_STORE_PATH": "store.path",
}


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    """Load the YAML config. Precedence: overrides > environment > file."""
    load_dotenv(ROOT / ".env")
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for env_var, key in ENV_OVERRIDES.items():
        if os.getenv(env_var):
            set_key(cfg, key, os.environ[env_var])
    for key, value in (overrides or {}).items():
        set_key(cfg, key, value)
    return cfg


def set_key(cfg: dict, dotted_key: str, value: Any) -> None:
    """set_key(cfg, "llm.model", "x") is cfg["llm"]["model"] = "x", creating levels if needed."""
    *parents, last = dotted_key.split(".")
    node = cfg
    for part in parents:
        node = node.setdefault(part, {})
    node[last] = copy.deepcopy(value)


def resolve_path(path: str | Path) -> Path:
    """Relative paths in the config are relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() or str(p) == ":memory:" else ROOT / p
