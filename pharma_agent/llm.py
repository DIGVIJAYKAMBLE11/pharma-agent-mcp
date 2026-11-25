"""Chat models, built from the config. Groq is the only real provider (free tier).

Three roles, so the work spreads over two Groq models (the free tier limits tokens per
minute *per model*):

    supervisor   decides which specialists to call            (small prompt, one call)
    specialist   tool-calling agents                          (most calls and tokens)
    synthesizer  writes the final structured, cited answer    (one call)

Add a provider with @LLMS.register("name") and set `llm.provider: name` in the config.
"""

import os
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from .fake_llm import ScriptedChatModel
from .registry import LLMS


@dataclass
class LLMRoles:
    supervisor: BaseChatModel
    specialist: BaseChatModel
    synthesizer: BaseChatModel


@LLMS.register("groq")
def groq_chat(model: str, temperature: float = 0, max_retries: int = 6, reasoning_effort: str | None = None, **_):
    from langchain_groq import ChatGroq

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste your free key from "
            "https://console.groq.com/keys (or use llm.provider: fake to run offline)."
        )
    kwargs = {"model": model, "temperature": temperature, "max_retries": max_retries}
    if reasoning_effort and model.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatGroq(**kwargs)


@LLMS.register("fake")
def fake_chat(**_):
    return ScriptedChatModel()


def build_llms(llm_cfg: dict) -> LLMRoles:
    """One chat model per role. `llm.model`, if set, overrides every role's model."""
    provider = llm_cfg.get("provider", "groq")
    roles = dict(llm_cfg.get("roles", {}))
    if llm_cfg.get("model"):
        roles = {role: llm_cfg["model"] for role in ("supervisor", "specialist", "synthesizer")}
    common = {k: v for k, v in llm_cfg.items() if k not in {"provider", "roles", "model"}}
    built: dict[str, BaseChatModel] = {}
    cache: dict[str, BaseChatModel] = {}
    for role in ("supervisor", "specialist", "synthesizer"):
        model = roles.get(role, "")
        if model not in cache:
            cache[model] = LLMS.create(provider, model=model, **common)
        built[role] = cache[model]
    return LLMRoles(**built)
