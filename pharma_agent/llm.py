"""Chat models, built from the config. Groq (free tier) is the only provider.

Three roles, so the work spreads over two Groq models (the free tier limits tokens per
minute *per model*):

    supervisor   decides which specialists to call            (small prompt, one call)
    specialist   tool-calling agents                          (most calls and tokens)
    synthesizer  writes the final structured, cited answer    (one call)

Free-tier daily limits: each Groq model also has its own tokens-per-day cap (200k for gpt-oss).
A live run hit it on gpt-oss-20b, so every role gets `llm.fallbacks`: when a model answers with a
rate-limit error, the same call is retried on the next model (each has its own daily allowance).

To add another provider later: @LLMS.register("name") a factory, then set `llm.provider: name`.
"""

import os
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from .registry import LLMS


@dataclass
class LLMRoles:
    """Each role is a chat model, or a chat model with fallbacks (bind_tools and
    with_structured_output pass through to every model in the chain)."""
    supervisor: Runnable
    specialist: Runnable
    synthesizer: Runnable


@LLMS.register("groq")
def groq_chat(model: str, temperature: float = 0, max_retries: int = 6, reasoning_effort: str | None = None, **_):
    from langchain_groq import ChatGroq

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste your free key from "
            "https://console.groq.com/keys"
        )
    kwargs = {"model": model, "temperature": temperature, "max_retries": max_retries}
    if reasoning_effort and model.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatGroq(**kwargs)


def build_llms(llm_cfg: dict) -> LLMRoles:
    """One chat model per role, each with the configured fallback models behind it.
    `llm.model`, if set, overrides every role's primary model."""
    provider = llm_cfg.get("provider", "groq")
    roles = dict(llm_cfg.get("roles", {}))
    if llm_cfg.get("model"):
        roles = {role: llm_cfg["model"] for role in ("supervisor", "specialist", "synthesizer")}
    fallbacks = list(llm_cfg.get("fallbacks", []))
    skip = {"provider", "roles", "model", "fallbacks", "structured_output_method"}
    common = {k: v for k, v in llm_cfg.items() if k not in skip}

    cache: dict[str, BaseChatModel] = {}

    def model(name: str) -> BaseChatModel:
        if name not in cache:
            cache[name] = LLMS.create(provider, model=name, **common)
        return cache[name]

    def with_fallbacks(primary: str) -> Runnable:
        backups = [model(name) for name in fallbacks if name != primary]
        if not backups:
            return model(primary)
        return model(primary).with_fallbacks(backups, exceptions_to_handle=rate_limit_errors(provider))

    return LLMRoles(**{role: with_fallbacks(roles[role]) for role in ("supervisor", "specialist", "synthesizer")})


def rate_limit_errors(provider: str) -> tuple[type[BaseException], ...]:
    """Only rate-limit errors switch to a fallback model; any other error is a real bug and surfaces."""
    if provider == "groq":
        from groq import RateLimitError

        return (RateLimitError,)
    return (Exception,)
