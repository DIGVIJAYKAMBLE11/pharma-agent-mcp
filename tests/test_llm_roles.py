"""LLM roles and fallbacks are wired as configured (building the clients makes no API call)."""

from groq import RateLimitError

from pharma_agent.llm import build_llms


def test_each_role_falls_back_to_other_models_on_rate_limit(cfg, groq_key):
    roles = build_llms(cfg["llm"])
    specialist = roles.specialist
    assert specialist.runnable.model_name == "openai/gpt-oss-20b"
    assert [m.model_name for m in specialist.fallbacks] == ["qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
    assert specialist.exceptions_to_handle == (RateLimitError,)
    # a role never falls back to its own model
    assert "openai/gpt-oss-120b" not in [m.model_name for m in roles.supervisor.fallbacks]
    # tools bind through the fallback chain, so ReAct agents keep working on any model
    assert type(specialist.bind_tools([])).__name__ == "RunnableWithFallbacks"
