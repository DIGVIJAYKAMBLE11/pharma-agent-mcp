"""A specialist = one ReAct agent (LLM + a few tools + a role prompt), defined in the config.

    label_agent:
      description: ...        # what the supervisor reads when routing
      tools: [search_drug_labels, get_adverse_event_counts]
      prompt: ...             # the agent's role

To add a specialist, add an entry under `agents:` in configs/default.yaml.
"""

from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from ..schemas import Evidence
from ..tools import content_to_text, parse_evidence
from .prompts import SPECIALIST_RULES


@dataclass
class SpecialistSpec:
    name: str
    description: str
    prompt: str
    tools: list[str] = field(default_factory=list)


class Specialist:
    def __init__(self, spec: SpecialistSpec, llm: BaseChatModel, tools: list[BaseTool], max_steps: int = 7):
        self.spec = spec
        self.tools = tools
        self.max_steps = max_steps
        self.agent = create_react_agent(llm, tools, prompt=f"{spec.prompt}\n\n{SPECIALIST_RULES}")

    async def run(self, instruction: str) -> tuple[str, list[Evidence], list[dict]]:
        """Returns (summary text, evidence from every tool call, the tool calls made)."""
        result = await self.agent.ainvoke(
            {"messages": [HumanMessage(instruction)]},
            config={"recursion_limit": self.max_steps},
        )
        messages = result["messages"]
        evidence = [e for m in messages if isinstance(m, ToolMessage) for e in parse_evidence(m.content)]
        calls = [{"tool": c["name"], "args": c["args"]} for m in messages for c in getattr(m, "tool_calls", []) or []]
        return content_to_text(messages[-1].content).strip(), evidence, calls


def build_specialists(
    agent_cfgs: dict, tools: list[BaseTool], llm: BaseChatModel, max_steps: int = 7
) -> dict[str, Specialist]:
    by_name = {t.name: t for t in tools}
    specialists = {}
    for name, c in agent_cfgs.items():
        missing = [t for t in c["tools"] if t not in by_name]
        if missing:
            raise KeyError(f"Agent '{name}' needs tools {missing}; available: {sorted(by_name)}")
        spec = SpecialistSpec(name=name, description=c["description"], prompt=c["prompt"], tools=c["tools"])
        specialists[name] = Specialist(spec, llm, [by_name[t] for t in c["tools"]], max_steps)
    return specialists
