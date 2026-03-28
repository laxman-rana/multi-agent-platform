from importlib import import_module
from types import ModuleType


# Map public agent names to module paths under the ecommerce domain.
AGENT_REGISTRY = {
    "support": "src.agents.ecommerce.support.agent",
}


def list_agents() -> list[str]:
    """Return available ecommerce agent names."""

    return sorted(AGENT_REGISTRY.keys())


def get_agent_module(agent_name: str) -> ModuleType:
    """Load and return an ecommerce agent module by registry name."""

    module_path = AGENT_REGISTRY.get(agent_name)
    if not module_path:
        raise ValueError(f"Unknown ecommerce agent: {agent_name}. Available: {', '.join(list_agents())}")
    return import_module(module_path)


def get_agent_graph(agent_name: str):
    """Return a compiled LangGraph instance for a named ecommerce agent."""

    module = get_agent_module(agent_name)
    return module.construct_graph()
