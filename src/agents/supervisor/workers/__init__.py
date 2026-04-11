"""
workers/__init__.py
-------------------
BaseWorker ABC, WorkerName enum, and the global worker registry.

Design:
- WorkerName (StrEnum) is the single source of truth for tool/node names.
  Using an enum instead of bare strings means a typo is an AttributeError
  at import time, not a silent routing failure at runtime.
- BaseWorker defines the minimum contract: name, description, input_schema,
  invoke(). The supervisor depends only on this abstraction — never on a
  concrete worker class (Dependency Inversion).
- The registry decouples registration from discovery. Workers register
  themselves at import time; the supervisor reads get_all_workers() without
  knowing which concrete classes exist (Open/Closed).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class WorkerName(str, Enum):
    """Canonical names for all registered supervisor workers.

    str + Enum with __str__ overridden to return the value — this mirrors the
    behaviour of StrEnum (Python 3.11+) while staying compatible with 3.10.
    str(WorkerName.OPPORTUNITY) == 'scan_opportunities', which is what
    LangChain uses as the tool name and LangGraph uses as the node name.

    Adding a new worker:
        1. Add a new member here.
        2. Create the worker module and call register() at module level.
        3. Import the module in workflow.py to trigger registration.
    """

    OPPORTUNITY = "scan_opportunities"
    PORTFOLIO = "analyze_portfolio"
    ECOMMERCE = "handle_support"

    def __str__(self) -> str:  # makes str(member) == member.value (like StrEnum)
        return self.value


class BaseWorker(ABC):
    """Abstract base class for all supervisor workers.

    Concrete workers must declare:
        name          WorkerName member — identifies the LLM tool and graph node.
        description   Natural-language text the supervisor LLM reads to decide
                      when to call this worker.
        input_schema  Pydantic BaseModel subclass defining the tool's parameters.
                      LangChain converts this to a JSON schema for the LLM.
        invoke()      Runs the underlying agent pipeline; always returns a plain
                      string that is safe to embed in a ToolMessage.

    Workers are stateless — a single instance is reused across all calls.
    """

    name: WorkerName
    description: str
    input_schema: type[BaseModel]

    @abstractmethod
    def invoke(self, **kwargs) -> str:
        """Execute the worker and return a string result."""
        ...

    def as_tool(self) -> StructuredTool:
        """Wrap this worker as a LangChain StructuredTool for LLM tool-binding.

        The supervisor calls get_all_tools() which calls as_tool() on every
        registered worker, so the LLM sees exactly one tool per worker.
        """
        return StructuredTool.from_function(
            func=self.invoke,
            name=str(self.name),
            description=self.description,
            args_schema=self.input_schema,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, BaseWorker] = {}


def register(worker_instance: BaseWorker) -> None:
    """Register a worker instance directly."""
    _REGISTRY[str(worker_instance.name)] = worker_instance
    logger.debug("[WorkerRegistry] Registered: %s", worker_instance.name)


def worker(cls: type[BaseWorker]) -> type[BaseWorker]:
    """Class decorator that instantiates and registers a BaseWorker subclass.

    Usage:
        @worker
        class MyWorker(BaseWorker):
            name = WorkerName.MY_WORKER
            ...

    Defining the class IS registering it — no separate register() call needed.
    """
    register(cls())
    return cls


def get_all_workers() -> list[BaseWorker]:
    """Return all registered workers in registration order."""
    return list(_REGISTRY.values())


def get_all_tools() -> list[StructuredTool]:
    """Return a LangChain StructuredTool for every registered worker."""
    return [w.as_tool() for w in get_all_workers()]


# ---------------------------------------------------------------------------
# Worker registration
# Worker modules are imported here — not in workflow.py — so registration
# is centralised in one place. Adding a new worker = one line here only.
# ---------------------------------------------------------------------------
from src.agents.supervisor.workers import opportunity_worker  # noqa: E402, F401
from src.agents.supervisor.workers import portfolio_worker    # noqa: E402, F401
from src.agents.supervisor.workers import ecommerce_worker   # noqa: E402, F401
