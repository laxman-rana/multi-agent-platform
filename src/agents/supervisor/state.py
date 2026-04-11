"""
state.py
--------
Shared state object for the supervisor/worker graph.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SupervisorState(TypedDict):
    """Shared state passed through every node in the supervisor/worker graph.

    messages
        Full ReAct message history built up as the supervisor loops:
            HumanMessage(query)
            → AIMessage(tool_call)
            → ToolMessage(worker result)
            → ...
            → AIMessage(final answer, no tool_calls)
        The add_messages reducer appends rather than overwrites, preserving
        the full conversation for the LLM's context window.

    worker_results
        Collected raw results keyed by WorkerName value. Useful for
        cross-agent chaining (e.g. pass opportunity scan results into the
        portfolio worker) and for post-run inspection.

    steps
        Monotonic counter incremented on every supervisor invocation.
        Guards against unbounded tool-call loops — graph forces END when
        steps >= _MAX_STEPS in workflow.py.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    worker_results: dict[str, str]
    steps: int
