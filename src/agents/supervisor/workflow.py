"""
workflow.py
-----------
Supervisor/Worker multi-agent orchestration graph.

The supervisor is a tool-calling LLM node that reads the user's query and
decides which worker agent to invoke — and in what order. Workers are thin
adapters over existing agent pipelines; none of those internals change.

Graph shape (ReAct loop):

    user query
        │
        ▼
   [supervisor] ──► tool_call ──► [opportunity_worker | portfolio_worker | ecommerce_worker]
        ▲                                              │
        └──────────────── ToolMessage ─────────────────┘
        │
        │  no tool_calls in response
        ▼
       END  (supervisor emits final AIMessage answer)

Run from the repository root:

    python -m src.agents.supervisor.workflow "Should I buy NVDA today?"
    python -m src.agents.supervisor.workflow "What should I do with my portfolio?"
    python -m src.agents.supervisor.workflow "My order #1234 is damaged, I want a refund"
    python -m src.agents.supervisor.workflow --model gpt-4o "Scan the US market for opportunities"

    # Cross-agent chaining — supervisor decides to call both agents:
    python -m src.agents.supervisor.workflow "Scan for opportunities and check if they fit my portfolio"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from src.llm import get_llm

from .state import SupervisorState

from .workers import WorkerName, get_all_tools, get_all_workers  # registration happens inside workers/

logger = logging.getLogger(__name__)

# Maximum supervisor → worker cycles before the graph forces END.
# Prevents unbounded loops when the LLM keeps emitting tool calls.
_MAX_STEPS = 6
_LOG_PREVIEW_CHARS = 120  # max characters shown in worker result preview log lines

_SYSTEM_PROMPT = """You are a multi-agent orchestrator. Your job is to fully satisfy the user's request \
by calling the appropriate worker agents — one at a time — before writing a final answer.

Rules:
1. Read the user's request carefully. Identify ALL distinct pieces of information needed.
2. Call the relevant worker(s) sequentially — do NOT answer until you have gathered all necessary data.
3. If the request requires output from multiple workers, call each one before writing your final answer.
4. Only write a final answer (no tool call) when all required information has been collected.
5. Synthesise the worker results into a single, clear, direct response for the user."""


# ---------------------------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------------------------


def _supervisor_node(state: SupervisorState) -> dict[str, Any]:
    """Invoke the LLM with all registered worker tools bound.

    If the response contains tool_calls, _should_continue routes to the
    matching worker node.  If it contains only text, the graph routes to END.
    """
    provider = os.getenv("PORTFOLIO_LLM_PROVIDER", "ollama")
    model = os.getenv("PORTFOLIO_LLM_MODEL") or None
    tools = get_all_tools()
    llm = get_llm(model_name=provider, model=model).bind_tools(tools)

    step = state["steps"] + 1
    logger.info("[Supervisor] Step %d — reasoning (provider=%s model=%s)", step, provider, model or "default")

    # Prepend the system prompt once — contains only orchestration rules.
    # Worker names and descriptions are already sent to the LLM via bind_tools()
    # as part of the function-calling schema — no need to repeat them here.
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=_SYSTEM_PROMPT)] + list(messages)

    response = llm.invoke(messages)

    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            logger.info(
                "[Supervisor] → Decided to invoke worker: '%s'  args: %s",
                tc["name"],
                tc["args"],
            )
    else:
        logger.info("[Supervisor] → No tool call — will synthesise final answer")

    return {
        "messages": [response],
        "steps": step,
    }


# ---------------------------------------------------------------------------
# Worker node factory
# ---------------------------------------------------------------------------


def _make_worker_node(worker):
    """Return a LangGraph node function bound to the given BaseWorker instance.

    Extracts the matching tool_call from the last AIMessage, delegates to
    worker.invoke(), and returns a ToolMessage so the supervisor can
    incorporate the result into its next reasoning step.
    """

    def node(state: SupervisorState) -> dict[str, Any]:
        last_msg = state["messages"][-1]
        tool_call = next(
            tc for tc in last_msg.tool_calls if tc["name"] == str(worker.name)
        )
        logger.info(
            "[Supervisor] ▶ Dispatching to worker: '%s'  args: %s",
            worker.name,
            tool_call["args"],
        )
        result = worker.invoke(**tool_call["args"])
        preview = result[:_LOG_PREVIEW_CHARS].replace("\n", " ") if len(result) > _LOG_PREVIEW_CHARS else result.replace("\n", " ")
        logger.info(
            "[Supervisor] ◀ Worker '%s' completed — result preview: %s%s",
            worker.name,
            preview,
            "..." if len(result) > _LOG_PREVIEW_CHARS else "",
        )
        return {
            "messages": [ToolMessage(content=result, tool_call_id=tool_call["id"])],
            "worker_results": {
                **state.get("worker_results", {}),
                str(worker.name): result,
            },
        }

    node.__name__ = f"{worker.name}_node"
    return node


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _should_continue(state: SupervisorState) -> str:
    """Decide the next node after the supervisor runs.

    Returns a WorkerName string so the conditional edge map routes directly
    to the matching worker node, or "end" to terminate the graph.

    Guards:
    - step limit:   forces END after _MAX_STEPS to prevent infinite loops.
    - unknown tool: forces END with a warning if the LLM hallucinates a name.
    """
    if state["steps"] >= _MAX_STEPS:
        logger.warning("[Supervisor] Max steps (%d) reached — forcing END.", _MAX_STEPS)
        return "end"

    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        tool_name = last.tool_calls[0]["name"]
        if tool_name in WorkerName._value2member_map_:
            logger.info("[Supervisor] Routing → worker node: '%s'", tool_name)
            return tool_name
        logger.warning(
            "[Supervisor] Unknown tool '%s' returned by LLM — forcing END.", tool_name
        )

    logger.info("[Supervisor] Routing → END (task complete)")
    return "end"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph():
    """Assemble and compile the supervisor/worker LangGraph.

    Structure:
        - One supervisor node (LLM with tools).
        - One node per registered worker, named after its WorkerName value.
        - Conditional edges from supervisor → worker (via _should_continue).
        - Unconditional edges from every worker back to supervisor (ReAct loop).
    """
    workers = get_all_workers()

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", _supervisor_node)

    for worker in workers:
        graph.add_node(str(worker.name), _make_worker_node(worker))

    graph.set_entry_point("supervisor")

    # Routing map: tool_name → node_name  +  "end" → END sentinel
    routing_map: dict[str, str] = {str(w.name): str(w.name) for w in workers}
    routing_map["end"] = END

    graph.add_conditional_edges("supervisor", _should_continue, routing_map)

    # All worker nodes loop back to the supervisor (ReAct pattern)
    for worker in workers:
        graph.add_edge(str(worker.name), "supervisor")

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(query: str, model: str | None = None) -> str:
    """Run the supervisor graph with a natural-language query.

    Parameters
    ----------
    query : The user's question or instruction in plain English.
    model : Optional model name override. Provider is inferred automatically
            from the model name (e.g. 'gpt-4o' → openai, 'llama3' → ollama).

    Returns
    -------
    The supervisor's final natural-language answer as a plain string.
    """
    if model:
        from src.llm.providers import infer_provider

        os.environ["PORTFOLIO_LLM_PROVIDER"] = infer_provider(model)
        os.environ["PORTFOLIO_LLM_MODEL"] = model

    compiled = build_graph()
    initial_state: SupervisorState = {
        "messages": [HumanMessage(content=query)],
        "worker_results": {},
        "steps": 0,
    }
    result = compiled.invoke(initial_state)

    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            return str(msg.content)

    return "Supervisor completed but returned no final answer."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _env_model = os.getenv("PORTFOLIO_LLM_MODEL")

    parser = argparse.ArgumentParser(
        description="Run the supervisor/worker multi-agent orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.agents.supervisor.workflow "Should I buy NVDA today?"
  python -m src.agents.supervisor.workflow "What should I do with my portfolio?"
  python -m src.agents.supervisor.workflow "My order #1234 is damaged, I need a refund"
  python -m src.agents.supervisor.workflow --model gpt-4o "Scan the US market for opportunities"

  # Cross-agent chaining:
  python -m src.agents.supervisor.workflow "Scan for BUY signals, then check if they suit my portfolio"
""",
    )
    parser.add_argument(
        "query",
        help="Natural-language query. The supervisor decides which agent(s) to invoke.",
    )
    parser.add_argument(
        "--model",
        default=_env_model,
        metavar="MODEL_NAME",
        help=(
            "LLM model override. Provider is inferred automatically from the model name. "
            f"(current: '{_env_model or 'default: gpt-oss:120b / ollama'}') "
            "Examples: gpt-4o, gemini-1.5-pro, llama3."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        answer = run(query=args.query, model=args.model)
    except ValueError as exc:
        print(f"\n[Configuration error] {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Supervisor workflow failed unexpectedly")
        print(f"\n[Error] {type(exc).__name__}: {exc}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(answer)
    print("=" * 60)
