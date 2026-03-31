"""
workflow.py
-----------
Entry point for the portfolio multi-agent analysis workflow.

Run from the repository root:
    python -m src.agents.portfolio.workflow
    python -m src.agents.portfolio.workflow --no-news
    python -m src.agents.portfolio.workflow --model gpt-4o
    python -m src.agents.portfolio.workflow --model gemini-1.5-pro
    python -m src.agents.portfolio.workflow --model llama3

Provider is inferred automatically from the model name.
"""

import argparse
import logging
import os
import time

from langgraph.graph import StateGraph, END

from src.agents.portfolio.state import PortfolioState
from src.observability import get_telemetry_logger
from src.agents.portfolio.subagents.portfolio_agent import PortfolioAgent
from src.agents.portfolio.subagents.risk_agent import RiskAgent
from src.agents.portfolio.subagents.market_agent import MarketAgent, VOLATILITY_THRESHOLD
from src.agents.portfolio.subagents.news_agent import NewsAgent
from src.agents.portfolio.subagents.decision_agent import DecisionAgent
from src.agents.portfolio.subagents.critic_agent import CriticAgent
from src.agents.portfolio.subagents.formatter_agent import FormatterAgent

logger = logging.getLogger(__name__)


def _passthrough(state: PortfolioState) -> PortfolioState:
    """
    No-op synchronisation node used as a fan-in barrier.

    LangGraph waits for ALL predecessor nodes to complete before invoking
    this node, so placing it after a parallel fan-out guarantees the state
    writes from every branch are merged before execution continues.
    """
    return state


def volatility_router(state: PortfolioState) -> str:
    """
    Conditional routing function called after the parallel sync barrier.

    Returns "high_volatility" if any ticker's volatility exceeds the
    threshold, otherwise returns "normal". These keys are mapped to
    node names in the graph below.
    """
    for insight in state.stock_insights.values():
        if insight.get("volatility", 0) > VOLATILITY_THRESHOLD:
            return "high_volatility"
    return "normal"


_MAX_CRITIC_RETRIES = 2


def critic_router(state: PortfolioState) -> str:
    """
    Conditional routing function called after CriticAgent.

    Returns "retry" to loop back to DecisionAgent when the critic rejected
    the decisions and we have not yet exhausted the retry budget.
    Returns "formatter" otherwise (approved, or retry budget spent).
    """
    if (
        not state.critic_feedback.get("approved", True)
        and state.critic_retry_count < _MAX_CRITIC_RETRIES
    ):
        return "retry"
    return "formatter"


def _increment_retry(state: PortfolioState) -> PortfolioState:
    state.critic_retry_count += 1
    logger.info(
        "[Graph] Critic loop retry #%d/%d",
        state.critic_retry_count,
        _MAX_CRITIC_RETRIES,
    )
    return state


def _risk_node(state: PortfolioState) -> dict:
    """
    Thin adapter so the parallel risk branch only emits the fields it owns.

    LangGraph's LastValue channel raises InvalidUpdateError when two parallel
    branches both return the full state object — it sees concurrent writes to
    every field, including ones neither branch touched.  Returning only the
    written keys avoids the conflict.
    """
    updated = RiskAgent().run(state)
    return {
        "risk_metrics":     updated.risk_metrics,
        "sector_allocation": updated.sector_allocation,
    }


def _market_node(state: PortfolioState) -> dict:
    """Same adapter pattern for the parallel market branch."""
    updated = MarketAgent().run(state)
    return {"stock_insights": updated.stock_insights}


def build_graph(skip_news: bool = False):
    """
    Assemble and wire the agent graph using LangGraph StateGraph.

    RiskAgent and MarketAgent only read state.portfolio and write to
    completely separate fields (risk_metrics vs stock_insights), so they
    run in parallel as a fan-out from PortfolioAgent. A no-op "sync" node
    acts as the fan-in barrier — LangGraph waits for both branches before
    proceeding.

    The two parallel nodes register as thin dict-returning wrappers
    (_risk_node / _market_node) instead of the agent .run() methods directly.
    This prevents LangGraph's LastValue channel from seeing concurrent writes
    to shared read-only fields (user_profile, portfolio, etc.).

    Default flow (skip_news=False):

        PortfolioAgent ──┬──► RiskAgent ───────────────────────────────┐
                         │                                              ▼
                         └──► MarketAgent ──────────────────────► sync (fan-in)
                                                                        │
                                               (conditional) ───────────┤
                                               volatility > 30%         ├──► NewsAgent ──┐
                                               all within threshold ─────┤               │
                                                                         └───────────────┼──► DecisionAgent
                                                                                         │         │
                                                                                         │    CriticAgent
                                                                                         │    (loop ≤2x)
                                                                                         │         │
                                                                                         └── FormatterAgent

    With --no-news (skip_news=True):
        PortfolioAgent → [RiskAgent ‖ MarketAgent] → sync → DecisionAgent → CriticAgent → FormatterAgent
    """
    graph = StateGraph(PortfolioState)

    # Instantiate agents once — all are stateless, so a single instance per run is fine.
    # Node names must not collide with PortfolioState field names;
    # "portfolio" and "news" are state fields, so we suffix them with "_agent".
    graph.add_node("portfolio_agent", PortfolioAgent().run)
    graph.add_node("risk",            _risk_node)
    graph.add_node("market",          _market_node)
    graph.add_node("sync",            _passthrough)   # fan-in barrier
    graph.add_node("news_agent",      NewsAgent().run)
    graph.add_node("decision",        DecisionAgent().run)
    graph.add_node("critic",          CriticAgent().run)
    graph.add_node("retry_handler",   _increment_retry)
    graph.add_node("formatter",       FormatterAgent().run)

    # Fan-out: RiskAgent and MarketAgent run in parallel.
    # They share only the read-only state.portfolio and write to distinct fields
    # (risk_metrics/sector_allocation vs stock_insights) — no conflicts.
    graph.set_entry_point("portfolio_agent")
    graph.add_edge("portfolio_agent", "risk")
    graph.add_edge("portfolio_agent", "market")

    # Fan-in: sync waits for both branches before continuing.
    graph.add_edge("risk",   "sync")
    graph.add_edge("market", "sync")

    if skip_news:
        # Bypass NewsAgent entirely
        graph.add_edge("sync", "decision")
    else:
        # Conditional branch once both data sources are ready
        graph.add_conditional_edges(
            "sync",
            volatility_router,
            {
                "high_volatility": "news_agent",
                "normal":          "decision",
            },
        )
        graph.add_edge("news_agent", "decision")

    graph.add_edge("decision", "critic")
    graph.add_conditional_edges(
        "critic",
        critic_router,
        {
            "retry":     "retry_handler",
            "formatter": "formatter",
        },
    )
    graph.add_edge("retry_handler", "decision")
    graph.add_edge("formatter", END)

    return graph.compile()


def main(skip_news: bool = False, model: str | None = None) -> None:
    # Resolve model → infer provider automatically.
    # If --model is not given, fall back to PORTFOLIO_LLM_MODEL env var, then
    # default to the ollama built-in default.
    from src.llm.providers import _DEFAULT_MODELS, infer_provider

    _env_model = os.getenv("PORTFOLIO_LLM_MODEL")
    resolved_model = model or _env_model

    if resolved_model:
        try:
            resolved_provider = infer_provider(resolved_model)
        except ValueError as exc:
            print(f"\n[Configuration error] {exc}")
            raise SystemExit(1) from None
        os.environ["PORTFOLIO_LLM_PROVIDER"] = resolved_provider
        os.environ["PORTFOLIO_LLM_MODEL"] = resolved_model
    else:
        # No model specified — use defaults
        resolved_provider = os.getenv("PORTFOLIO_LLM_PROVIDER", "ollama")
        resolved_model = _DEFAULT_MODELS.get(resolved_provider, "(default)")

    _provider = resolved_provider
    _model = resolved_model

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )
    telemetry = get_telemetry_logger()

    logger.info("=" * 60)
    logger.info("STARTING PORTFOLIO ANALYSIS WORKFLOW")
    logger.info("=" * 60)
    logger.info("Provider : %s  |  Model : %s", _provider, _model)

    telemetry.log_event("workflow_start", {"skip_news": skip_news})
    start = time.monotonic()

    try:
        compiled = build_graph(skip_news=skip_news)
        result = compiled.invoke(PortfolioState())
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        print(f"\n[Configuration error] {exc}")
        raise SystemExit(1) from None
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        print(
            f"\n[Missing dependency] {exc}\n"
            f"Install the required package for provider '{_provider}' and retry.\n"
            f"  ollama : pip install langchain-ollama\n"
            f"  openai : pip install langchain-openai\n"
            f"  google : pip install langchain-google-genai"
        )
        raise SystemExit(1) from None
    except Exception as exc:
        logger.exception("Workflow failed unexpectedly")
        print(f"\n[Workflow error] {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None

    final_state = result if isinstance(result, PortfolioState) else PortfolioState(**result)

    elapsed = round(time.monotonic() - start, 2)
    decisions = final_state.decisions
    telemetry.log_event(
        "workflow_complete",
        {
            "duration_seconds": elapsed,
            "tickers": list(decisions.keys()),
            "actions": {t: d["action"] for t, d in decisions.items()},
            "approved": final_state.critic_feedback.get("approved", True),
            "warnings": len(final_state.critic_feedback.get("warnings", [])),
        },
    )

    logger.info("Workflow completed in %.2fs", elapsed)
    print("\n" + final_state.final_output)


if __name__ == "__main__":
    _env_model = os.getenv("PORTFOLIO_LLM_MODEL")

    parser = argparse.ArgumentParser(description="Run the portfolio analysis workflow.")
    parser.add_argument(
        "--no-news",
        action="store_true",
        help="Skip the NewsAgent even when high-volatility tickers are detected.",
    )
    parser.add_argument(
        "--model",
        default=_env_model,
        metavar="MODEL_NAME",
        help=(
            "Model to use. Provider is inferred automatically from the model name. "
            f"(current: '{_env_model or 'default: gpt-oss:120b / ollama'}') "
            "Examples: gpt-4o, gemini-1.5-pro, llama3, gpt-4-turbo."
        ),
    )
    args = parser.parse_args()
    main(skip_news=args.no_news, model=args.model)
