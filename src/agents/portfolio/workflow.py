"""
workflow.py
-----------
Entry point for the portfolio multi-agent analysis workflow.

Run from the repository root:
    python -m src.agents.portfolio.workflow
    python -m src.agents.portfolio.workflow --no-news
"""

import argparse
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


def volatility_router(state: PortfolioState) -> str:
    """
    Conditional routing function called after MarketAgent.

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
    print(
        f"  [Graph] Critic loop retry #{state.critic_retry_count}/{_MAX_CRITIC_RETRIES}"
    )
    return state


def build_graph(skip_news: bool = False):
    """
    Assemble and wire the agent graph using LangGraph StateGraph.

    Default flow (skip_news=False):
        PortfolioAgent → RiskAgent → MarketAgent
                                          │
                           (conditional) ─┤ volatility > 30%  ──► NewsAgent ──┐
                                          │                                    │
                                          └─ all within threshold ─────────────┤
                                                                               ▼
                                                                        DecisionAgent
                                                                               │
                                                                        CriticAgent
                                                                               │
                                                                       FormatterAgent

    With --no-news (skip_news=True):
        PortfolioAgent → RiskAgent → MarketAgent → DecisionAgent → CriticAgent → FormatterAgent
    """
    graph = StateGraph(PortfolioState)

    # Instantiate agents once — all are stateless, so a single instance per run is fine.
    # Node names must not collide with PortfolioState field names;
    # "portfolio" and "news" are state fields, so we suffix them with "_agent".
    graph.add_node("portfolio_agent", PortfolioAgent().run)
    graph.add_node("risk",            RiskAgent().run)
    graph.add_node("market",          MarketAgent().run)
    graph.add_node("news_agent",      NewsAgent().run)
    graph.add_node("decision",        DecisionAgent().run)
    graph.add_node("critic",          CriticAgent().run)
    graph.add_node("retry_handler",   _increment_retry)
    graph.add_node("formatter",       FormatterAgent().run)

    # Entry point and linear edges
    graph.set_entry_point("portfolio_agent")
    graph.add_edge("portfolio_agent", "risk")
    graph.add_edge("risk",            "market")

    if skip_news:
        # Bypass NewsAgent entirely
        graph.add_edge("market", "decision")
    else:
        # Conditional branch after market data is fetched
        graph.add_conditional_edges(
            "market",
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


def main(skip_news: bool = False) -> None:
    telemetry = get_telemetry_logger()

    print("\n" + "=" * 70)
    print("  STARTING PORTFOLIO ANALYSIS WORKFLOW")
    print("=" * 70)

    telemetry.log_event("workflow_start", {"skip_news": skip_news})
    start = time.monotonic()

    compiled = build_graph(skip_news=skip_news)
    result = compiled.invoke(PortfolioState())

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

    print("\n" + final_state.final_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the portfolio analysis workflow.")
    parser.add_argument(
        "--no-news",
        action="store_true",
        help="Skip the NewsAgent even when high-volatility tickers are detected.",
    )
    args = parser.parse_args()
    main(skip_news=args.no_news)
