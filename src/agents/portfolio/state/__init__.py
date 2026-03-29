from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PortfolioState:
    """
    Central state object passed through every node of the agent graph.
    Each agent reads from and writes to this shared object.
    """

    # Who the investor is and their risk preferences
    user_profile: Dict[str, Any] = field(default_factory=dict)

    # Raw positions: list of {ticker, shares, avg_cost, sector}
    portfolio: List[Dict[str, Any]] = field(default_factory=list)

    # Sector allocation percentages derived from portfolio values
    sector_allocation: Dict[str, float] = field(default_factory=dict)

    # Portfolio-level risk metrics (concentration, volatility, PnL, etc.)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)

    # Per-ticker market data enriched with cost basis context
    stock_insights: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Per-ticker news articles when high-volatility routing is triggered
    news: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)

    # LLM-generated decisions: ticker → {action, reason, confidence, gain_pct}
    decisions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Critic validation result: approved flag + per-ticker issues + warnings
    critic_feedback: Dict[str, Any] = field(default_factory=dict)

    # Number of times DecisionAgent has been re-run due to critic rejection.
    # Managed by the graph layer (workflow.py) — agents must not write this field.
    critic_retry_count: int = 0

    # Final human-readable report produced by FormatterAgent
    final_output: str = ""
