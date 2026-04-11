from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.agents.portfolio.models import (
    CriticFeedback,
    PortfolioAction,
    Position,
    RiskMetrics,
    StockDecision,
    StockInsight,
    NewsArticle,
    UserProfile,
)


@dataclass
class PortfolioState:
    """
    Central state object passed through every node of the agent graph.
    Each agent reads from and writes to this shared object.
    """

    # Who the investor is and their risk preferences
    user_profile: UserProfile = field(default_factory=UserProfile)

    # Raw positions
    portfolio: List[Position] = field(default_factory=list)

    # Sector allocation percentages derived from portfolio values
    sector_allocation: Dict[str, float] = field(default_factory=dict)

    # Portfolio-level risk metrics (concentration, volatility, PnL, etc.)
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)

    # Per-ticker market data enriched with cost basis context
    stock_insights: Dict[str, StockInsight] = field(default_factory=dict)

    # Per-ticker news articles when high-volatility routing is triggered
    news: Dict[str, List[NewsArticle]] = field(default_factory=dict)

    # LLM-generated decisions: ticker → StockDecision
    decisions: Dict[str, StockDecision] = field(default_factory=dict)

    # Critic validation result
    critic_feedback: CriticFeedback = field(default_factory=CriticFeedback)

    # Portfolio-level rebalance recommendation computed by DecisionAgent
    portfolio_action: Optional[PortfolioAction] = None

    # Number of times DecisionAgent has been re-run due to critic rejection.
    # Managed by the graph layer (workflow.py) — agents must not write this field.
    critic_retry_count: int = 0

    # Final human-readable report produced by FormatterAgent
    final_output: str = ""
