"""
models.py
---------
Typed domain models for the portfolio agent's state fields.

Replacing every ``Dict[str, Any]`` in PortfolioState with these models
makes read/write contracts explicit, enables IDE auto-complete, and surfaces
schema mismatches at instantiation time rather than at runtime dict access.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Investor / position inputs
# ---------------------------------------------------------------------------


class UserProfile(BaseModel):
    name: str = ""
    risk_tolerance: str = ""
    investment_horizon: str = ""
    total_invested: float = 0.0


class Position(BaseModel):
    ticker: str
    shares: float
    avg_cost: float
    sector: str


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


class StockInsight(BaseModel):
    """
    Per-ticker market data enriched by MarketAgent with portfolio context.

    ``week_52_high`` / ``week_52_low`` are stored under the aliases
    ``"52w_high"`` / ``"52w_low"`` so the raw data returned by market_tools
    can be validated without any key renaming.
    """

    price: float = 0.0
    change_pct: float = 0.0
    volatility: float = 0.0
    avg_cost: float = 0.0
    shares: float = 0.0
    unrealized_pnl: float = 0.0
    sector: str = ""
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    week_52_high: float = Field(default=0.0, alias="52w_high")
    week_52_low: float = Field(default=0.0, alias="52w_low")
    volume: int = 0
    avg_volume: int = 0

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------


class NewsArticle(BaseModel):
    headline: str
    sentiment: str


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


class StockDecision(BaseModel):
    action: Literal["EXIT", "REDUCE", "HOLD", "DOUBLE_DOWN"]
    confidence: Literal["high", "moderate", "low"]
    reason: str
    allocation_change: str = "0%"
    gain_pct: float = 0.0


# ---------------------------------------------------------------------------
# Critic feedback
# ---------------------------------------------------------------------------


class CriticTickerFeedback(BaseModel):
    status: Literal["ok", "flagged"]
    issues: List[str] = []


class CriticFeedback(BaseModel):
    approved: bool = True
    warnings: List[str] = []
    per_ticker: dict[str, CriticTickerFeedback] = {}
    feedback: str = ""


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------


class RiskMetrics(BaseModel):
    total_portfolio_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    weighted_volatility: float = 0.0
    sector_allocation: dict[str, float] = {}
    stock_allocation: dict[str, float] = {}
    concentration_risk: str = ""
    top_sector: str = ""
    top_stock: str = ""
    high_concentration: bool = False
    high_concentration_stocks: List[str] = []


# ---------------------------------------------------------------------------
# Portfolio action (rebalancing recommendation)
# ---------------------------------------------------------------------------


class PortfolioAction(BaseModel):
    rebalance: bool = False
    reduce_sector: str = ""
    current_exposure: float = 0.0
    target_exposure: float = 0.0
    priority_exits: List[str] = []
    add_diversification: bool = False
    missing_sectors: List[str] = []
    summary: str = "Portfolio allocation is within acceptable bounds."
