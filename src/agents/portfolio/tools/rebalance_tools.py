"""
rebalance_tools.py
------------------
Deterministic portfolio-level action engine.

Analyses completed per-stock decisions alongside sector allocation and
concentration metrics to produce a single `portfolio_action` recommendation
that answers the question the per-ticker loop cannot:

    "What should we do with the PORTFOLIO as a whole?"

No LLM involved — all logic is rule-based and reproducible.
"""

from typing import Dict, List

from src.agents.portfolio.models import (
    PortfolioAction,
    Position,
    RiskMetrics,
    StockDecision,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum acceptable single-sector weight for a moderate-risk investor.
_SECTOR_TARGET_MAX_PCT = 60.0

# If the top sector exceeds this level it's flagged as critically overweight.
_SECTOR_CRITICAL_PCT = 80.0

# Minimum number of sectors for adequate diversification.
_MIN_SECTORS = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_portfolio_action(
    sector_allocation: Dict[str, float],
    decisions: Dict[str, StockDecision],
    risk_metrics: RiskMetrics,
    portfolio: List[Position],
) -> PortfolioAction:
    """
    Derive a portfolio-level action from current state.

    Parameters
    ----------
    sector_allocation : sector → current weight (%)
    decisions         : ticker → {action, confidence, reason, gain_pct}
    risk_metrics      : output of calculate_risk()
    portfolio         : raw positions list (for sector membership lookup)

    Returns
    -------
    dict with keys:
        rebalance          bool        — True when action is recommended
        reduce_sector      str | None  — overweight sector to trim
        current_exposure   float|None  — current sector weight (%)
        target_exposure    str | None  — e.g. "60%"
        priority_exits     list[str]   — tickers in reduce_sector already flagged EXIT
        add_diversification bool       — True when < _MIN_SECTORS sectors present
        missing_sectors    list[str]   — broad sectors that are absent
        summary            str         — one human-readable sentence
    """
    rebalance = False
    reduce_sector: str = ""
    current_exposure: float = 0.0
    target_exposure: float = 0.0
    priority_exits: List[str] = []
    add_diversification = False
    missing_sectors: List[str] = []
    summary = "Portfolio allocation is within acceptable bounds."

    # ── 1. Sector concentration check ────────────────────────────────────
    if sector_allocation:
        top_sector = max(sector_allocation, key=lambda s: sector_allocation[s])
        top_pct    = sector_allocation[top_sector]

        if top_pct > _SECTOR_TARGET_MAX_PCT:
            rebalance = True
            reduce_sector = top_sector
            current_exposure = round(top_pct, 1)
            target_exposure = float(_SECTOR_TARGET_MAX_PCT)

            # Tickers in the overweight sector that DecisionAgent already flagged EXIT
            sector_map: Dict[str, str] = {
                pos.ticker: pos.sector for pos in portfolio
            }
            priority_exits = [
                ticker
                for ticker, d in decisions.items()
                if d.action == "EXIT"
                and sector_map.get(ticker) == top_sector
            ]

            severity = "critically" if top_pct >= _SECTOR_CRITICAL_PCT else "significantly"
            summary = (
                f"{top_sector} is {severity} overweight at {top_pct:.1f}% "
                f"(target ≤ {int(_SECTOR_TARGET_MAX_PCT)}%). "
                + (
                    f"Priority exits: {', '.join(priority_exits)}."
                    if priority_exits
                    else "Consider trimming positions to rebalance."
                )
            )

    # ── 2. Diversification check ──────────────────────────────────────────
    _BROAD_SECTORS = [
        "Technology", "Healthcare", "Financials", "Consumer Discretionary",
        "Consumer Staples", "Industrials", "Energy", "Utilities",
        "Real Estate", "Materials", "Communication Services",
    ]
    present = set(sector_allocation.keys())
    if len(present) < _MIN_SECTORS:
        add_diversification = True
        missing_sectors = [s for s in _BROAD_SECTORS if s not in present][:4]
        diversify_note = (
            f"Only {len(present)} sector(s) present — consider adding: "
            + ", ".join(missing_sectors) + "."
        )
        summary = summary.rstrip(".") + ". " + diversify_note

    return PortfolioAction(
        rebalance=rebalance,
        reduce_sector=reduce_sector,
        current_exposure=current_exposure,
        target_exposure=target_exposure,
        priority_exits=priority_exits,
        add_diversification=add_diversification,
        missing_sectors=missing_sectors,
        summary=summary,
    )
