from typing import Dict, List

from src.agents.portfolio.models import Position, RiskMetrics, StockInsight


def calculate_risk(
    positions: List[Position],
    stock_insights: Dict[str, StockInsight],
) -> RiskMetrics:
    """
    Calculate portfolio-level risk metrics from positions and market data.

    Returns:
        total_portfolio_value, unrealized_pnl, unrealized_pnl_pct,
        weighted_volatility, sector_allocation, stock_allocation,
        concentration_risk label, top_stock, high_concentration flag.
    """
    total_value = 0.0
    sector_values: Dict[str, float] = {}
    stock_market_values: Dict[str, float] = {}  # per-ticker market value for concentration
    weighted_volatility = 0.0
    unrealized_pnl = 0.0

    for pos in positions:
        ticker = pos.ticker
        shares = pos.shares
        avg_cost = pos.avg_cost
        sector = pos.sector

        # Use current price from insights if available; fallback to avg_cost
        insight = stock_insights.get(ticker)
        current_price = insight.price if insight else avg_cost
        volatility = insight.volatility if insight else 0.20

        market_value = shares * current_price
        cost_basis = shares * avg_cost

        total_value += market_value
        stock_market_values[ticker] = market_value
        sector_values[sector] = sector_values.get(sector, 0.0) + market_value
        unrealized_pnl += market_value - cost_basis
        weighted_volatility += market_value * volatility

    if total_value == 0:
        return RiskMetrics()

    # Normalize weighted volatility by total portfolio value
    weighted_volatility /= total_value

    sector_allocation = {
        sector: round((val / total_value) * 100, 2)
        for sector, val in sector_values.items()
    }

    # Per-stock weight — used by DecisionAgent to flag concentration risk in prompts
    stock_allocation = {
        ticker: round((val / total_value) * 100, 2)
        for ticker, val in stock_market_values.items()
    }

    top_sector = max(sector_allocation, key=lambda s: sector_allocation[s])
    top_pct = sector_allocation[top_sector]

    top_stock = max(stock_allocation, key=lambda t: stock_allocation[t])

    # Stocks whose individual weight exceeds 40% of the portfolio
    high_concentration_stocks = [
        t for t, pct in stock_allocation.items() if pct > 40
    ]

    if top_pct > 40:
        concentration_risk = "high"
    elif top_pct > 25:
        concentration_risk = "moderate"
    else:
        concentration_risk = "low"

    return RiskMetrics(
        total_portfolio_value=round(total_value, 2),
        unrealized_pnl=round(unrealized_pnl, 2),
        unrealized_pnl_pct=round(
            (unrealized_pnl / (total_value - unrealized_pnl)) * 100, 2
        ),
        weighted_volatility=round(weighted_volatility, 4),
        sector_allocation=sector_allocation,
        stock_allocation=stock_allocation,
        top_sector=top_sector,
        top_stock=top_stock,
        high_concentration_stocks=high_concentration_stocks,
        high_concentration=len(high_concentration_stocks) > 0,
        concentration_risk=concentration_risk,
    )
