from typing import Any, Dict, List


def calculate_risk(
    positions: List[Dict[str, Any]],
    stock_insights: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate portfolio-level risk metrics from positions and market data.

    Returns:
        total_portfolio_value, unrealized_pnl, unrealized_pnl_pct,
        weighted_volatility, sector_allocation, concentration_risk label.
    """
    total_value = 0.0
    sector_values: Dict[str, float] = {}
    weighted_volatility = 0.0
    unrealized_pnl = 0.0

    for pos in positions:
        ticker = pos["ticker"]
        shares = pos["shares"]
        avg_cost = pos["avg_cost"]
        sector = pos["sector"]

        # Use current price from insights if available; fallback to avg_cost
        current_price = stock_insights.get(ticker, {}).get("price", avg_cost)
        volatility = stock_insights.get(ticker, {}).get("volatility", 0.20)

        market_value = shares * current_price
        cost_basis = shares * avg_cost

        total_value += market_value
        sector_values[sector] = sector_values.get(sector, 0.0) + market_value
        unrealized_pnl += market_value - cost_basis
        weighted_volatility += market_value * volatility

    if total_value == 0:
        return {}

    # Normalize weighted volatility by total portfolio value
    weighted_volatility /= total_value

    sector_allocation = {
        sector: round((val / total_value) * 100, 2)
        for sector, val in sector_values.items()
    }

    top_sector = max(sector_allocation, key=lambda s: sector_allocation[s])
    top_pct = sector_allocation[top_sector]

    if top_pct > 40:
        concentration_risk = "high"
    elif top_pct > 25:
        concentration_risk = "moderate"
    else:
        concentration_risk = "low"

    return {
        "total_portfolio_value": round(total_value, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "unrealized_pnl_pct": round(
            (unrealized_pnl / (total_value - unrealized_pnl)) * 100, 2
        ),
        "weighted_volatility": round(weighted_volatility, 4),
        "sector_allocation": sector_allocation,
        "top_sector": top_sector,
        "concentration_risk": concentration_risk,
    }
