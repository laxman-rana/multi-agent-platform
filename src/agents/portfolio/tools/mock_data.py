"""
mock_data.py
------------
Mock investor profile and portfolio positions used during development.

To swap for a real source, replace the get_portfolio() function body in
portfolio_tools.py (e.g. brokerage API, CSV loader, or database query).
"""

from typing import Any, Dict


# ---------------------------------------------------------------------------
# Investor profile + positions
# ---------------------------------------------------------------------------

PORTFOLIO: Dict[str, Any] = {
    "user_profile": {
        "name": "Alex Johnson",
        "risk_tolerance": "moderate",
        "investment_horizon": "5 years",
        "total_invested": 52_000.0,
    },
    "positions": [
        {"ticker": "META", "shares":  2, "avg_cost": 586.0, "sector": "Technology"},
        {"ticker": "MSFT", "shares": 15, "avg_cost": 403.0, "sector": "Technology"},
        {"ticker": "NVDA",  "shares": 30, "avg_cost":  173.0, "sector": "Technology"},
        {"ticker": "HOOD", "shares": 10, "avg_cost": 75.0, "sector": "Financials"},
        {"ticker": "SOFI",  "shares": 25, "avg_cost": 18.0, "sector": "Financials"},
        {"ticker": "AMZN", "shares":  8, "avg_cost": 209.0, "sector": "Technology"},
    ],
}
