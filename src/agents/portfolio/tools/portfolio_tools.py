from typing import Any, Dict

from src.agents.portfolio.tools.mock_data import PORTFOLIO


def get_portfolio() -> Dict[str, Any]:
    """
    Return portfolio data for the current investor.
    Mock data is sourced from tools/mock_data.py.
    In production, replace with a brokerage API, CSV loader, or database query.
    """
    return PORTFOLIO
