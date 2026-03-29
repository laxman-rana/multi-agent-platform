import logging

from src.agents.portfolio.state import PortfolioState
from src.agents.portfolio.tools.portfolio_tools import get_portfolio

logger = logging.getLogger(__name__)


class PortfolioAgent:
    """
    First node in the graph.
    Loads user profile and portfolio positions into state.
    """

    def run(self, state: PortfolioState) -> PortfolioState:
        data = get_portfolio()
        state.user_profile = data["user_profile"]
        state.portfolio = data["positions"]
        logger.info(
            "[PortfolioAgent] Loaded %d positions for %s (risk: %s)",
            len(state.portfolio),
            state.user_profile["name"],
            state.user_profile["risk_tolerance"],
        )
        return state
