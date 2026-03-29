from src.agents.portfolio.state import PortfolioState
from src.agents.portfolio.tools.portfolio_tools import get_portfolio


class PortfolioAgent:
    """
    First node in the graph.
    Loads user profile and portfolio positions into state.
    """

    def run(self, state: PortfolioState) -> PortfolioState:
        data = get_portfolio()
        state.user_profile = data["user_profile"]
        state.portfolio = data["positions"]
        print(
            f"  Loaded {len(state.portfolio)} positions "
            f"for {state.user_profile['name']} "
            f"(risk: {state.user_profile['risk_tolerance']})"
        )
        return state
