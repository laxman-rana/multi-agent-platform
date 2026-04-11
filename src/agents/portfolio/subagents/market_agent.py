import logging

from src.agents.base import BaseAgent
from src.agents.portfolio.models import StockInsight
from src.agents.portfolio.state import PortfolioState
from src.agents.portfolio.tools.market_tools import get_stock_data
from src.observability import get_telemetry_logger

logger = logging.getLogger(__name__)

# Tickers with volatility above this value trigger the NewsAgent branch
VOLATILITY_THRESHOLD = 0.30


class MarketAgent(BaseAgent):
    """
    Fetches current market data for every position in the portfolio.
    Populates state.stock_insights with enriched per-ticker data.
    The graph engine uses this node's output for the conditional routing decision.
    """

    def run(self, state: PortfolioState) -> PortfolioState:
        insights: dict = {}

        telemetry = get_telemetry_logger()

        for pos in state.portfolio:
            ticker = pos.ticker
            data = get_stock_data(ticker)
            telemetry.log_tool_usage(
                "get_stock_data",
                {"ticker": ticker},
                {"price": data.get("price"), "volatility": data.get("volatility"), "change_pct": data.get("change_pct")},
            )

            # Enrich market data with portfolio cost-basis context
            data["avg_cost"] = pos.avg_cost
            data["shares"] = pos.shares
            data["sector"] = pos.sector
            data["unrealized_pnl"] = round(
                (data["price"] - pos.avg_cost) * pos.shares, 2
            )
            insights[ticker] = StockInsight.model_validate(data)

        state.stock_insights = insights

        high_vol = [t for t, d in insights.items() if d.volatility > VOLATILITY_THRESHOLD]
        logger.info(
            "[MarketAgent] Fetched data for %d tickers | High-volatility (>%s): %s",
            len(insights),
            f"{VOLATILITY_THRESHOLD:.0%}",
            high_vol or ["none"],
        )
        return state
