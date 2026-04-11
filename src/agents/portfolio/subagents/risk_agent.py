import logging

from src.agents.base import BaseAgent
from src.agents.portfolio.state import PortfolioState
from src.agents.portfolio.tools.risk_tools import calculate_risk
from src.observability import get_telemetry_logger

logger = logging.getLogger(__name__)


class RiskAgent(BaseAgent):
    """
    Calculates portfolio-level risk metrics and sector allocation.
    Requires positions already loaded by PortfolioAgent.
    Stock insights are optional at this stage; RiskAgent can run
    with empty insights and will be re-evaluated after MarketAgent.
    """

    def run(self, state: PortfolioState) -> PortfolioState:
        metrics = calculate_risk(state.portfolio, state.stock_insights)
        state.risk_metrics = metrics
        state.sector_allocation = metrics.sector_allocation

        high_conc_stocks = metrics.high_concentration_stocks

        get_telemetry_logger().log_event(
            "risk_calculated",
            {
                "total_portfolio_value": metrics.total_portfolio_value,
                "unrealized_pnl": metrics.unrealized_pnl,
                "unrealized_pnl_pct": metrics.unrealized_pnl_pct,
                "weighted_volatility": metrics.weighted_volatility,
                "concentration_risk": metrics.concentration_risk,
                "top_sector": metrics.top_sector,
                "top_stock": metrics.top_stock,
                "high_concentration": metrics.high_concentration,
                "high_concentration_stocks": high_conc_stocks,
            },
        )

        conc_note = (
            f" ⚠ High-concentration stocks: {high_conc_stocks}" if high_conc_stocks else ""
        )
        logger.info(
            "[RiskAgent] Concentration: %s | Top sector: %s (%.1f%%) | Top stock: %s (%.1f%%) | Value: $%s%s",
            metrics.concentration_risk.upper(),
            metrics.top_sector,
            metrics.sector_allocation.get(metrics.top_sector, 0),
            metrics.top_stock,
            metrics.stock_allocation.get(metrics.top_stock, 0),
            f"{metrics.total_portfolio_value:,.2f}",
            conc_note,
        )
        return state
