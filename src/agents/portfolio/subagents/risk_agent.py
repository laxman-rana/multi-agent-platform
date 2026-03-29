import logging

from src.agents.portfolio.state import PortfolioState
from src.agents.portfolio.tools.risk_tools import calculate_risk
from src.observability import get_telemetry_logger

logger = logging.getLogger(__name__)


class RiskAgent:
    """
    Calculates portfolio-level risk metrics and sector allocation.
    Requires positions already loaded by PortfolioAgent.
    Stock insights are optional at this stage; RiskAgent can run
    with empty insights and will be re-evaluated after MarketAgent.
    """

    def run(self, state: PortfolioState) -> PortfolioState:
        metrics = calculate_risk(state.portfolio, state.stock_insights)
        state.risk_metrics = metrics
        state.sector_allocation = metrics.get("sector_allocation", {})

        high_conc_stocks = metrics.get("high_concentration_stocks", [])

        get_telemetry_logger().log_event(
            "risk_calculated",
            {
                "total_portfolio_value": metrics.get("total_portfolio_value"),
                "unrealized_pnl": metrics.get("unrealized_pnl"),
                "unrealized_pnl_pct": metrics.get("unrealized_pnl_pct"),
                "weighted_volatility": metrics.get("weighted_volatility"),
                "concentration_risk": metrics.get("concentration_risk"),
                "top_sector": metrics.get("top_sector"),
                "top_stock": metrics.get("top_stock"),
                "high_concentration": metrics.get("high_concentration"),
                "high_concentration_stocks": high_conc_stocks,
            },
        )

        conc_note = (
            f" ⚠ High-concentration stocks: {high_conc_stocks}" if high_conc_stocks else ""
        )
        stock_alloc = metrics.get("stock_allocation", {})
        top_stock = metrics.get("top_stock", "")
        logger.info(
            "[RiskAgent] Concentration: %s | Top sector: %s (%.1f%%) | Top stock: %s (%.1f%%) | Value: $%s%s",
            metrics.get("concentration_risk", "unknown").upper(),
            metrics.get("top_sector"),
            metrics.get("sector_allocation", {}).get(metrics.get("top_sector", ""), 0),
            top_stock,
            stock_alloc.get(top_stock, 0),
            f"{metrics.get('total_portfolio_value', 0):,.2f}",
            conc_note,
        )
        return state
