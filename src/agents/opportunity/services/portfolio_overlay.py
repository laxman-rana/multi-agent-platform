import logging
from typing import Any, Dict, List, Tuple

from src.agents.opportunity.state import OpportunityState

logger = logging.getLogger(__name__)


class PortfolioOverlayPolicy:
    def __init__(
        self,
        max_sector_exposure: float = 60.0,
        max_position_weight: float = 10.0,
    ) -> None:
        self._max_sector_exposure = max_sector_exposure
        self._max_position_weight = max_position_weight

    def apply(self, state: OpportunityState) -> OpportunityState:
        portfolio_context = state.portfolio_context
        sector_allocation: Dict[str, float] = portfolio_context.get("sector_allocation", {})
        position_weights: Dict[str, float] = portfolio_context.get("position_weights", {})
        cash_available: float = portfolio_context.get("cash_available", float("inf"))

        if cash_available <= 0 and not state.ignore_cash_check:
            logger.warning("[PortfolioOverlayPolicy] cash_available=%.2f — skipping all candidates", cash_available)
            state.blocked_no_cash = list(state.candidates) + list(state.skipped_cooldown)
            state.skipped_cooldown = []
            state.candidates = []
            state.buy_opportunities = []
            return state

        for ticker in state.candidates:
            market_data = state.market_data[ticker]
            sector = market_data.get("sector", "Unknown")
            sector_pct = sector_allocation.get(sector, 0.0)
            position_pct = position_weights.get(ticker, 0.0)
            if sector_pct > self._max_sector_exposure:
                logger.warning(
                    "[PortfolioOverlayPolicy] %s — sector '%s' at %.1f%% exceeds %.0f%% cap (warning only)",
                    ticker,
                    sector,
                    sector_pct,
                    self._max_sector_exposure,
                )
            if position_pct > self._max_position_weight:
                logger.warning(
                    "[PortfolioOverlayPolicy] %s — existing position %.1f%% exceeds %.0f%% cap (warning only)",
                    ticker,
                    position_pct,
                    self._max_position_weight,
                )
        return state
