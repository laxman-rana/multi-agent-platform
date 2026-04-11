"""
portfolio_worker.py
-------------------
Supervisor worker adapter over the Portfolio multi-agent pipeline.

Wraps src.agents.portfolio.workflow.build_graph() — no changes to
the underlying pipeline.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from src.agents.portfolio.state import PortfolioState
from src.agents.portfolio.workflow import build_graph

from . import BaseWorker, WorkerName, worker

logger = logging.getLogger(__name__)


class PortfolioInput(BaseModel):
    skip_news: bool = Field(
        False,
        description=(
            "Set to true to skip news sentiment analysis even when high-volatility "
            "tickers are detected. Useful for faster analysis or when news data is unavailable."
        ),
    )


@worker
class PortfolioWorker(BaseWorker):
    """Analyses the investor's portfolio and produces HOLD/EXIT/DOUBLE_DOWN decisions."""

    name = WorkerName.PORTFOLIO
    description = (
        "Analyse the current equity portfolio and produce structured HOLD, EXIT, or DOUBLE_DOWN "
        "trade decisions with confidence levels and reasoning. Use this when the user asks about "
        "their portfolio performance, what to do with their holdings, risk exposure, or "
        "rebalancing recommendations. Returns a formatted trade-decision report."
    )
    input_schema = PortfolioInput

    def invoke(self, skip_news: bool = False) -> str:
        logger.info("[PortfolioWorker] Running portfolio analysis — skip_news=%s", skip_news)
        compiled = build_graph(skip_news=skip_news)
        result = compiled.invoke(PortfolioState())
        state = result if isinstance(result, PortfolioState) else PortfolioState(**result)
        return state.final_output or "Portfolio analysis completed but produced no output."



