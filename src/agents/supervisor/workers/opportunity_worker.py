"""
opportunity_worker.py
---------------------
Supervisor worker adapter over the AlphaScannerAgent pipeline.

Wraps src.agents.opportunity.workflow.trigger_scan() — no changes to
the underlying pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import BaseModel, Field

from src.agents.opportunity.workflow import trigger_scan

from . import BaseWorker, WorkerName, worker

logger = logging.getLogger(__name__)


class OpportunityInput(BaseModel):
    tickers: Optional[list[str]] = Field(
        None,
        description=(
            "Explicit list of ticker symbols to scan, e.g. ['AAPL', 'NVDA']. "
            "Leave empty to scan the top liquid universe automatically."
        ),
    )
    market: str = Field(
        "US",
        description="Market to scan. 'US' for S&P 500/NYSE (default), 'IN' for NIFTY 50/NSE.",
    )
    top_n: Optional[int] = Field(
        None,
        description="When tickers is None, scan the top-N most liquid stocks. Defaults to 200.",
    )


@worker
class OpportunityWorker(BaseWorker):
    """Scans live market data for high-quality BUY signals."""

    name = WorkerName.OPPORTUNITY
    description = (
        "Scan the live market for BUY signal opportunities using quantitative signals "
        "and LLM-based news sentiment analysis. Use this when the user asks which stocks "
        "to buy, wants to find market opportunities, or requests a BUY signal scan. "
        "Returns a ranked list of candidates with confidence scores and news catalysts."
    )
    input_schema = OpportunityInput

    def invoke(
        self,
        tickers: Optional[list[str]] = None,
        market: str = "US",
        top_n: Optional[int] = None,
    ) -> str:
        logger.info(
            "[OpportunityWorker] Scanning — tickers=%s market=%s top_n=%s",
            tickers,
            market,
            top_n,
        )
        results = trigger_scan(tickers=tickers, top_n=top_n, market=market)
        if not results:
            return json.dumps({"message": "No BUY signals found for the given scan parameters."})
        return json.dumps(results, default=str)



