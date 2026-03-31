from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class OpportunityState:
    """
    Central state object passed through every node of the AlphaScannerAgent graph.
    Each agent reads from and writes to this shared object.

    recent_signals persists across batch-scan iterations so the cooldown guard
    correctly prevents the same ticker from re-emitting within _COOLDOWN_MINUTES.
    """

    # Tickers to evaluate in this scan cycle
    watchlist: List[str] = field(default_factory=list)

    # Caller-supplied portfolio context used for guardrail checks:
    #   sector_allocation : {sector: pct_float}  — e.g. {"Technology": 45.2}
    #   position_weights  : {ticker: pct_float}  — existing position size
    #   cash_available    : float                 — deployable capital
    #   total_positions   : int
    #   top_holding_weight: float
    portfolio_context: Dict[str, Any] = field(default_factory=dict)

    # Extended market data per ticker (price, change_pct, volatility,
    # pe_ratio, forward_pe, 52w_high, 52w_low, sector, volume, avg_volume)
    market_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Tickers that passed the PreFilterEngine lightweight checks
    prefiltered: List[str] = field(default_factory=list)

    # SignalEngine output per ticker: {score, signals, tier}
    signals: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Tickers with score >= 1 and a clear cooldown window
    candidates: List[str] = field(default_factory=list)

    # LLM output per candidate ticker: {action, confidence, entry_quality, reason, type, score}
    decisions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # News sentiment per ticker (populated by NewsNode for candidates only).
    # {ticker: {sentiment: "positive"|"neutral"|"negative", catalyst: str, headline_count: int}}
    news_sentiment: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Final sorted list of BUY opportunities
    buy_opportunities: List[Dict[str, Any]] = field(default_factory=list)

    # Tickers whose market-data fetch failed: {ticker: error_message}
    scan_errors: Dict[str, str] = field(default_factory=dict)

    # Cooldown tracker: ticker -> ISO-format UTC timestamp of last emitted BUY signal.
    # Preserved across batch-scan iterations; reset only on process restart.
    recent_signals: Dict[str, str] = field(default_factory=dict)
