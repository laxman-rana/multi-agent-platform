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

    # Tickers that passed score but were suppressed by cooldown in the scanner.
    skipped_cooldown: List[str] = field(default_factory=list)

    # Tickers that passed score+cooldown but were blocked because no deployable
    # capital was available in the portfolio context.
    blocked_no_cash: List[str] = field(default_factory=list)

    # When True (default), the zero-cash guard is bypassed so the scan produces
    # BUY signals regardless of portfolio cash balance.  Set to False to enforce
    # the hard stop (useful in live-trading mode where capital is limited).
    ignore_cash_check: bool = True

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

    # Price and score captured at the time each BUY signal was emitted.
    # Used by the freshness bypass: if price has dropped ≥5% since the last
    # signal, or a new capitulation event fires, the cooldown is lifted so a
    # new lower entry zone can be re-evaluated.
    recent_signal_context: Dict[str, Any] = field(default_factory=dict)
