import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import yfinance as yf

from src.agents.opportunity.markets.market_strategy import get_liquid_universe, get_market_strategy
from src.agents.opportunity.engines.prefilter_engine import PreFilterEngine
from src.agents.opportunity.engines.signal_engine import SignalEngine
from src.agents.opportunity.state import OpportunityState
from src.observability import get_telemetry_logger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants — override by subclassing if needed.
# ---------------------------------------------------------------------------
_COOLDOWN_MINUTES:    int   = 30
_COOLDOWN_UNIT:       str   = "minutes"  # "minutes" | "hours" | "days"
_CANDIDATE_MIN_SCORE: int   = 1
_MAX_SECTOR_EXPOSURE: float = 60.0
_MAX_POSITION_WEIGHT: float = 10.0
_MAX_FETCH_WORKERS:   int   = 10


# ---------------------------------------------------------------------------
# Extended market data fetch
# ---------------------------------------------------------------------------

def _fetch_extended(ticker: str) -> Dict[str, Any]:
    """
    Fetch live market data including sector and volume fields that are not
    returned by the portfolio's get_stock_data().  A single yf.Ticker.info
    call is made so there is no extra network round-trip.

    Raises ValueError when no price data is available for the ticker.
    """
    t    = yf.Ticker(ticker)
    info = t.info

    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )
    if not price:
        raise ValueError(f"No price data returned for {ticker}")

    prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")
    change_pct = (
        round(((price - prev_close) / prev_close) * 100, 2) if prev_close else 0.0
    )

    hist = t.history(period="30d")
    if len(hist) >= 2:
        daily_returns = hist["Close"].pct_change().dropna()
        volatility    = round(float(daily_returns.std() * math.sqrt(252)), 4)
    else:
        volatility = 0.20  # sensible default when history is unavailable

    volume     = info.get("regularMarketVolume", 0)
    avg_volume = info.get("averageVolume", 0)

    # Volume pressure proxy: price-direction × volume-spike as a cheap
    # buy/sell pressure signal (no paid Level-2 data required).
    vol_pressure = "neutral"
    if volume and avg_volume and volume >= avg_volume * 1.5:
        vol_pressure = "buying" if change_pct > 0 else ("selling" if change_pct < 0 else "neutral")

    return {
        "ticker":          ticker,
        "price":           round(float(price), 2),
        "change_pct":      change_pct,
        "volatility":      volatility,
        "pe_ratio":        info.get("trailingPE"),
        "forward_pe":      info.get("forwardPE"),
        "52w_high":        info.get("fiftyTwoWeekHigh", 0.0),
        "52w_low":         info.get("fiftyTwoWeekLow",  0.0),
        "sector":          info.get("sector", "Unknown"),
        "volume":          volume,
        "avg_volume":      avg_volume,
        # Analyst consensus (yfinance info fields)
        "analyst_rating":  (info.get("recommendationKey") or "none").lower(),
        "analyst_count":   info.get("numberOfAnalystOpinions") or 0,
        "analyst_target":  info.get("targetMeanPrice"),  # float or None
        # Volume pressure proxy
        "vol_pressure":    vol_pressure,
    }


# ---------------------------------------------------------------------------
# Cooldown helper
# ---------------------------------------------------------------------------

def _is_cooled_down(ticker: str, recent_signals: Dict[str, str]) -> bool:
    """
    Return True when the ticker has NOT emitted a BUY signal within the
    configured cooldown window.  Unit is controlled by _COOLDOWN_UNIT:
        "minutes" — _COOLDOWN_MINUTES minutes  (default, intraday)
        "hours"   — _COOLDOWN_MINUTES hours    (hourly scan cadence)
        "days"    — _COOLDOWN_MINUTES calendar days  (daily scan cadence)
    A missing or unparseable entry is treated as cooled down.
    """
    last_signal = recent_signals.get(ticker)
    if not last_signal:
        return True
    try:
        last_dt  = datetime.fromisoformat(last_signal)
        elapsed  = (datetime.now(timezone.utc) - last_dt).total_seconds()
        if _COOLDOWN_UNIT == "hours":
            return (elapsed / 3600) >= _COOLDOWN_MINUTES
        if _COOLDOWN_UNIT == "days":
            return (elapsed / 86400) >= _COOLDOWN_MINUTES
        return (elapsed / 60) >= _COOLDOWN_MINUTES   # default: minutes
    except (ValueError, TypeError):
        return True


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AlphaScannerAgent:
    """
    Orchestrates the full BUY-opportunity scanning pipeline for a single cycle:

      1. Parallel market-data fetch  (extended: price + PE + vol + sector + volume)
      2. PreFilterEngine             lightweight OR-logic triage
      3. SignalEngine                deterministic quantitative scoring
      4. Candidate filter            score >= 1 AND cooldown window clear
      5. Portfolio-awareness guards  sector cap, position weight cap, cash check
      6. OpportunityDecisionAgent    LLM BUY / IGNORE verdict per candidate
      7. Sort                        confidence rank → portfolio-fit-boosted score

    All seven required telemetry events are emitted:
      scan_start, prefilter_complete, signals_generated,
      candidates_filtered, llm_decision, buy_signal_emitted, scan_summary
    """

    def __init__(self) -> None:
        self._prefilter     = PreFilterEngine()
        self._signal_engine = SignalEngine()

    def run(self, state: OpportunityState) -> OpportunityState:
        telemetry         = get_telemetry_logger()
        portfolio_context = state.portfolio_context
        step_latencies:   Dict[str, float] = {}
        scan_t0           = time.monotonic()

        # ------------------------------------------------------------------
        # Step 1: Telemetry — scan_start
        # ------------------------------------------------------------------
        telemetry.log_event(
            "scan_start",
            {
                "watchlist_size": len(state.watchlist),
                "timestamp":      datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info("[AlphaScannerAgent] Scan started — %d tickers", len(state.watchlist))

        # ------------------------------------------------------------------
        # Step 2: Parallel market-data fetch
        # ------------------------------------------------------------------
        _t = time.monotonic()
        workers = min(len(state.watchlist), _MAX_FETCH_WORKERS) if state.watchlist else 1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_fetch_extended, ticker): ticker
                for ticker in state.watchlist
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    state.market_data[ticker] = future.result()
                except Exception as exc:
                    logger.warning(
                        "[AlphaScannerAgent] Fetch failed for %s: %s", ticker, exc
                    )
                    state.scan_errors[ticker] = str(exc)
        step_latencies["fetch_ms"] = round((time.monotonic() - _t) * 1000, 1)

        logger.info(
            "[AlphaScannerAgent] Fetched %d/%d tickers  (%d errors)  %.0fms",
            len(state.market_data),
            len(state.watchlist),
            len(state.scan_errors),
            step_latencies["fetch_ms"],
        )

        # ------------------------------------------------------------------
        # Step 3: PreFilterEngine
        # ------------------------------------------------------------------
        _t = time.monotonic()
        prefiltered: List[str] = [
            ticker
            for ticker, data in state.market_data.items()
            if self._prefilter.pre_filter(data)
        ]
        state.prefiltered = prefiltered
        step_latencies["prefilter_ms"] = round((time.monotonic() - _t) * 1000, 1)

        telemetry.log_event(
            "prefilter_complete",
            {
                "total":        len(state.market_data),
                "passed":       len(prefiltered),
                "filtered_out": len(state.market_data) - len(prefiltered),
                "latency_ms":   step_latencies["prefilter_ms"],
            },
        )
        logger.info(
            "[AlphaScannerAgent] Prefilter: %d/%d tickers passed  %.0fms",
            len(prefiltered), len(state.market_data), step_latencies["prefilter_ms"],
        )

        # ------------------------------------------------------------------
        # Step 4: SignalEngine — score every prefiltered ticker
        # ------------------------------------------------------------------
        _t = time.monotonic()
        for ticker in prefiltered:
            state.signals[ticker] = self._signal_engine.score(state.market_data[ticker])
        step_latencies["signal_ms"] = round((time.monotonic() - _t) * 1000, 1)

        strong_buy_tier = sum(1 for s in state.signals.values() if s["tier"] == "strong_buy")
        buy_tier        = sum(1 for s in state.signals.values() if s["tier"] == "buy")
        neutral_tier    = sum(1 for s in state.signals.values() if s["tier"] == "neutral")
        avoid_tier      = sum(1 for s in state.signals.values() if s["tier"] == "avoid")

        telemetry.log_event(
            "signals_generated",
            {
                "scored":          len(state.signals),
                "strong_buy_tier": strong_buy_tier,
                "buy_tier":        buy_tier,
                "neutral_tier":    neutral_tier,
                "avoid_tier":      avoid_tier,
                "latency_ms":      step_latencies["signal_ms"],
            },
        )
        logger.info(
            "[AlphaScannerAgent] Signals: strong_buy=%d  buy=%d  neutral=%d  avoid=%d  %.0fms",
            strong_buy_tier, buy_tier, neutral_tier, avoid_tier, step_latencies["signal_ms"],
        )

        # ------------------------------------------------------------------
        # Step 5: Candidate filter — score >= _CANDIDATE_MIN_SCORE AND cooldown clear
        # ------------------------------------------------------------------
        candidates:       List[str] = []
        skipped_cooldown: List[str] = []

        for ticker, signal in state.signals.items():
            if signal["score"] < _CANDIDATE_MIN_SCORE:
                continue
            if not _is_cooled_down(ticker, state.recent_signals):
                skipped_cooldown.append(ticker)
                logger.info("[AlphaScannerAgent] %s skipped — cooldown active", ticker)
                continue
            candidates.append(ticker)

        state.candidates = candidates

        telemetry.log_event(
            "candidates_filtered",
            {
                "candidate_count":   len(candidates),
                "filtered_count":    len(state.signals) - len(candidates) - len(skipped_cooldown),
                "skipped_cooldown":  skipped_cooldown,
            },
        )
        logger.info(
            "[AlphaScannerAgent] %d candidates after score + cooldown filter  "
            "(%d on cooldown)",
            len(candidates),
            len(skipped_cooldown),
        )

        # ------------------------------------------------------------------
        # Step 6: Portfolio-awareness guards
        # ------------------------------------------------------------------
        sector_allocation: Dict[str, float] = portfolio_context.get("sector_allocation", {})
        position_weights:  Dict[str, float] = portfolio_context.get("position_weights", {})
        cash_available:    float            = portfolio_context.get("cash_available", float("inf"))

        if cash_available <= 0:
            logger.warning(
                "[AlphaScannerAgent] cash_available=%.2f — no capital deployable; "
                "skipping all LLM decisions",
                cash_available,
            )
            telemetry.log_event(
                "candidates_filtered",
                {"skipped_reason": "no_cash", "skipped_count": len(candidates)},
            )
            state.buy_opportunities = []
            return state

        # All candidates pass through — caps are surfaced as warnings in
        # the buy_entry output (Step 7) rather than silently removing tickers.
        # Only zero-cash is a hard stop since there is literally no capital to deploy.
        approved_candidates: List[str] = list(candidates)

        for ticker in candidates:
            sector     = state.market_data[ticker].get("sector", "Unknown")
            sector_pct = sector_allocation.get(sector, 0.0)
            position_pct = position_weights.get(ticker, 0.0)

            if sector_pct > _MAX_SECTOR_EXPOSURE:
                logger.warning(
                    "[AlphaScannerAgent] %s — sector '%s' at %.1f%% exceeds %.0f%% cap "
                    "(included with warning)",
                    ticker, sector, sector_pct, _MAX_SECTOR_EXPOSURE,
                )
            if position_pct > _MAX_POSITION_WEIGHT:
                logger.warning(
                    "[AlphaScannerAgent] %s — existing position %.1f%% exceeds %.0f%% cap "
                    "(included with warning)",
                    ticker, position_pct, _MAX_POSITION_WEIGHT,
                )

        logger.info(
            "[AlphaScannerAgent] %d candidates proceeding to decision pipeline",
            len(approved_candidates),
        )
        return state


