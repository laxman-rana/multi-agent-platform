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
_COOLDOWN_MINUTES:        int   = 30
_COOLDOWN_UNIT:           str   = "minutes"  # "minutes" | "hours" | "days"
_CANDIDATE_MIN_SCORE:     int   = 2   # score must be ≥2 ("buy" tier) to reach LLM
_MAX_SECTOR_EXPOSURE:     float = 60.0
_MAX_POSITION_WEIGHT:     float = 10.0
_MAX_FETCH_WORKERS:       int   = 10
# Event-override: bypass score gate for extreme moves near the 52-week low.
# These are potential crash-dip / panic-selling / earnings-reaction setups
# that a pure score filter would silently discard.
_EVENT_OVERRIDE_MIN_MOVE: float = 8.0   # abs(change_pct) threshold (%)
_EVENT_OVERRIDE_52W_BAND: float = 0.30  # price within lower 30% of 52w range
_MAX_CONCURRENT_BUYS:     int   = 3     # max new positions to open per scan cycle
_MAX_SIGNAL_SCORE:        float = 5.0   # theoretical max raw signal score (normalization denominator)


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

def _is_event_override(mdata: Dict[str, Any]) -> bool:
    """
    Return True when a ticker qualifies for an event override regardless of
    its quantitative score.  Criteria (both must hold):
      1. abs(change_pct) >= _EVENT_OVERRIDE_MIN_MOVE (default 8%) — large
         intraday move signalling a potential crash-dip or earnings reaction.
      2. Price is within the lower _EVENT_OVERRIDE_52W_BAND of its 52-week
         range (default: lower 30%) — mean-reversion entry zone.
    These stocks are forwarded to the LLM with override_reason="extreme_move"
    so the model can apply discretion rather than being silently ignored.
    """
    change_pct = mdata.get("change_pct", 0.0)
    price      = mdata.get("price", 0.0)
    high_52w   = mdata.get("52w_high", 0.0)
    low_52w    = mdata.get("52w_low",  0.0)

    if abs(change_pct) < _EVENT_OVERRIDE_MIN_MOVE:
        return False
    if not (high_52w and low_52w and high_52w > low_52w):
        return False
    range_pct = (price - low_52w) / (high_52w - low_52w)
    return range_pct <= _EVENT_OVERRIDE_52W_BAND


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


def _is_fresh_despite_cooldown(
    ticker: str,
    mdata: Dict[str, Any],
    recent_signal_context: Dict[str, Any],
) -> bool:
    """
    Return True when the ticker is within its cooldown window but a new
    material catalyst has emerged since the last BUY signal was emitted:

      1. Capitulation event (drop ≥8% on ≥3× avg volume): panic-selling
         exhaustion is always a fresh setup regardless of signal recency.
      2. Significant new dip: price has fallen ≥5% since the price recorded
         at last signal, opening a new lower entry zone.
    """
    from src.agents.opportunity.engines.signal_engine import _THRESHOLDS  # noqa: PLC0415
    change_pct = mdata.get("change_pct", 0.0)
    volume     = mdata.get("volume", 0)
    avg_volume = mdata.get("avg_volume", 0)

    # Capitulation always warrants re-evaluation regardless of cooldown.
    if (
        change_pct <= -_THRESHOLDS["capitulation_move_pct"]
        and avg_volume > 0
        and volume / avg_volume >= _THRESHOLDS["capitulation_vol_ratio"]
    ):
        return True

    # Significant new dip since the last signal price.
    ctx = recent_signal_context.get(ticker)
    if ctx:
        prev_price = ctx.get("price", 0.0)
        curr_price = mdata.get("price", 0.0)
        if prev_price > 0 and curr_price > 0:
            drop_since_pct = (curr_price - prev_price) / prev_price * 100
            if drop_since_pct <= -5.0:
                return True

    return False


def _compute_opportunity_score(
    ticker: str,
    mdata: Dict[str, Any],
    signal: Dict[str, Any],
    recent_signal_context: Dict[str, Any],
) -> float:
    """
    Composite ranking metric that blends four orthogonal dimensions:

      0.4 × normalized_signal_score   raw_score / _MAX_SIGNAL_SCORE → [0, 1]
      0.3 × analyst_upside            analyst mean-target upside, 50% upside = 1.0
      0.2 × volume_spike              volume / avg_volume ratio, 10× = 1.0
      0.1 × freshness                 1.0 = brand-new idea, 0.0 = recently acted on

    All components are clamped to [0, 1] before weighting so a single extreme
    value cannot dominate the composite.  Returns a float in [0.0, 1.0].
    """
    # Component 1: signal strength (normalized)
    raw_score  = signal.get("score", 0)
    norm_score = max(0.0, min(1.0, raw_score / _MAX_SIGNAL_SCORE))

    # Component 2: analyst upside (normalized: 50% upside → 1.0)
    analyst_target = mdata.get("analyst_target")
    price          = mdata.get("price", 0.0)
    if analyst_target and price and price > 0:
        upside              = (float(analyst_target) - price) / price
        analyst_upside_norm = min(1.0, max(0.0, upside / 0.5))
    else:
        analyst_upside_norm = 0.0

    # Component 3: volume spike (normalized: 10× avg volume → 1.0)
    volume     = mdata.get("volume", 0)
    avg_volume = mdata.get("avg_volume", 0)
    if avg_volume and avg_volume > 0:
        volume_spike_norm = min(1.0, (volume / avg_volume) / 10.0)
    else:
        volume_spike_norm = 0.0

    # Component 4: freshness (1.0 = never signalled before, 0.0 = recently acted on)
    freshness = 0.0 if ticker in recent_signal_context else 1.0

    return round(
        0.4 * norm_score
        + 0.3 * analyst_upside_norm
        + 0.2 * volume_spike_norm
        + 0.1 * freshness,
        4,
    )


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
        # Step 5: Candidate filter — score >= _CANDIDATE_MIN_SCORE  OR  event
        #         override (large intraday move + price near 52-week low).
        # Event overrides are tagged with signal["override_reason"] so the
        # LLM and display layer receive the full context.
        # ------------------------------------------------------------------
        candidates:       List[str] = []
        skipped_cooldown: List[str] = []
        event_overrides:  List[str] = []

        # Sort score-qualified tickers by opportunity_score descending — highest
        # conviction first.  Ranking before the cooldown gate ensures the best
        # setups always reach the LLM ahead of marginal ones, regardless of dict
        # iteration order.  opportunity_score blends signal strength, analyst
        # upside, volume spike, and idea freshness into a single composite rank.
        _qualified = [
            (ticker, sig)
            for ticker, sig in state.signals.items()
            if sig["score"] >= _CANDIDATE_MIN_SCORE
            or _is_event_override(state.market_data.get(ticker, {}))
        ]
        for ticker, sig in _qualified:
            sig["opportunity_score"] = _compute_opportunity_score(
                ticker,
                state.market_data.get(ticker, {}),
                sig,
                state.recent_signal_context,
            )

        scored_queue = sorted(
            _qualified,
            key=lambda x: x[1]["opportunity_score"],
            reverse=True,
        )

        for ticker, signal in scored_queue:
            mdata       = state.market_data.get(ticker, {})
            is_override = _is_event_override(mdata)

            if not _is_cooled_down(ticker, state.recent_signals):
                # Freshness bypass: a new capitulation or significant price drop
                # re-opens the ticker for re-evaluation even within cooldown.
                if _is_fresh_despite_cooldown(ticker, mdata, state.recent_signal_context):
                    logger.info(
                        "[AlphaScannerAgent] %s — cooldown bypassed: fresh catalyst (%.1f%%)",
                        ticker, mdata.get("change_pct", 0.0),
                    )
                    signal["override_reason"] = signal.get("override_reason") or "fresh_catalyst"
                else:
                    skipped_cooldown.append(ticker)
                    logger.info("[AlphaScannerAgent] %s skipped — cooldown active", ticker)
                    continue

            if is_override and signal["score"] < _CANDIDATE_MIN_SCORE:
                signal["override_reason"] = "extreme_move"
                event_overrides.append(ticker)
                logger.info(
                    "[AlphaScannerAgent] %s — event override: %.1f%% move near 52w low",
                    ticker, mdata.get("change_pct", 0.0),
                )

            candidates.append(ticker)

        state.candidates = candidates
        state.skipped_cooldown = list(skipped_cooldown)
        state.blocked_no_cash = []

        telemetry.log_event(
            "candidates_filtered",
            {
                "candidate_count":  len(candidates),
                "filtered_count":   len(state.signals) - len(candidates) - len(skipped_cooldown),
                "skipped_cooldown": skipped_cooldown,
                "event_overrides":  event_overrides,
            },
        )
        logger.info(
            "[AlphaScannerAgent] %d candidates after score + cooldown filter  "
            "(%d on cooldown, %d event overrides)",
            len(candidates),
            len(skipped_cooldown),
            len(event_overrides),
        )

        # ------------------------------------------------------------------
        # Step 6: Portfolio-awareness guards
        # ------------------------------------------------------------------
        sector_allocation: Dict[str, float] = portfolio_context.get("sector_allocation", {})
        position_weights:  Dict[str, float] = portfolio_context.get("position_weights", {})
        cash_available:    float            = portfolio_context.get("cash_available", float("inf"))

        if cash_available <= 0 and not state.ignore_cash_check:
            logger.warning(
                "[AlphaScannerAgent] cash_available=%.2f — no capital deployable; "
                "skipping all LLM decisions  (pass ignore_cash_check=True to override)",
                cash_available,
            )
            telemetry.log_event(
                "candidates_filtered",
                {"skipped_reason": "no_cash", "skipped_count": len(candidates) + len(skipped_cooldown)},
            )
            # All score-qualified tickers — both candidates and cooldown-suppressed —
            # are reported as blocked_no_cash.  Cash was the real blocking factor;
            # the cooldown distinction is irrelevant when there is nothing to deploy.
            state.blocked_no_cash = list(candidates) + list(skipped_cooldown)
            state.skipped_cooldown = []   # reclassified: no-cash drove suppression
            # Clear candidates so NewsNode and DecisionNode are no-ops.
            state.candidates = []
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


