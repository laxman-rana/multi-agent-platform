"""
workflow.py
-----------
Entry point for the AlphaScannerAgent quality-first BUY-opportunity workflow.

Run from the repository root:

    # Single scan — US market (large + mid cap), useful outside market hours for testing
    python -m src.agents.opportunity.workflow --tickers AAPL MSFT NVDA --once

    # Continuous batch scan every 15 minutes while the US market is open
    python -m src.agents.opportunity.workflow --tickers AAPL MSFT NVDA

    # Scan Indian market (NIFTY 50) — top 50 stocks, single shot
    python -m src.agents.opportunity.workflow --top-n 50 --market IN --once

    # Override the LLM model (provider is inferred automatically)
    python -m src.agents.opportunity.workflow --tickers AAPL MSFT --model gpt-4o --once

Environment variables
---------------------
ALPHA_SCANNER_LLM_MODEL   Override the decision LLM (provider auto-inferred).
PORTFOLIO_LLM_MODEL       Fallback model when ALPHA_SCANNER_LLM_MODEL is unset.
PORTFOLIO_LLM_PROVIDER    Fallback provider (default: ollama).
"""

import argparse
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from src.agents.opportunity.nodes.alpha_scanner_agent import AlphaScannerAgent
from src.agents.opportunity.nodes.news_node import NewsNode
from src.agents.opportunity.nodes.decision_node import DecisionNode
from src.agents.opportunity.markets.market_strategy import get_liquid_universe, get_market_strategy
from src.agents.opportunity.providers.factory import create_market_data_provider
from src.agents.opportunity.state import OpportunityState
from src.observability import get_telemetry_logger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Market-hours helper
# ---------------------------------------------------------------------------


def is_market_open(market: str = "US") -> bool:
    """Return True when the target market's primary exchange is currently open.

    Delegates to the market's strategy — no market-specific logic lives here.
    US : NYSE/NASDAQ  Mon–Fri 09:30–16:00 ET
    IN : NSE (India)  Mon–Fri 09:15–15:30 IST
    """
    return get_market_strategy(market).is_open()


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """
    Compile the 3-node AlphaScannerAgent LangGraph.

    scanner  : fetch market data + prefilter + signal + candidate filter + guards
    news     : fetch headlines + LLM sentiment per candidate  (NewsNode)
    decision : LLM BUY/IGNORE per candidate + portfolio warnings + sort  (DecisionNode)

        [scanner] → [news] → [decision] → END
    """
    graph = StateGraph(OpportunityState)
    graph.add_node("scanner",  AlphaScannerAgent(create_market_data_provider()).run)
    graph.add_node("news",     NewsNode().run)
    graph.add_node("decision", DecisionNode().run)
    graph.set_entry_point("scanner")
    graph.add_edge("scanner",  "news")
    graph.add_edge("news",     "decision")
    graph.add_edge("decision", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Public scan API
# ---------------------------------------------------------------------------

def trigger_scan(
    tickers: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    market: str = "US",
) -> List[Dict[str, Any]]:
    """
    Single-shot scan.  Safe to call from external event-driven hooks
    (e.g. a price-spike webhook, a volatility alert, a scheduler).

    Parameters
    ----------
    tickers : explicit list of ticker symbols; if None and top_n is
              set, the dynamic liquid universe is used instead.
    top_n   : when tickers is None, scan the top-N most liquid tickers
              from the built-in universe (default: 200).
    market  : "US" for large + mid cap / NYSE (default), "IN" for NIFTY 50 / NSE.

    Returns
    -------
    Sorted list of BUY opportunity dicts (may be empty).
    """
    resolved_tickers = tickers or get_liquid_universe(top_n or 200, market)
    compiled      = build_graph()
    initial_state = OpportunityState(watchlist=resolved_tickers)
    result      = compiled.invoke(initial_state)
    final_state = result if isinstance(result, OpportunityState) else OpportunityState(**result)
    return final_state.buy_opportunities


def run_batch_scan(
    tickers: Optional[List[str]] = None,
    interval_minutes: int = 15,
    top_n: Optional[int] = None,
    market: str = "US",
) -> None:
    """
    Repeated batch scan that runs continuously while the market is open.

    The recent_signals cooldown dict is carried forward between iterations
    so the same ticker cannot re-emit a BUY signal within _COOLDOWN_MINUTES,
    even across multiple scan cycles.

    Blocks until the market closes or the process is interrupted.
    """
    resolved_tickers = tickers or get_liquid_universe(top_n or 200, market)
    compiled  = build_graph()
    telemetry = get_telemetry_logger()

    # Cooldown tracker persists across iterations; all other state is fresh
    # each cycle so stale market data is never re-used.
    recent_signals:         Dict[str, str]     = {}
    recent_signal_context:  Dict[str, Any]     = {}

    logger.info(
        "[Workflow] Starting batch scan — %d tickers, interval=%d min",
        len(resolved_tickers), interval_minutes,
    )

    while is_market_open(market):
        state_input = OpportunityState(
            watchlist=resolved_tickers,
            recent_signals=dict(recent_signals),          # shallow copy preserves cooldown
            recent_signal_context=dict(recent_signal_context),  # preserves freshness context
        )

        result = compiled.invoke(state_input)
        final  = result if isinstance(result, OpportunityState) else OpportunityState(**result)

        # Carry forward updated cooldown timestamps and freshness context
        recent_signals        = final.recent_signals
        recent_signal_context = final.recent_signal_context

        opportunities = final.buy_opportunities
        telemetry.log_event(
            "batch_scan_cycle",
            {
                "buy_count": len(opportunities),
                "tickers":   [o["ticker"] for o in opportunities],
            },
        )

        if opportunities:
            logger.info("[Workflow] %d quality-ranked BUY opportunities this cycle:", len(opportunities))
            for opp in opportunities:
                logger.info(
                    "  %-6s | %-8s | q=%+d | %s",
                    opp["ticker"],
                    opp["confidence"],
                    opp.get("quality_score", opp["score"]),
                    opp["reason"][:80],
                )
        else:
            logger.info("[Workflow] No BUY opportunities this cycle.")

        logger.info("[Workflow] Next scan in %d minutes.", interval_minutes)
        time.sleep(interval_minutes * 60)

    logger.info("[Workflow] Market closed — batch scan ended.")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main(
    tickers: Optional[List[str]] = None,
    interval_minutes: int = 15,
    model: Optional[str] = None,
    once: bool = False,
    top_n: Optional[int] = None,
    market: str = "US",
    verbose: bool = False,
) -> None:
    from src.llm.providers import _DEFAULT_MODELS, infer_provider

    _env_model    = os.getenv("ALPHA_SCANNER_LLM_MODEL") or os.getenv("PORTFOLIO_LLM_MODEL")
    resolved_model = model or _env_model

    if resolved_model:
        try:
            resolved_provider = infer_provider(resolved_model)
        except ValueError as exc:
            print(f"\n[Configuration error] {exc}")
            raise SystemExit(1) from None
        os.environ["ALPHA_SCANNER_LLM_MODEL"] = resolved_model
        os.environ["PORTFOLIO_LLM_PROVIDER"]  = resolved_provider
    else:
        resolved_provider = os.getenv("PORTFOLIO_LLM_PROVIDER", "ollama")
        resolved_model    = _DEFAULT_MODELS.get(resolved_provider, "(default)")

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    telemetry = get_telemetry_logger()
    telemetry.log_event(
        "workflow_start",
        {"provider": resolved_provider, "model": resolved_model},
    )

    logger.info("=" * 60)
    logger.info("STARTING QUALITY STOCK WORKFLOW")
    logger.info("=" * 60)
    logger.info("Provider : %s  |  Model : %s", resolved_provider, resolved_model)
    logger.info("Market   : %s", "US (Large + Mid Cap / NYSE)" if market.upper() == "US" else "IN (NIFTY 50 / NSE)")
    logger.info("Tickers  : %s", tickers or f"[dynamic top-{top_n or 200}]")

    if not tickers and not top_n:
        print("[Error] Provide --tickers AAPL MSFT or --top-n N to scan the liquid universe")
        raise SystemExit(1)

    try:
        if once:
            final_state = _trigger_scan_full(tickers, top_n, market)
            _print_opportunities(final_state.buy_opportunities)
            _print_ignored(final_state)
            if verbose:
                _print_scan_digest(final_state)
        else:
            run_batch_scan(tickers, interval_minutes, top_n, market)

    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        print(f"\n[Configuration error] {exc}")
        raise SystemExit(1) from None
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        print(
            f"\n[Missing dependency] {exc}\n"
            "Install the required package and retry.\n"
            "  ollama : pip install langchain-ollama\n"
            "  openai : pip install langchain-openai\n"
            "  google : pip install langchain-google-genai"
        )
        raise SystemExit(1) from None
    except Exception as exc:
        logger.exception("Workflow failed unexpectedly")
        print(f"\n[Workflow error] {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None


def _prefilter_fail_reasons(mdata: Dict[str, Any]) -> str:
    """Return a compact string explaining why a ticker failed the PreFilter."""
    from src.agents.opportunity.engines.prefilter_engine import _PREFILTER_THRESHOLDS  # noqa: PLC0415
    price      = mdata.get("price", 0.0)
    change_pct = mdata.get("change_pct", 0.0)
    volatility = mdata.get("volatility", 0.0)
    high       = mdata.get("52w_high", 0.0)
    low        = mdata.get("52w_low",  0.0)
    volume     = mdata.get("volume", 0)
    avg_volume = mdata.get("avg_volume", 0)

    vol_ratio = (volume / avg_volume) if avg_volume > 0 else 0.0
    parts = [
        f"move={change_pct:+.1f}% (need \u00b13%)",
        f"vol-ratio={vol_ratio:.1f}x (need 1.5x)",
        f"ann-vol={volatility:.0%} (need >35%)",
    ]
    if high and low and high > low:
        lower_band = low + _PREFILTER_THRESHOLDS["lower_band_pct"] * (high - low)
        parts.append(f"price=${price:.2f} vs 52w-lower-band=${lower_band:.2f}")
    return "  |  ".join(parts)


def _trigger_scan_full(
    tickers: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    market: str = "US",
) -> "OpportunityState":
    """Same as trigger_scan but returns the full OpportunityState for diagnostics."""
    resolved_tickers = tickers or get_liquid_universe(top_n or 200, market)
    compiled      = build_graph()
    initial_state = OpportunityState(watchlist=resolved_tickers)
    result = compiled.invoke(initial_state)
    return result if isinstance(result, OpportunityState) else OpportunityState(**result)


def _print_scan_digest(state: "OpportunityState") -> None:
    """Print a per-ticker diagnostic table showing where each ticker exited the pipeline."""
    from src.agents.opportunity.nodes.alpha_scanner_agent import _CANDIDATE_MIN_SCORE  # noqa: PLC0415
    divider = "-" * 74
    em_dash = "\u2014"
    print(f"\n{divider}")
    print(f"SCAN DIGEST {em_dash} {len(state.watchlist)} ticker(s) traced through pipeline")
    print(divider)
    print(f"  {'Ticker':<12}  {'Stage':<20}  {'Score/Tier':<14}  Detail")
    print(f"  {'-'*10:<12}  {'-'*18:<20}  {'-'*12:<14}  ------")

    for ticker in sorted(state.watchlist):
        if ticker in state.scan_errors:
            err = state.scan_errors[ticker][:55]
            stage = "\u274c FETCH FAILED"
            dash  = "\u2014"
            print(f"  {ticker:<12}  {stage:<20}  {dash:<14}  {err}")
            continue
        if ticker not in state.market_data:
            stage = "\u274c NO DATA"
            dash  = "\u2014"
            print(f"  {ticker:<12}  {stage:<20}  {dash:<14}")
            continue
        if ticker not in state.prefiltered:
            reasons = _prefilter_fail_reasons(state.market_data[ticker])
            stage   = "\u274c PREFILTER FAIL"
            dash    = "\u2014"
            print(f"  {ticker:<12}  {stage:<20}  {dash:<14}  {reasons}")
            continue
        if ticker not in state.signals:
            stage = "\u26a0\ufe0f  SIGNAL MISSING"
            dash  = "\u2014"
            print(f"  {ticker:<12}  {stage:<20}  {dash:<14}")
            continue
        sig        = state.signals[ticker]
        score_str  = f"{sig['score']:+d} ({sig['tier']})"
        if sig["score"] < _CANDIDATE_MIN_SCORE and not sig.get("override_reason"):
            stage = "\u274c SCORE TOO LOW"
            print(f"  {ticker:<12}  {stage:<20}  {score_str:<14}  score < {_CANDIDATE_MIN_SCORE}, not sent to LLM")
            continue
        if ticker in state.skipped_cooldown:
            stage = "\u23f3 ON COOLDOWN"
            print(f"  {ticker:<12}  {stage:<20}  {score_str:<14}")
            continue
        if ticker not in state.decisions:
            stage = "\u23f3 ON COOLDOWN"
            print(f"  {ticker:<12}  {stage:<20}  {score_str:<14}")
            continue
        decision   = state.decisions[ticker]
        action     = decision.get("action", "?")
        confidence = decision.get("confidence", "?")
        icon       = "\u2705" if action == "BUY" else "\u23e9"
        stage      = icon + " LLM: " + action + "/" + confidence
        print(f"  {ticker:<12}  {stage:<20}  {score_str:<14}  {decision.get('reason','')[:55]}")

    print(divider)


def _build_low_score_reason(
    ticker: str,
    sig: Dict[str, Any],
    state: "OpportunityState",
) -> str:
    """Construct a human-readable explanation for a ticker that scored too low."""
    score   = sig.get("quality_score", sig.get("score", 0))
    signals = sig.get("quality_signals", sig.get("signals", []))
    mdata   = state.market_data.get(ticker, {})
    news    = state.news_sentiment.get(ticker)

    analyst    = (mdata.get("analyst_rating") or "").replace("_", " ").strip()
    change_pct = mdata.get("change_pct")
    pe         = mdata.get("pe_ratio")
    fpe        = mdata.get("forward_pe")
    volatility = mdata.get("volatility")

    parts: List[str] = []

    # Signal balance
    if score < 0:
        parts.append("quality risks outweigh quality signals")
    elif signals:
        parts.append("fundamentals are mixed — no durable quality thesis dominates")
    else:
        parts.append("no meaningful quality signals detected")

    # Analyst sentiment
    if analyst in ("sell", "strong sell", "underperform"):
        parts.append(f"analyst consensus is {analyst}")
    elif analyst in ("hold", "neutral"):
        parts.append(f"analyst consensus is {analyst} with no clear upside target")

    # Valuation trend
    if pe and fpe:
        if fpe >= pe:
            parts.append(f"valuation does not improve (fwd P/E {fpe:.1f} ≥ trailing {pe:.1f})")

    # Price movement
    if change_pct is not None and abs(change_pct) < 1.0:
        parts.append(f"minimal price movement today ({change_pct:+.1f}%)")

    # News sentiment
    if news and news.get("sentiment", "neutral").lower() in ("negative", "bearish"):
        parts.append("negative news sentiment")

    if volatility is not None and volatility > 0.45:
        parts.append(f"high volatility ({volatility:.0%}) adds execution risk")

    if not parts:
        parts.append(f"insufficient quality conviction (score {score:+d})")

    return "; ".join(parts).capitalize() + "."



def _missing_quality_signals(sig: Dict[str, Any]) -> str:
    """
    For a watchlist-alert ticker, describe what quality signal(s) would
    push it over the BUY threshold — helps the user know what to watch for.
    """
    active = {s.lower() for s in sig.get("quality_signals", sig.get("signals", []))}

    missing: List[str] = []

    # Check which Tier 1 signals are absent
    has_profit  = any("profit margin" in s for s in active)
    has_roe     = any("return on equity" in s for s in active)
    has_fcf     = any("fcf yield" in s for s in active)
    has_opmargin = any("operating margin" in s for s in active)
    has_balance = any("debt-to-equity" in s and "investment-grade" in s for s in active)

    if not has_profit:
        missing.append("profit margin ≥10% (durable profitability)")
    if not has_roe:
        missing.append("ROE ≥15% (capital efficiency)")
    if not has_fcf:
        missing.append("FCF yield ≥2% (self-funded growth)")
    if not has_opmargin:
        missing.append("operating margin ≥12% (pricing power)")
    if not has_balance and not any("leverage" in s for s in active):
        missing.append("D/E ≤0.75 (investment-grade balance sheet)")

    if not missing:
        return "confirm margin or FCF improvement in next earnings"

    return " | ".join(missing[:2])  # show top 2 gaps only


def _print_ticker_detail(
    ticker: str,
    sig: Dict[str, Any],
    state: "OpportunityState",
    reason: str,
    confidence: Optional[str] = None,
    decision: Optional[Dict[str, Any]] = None,
) -> None:
    """Print a full detail block for one ticker (shared by all rejection groups)."""
    score     = sig.get("score", 0)
    quality_score = sig.get("quality_score", score)
    score_str = f"{score:+d}" if isinstance(score, int) else str(score)
    quality_str = f"{quality_score:+d}" if isinstance(quality_score, int) else str(quality_score)
    tier      = sig.get("tier", "?")
    quality_tier = sig.get("quality_tier", tier)
    mdata     = state.market_data.get(ticker, {})
    news      = state.news_sentiment.get(ticker)

    signals_text = (
        "\n".join(f"      - {s}" for s in sig.get("signals", []))
        or "      (no signals fired)"
    )

    analyst_rating = (mdata.get("analyst_rating") or "none").replace("_", " ")
    analyst_count  = mdata.get("analyst_count", 0) or 0
    analyst_line   = (
        f"{analyst_rating}  ({analyst_count} analyst(s))"
        if analyst_count else "no analyst coverage"
    )

    price       = mdata.get("price")
    change_pct  = mdata.get("change_pct")
    pe_trailing = mdata.get("pe_ratio")
    pe_forward  = mdata.get("forward_pe")
    sector      = mdata.get("sector", "")
    market_cap  = mdata.get("market_cap")

    price_line = ""
    if price is not None:
        change_str = f"  ({change_pct:+.2f}% today)" if change_pct is not None else ""
        price_line = f"\n  Price      : ${price:.2f}{change_str}"

    valuation_line = ""
    if pe_trailing or pe_forward:
        parts = []
        if pe_trailing:
            parts.append(f"P/E {pe_trailing:.1f}")
        if pe_forward:
            parts.append(f"Fwd P/E {pe_forward:.1f}")
        valuation_line = f"\n  Valuation  : {' | '.join(parts)}"

    cap_line = ""
    if market_cap:
        cap_b = market_cap / 1e9
        cap_line = f"\n  Market Cap : ${cap_b:.1f}B"

    sector_line = f"\n  Sector     : {sector}" if sector else ""

    news_line = ""
    if news:
        sentiment = news.get("sentiment", "neutral").upper()
        catalyst  = news.get("catalyst", "")
        headlines = news.get("headlines", [])
        news_line = f"\n  News       : {sentiment}  —  {catalyst}"
        if headlines:
            for h in headlines[:3]:
                news_line += f"\n               • {h}"

    confidence_line = f"\n  Confidence : {confidence}" if confidence else ""
    override        = sig.get("override_reason")
    override_line   = f"\n  \u26a1 Override  : EVENT OVERRIDE \u2014 large move near 52-week low (bypass score gate)" if override else ""

    risk_level    = decision.get("risk_level", "") if decision else ""
    time_horizon  = decision.get("time_horizon_bias", "") if decision else ""
    news_impact   = decision.get("news_impact", "") if decision else ""
    risk_factors  = decision.get("risk_factors", []) if decision else []
    risk_line     = f"\n  Risk       : {risk_level.upper()}  |  Horizon: {time_horizon}  |  News impact: {news_impact}" if risk_level else ""
    risk_fac_text = ("\n" + "\n".join(f"               - {r}" for r in risk_factors)) if risk_factors else ""

    # "What would change this" — missing Tier 1 signals
    watch_for = _missing_quality_signals(sig)

    print(
        f"\n  Ticker     : {ticker}{price_line}{cap_line}{sector_line}"
        f"\n  Score      : {quality_str}  ({quality_tier})"
        f"\n  Legacy     : {score_str}  ({tier}){confidence_line}{override_line}"
        f"\n  Analyst    : {analyst_line}{valuation_line}{news_line}"
        f"{risk_line}{risk_fac_text}"
        f"\n  Reason     : {reason}"
        f"\n  Watch for  : {watch_for}"
        f"\n  Signals    :\n{signals_text}"
    )


def _print_ignored(state: "OpportunityState") -> None:
    """Print all scanned tickers that were not recommended, grouped by rejection stage."""
    from src.agents.opportunity.nodes.alpha_scanner_agent import _CANDIDATE_MIN_SCORE  # noqa: PLC0415

    buy_tickers = {opp["ticker"] for opp in state.buy_opportunities}
    not_recommended = [t for t in state.watchlist if t not in buy_tickers]

    if not not_recommended:
        return

    # Categorise each non-BUY ticker by the stage where it was eliminated
    llm_ignored: list = []      # reached LLM → IGNORE
    low_score: list   = []      # signals fired but score < minimum threshold
    on_cooldown: list = []      # score OK but suppressed by cooldown
    prefilter_fail: list = []   # failed prefilter (no significant activity)
    errors: list = []           # data fetch or signal error

    for ticker in not_recommended:
        if ticker in state.scan_errors or ticker not in state.market_data:
            errors.append(ticker)
            continue
        if ticker not in state.prefiltered:
            prefilter_fail.append(ticker)
            continue
        if ticker not in state.signals:
            errors.append(ticker)
            continue
        sig   = state.signals[ticker]
        score = sig.get("score", 0)
        if score < _CANDIDATE_MIN_SCORE and not sig.get("override_reason"):
            low_score.append((ticker, sig))
            continue
        if ticker in state.skipped_cooldown:
            on_cooldown.append((ticker, sig))
            continue
        if ticker not in state.decisions:
            on_cooldown.append((ticker, sig))
            continue
        dec = state.decisions[ticker]
        if dec.get("action") == "IGNORE":
            llm_ignored.append((ticker, sig, dec))

    # Also gather capped tickers (qualified BUY but excluded by position cap)
    capped = getattr(state, "capped_opportunities", [])

    divider = "=" * 60
    print(f"\n{divider}")
    print(f"STOCKS SCANNED — NOT RECOMMENDED ({len(not_recommended)} ticker(s))")
    print(divider)

    # 0. Capped opportunities — qualified BUY, excluded only by max_concurrent_buys
    if capped:
        capped_sorted = sorted(capped, key=lambda x: x.get("quality_score", x.get("score", 0)), reverse=True)
        print(f"\n  [ \U0001f4cc QUALIFIED BUT CAPPED — strong signal, position limit reached: {len(capped)} ]")
        for opp in capped_sorted:
            t         = opp["ticker"]
            qs        = opp.get("quality_score", opp["score"])
            qt        = opp.get("quality_tier", "")
            dec       = opp.get("decision", "BUY")
            thesis    = opp.get("thesis_type", "").replace("_", " ").title()
            conf      = opp.get("confidence", "")
            rb        = opp.get("risk_breakdown", {})
            reason    = opp.get("reason", "")
            ps        = opp.get("position_sizing", {})
            triggers  = opp.get("entry_triggers", [])
            ks        = opp.get("key_signals", [])
            mdata     = state.market_data.get(t, {})
            price     = mdata.get("price", 0.0)
            change_pct = mdata.get("change_pct", 0.0)
            fwd_pe    = mdata.get("forward_pe")
            fwd_pe_s  = f"{fwd_pe:.1f}" if fwd_pe else "N/A"
            h52, l52  = mdata.get("52w_high", 0.0), mdata.get("52w_low", 0.0)
            range_pos = f"{(price - l52) / (h52 - l52):.0%}" if h52 > l52 else "N/A"
            v_ = rb.get("volatility_risk",  "?").upper()
            f_ = rb.get("fundamental_risk", "?").upper()
            s_ = rb.get("sentiment_risk",   "?").upper()
            ps_type  = ps.get("type", "starter").upper()
            ps_range = ps.get("range", "")
            print(
                f"\n  Ticker     : {t}"
                f"\n  Decision   : {dec}  ({qt})  — capped by position limit, NOT a quality issue"
                f"\n  Thesis     : {thesis}  |  Confidence: {conf}"
                f"\n  Score      : {qs:+d}"
                f"\n  Price      : ${price:.2f}  ({change_pct:+.2f}%)  |  52w position: {range_pos}"
                f"\n  Fwd P/E    : {fwd_pe_s}"
                f"\n  Risk       : Volatility={v_}  |  Fundamental={f_}  |  Sentiment={s_}"
                f"\n  Would size : {ps_type}  {ps_range}  if cap were raised"
            )
            if ks:
                print("  Key Signals:")
                for s in ks:
                    print(f"      \u2713  {s}")
            if triggers:
                print("  Why still interesting:")
                for t_ in triggers:
                    print(f"      \u2192  {t_}")
            if reason:
                print(f"  Reason     : {reason}")
            print(f"  Action     : Monitor — add to next scan cycle if position cap is raised")

    # 1. LLM IGNOREs — full detail (highest score first)
    if llm_ignored:
        llm_ignored.sort(key=lambda x: x[1].get("score", 0), reverse=True)
        print(f"\n  [ LLM reviewed and declined: {len(llm_ignored)} ]")
        for ticker, sig, dec in llm_ignored:
            _print_ticker_detail(
                ticker, sig, state,
                reason=dec.get("reason", "N/A"),
                confidence=dec.get("confidence"),
                decision=dec,
            )

    # 2. Split low_score into watchlist alerts vs genuinely weak
    _WATCHLIST_ALERT_SCORE = _CANDIDATE_MIN_SCORE - 1   # == 4 with current threshold
    _WATCHLIST_52W_PCT     = 0.05                        # bottom 5% of 52w range

    watchlist_alerts: list = []
    genuine_low: list = []

    for ticker, sig in low_score:
        score_val = sig.get("score", 0)
        mdata     = state.market_data.get(ticker, {})
        price     = mdata.get("price", 0.0)
        high_52   = mdata.get("52w_high", 0.0)
        low_52    = mdata.get("52w_low",  0.0)
        in_bottom = False
        if high_52 and low_52 and high_52 > low_52 and price:
            pct_in_range = (price - low_52) / (high_52 - low_52)
            in_bottom = pct_in_range <= _WATCHLIST_52W_PCT
        if score_val == _WATCHLIST_ALERT_SCORE and in_bottom:
            watchlist_alerts.append((ticker, sig))
        else:
            genuine_low.append((ticker, sig))

    # 2a. Watchlist alerts — rich detail block
    if watchlist_alerts:
        watchlist_alerts.sort(key=lambda x: x[1].get("score", 0), reverse=True)
        print(f"\n  [ \u26a1 WATCHLIST ALERT \u2014 near 52w low, one quality signal away from BUY: {len(watchlist_alerts)} ]")
        for ticker, sig in watchlist_alerts:
            mdata    = state.market_data.get(ticker, {})
            price    = mdata.get("price", 0.0)
            high_52  = mdata.get("52w_high", 0.0)
            low_52   = mdata.get("52w_low", 0.0)
            pct_pos  = f"{(price - low_52) / (high_52 - low_52):.0%}" if high_52 > low_52 else "N/A"
            fwd_pe   = mdata.get("forward_pe")
            fwd_pe_s = f"{fwd_pe:.1f}" if fwd_pe else "N/A"
            a_count  = mdata.get("analyst_count", 0) or 0
            a_tgt    = mdata.get("analyst_target")
            a_upside = f"+{((a_tgt - price) / price):.0%}" if a_tgt and price else "N/A"
            rev_g    = mdata.get("revenue_growth")
            rev_s    = f"{rev_g:.0%}" if rev_g is not None else "N/A"
            sig_score = sig.get("score", 0)
            sig_tier  = sig.get("quality_tier", "watchlist")
            missing   = _missing_quality_signals(sig)
            print(
                f"\n  Ticker     : {ticker}"
                f"\n  Price      : ${price:.2f}  |  52w position: {pct_pos} of range (bottom {int(_WATCHLIST_52W_PCT*100)}%)"
                f"\n  Score      : {sig_score:+d}  ({sig_tier})  \u2014  ONE signal short of BUY threshold ({_CANDIDATE_MIN_SCORE})"
                f"\n  Fwd P/E    : {fwd_pe_s}  |  Revenue growth: {rev_s}  |  Analyst upside: {a_upside} ({a_count} analysts)"
                f"\n  Watch for  : {missing}"
                f"\n  Signals    :"
            )
            for s in sig.get("quality_signals", sig.get("signals", [])):
                print(f"      - {s}")

    # 2b. Genuinely low score — full detail
    if genuine_low:
        genuine_low.sort(key=lambda x: x[1].get("score", 0), reverse=True)
        print(f"\n  [ Passed activity filter but signal score too low (< {_CANDIDATE_MIN_SCORE}): {len(genuine_low)} ]")
        for ticker, sig in genuine_low:
            _print_ticker_detail(ticker, sig, state, reason=_build_low_score_reason(ticker, sig, state))

    if on_cooldown:
        on_cooldown.sort(key=lambda x: x[1].get("score", 0), reverse=True)
        print(f"\n  [ Score meets threshold but suppressed by cooldown this cycle: {len(on_cooldown)} ]")
        for ticker, sig in on_cooldown:
            _print_ticker_detail(ticker, sig, state, reason="Already signalled recently — cooldown active.")

    # 3. Prefilter failures — summary line listing all tickers
    if prefilter_fail:
        n = len(prefilter_fail)
        names = ", ".join(sorted(prefilter_fail))
        print(
            f"\n  [ {n} ticker(s) skipped — no significant price movement or volume spike today ]\n"
            f"    {names}"
        )

    # 4. Data / signal errors
    if errors:
        print(f"\n  [ {len(errors)} ticker(s) had data errors: {', '.join(sorted(errors))} ]")

    print(f"\n{divider}\n")


def _print_opportunities(opportunities: List[Dict[str, Any]]) -> None:
    """Pretty-print the final BUY opportunity list to stdout."""
    heavy   = "=" * 60
    light   = "-" * 60

    print(f"\n{heavy}")
    print(f"QUALITY STOCK SCANNER — BUY SIGNALS ({len(opportunities)} found)")
    print(heavy)

    if not opportunities:
        print("  No BUY signals identified in this scan.")
        print(f"\n{heavy}\n")
        return

    for opp in opportunities:
        ticker        = opp["ticker"]
        score         = opp["score"]
        quality_score = opp.get("quality_score", score)
        quality_tier  = opp.get("quality_tier", "")
        opp_score     = opp.get("opportunity_score")
        tier          = opp.get("tier", "")
        confidence    = opp["confidence"]
        entry_quality = opp["entry_quality"]
        otype         = opp["type"]
        reason        = opp["reason"]
        sector        = opp.get("sector", "Unknown")
        news_sent     = opp.get("news_sentiment", "N/A").upper()
        news_cat      = opp.get("news_catalyst", "")
        signals       = opp.get("signals", [])

        # ── Section 1: signal header ──────────────────────────────────────
        print(f"\n{heavy}")
        print("QUALITY SIGNAL — OPPORTUNITY DETECTED")
        print(heavy)

        decision_tier  = opp.get("decision", opp["action"])
        _raw_thesis    = opp.get("thesis_type", "").strip()
        if not _raw_thesis:
            # Fallback: derive from internal opportunity type
            _otype_map = {
                "elite_compounder":       "quality_compounder",
                "quality_value_compounder": "quality_compounder",
                "compounder":             "quality_compounder",
                "quality_value":          "value_play",
                "quality_watchlist":      "high_growth_speculative",
            }
            _raw_thesis = _otype_map.get(opp.get("type", ""), "high_growth_speculative")
        thesis_type = _raw_thesis.replace("_", " ").title()
        risk_level     = opp.get("risk_level", "").upper()
        risk_breakdown = opp.get("risk_breakdown", {})
        time_horizon   = opp.get("time_horizon_bias", "")
        news_impact    = opp.get("news_impact", "")
        key_signals    = opp.get("key_signals", opp.get("key_supporting_signals", []))
        entry_triggers = opp.get("entry_triggers", [])
        position_sizing = opp.get("position_sizing", {})
        opp_score_line = f"  Opportunity Score : {opp_score:.4f}" if opp_score is not None else ""

        print(
            f"\n  Ticker        : {ticker}\n"
            f"  Decision      : {decision_tier}\n"
            f"  Thesis        : {thesis_type}\n"
            f"  Confidence    : {confidence}  |  Entry Quality: {entry_quality}\n"
            f"  Quality Score : {quality_score:+d}  ({quality_tier})\n"
            f"  Sector        : {sector}\n"
        )
        if opp_score is not None:
            print(f"  Opportunity Score : {opp_score:.4f}")

        if risk_breakdown:
            v_  = risk_breakdown.get("volatility_risk",  "?").upper()
            f_  = risk_breakdown.get("fundamental_risk", "?").upper()
            s_  = risk_breakdown.get("sentiment_risk",   "?").upper()
            print(
                f"\n  Risk          : Volatility={v_}  |  Fundamental={f_}  |  Sentiment={s_}\n"
                f"  Horizon       : {time_horizon}  |  News impact: {news_impact}"
            )

        if position_sizing:
            ps_type  = position_sizing.get("type", "").upper()
            ps_range = position_sizing.get("range", "")
            ps_why   = position_sizing.get("rationale", "")
            print(f"\n  Position Size : {ps_type}  {ps_range}  —  {ps_why}")

        if key_signals:
            print("\n  Key Signals (Tier 1):")
            for s in key_signals:
                print(f"      ✓  {s}")

        if entry_triggers:
            print("\n  Why Now:")
            for t in entry_triggers:
                print(f"      →  {t}")

        # News
        if news_sent and news_sent != "N/A":
            print(f"  News          : {news_sent}  —  {news_cat}")

        # Signals
        if signals:
            print("\n  Quality Signals:")
            for s in opp.get("quality_signals", signals):
                print(f"      - {s}")

        # Reason block
        print(f"\n  Reason:\n  {reason}\n")

        # ── Section 2: execution status ───────────────────────────────────
        print(light)
        print("EXECUTION STATUS")
        print(light)

        print(f"\n  Actionable    : YES\n")
        _ps_type  = position_sizing.get("type", "starter").upper() if position_sizing else "STARTER"
        _ps_range = position_sizing.get("range", "") if position_sizing else ""
        print("\n  Suggested Actions:")
        print(f"      - Allocate {_ps_type} position ({_ps_range}) — scale on fundamental confirmation")
        print(f"      - Validate thesis durability and portfolio sector exposure")
        print(f"      - Set price alert at 52w low as stop-loss reference")
        print()

    print(f"{heavy}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _env_model = os.getenv("ALPHA_SCANNER_LLM_MODEL") or os.getenv("PORTFOLIO_LLM_MODEL")

    parser = argparse.ArgumentParser(
        description="Run the AlphaScannerAgent quality-stock BUY-opportunity scan."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="Space-separated ticker symbols to scan.  E.g. --tickers AAPL MSFT NVDA",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        dest="top_n",
        metavar="N",
        help=(
            "Scan the top-N most liquid tickers from the built-in universe "
            "instead of an explicit watchlist (e.g. --top-n 50). "
            "Mutually usable with --tickers (--tickers takes precedence)."
        ),
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        metavar="MINUTES",
        help="Minutes between scans in continuous mode (default: 15).",
    )
    parser.add_argument(
        "--model",
        default=_env_model,
        metavar="MODEL_NAME",
        help=(
            "LLM model to use. Provider is inferred automatically. "
            f"(current: '{_env_model or 'default: gpt-oss:120b / ollama'}') "
            "Examples: gpt-4o, gemini-1.5-pro, llama3."
        ),
    )
    parser.add_argument(
        "--market",
        choices=["US", "IN", "IN_MID", "IN_SMALL"],
        default="US",
        metavar="MARKET",
        help=(
            "Market universe to scan (default: 'US'). "
            "'US' = Large + Mid Cap / NYSE, "
            "'IN' = NIFTY 50 / NSE, "
            "'IN_MID' = NIFTY MIDCAP 100 / NSE, "
            "'IN_SMALL' = NIFTY SMALLCAP 100 / NSE.  "
            "Applies to --top-n universe selection and market-hours check."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Enable verbose output: show DEBUG-level prefilter reasons and a "
            "per-ticker pipeline digest after the scan (only applies with --once)."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan and exit (useful for testing outside market hours).",
    )

    args = parser.parse_args()
    main(
        tickers=args.tickers,
        interval_minutes=args.interval,
        model=args.model,
        once=args.once,
        top_n=args.top_n,
        market=args.market,
        verbose=args.verbose,
    )
