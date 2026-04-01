"""
workflow.py
-----------
Entry point for the AlphaScannerAgent BUY-opportunity scanning workflow.

Run from the repository root:

    # Single scan — US market (S&P 500), useful outside market hours for testing
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

from src.agents.opportunity.nodes.alpha_scanner_agent import AlphaScannerAgent, _MAX_CONCURRENT_BUYS
from src.agents.opportunity.nodes.news_node import NewsNode
from src.agents.opportunity.nodes.decision_node import DecisionNode
from src.agents.opportunity.markets.market_strategy import get_liquid_universe, get_market_strategy
from src.agents.opportunity.state import OpportunityState
from src.agents.portfolio.state import PortfolioState
from src.agents.portfolio.subagents.portfolio_agent import PortfolioAgent
from src.agents.portfolio.subagents.market_agent import MarketAgent
from src.agents.portfolio.subagents.risk_agent import RiskAgent
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
# Portfolio context builder
# ---------------------------------------------------------------------------

def _fetch_portfolio_context() -> Dict[str, Any]:
    """
    Fetch live portfolio data from PortfolioAgent and RiskAgent.
    
    Builds a portfolio_context dict with:
      - position_weights: ticker → % of portfolio
      - sector_allocation: sector → % of portfolio  
      - cash_available: float (deployable capital)
      
    Returns empty dict if portfolio agent fails, allowing scan to continue.
    """
    try:
        # Initialize empty portfolio state
        port_state = PortfolioState()
        
        # Run PortfolioAgent to load positions
        port_agent = PortfolioAgent()
        port_state = port_agent.run(port_state)
        
        # Run MarketAgent to fetch live prices — required so RiskAgent
        # computes allocation % from real market values, not avg_cost.
        # Without this, a 15-share MSFT position at $403 avg_cost looks
        # ~40% of portfolio instead of its true current-price weighting.
        market_agent = MarketAgent()
        port_state = market_agent.run(port_state)
        
        # Run RiskAgent to compute allocations
        risk_agent = RiskAgent()
        port_state = risk_agent.run(port_state)
        
        # Extract portfolio context from state
        sector_alloc = port_state.sector_allocation or {}
        stock_alloc = port_state.risk_metrics.get("stock_allocation", {}) if port_state.risk_metrics else {}
        cash_avail = port_state.risk_metrics.get("cash_balance", 0.0) if port_state.risk_metrics else 0.0

        portfolio_context = {
            "position_weights": stock_alloc,
            "sector_allocation": sector_alloc,
            "cash_available": cash_avail,
        }
        
        logger.info(
            "[Portfolio] Loaded %d positions, %d sectors, cash=%.0f",
            len(stock_alloc), len(sector_alloc), cash_avail
        )
        return portfolio_context
        
    except Exception as exc:
        logger.warning(
            "[Portfolio] Failed to fetch portfolio context: %s. Continuing with empty portfolio.", exc
        )
        return {}


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
    graph.add_node("scanner",  AlphaScannerAgent().run)
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
    portfolio_context: Optional[Dict[str, Any]] = None,
    top_n: Optional[int] = None,
    market: str = "US",
) -> List[Dict[str, Any]]:
    """
    Single-shot scan.  Safe to call from external event-driven hooks
    (e.g. a price-spike webhook, a volatility alert, a scheduler).

    Parameters
    ----------
    tickers           : explicit list of ticker symbols; if None and top_n is
                        set, the dynamic liquid universe is used instead.
    portfolio_context : optional dict with sector_allocation, position_weights,
                        cash_available, total_positions, top_holding_weight
    top_n             : when tickers is None, scan the top-N most liquid tickers
                        from the built-in universe (default: 200).
    market            : "US" for S&P 500 / NYSE (default), "IN" for NIFTY 50 / NSE.

    Returns
    -------
    Sorted list of BUY opportunity dicts (may be empty).
    """
    resolved_tickers = tickers or get_liquid_universe(top_n or 200, market)
    compiled      = build_graph()
    initial_state = OpportunityState(
        watchlist=resolved_tickers,
        portfolio_context=portfolio_context or {},
    )
    result      = compiled.invoke(initial_state)
    final_state = result if isinstance(result, OpportunityState) else OpportunityState(**result)
    return final_state.buy_opportunities


def run_batch_scan(
    tickers: Optional[List[str]] = None,
    portfolio_context: Optional[Dict[str, Any]] = None,
    interval_minutes: int = 15,
    top_n: Optional[int] = None,
    market: str = "US",
    ignore_cash_check: bool = True,
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
            portfolio_context=portfolio_context or {},
            recent_signals=dict(recent_signals),          # shallow copy preserves cooldown
            recent_signal_context=dict(recent_signal_context),  # preserves freshness context
            ignore_cash_check=ignore_cash_check,
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
            logger.info("[Workflow] %d BUY opportunities this cycle:", len(opportunities))
            for opp in opportunities:
                logger.info(
                    "  %-6s | %-8s | score=%+d | %s",
                    opp["ticker"],
                    opp["confidence"],
                    opp["score"],
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
    portfolio_context: Optional[Dict[str, Any]] = None,
    interval_minutes: int = 15,
    model: Optional[str] = None,
    once: bool = False,
    top_n: Optional[int] = None,
    market: str = "US",
    verbose: bool = False,
    ignore_cash_check: bool = True,
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
    logger.info("STARTING ALPHA SCANNER WORKFLOW")
    logger.info("=" * 60)
    logger.info("Provider : %s  |  Model : %s", resolved_provider, resolved_model)
    logger.info("Market   : %s", "US (S&P 500 / NYSE)" if market.upper() == "US" else "IN (NIFTY 50 / NSE)")
    logger.info("Tickers  : %s", tickers or f"[dynamic top-{top_n or 200}]")

    if not tickers and not top_n:
        print("[Error] Provide --tickers AAPL MSFT or --top-n N to scan the liquid universe")
        raise SystemExit(1)

    # Fetch portfolio context if not explicitly provided
    if not portfolio_context:
        logger.info("Fetching portfolio context from PortfolioAgent...")
        portfolio_context = _fetch_portfolio_context()

    try:
        if once:
            final_state = _trigger_scan_full(tickers, portfolio_context, top_n, market, ignore_cash_check)
            _print_opportunities(final_state.buy_opportunities)
            _print_ignored(final_state)
            if verbose:
                _print_scan_digest(final_state)
        else:
            run_batch_scan(tickers, portfolio_context, interval_minutes, top_n, market, ignore_cash_check)

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
    portfolio_context: Optional[Dict[str, Any]] = None,
    top_n: Optional[int] = None,
    market: str = "US",
    ignore_cash_check: bool = True,
) -> "OpportunityState":
    """Same as trigger_scan but returns the full OpportunityState for diagnostics."""
    resolved_tickers = tickers or get_liquid_universe(top_n or 200, market)
    compiled      = build_graph()
    initial_state = OpportunityState(
        watchlist=resolved_tickers,
        portfolio_context=portfolio_context or {},
        ignore_cash_check=ignore_cash_check,
    )
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
        if ticker in state.blocked_no_cash:
            stage = "\ud83d\udcb0 NO CASH"
            print(f"  {ticker:<12}  {stage:<20}  {score_str:<14}  no deployable capital, not sent to LLM")
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
    score   = sig.get("score", 0)
    signals = sig.get("signals", [])
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
        parts.append("bearish signals outweigh bullish")
    elif signals:
        parts.append("signals are mixed — no dominant bullish catalyst")
    else:
        parts.append("no meaningful buy signals detected")

    # Analyst sentiment
    if analyst in ("sell", "strong sell", "underperform"):
        parts.append(f"analyst consensus is {analyst}")
    elif analyst in ("hold", "neutral"):
        parts.append(f"analyst consensus is {analyst} with no clear upside target")

    # Valuation trend
    if pe and fpe:
        if fpe >= pe:
            parts.append(
                f"earnings growth not priced in (fwd P/E {fpe:.1f} ≥ trailing {pe:.1f})"
            )

    # Price movement
    if change_pct is not None and abs(change_pct) < 1.0:
        parts.append(f"minimal price movement today ({change_pct:+.1f}%)")

    # News sentiment
    if news and news.get("sentiment", "neutral").lower() in ("negative", "bearish"):
        parts.append("negative news sentiment")

    # Volatility — low vol means less opportunity
    if volatility is not None and volatility < 0.20:
        parts.append(f"low volatility ({volatility:.0%}) — limited near-term price catalyst")

    if not parts:
        parts.append(f"insufficient conviction (score {score:+d})")

    return "; ".join(parts).capitalize() + "."


def _print_ticker_detail(
    ticker: str,
    sig: Dict[str, Any],
    state: "OpportunityState",
    reason: str,
    confidence: Optional[str] = None,
) -> None:
    """Print a full detail block for one ticker (shared by all rejection groups)."""
    score     = sig.get("score", 0)
    score_str = f"{score:+d}" if isinstance(score, int) else str(score)
    tier      = sig.get("tier", "?")
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

    print(
        f"\n  Ticker     : {ticker}{price_line}{cap_line}{sector_line}"
        f"\n  Score      : {score_str}  ({tier}){confidence_line}{override_line}"
        f"\n  Analyst    : {analyst_line}{valuation_line}{news_line}"
        f"\n  Reason     : {reason}"
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
    no_cash: list = []          # score OK but blocked because cash_available <= 0
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
        if ticker in state.blocked_no_cash:
            no_cash.append((ticker, sig))
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

    divider = "=" * 60
    print(f"\n{divider}")
    print(f"STOCKS SCANNED — NOT RECOMMENDED ({len(not_recommended)} ticker(s))")
    print(divider)

    # 1. LLM IGNOREs — full detail (highest score first)
    if llm_ignored:
        llm_ignored.sort(key=lambda x: x[1].get("score", 0), reverse=True)
        print(f"\n  [ LLM reviewed and declined: {len(llm_ignored)} ]")
        for ticker, sig, dec in llm_ignored:
            _print_ticker_detail(
                ticker, sig, state,
                reason=dec.get("reason", "N/A"),
                confidence=dec.get("confidence"),
            )

    # 2. Low score / cooldown — full detail block (same format as LLM IGNORE)
    if low_score:
        low_score.sort(key=lambda x: x[1].get("score", 0), reverse=True)
        print(f"\n  [ Passed activity filter but signal score too low (< {_CANDIDATE_MIN_SCORE}): {len(low_score)} ]")
        for ticker, sig in low_score:
            _print_ticker_detail(ticker, sig, state, reason=_build_low_score_reason(ticker, sig, state))

    if no_cash:
        no_cash.sort(key=lambda x: x[1].get("score", 0), reverse=True)
        cash_available = state.portfolio_context.get("cash_available", 0.0)
        print(f"\n  [ Score meets threshold but blocked by no deployable capital: {len(no_cash)} ]")
        for ticker, sig in no_cash:
            _print_ticker_detail(
                ticker,
                sig,
                state,
                reason=f"No deployable capital available (cash=${cash_available:,.0f}) — not sent to LLM.",
            )

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
    print(f"ALPHA SCANNER — SIGNALS DETECTED ({len(opportunities)} found)")
    print(heavy)

    if not opportunities:
        print("  No BUY signals identified in this scan.")
        print(f"\n{heavy}\n")
        return

    for opp in opportunities:
        ticker       = opp["ticker"]
        score        = opp["score"]
        opp_score    = opp.get("opportunity_score")
        tier         = opp.get("tier", "")
        confidence   = opp["confidence"]
        entry_quality = opp["entry_quality"]
        otype        = opp["type"]
        reason       = opp["reason"]
        sector       = opp.get("sector", "Unknown")
        current_pos  = opp.get("current_position_pct", 0.0)
        sector_alloc = opp.get("sector_allocation_pct", 0.0)
        pos_size     = opp.get("suggested_position_size")
        news_sent    = opp.get("news_sentiment", "N/A").upper()
        news_cat     = opp.get("news_catalyst", "")
        signals      = opp.get("signals", [])
        warnings     = opp.get("portfolio_warnings", [])
        hints        = opp.get("portfolio_hints", [])

        # ── Section 1: signal header ──────────────────────────────────────
        print(f"\n{heavy}")
        print("ALPHA SIGNAL — OPPORTUNITY DETECTED")
        print(heavy)

        opp_score_line = f"  Opportunity Score : {opp_score:.4f}" if opp_score is not None else ""
        print(
            f"\n  Ticker        : {ticker}\n"
            f"  Signal        : {opp['action']}\n"
            f"  Confidence    : {confidence}\n"
            f"  Entry Quality : {entry_quality}\n"
            f"  Score         : {score:+d}"
            + (f"  {opp_score_line}" if opp_score is not None else "") + "\n"
            f"  Type          : {otype}\n"
            f"  Sector        : {sector}  (held: {current_pos:.1f}%  |  sector alloc: {sector_alloc:.1f}%)\n"
        )

        # News
        if news_sent and news_sent != "N/A":
            print(f"  News          : {news_sent}  —  {news_cat}")

        # Signals
        if signals:
            print("\n  Quantitative Signals:")
            for s in signals:
                print(f"      - {s}")

        # Reason block
        print(f"\n  Reason:\n  {reason}\n")

        # ── Section 2: portfolio constraints ─────────────────────────────
        print(light)
        print("PORTFOLIO CONSTRAINTS (Advisory)")
        print(light)

        if not warnings and not hints:
            print("\n  No portfolio constraints detected.\n")
        else:
            if warnings:
                print()
                for w in warnings:
                    print(f"  {w}")
            if hints:
                print()
                for h in hints:
                    print(f"  {h}")
            print()

        # ── Section 3: execution status ───────────────────────────────────
        print(light)
        print("EXECUTION STATUS")
        print(light)

        # Actionable = no blocking warnings AND capital available
        has_cap_warning = any(
            kw in w for w in warnings
            for kw in ("POSITION CAP", "SECTOR CAP", "LIMITED CAPITAL")
        )
        actionable = not has_cap_warning

        print(f"\n  Actionable    : {'YES' if actionable else 'NO'}\n")
        if pos_size is not None:
            print(f"  Suggested size: ${pos_size:,.0f}  (cash ÷ {_MAX_CONCURRENT_BUYS} max buys)")

        print("\n  Suggested Actions:")
        if actionable:
            print(f"      - Execute BUY for {ticker}")
            if pos_size is not None:
                print(f"      - Limit position to ${pos_size:,.0f}")
            print(f"      - Set stop-loss below 52-week low")
        else:
            if any("POSITION CAP" in w for w in warnings):
                print(f"      - Consider trimming existing {ticker} position first")
            if any("SECTOR CAP" in w for w in warnings):
                print(f"      - Rebalance {sector} sector exposure before adding")
            if any("LIMITED CAPITAL" in w for w in warnings):
                print(f"      - Deploy capital when available")
            print(f"      - Add {ticker} to watchlist for next rebalance")
        print()

    print(f"{heavy}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _env_model = os.getenv("ALPHA_SCANNER_LLM_MODEL") or os.getenv("PORTFOLIO_LLM_MODEL")

    parser = argparse.ArgumentParser(
        description="Run the AlphaScannerAgent BUY-opportunity scan."
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
            "'US' = S&P 500 / NYSE, "
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
    parser.add_argument(
        "--enforce-cash",
        action="store_true",
        dest="enforce_cash",
        help=(
            "Enforce the zero-cash guard: skip all LLM decisions when "
            "portfolio cash_available <= 0.  By default this guard is disabled "
            "so the scan always runs regardless of available capital."
        ),
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
        ignore_cash_check=not args.enforce_cash,
    )
