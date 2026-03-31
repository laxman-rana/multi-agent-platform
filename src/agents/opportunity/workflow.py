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

from src.agents.opportunity.nodes.alpha_scanner_agent import AlphaScannerAgent
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
        cash_avail = port_state.risk_metrics.get("cash_balance", float("inf")) if port_state.risk_metrics else float("inf")
        
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
    recent_signals: Dict[str, str] = {}

    logger.info(
        "[Workflow] Starting batch scan — %d tickers, interval=%d min",
        len(resolved_tickers), interval_minutes,
    )

    while is_market_open(market):
        state_input = OpportunityState(
            watchlist=resolved_tickers,
            portfolio_context=portfolio_context or {},
            recent_signals=dict(recent_signals),   # shallow copy preserves cooldown
        )

        result = compiled.invoke(state_input)
        final  = result if isinstance(result, OpportunityState) else OpportunityState(**result)

        # Carry forward updated cooldown timestamps
        recent_signals = final.recent_signals

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
            if verbose:
                final_state = _trigger_scan_full(tickers, portfolio_context, top_n, market)
                _print_opportunities(final_state.buy_opportunities)
                _print_scan_digest(final_state)
            else:
                opportunities = trigger_scan(tickers, portfolio_context, top_n, market)
                _print_opportunities(opportunities)
        else:
            run_batch_scan(tickers, portfolio_context, interval_minutes, top_n, market)

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
) -> "OpportunityState":
    """Same as trigger_scan but returns the full OpportunityState for diagnostics."""
    resolved_tickers = tickers or get_liquid_universe(top_n or 200, market)
    compiled      = build_graph()
    initial_state = OpportunityState(
        watchlist=resolved_tickers,
        portfolio_context=portfolio_context or {},
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
        if sig["score"] < _CANDIDATE_MIN_SCORE:
            stage = "\u274c SCORE TOO LOW"
            print(f"  {ticker:<12}  {stage:<20}  {score_str:<14}  score < {_CANDIDATE_MIN_SCORE}, not sent to LLM")
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


def _print_opportunities(opportunities: List[Dict[str, Any]]) -> None:
    """Pretty-print the final BUY opportunity list to stdout."""
    divider = "=" * 60
    print(f"\n{divider}")
    print(f"ALPHA SCANNER — BUY OPPORTUNITIES ({len(opportunities)} found)")
    print(divider)

    if not opportunities:
        print("  No BUY opportunities identified in this scan.")
    else:
        for opp in opportunities:
            signals_text = (
                "\n".join(f"      - {s}" for s in opp.get("signals", []))
                or "      (none)"
            )
            
            # Build portfolio context section
            warnings = opp.get("portfolio_warnings", [])
            hints = opp.get("portfolio_hints", [])
            portfolio_text = ""
            
            if warnings or hints:
                portfolio_text = "\n  Portfolio Context:"
                for warning in warnings:
                    portfolio_text += f"\n      {warning}"
                for hint in hints:
                    portfolio_text += f"\n      {hint}"
            
            sector = opp.get("sector", "Unknown")
            current_pos = opp.get("current_position_pct", 0.0)
            sector_alloc = opp.get("sector_allocation_pct", 0.0)
            
            print(
                f"\n  Ticker        : {opp['ticker']}\n"
                f"  Action        : {opp['action']}\n"
                f"  Confidence    : {opp['confidence']}\n"
                f"  Entry quality : {opp['entry_quality']}\n"
                f"  Type          : {opp['type']}\n"
                f"  Score         : {opp['score']:+d}\n"
                f"  Sector        : {sector} (current alloc: {sector_alloc:.1f}%)\n"
                f"  Position      : {current_pos:.1f}% currently held\n"
                f"  News          : {opp.get('news_sentiment', 'N/A').upper()}  —  {opp.get('news_catalyst', '')}\n"
                f"  Reason        : {opp['reason']}\n"
                f"  Signals       :\n{signals_text}{portfolio_text}"
            )

    print(f"\n{divider}\n")


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
