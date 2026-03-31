"""
decision_node.py
----------------
DecisionNode: LangGraph node that runs the per-candidate LLM BUY/IGNORE
decisions, builds portfolio-context warnings, emits scan telemetry, and
sorts the final buy_opportunities list.

Extracted from AlphaScannerAgent so each pipeline stage has one responsibility:

  [scanner]       steps 1-5: fetch, prefilter, signal, candidate filter, guards
  [news_node]     step  6  : headline fetch + LLM sentiment
  [decision_node] steps 7-9: LLM decisions, portfolio warnings, sort, summary

State contract
--------------
  Reads  : state.candidates, state.signals, state.market_data,
           state.news_sentiment, state.portfolio_context
  Writes : state.decisions, state.recent_signals, state.buy_opportunities
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.agents.opportunity.nodes.alpha_scanner_agent import _MAX_SECTOR_EXPOSURE, _MAX_POSITION_WEIGHT
from src.agents.opportunity.engines.decision_agent import OpportunityDecisionAgent
from src.agents.opportunity.state import OpportunityState
from src.observability import get_telemetry_logger

logger = logging.getLogger(__name__)

_MAX_LLM_WORKERS:           int   = 5
_DECISION_CACHE_TTL_MINUTES: int  = 15
_CONFIDENCE_RANK:    Dict[str, int]   = {"high": 0, "moderate": 1, "low": 2}
_SECTOR_UNDERWEIGHT_BOOST:   float   = 0.5
_SECTOR_TARGET_WEIGHT:       float   = 20.0


class DecisionNode:
    """
    LangGraph node: LLM BUY/IGNORE per candidate → portfolio warnings → sort.

    Instance-level decision cache (TTL: _DECISION_CACHE_TTL_MINUTES) avoids
    redundant LLM calls when the same ticker re-enters candidates with the
    same score+type within a scan cycle.
    """

    def __init__(self) -> None:
        self._decision_agent = OpportunityDecisionAgent()
        # ticker:score:type  →  (decision_dict, monotonic_timestamp)
        self._decision_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}

    def run(self, state: OpportunityState) -> OpportunityState:
        if not state.candidates:
            return state

        telemetry         = get_telemetry_logger()
        portfolio_context = state.portfolio_context
        sector_allocation: Dict[str, float] = portfolio_context.get("sector_allocation", {})
        position_weights:  Dict[str, float] = portfolio_context.get("position_weights", {})
        cash_available:    float            = portfolio_context.get("cash_available", float("inf"))

        approved_candidates = state.candidates
        now_iso      = datetime.now(timezone.utc).isoformat()
        cache_cutoff = time.monotonic() - _DECISION_CACHE_TTL_MINUTES * 60
        node_t0      = time.monotonic()

        # ── per-ticker LLM calls ──────────────────────────────────────────

        def _cache_key(tkr: str) -> str:
            sig = state.signals[tkr]
            return f"{tkr}:{sig['score']}:{sig['type']}"

        def _decide(tkr: str) -> Tuple[str, Dict[str, Any], float]:
            key   = _cache_key(tkr)
            entry = self._decision_cache.get(key)
            if entry and entry[1] >= cache_cutoff:
                logger.debug("[DecisionNode] cache hit: %s", tkr)
                return tkr, entry[0], 0.0

            mdata      = state.market_data[tkr]
            sigres     = state.signals[tkr]
            news_sent  = state.news_sentiment.get(tkr)
            t0         = time.monotonic()
            dec        = self._decision_agent.run(
                tkr, mdata, sigres, sigres["type"], news_sentiment=news_sent
            )
            lat        = round((time.monotonic() - t0) * 1000, 1)
            self._decision_cache[key] = (dec, time.monotonic())
            return tkr, dec, lat

        n_workers = min(len(approved_candidates), _MAX_LLM_WORKERS) or 1
        raw_decisions: Dict[str, Tuple[Dict[str, Any], float]] = {}

        _t = time.monotonic()
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_decide, t): t for t in approved_candidates}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    _, decision, latency_ms = fut.result()
                    raw_decisions[t] = (decision, latency_ms)
                except Exception as exc:
                    logger.warning("[DecisionNode] LLM decision failed for %s: %s", t, exc)
                    raw_decisions[t] = (
                        {
                            "action":        "IGNORE",
                            "confidence":    "low",
                            "entry_quality": "weak",
                            "reason":        f"LLM error: {exc}",
                            "type":          state.signals[t]["type"],
                            "score":         state.signals[t]["score"],
                            "ticker":        t,
                        },
                        0.0,
                    )
        llm_total_ms = round((time.monotonic() - _t) * 1000, 1)

        # ── emit telemetry + build buy entries ────────────────────────────

        for ticker in approved_candidates:
            decision, latency_ms = raw_decisions[ticker]
            signal_result        = state.signals[ticker]
            mkt                  = state.market_data[ticker]
            sector               = mkt.get("sector", "Unknown")
            current_position     = position_weights.get(ticker, 0.0)
            current_sector_pct   = sector_allocation.get(sector, 0.0)

            telemetry.log_event(
                "llm_decision",
                {
                    "ticker":     ticker,
                    "score":      signal_result["score"],
                    "action":     decision["action"],
                    "confidence": decision["confidence"],
                    "latency_ms": latency_ms,
                },
            )
            logger.info(
                "[DecisionNode] %s → %s (%s) | score=%+d | %.0fms%s",
                ticker,
                decision["action"],
                decision["confidence"],
                signal_result["score"],
                latency_ms,
                " [cached]" if latency_ms == 0.0 else "",
            )

            state.decisions[ticker] = decision

            if decision["action"] != "BUY":
                continue

            state.recent_signals[ticker] = now_iso

            # ── portfolio concentration warnings ──────────────────────────
            portfolio_warnings: List[str] = []
            portfolio_hints:    List[str] = []

            if current_position > 0:
                if current_position > _MAX_POSITION_WEIGHT:
                    portfolio_warnings.append(
                        f"⚠️  POSITION CAP EXCEEDED: {ticker} is already {current_position:.1f}% "
                        f"of portfolio (cap: {_MAX_POSITION_WEIGHT:.0f}%). "
                        f"Adding more increases concentration risk."
                    )
                elif current_position > _MAX_POSITION_WEIGHT * 0.8:
                    portfolio_warnings.append(
                        f"⚠️  CONCENTRATION RISK: {ticker} is at {current_position:.1f}% "
                        f"of portfolio (cap: {_MAX_POSITION_WEIGHT:.0f}%). Approaching limit."
                    )
                else:
                    portfolio_hints.append(
                        f"ℹ️  Current position: {current_position:.1f}%. "
                        f"Additional BUY will increase exposure to {ticker}."
                    )
            else:
                portfolio_hints.append(
                    f"✓ NEW POSITION: {ticker} not currently held; clean entry opportunity."
                )

            sector_gap = _SECTOR_TARGET_WEIGHT - current_sector_pct
            if current_sector_pct > _MAX_SECTOR_EXPOSURE:
                portfolio_warnings.append(
                    f"⚠️  SECTOR CAP EXCEEDED: {sector} is at {current_sector_pct:.1f}% "
                    f"(hard cap: {_MAX_SECTOR_EXPOSURE:.0f}%). Adding BUY significantly "
                    f"increases sector concentration risk."
                )
            elif sector_gap < 0:
                portfolio_warnings.append(
                    f"⚠️  SECTOR OVERWEIGHT: {sector} is at {current_sector_pct:.1f}% "
                    f"(target: {_SECTOR_TARGET_WEIGHT:.0f}%). "
                    f"BUY will further increase sector concentration."
                )
            elif sector_gap > 5.0:
                portfolio_hints.append(
                    f"✓ DIVERSIFICATION OPPORTUNITY: {sector} is {sector_gap:.1f}% BELOW target. "
                    f"BUY supports portfolio rebalancing toward {_SECTOR_TARGET_WEIGHT:.0f}%."
                )
            else:
                portfolio_hints.append(
                    f"ℹ️  Sector fit: {sector} is at {current_sector_pct:.1f}% "
                    f"(target: {_SECTOR_TARGET_WEIGHT:.0f}%). BUY is aligned with target allocation."
                )

            if cash_available < float("inf"):
                approx_total = cash_available + sum(position_weights.values()) * 10
                cash_pct     = (cash_available / approx_total * 100) if approx_total > 0 else 100
                if cash_pct < 10.0:
                    portfolio_warnings.append(
                        f"⚠️  LIMITED CAPITAL: Only ${cash_available:,.0f} available "
                        f"({cash_pct:.0f}%). Full position size NOT possible."
                    )
                elif cash_pct < 25.0:
                    portfolio_hints.append(
                        f"ℹ️  TIGHT CAPITAL: {cash_pct:.0f}% of capital available. "
                        f"Recommend max {_MAX_POSITION_WEIGHT / 2:.1f}% position size."
                    )
                else:
                    portfolio_hints.append(
                        f"✓ CAPITAL AVAILABLE: {cash_pct:.0f}% of capital deployable. "
                        f"Full position sizing up to {_MAX_POSITION_WEIGHT:.0f}% is possible."
                    )
            else:
                portfolio_hints.append("✓ UNLIMITED CAPITAL: Full position sizing possible.")

            news_sent = state.news_sentiment.get(ticker, {})
            buy_entry = {
                "ticker":                ticker,
                "action":                "BUY",
                "confidence":            decision["confidence"],
                "entry_quality":         decision["entry_quality"],
                "reason":                decision["reason"],
                "type":                  decision["type"],
                "score":                 signal_result["score"],
                "signals":               signal_result["signals"],
                "sector":                sector,
                "current_position_pct":  current_position,
                "sector_allocation_pct": current_sector_pct,
                "portfolio_warnings":    portfolio_warnings,
                "portfolio_hints":       portfolio_hints,
                "news_sentiment":        news_sent.get("sentiment", "neutral"),
                "news_catalyst":         news_sent.get("catalyst", ""),
            }
            state.buy_opportunities.append(buy_entry)

            telemetry.log_event(
                "buy_signal_emitted",
                {
                    "ticker":                ticker,
                    "score":                 signal_result["score"],
                    "confidence":            decision["confidence"],
                    "entry_quality":         decision["entry_quality"],
                    "type":                  decision["type"],
                    "sector":                sector,
                    "current_position_pct":  current_position,
                    "sector_allocation_pct": current_sector_pct,
                    "news_sentiment":        news_sent.get("sentiment", "neutral"),
                    "warnings_count":        len(portfolio_warnings),
                    "hints_count":           len(portfolio_hints),
                },
            )
            logger.info(
                "[DecisionNode] BUY emitted: %s | %s | score=%+d | "
                "pos=%.1f%% sector=%.1f%% | ⚠️ %d warnings ℹ️ %d hints",
                ticker, decision["confidence"], signal_result["score"],
                current_position, current_sector_pct,
                len(portfolio_warnings), len(portfolio_hints),
            )

        # ── sort: confidence rank → portfolio-fit-boosted score desc ──────

        def _fit_boost(opp: Dict[str, Any]) -> float:
            s       = state.market_data.get(opp["ticker"], {}).get("sector", "Unknown")
            current = sector_allocation.get(s, 0.0)
            gap     = max(0.0, _SECTOR_TARGET_WEIGHT - current)
            return gap * _SECTOR_UNDERWEIGHT_BOOST

        state.buy_opportunities.sort(
            key=lambda x: (
                _CONFIDENCE_RANK.get(x["confidence"], 99),
                -(x["score"] + _fit_boost(x)),
            )
        )

        # ── scan summary telemetry ─────────────────────────────────────────
        total_ms = round((time.monotonic() - node_t0) * 1000, 1)
        telemetry.log_event(
            "scan_summary",
            {
                "timestamp":         datetime.now(timezone.utc).isoformat(),
                "watchlist_size":     len(state.watchlist),
                "fetched":           len(state.market_data),
                "fetch_errors":      len(state.scan_errors),
                "prefiltered":       len(state.prefiltered),
                "scored":            len(state.signals),
                "candidates":        len(approved_candidates),
                "buy_opportunities": len(state.buy_opportunities),
                "llm_total_ms":      llm_total_ms,
                "total_ms":          total_ms,
                "buy_tickers":       [o["ticker"] for o in state.buy_opportunities],
            },
        )
        logger.info(
            "[DecisionNode] Scan complete — %d BUY opportunit%s | llm=%.0fms",
            len(state.buy_opportunities),
            "y" if len(state.buy_opportunities) == 1 else "ies",
            llm_total_ms,
        )
        return state
