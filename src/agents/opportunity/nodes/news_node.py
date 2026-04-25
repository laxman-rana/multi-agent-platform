"""
news_node.py
------------
NewsNode: LangGraph node that fetches recent news headlines via yfinance and
runs a single LLM call per candidate ticker to produce a sentiment summary
(positive / neutral / negative) plus a one-line catalyst.

Only *candidates* (score >= 1, cooldown clear) are processed.  Fetching news
for every ticker in a 200-stock universe would be wasteful and slow.

State contract
--------------
  Reads  : state.candidates
  Writes : state.news_sentiment   — {ticker: {sentiment, catalyst, headline_count}}
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional

import yfinance as yf
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.opportunity.state import OpportunityState
from src.agents.opportunity.engines.signal_engine import _quality_tier as _score_to_quality_tier, _tier as _score_to_tier
from src.llm import get_llm, get_provider, infer_provider

logger = logging.getLogger(__name__)

_MAX_HEADLINES  = 5   # headlines per ticker sent to the LLM
_MAX_NEWS_WORKERS = 3  # max parallel LLM calls (Ollama concurrency caps this further)

# Fundamental risk guardrail: big drop + bad news = falling knife, not a buy.
_FUNDAMENTAL_RISK_DROP_PCT: float = 10.0  # abs(change_pct) threshold
_FUNDAMENTAL_RISK_PENALTY:  int   = -2    # score adjustment applied

_NEWS_SYSTEM_PROMPT = """You are a financial news analyst.

Given a list of recent news headlines for a publicly traded stock, determine:
  1. Overall sentiment: "positive", "neutral", or "negative"
  2. The primary catalyst in ONE concise sentence — what is the main story moving this stock?

Rules:
  - If there are no meaningful or relevant headlines, return "neutral" and
    "No significant news catalyst."
  - Do NOT speculate beyond the headline text.
  - Keep the catalyst sentence under 20 words.

Respond ONLY with valid JSON (no markdown, no extra text):
{"sentiment": "positive" | "neutral" | "negative", "catalyst": "one sentence"}"""


@lru_cache(maxsize=1)
def _get_news_llm():
    """Lazy LLM singleton — shares provider config with the scanner."""
    model    = os.getenv("ALPHA_SCANNER_LLM_MODEL") or os.getenv("PORTFOLIO_LLM_MODEL") or None
    provider = os.getenv("PORTFOLIO_LLM_PROVIDER", "ollama")
    if model:
        try:
            provider = infer_provider(model)
        except ValueError:
            pass
    return get_llm(model_name=provider, model=model)


def _fetch_headlines(ticker: str) -> List[str]:
    """Return up to _MAX_HEADLINES recent news titles for ticker via yfinance."""
    try:
        items = yf.Ticker(ticker).news or []
        titles = []
        for item in items[:_MAX_HEADLINES]:
            # yfinance >=0.2.x wraps titles under content.title
            title = (
                (item.get("content") or {}).get("title")
                or item.get("title")
                or ""
            )
            if title:
                titles.append(title)
        return titles
    except Exception as exc:
        logger.debug("[NewsNode] headline fetch failed for %s: %s", ticker, exc)
        return []


def _summarise_news(ticker: str, headlines: List[str]) -> Dict[str, Any]:
    """Run one LLM call to classify sentiment + extract a catalyst sentence."""
    if not headlines:
        return {
            "sentiment":     "neutral",
            "catalyst":      "No recent news found.",
            "headline_count": 0,
        }

    human_text = f"Ticker: {ticker}\nRecent headlines:\n" + "\n".join(
        f"- {h}" for h in headlines
    )
    try:
        llm      = _get_news_llm()
        messages = [SystemMessage(content=_NEWS_SYSTEM_PROMPT), HumanMessage(content=human_text)]
        response = None
        for _retry in range(3):
            try:
                response = llm.invoke(messages)
                break
            except Exception as exc:
                if "429" in str(exc) and _retry < 2:
                    _wait = 2.0 * (2 ** _retry)
                    logger.warning(
                        "[NewsNode] 429 rate-limit for %s — retry %d in %.0fs",
                        ticker, _retry + 1, _wait,
                    )
                    time.sleep(_wait)
                else:
                    raise
        raw = response.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw   = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

        parsed    = json.loads(raw)
        sentiment = str(parsed.get("sentiment", "neutral")).lower()
        if sentiment not in ("positive", "neutral", "negative"):
            sentiment = "neutral"
        catalyst = str(parsed.get("catalyst", "")).strip() or "No notable catalyst."
        return {
            "sentiment":      sentiment,
            "catalyst":       catalyst,
            "headline_count": len(headlines),
        }
    except Exception as exc:
        logger.warning("[NewsNode] LLM sentiment failed for %s: %s", ticker, exc)
        return {
            "sentiment":      "neutral",
            "catalyst":       "News analysis unavailable.",
            "headline_count": len(headlines),
        }


def _apply_fundamental_risk_penalty(state: "OpportunityState") -> "OpportunityState":
    """
    Post-news guardrail: downgrade score by −2 when a large price drop coincides
    with negative news sentiment.

    Prevents the capitulation signal (+2) from producing a BUY when the drop is
    caused by fraud, earnings collapse, or structural decline rather than
    indiscriminate panic.  Rule: if abs(change_pct) ≥ 10% AND sentiment == "negative",
    the stock is a falling knife — penalise the score so it needs stronger
    fundamental support to cross the BUY threshold.
    """
    for ticker in state.candidates:
        news = state.news_sentiment.get(ticker)
        if not news or news.get("sentiment") != "negative":
            continue

        mdata      = state.market_data.get(ticker, {})
        change_pct = mdata.get("change_pct", 0.0) or 0.0
        if change_pct > -_FUNDAMENTAL_RISK_DROP_PCT:
            continue

        sig = state.signals.get(ticker)
        if sig is None:
            continue

        old_score = sig["score"]
        new_score = old_score + _FUNDAMENTAL_RISK_PENALTY
        sig["score"] = new_score
        sig["quality_score"] = new_score
        sig["tier"] = _score_to_tier(new_score)
        sig["quality_tier"] = _score_to_quality_tier(new_score)
        catalyst_snippet = (news.get("catalyst") or "")[:70]
        risk_signal = (
            f"\u26a0 Fundamental risk: {change_pct:.1f}% drop + negative news "
            f"\u2192 score {_FUNDAMENTAL_RISK_PENALTY:+d} "
            f"({catalyst_snippet})"
        )
        sig["signals"] = sig.get("signals", []) + [risk_signal]
        sig["quality_signals"] = sig.get("quality_signals", []) + [risk_signal]
        logger.warning(
            "[NewsNode] \u26a0 Fundamental risk — %s: drop=%.1f%%  negative news  "
            "score %+d \u2192 %+d  (falling knife guard)",
            ticker, change_pct, old_score, new_score,
        )

    return state


class NewsNode:
    """
    LangGraph node: fetch headlines + LLM sentiment for each candidate ticker.

    Position in graph:  scanner → news → decision
    Reads  state.candidates (set by AlphaScannerAgent).
    Writes state.news_sentiment.
    """

    def run(self, state: OpportunityState) -> OpportunityState:
        if not state.candidates:
            logger.debug("[NewsNode] No candidates — skipping news fetch.")
            return state

        # Respect provider concurrency limit (Ollama = 1 concurrent request)
        _provider_name = os.getenv("PORTFOLIO_LLM_PROVIDER", "ollama")
        _max_workers   = min(
            len(state.candidates),
            get_provider(_provider_name).max_concurrency,
            _MAX_NEWS_WORKERS,
        ) or 1

        logger.info(
            "[NewsNode] Fetching news for %d candidate(s)  (workers=%d)",
            len(state.candidates), _max_workers,
        )

        def _process(ticker: str):
            headlines = _fetch_headlines(ticker)
            summary   = _summarise_news(ticker, headlines)
            return ticker, summary

        with ThreadPoolExecutor(max_workers=_max_workers) as pool:
            futures = {pool.submit(_process, t): t for t in state.candidates}
            for future in as_completed(futures):
                ticker, summary = future.result()
                state.news_sentiment[ticker] = summary
                logger.info(
                    "[NewsNode] %-6s  sentiment=%-8s  headlines=%d  catalyst=%s",
                    ticker,
                    summary["sentiment"],
                    summary["headline_count"],
                    summary["catalyst"][:70],
                )

        # After all sentiments are known, apply the falling-knife guardrail.
        state = _apply_fundamental_risk_penalty(state)
        return state
