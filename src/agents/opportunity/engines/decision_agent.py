import json
import logging
import os
import time
from functools import lru_cache
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_llm, infer_provider
from src.observability import get_telemetry_logger

logger = logging.getLogger(__name__)

_VALID_ACTIONS       = frozenset({"BUY", "IGNORE"})
_VALID_CONFIDENCE    = frozenset({"high", "moderate", "low"})
_VALID_ENTRY_QUALITY = frozenset({"strong", "moderate", "weak"})

_SYSTEM_PROMPT = """You are a conservative buy-side equity analyst identifying NEW entry opportunities.

CRITICAL CONSTRAINTS:
- You ONLY decide whether to BUY or IGNORE a stock.
- You MUST NOT suggest selling, reducing, or exiting any position.
- You are evaluating new entries — not managing existing holdings.
- Be conservative. High conviction is required for a BUY recommendation.

--------------------------------------------------
INPUT
--------------------------------------------------
You will receive:
  - A QUANTITATIVE SCORE block (deterministic, computed before this prompt)
  - Live market data: price, P/E ratios, volatility, 52-week range, sector, volume

The quantitative score is your PRIMARY ANCHOR. Do not contradict it without
citing a specific reason from the market data.

--------------------------------------------------
DECISION GUIDELINES
--------------------------------------------------

  BUY:
    - Score >= +1 with at least one fundamental positive signal
    - Valuation is reasonable (forward P/E improving or near 52w low)
    - Risk/reward is favourable for a new position
    - Volatility is acceptable for the entry size
    - Analyst consensus (when available) is not bearish

  IGNORE:
    - Score <= 0, or score +1 paired with elevated volatility and no valuation support
    - Conflicting signals with no clear directional thesis
    - Insufficient data to form a confident view
    - Price chasing near 52-week high
    - Analyst consensus is sell/underperform (unless strong quantitative buy signal)

--------------------------------------------------
CONFIDENCE
--------------------------------------------------

  high     — Multiple strong signals align clearly; high conviction
  moderate — Mixed signals but one side clearly stronger
  low      — Weak or conflicting signals, or notable data gaps

--------------------------------------------------
ENTRY QUALITY
--------------------------------------------------

  strong   — Ideal entry: score >= +2, low volatility, in value zone
  moderate — Acceptable: score +1, mixed signals, manageable risk
  weak     — Marginal: barely passes threshold, elevated risk

--------------------------------------------------
ADDITIONAL CONTEXT (when provided)
--------------------------------------------------

  Analyst consensus : Wall Street analyst rating + coverage count + mean price target.
    - Use as a secondary confirmation signal, not a primary override.
    - A bullish rating (buy/strong_buy) with a meaningful upside target supports BUY.
    - A bearish rating (underperform/sell) is a red flag even with a positive quant score.

  Volume pressure : A proxy for institutional flow derived from price direction × volume spike.
    - "buying"  = price up on elevated volume  → demand signal
    - "selling" = price down on elevated volume → supply / distribution signal
    - "neutral" = no significant volume spike

  News sentiment : LLM-classified headline sentiment (positive / neutral / negative)
    and the primary catalyst (one sentence).
    - Consider whether the catalyst is a durable thesis driver or a one-day event.
    - Negative news can justify IGNORE even when the quant score is positive.
    - Positive news corroborates BUY but does not override a score <= 0.

--------------------------------------------------

--------------------------------------------------
RULES
--------------------------------------------------
- If data is insufficient → IGNORE with low confidence; state what is missing
- Reference the score tier in your reason
- Keep reason factual, concise, one sentence

Respond ONLY with a valid JSON object. No markdown, no text outside the JSON.

{
  "action": "BUY" | "IGNORE",
  "confidence": "high" | "moderate" | "low",
  "entry_quality": "strong" | "moderate" | "weak",
  "reason": "one concise sentence"
}

When action is IGNORE, still populate entry_quality with your best
assessment — it describes the stock's entry profile, not a purchase decision."""


@lru_cache(maxsize=1)
def _get_decision_llm():
    """
    Lazy LLM singleton for the opportunity decision agent.

    Resolution order:
      1. ALPHA_SCANNER_LLM_MODEL  → provider inferred via infer_provider()
      2. PORTFOLIO_LLM_MODEL      → provider inferred via infer_provider()
      3. PORTFOLIO_LLM_PROVIDER   env var (default: "ollama") with provider default model
    """
    model    = os.getenv("ALPHA_SCANNER_LLM_MODEL") or os.getenv("PORTFOLIO_LLM_MODEL") or None
    provider = os.getenv("PORTFOLIO_LLM_PROVIDER", "ollama")

    if model:
        try:
            provider = infer_provider(model)
        except ValueError:
            # Unknown model — fall back to the explicit provider env var
            pass

    return get_llm(model_name=provider, model=model)


def _build_human_message(
    ticker: str,
    market_data: Dict[str, Any],
    signal_result: Dict[str, Any],
    opportunity_type: str = "momentum",
    news_sentiment: Optional[Dict[str, Any]] = None,
) -> str:
    score      = signal_result["score"]
    tier       = signal_result["tier"].upper()
    sign       = "+" if score > 0 else ""
    signals_text = "\n".join(
        f"    {s}" for s in signal_result["signals"]
    ) or "    (no signals fired)"

    pe_trailing = market_data.get("pe_ratio")
    pe_forward  = market_data.get("forward_pe")
    pe_trailing_str = f"{pe_trailing:.1f}" if pe_trailing is not None else "N/A"
    pe_forward_str  = f"{pe_forward:.1f}"  if pe_forward  is not None else "N/A"

    # Analyst consensus block
    analyst_rating = (market_data.get("analyst_rating") or "none").replace("_", " ")
    analyst_count  = market_data.get("analyst_count", 0) or 0
    analyst_target = market_data.get("analyst_target")
    analyst_target_str = f"${analyst_target:.2f}" if analyst_target else "N/A"
    if analyst_count >= 1:
        analyst_line = f"{analyst_rating}  ({analyst_count} analysts, target {analyst_target_str})"
    else:
        analyst_line = "N/A  (no analyst coverage)"

    # News sentiment block (optional — only present when NewsNode ran)
    if news_sentiment:
        news_block = (
            f"News sentiment   : {news_sentiment.get('sentiment', 'neutral').upper()}  "
            f"({news_sentiment.get('headline_count', 0)} headlines)\n"
            f"News catalyst    : {news_sentiment.get('catalyst', 'N/A')}\n"
        )
    else:
        news_block = ""

    score_block = (
        "--------------------------------------------------\n"
        "QUANTITATIVE SCORE (deterministic — computed before LLM)\n"
        "--------------------------------------------------\n"
        f"  Total score : {sign}{score}  →  {tier}\n"
        f"  Signals:\n{signals_text}\n"
        "--------------------------------------------------"
    )

    return (
        f"{score_block}\n\n"
        f"Ticker           : {ticker}\n"
        f"Sector           : {market_data.get('sector', 'Unknown')}\n"
        f"Opportunity      : {opportunity_type}  (signal-derived)\n"
        f"Current price    : ${market_data.get('price', 0.0):.2f}\n"
        f"Daily change     : {market_data.get('change_pct', 0.0):+.1f}%\n"
        f"Volume pressure  : {market_data.get('vol_pressure', 'neutral').upper()}\n"
        f"Volatility       : {market_data.get('volatility', 0.0):.0%}  (annualised)\n"
        f"Trailing P/E     : {pe_trailing_str}\n"
        f"Forward P/E      : {pe_forward_str}\n"
        f"52w high / low   : ${market_data.get('52w_high', 0.0):.2f} / "
        f"${market_data.get('52w_low', 0.0):.2f}\n"
        f"Volume           : {market_data.get('volume', 0):,}  "
        f"(30d avg: {market_data.get('avg_volume', 0):,})\n"
        f"Analyst consensus: {analyst_line}\n"
        f"{news_block}"
        "\nShould this ticker be bought as a new position? Respond with JSON only."
    )


def _parse_llm_response(content: str) -> Dict[str, Any]:
    """
    Parse and validate the LLM's JSON output.
    Returns a safe IGNORE/low-confidence fallback on any error.
    """
    text = content.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text  = parts[1].lstrip("json").strip() if len(parts) > 1 else text
        text  = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "[OpportunityDecisionAgent] JSON parse error: %s | raw: %.200s", exc, content
        )
        return {
            "action":        "IGNORE",
            "confidence":    "low",
            "entry_quality": "weak",
            "reason":        "LLM response could not be parsed.",
        }

    action = str(parsed.get("action", "")).upper()
    if action not in _VALID_ACTIONS:
        logger.warning(
            "[OpportunityDecisionAgent] Unexpected action %r — defaulting to IGNORE", action
        )
        action = "IGNORE"

    confidence = str(parsed.get("confidence", "low")).lower()
    if confidence not in _VALID_CONFIDENCE:
        confidence = "low"

    entry_quality = str(parsed.get("entry_quality", "weak")).lower()
    if entry_quality not in _VALID_ENTRY_QUALITY:
        entry_quality = "weak"

    reason = str(parsed.get("reason", "")).strip() or "No reason provided."

    return {
        "action":        action,
        "confidence":    confidence,
        "entry_quality": entry_quality,
        "reason":        reason,
    }


class OpportunityDecisionAgent:
    """
    LLM-based BUY/IGNORE decision for a single candidate ticker.

    Called per-ticker by AlphaScannerAgent (not a LangGraph node directly).
    Uses a conservative system prompt that explicitly forbids SELL/REDUCE/EXIT.

    On any LLM or parse failure the method returns a safe IGNORE/low-confidence
    result so the scan pipeline continues without crashing.
    """

    def run(
        self,
        ticker: str,
        market_data: Dict[str, Any],
        signal_result: Dict[str, Any],
        opportunity_type: str = "momentum",
        news_sentiment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Invoke the LLM for a single ticker and return a validated decision dict.

        Parameters
        ----------
        opportunity_type : pre-classified type from SignalEngine ("dip_buy" | "value" | "momentum").
                           Injected into the human message as context; returned verbatim in the
                           result dict.  The LLM does NOT decide this.
        news_sentiment   : optional dict from NewsNode with keys sentiment, catalyst, headline_count.

        Returns
        -------
        dict with keys: ticker, action, confidence, entry_quality, reason, type, score
        """
        human_msg = _build_human_message(
            ticker, market_data, signal_result, opportunity_type, news_sentiment
        )

        messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_msg)]
        try:
            llm = _get_decision_llm()
            response = None
            for _retry in range(3):
                try:
                    response = llm.invoke(messages)
                    break
                except Exception as exc:
                    if "429" in str(exc) and _retry < 2:
                        _wait = 2.0 * (2 ** _retry)  # 2s, then 4s
                        logger.warning(
                            "[OpportunityDecisionAgent] 429 rate-limit for %s — retry %d in %.0fs",
                            ticker, _retry + 1, _wait,
                        )
                        time.sleep(_wait)
                    else:
                        raise
            raw    = response.content if hasattr(response, "content") else str(response)
            result = _parse_llm_response(raw)
        except Exception as exc:
            logger.warning(
                "[OpportunityDecisionAgent] LLM call failed for %s: %s", ticker, exc
            )
            result = {
                "action":        "IGNORE",
                "confidence":    "low",
                "entry_quality": "weak",
                "reason":        f"LLM unavailable: {exc}",
            }

        get_telemetry_logger().log_llm_interaction(human_msg, result.get("reason", ""))

        return {
            "ticker": ticker,
            "score":  signal_result["score"],
            "type":   opportunity_type,
            **result,
        }
