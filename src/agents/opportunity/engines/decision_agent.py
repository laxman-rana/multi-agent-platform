"""
decision_agent.py
-----------------
LLM risk interpreter for a single candidate ticker.

ARCHITECTURE
  - Deterministic scoring layer is SOURCE OF TRUTH.
  - LLM CANNOT override or downgrade the decision tier.
  - LLM role: thesis classification, structured risk, position sizing,
               entry triggers, and narrative.

Score → Decision (authoritative):
  +8 to +10  →  STRONG_BUY
  +5 to +7   →  BUY
  +3 to +4   →  WATCHLIST
  below +3   →  IGNORE

Signal hierarchy:
  Tier 1 (hard): revenue growth, earnings growth, margins, ROE, FCF, forward valuation
  Tier 2 (soft): news sentiment, price movement, volatility  — context only, never override

News classification:
  structural  → fundamental change (earnings impairment, contract loss, guidance cut)
  temporary   → market reaction (price drop, sentiment, analyst price-target cut)
"""

import json
import logging
import os
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_llm, infer_provider
from src.observability import get_telemetry_logger

logger = logging.getLogger(__name__)

# ── Deterministic score → decision ────────────────────────────────────────────
_SCORE_TO_DECISION = [
    (8, "STRONG_BUY"),
    (5, "BUY"),
    (3, "WATCHLIST"),
    (0, "IGNORE"),
]

_VALID_DECISIONS     = frozenset({"STRONG_BUY", "BUY", "WATCHLIST", "IGNORE"})
_VALID_CONFIDENCE    = frozenset({"high", "moderate", "low"})
_VALID_ENTRY_QUALITY = frozenset({"strong", "moderate", "weak"})
_VALID_RISK          = frozenset({"low", "medium", "high"})
_VALID_TIME_HORIZON  = frozenset({"short_term", "long_term"})
_VALID_NEWS_IMPACT   = frozenset({"structural", "temporary", "none"})
_VALID_THESIS        = frozenset({
    "quality_compounder",
    "high_growth_speculative",
    "value_play",
    "turnaround",
    "cyclical",
})
_VALID_POSITION_TYPE = frozenset({"core", "starter", "none"})
_BUY_DECISIONS       = frozenset({"STRONG_BUY", "BUY"})


def _score_to_decision(quality_score: int) -> str:
    for threshold, label in _SCORE_TO_DECISION:
        if quality_score >= threshold:
            return label
    return "IGNORE"


# ── Position sizing rules (deterministic — applied after LLM risk) ─────────────
def _derive_position_sizing(
    decision: str,
    risk_level: str,
    range_position_pct: float = 0.5,   # 0.0 = at 52w low, 1.0 = at 52w high
) -> Dict[str, str]:
    """
    Investment-grade position sizing — 3 inputs:
      decision          : STRONG_BUY | BUY | WATCHLIST | IGNORE
      risk_level        : low | medium | high
      range_position_pct: where price sits in the 52w range (0–1)

    Rules:
      STRONG_BUY + low/medium risk + range < 0.90  → core     5–10%
      STRONG_BUY + near 52w high  (range >= 0.90)  → starter  2–4%  (margin of safety compressed)
      STRONG_BUY + high risk                        → starter  2–4%
      BUY + range < 0.85                            → starter  1–3%
      BUY + range >= 0.85                           → starter  1–2%  (reduced sizing near highs)
      WATCHLIST / IGNORE                            → none     0%
    """
    near_high = range_position_pct >= 0.90
    at_high   = range_position_pct >= 0.85

    if decision == "STRONG_BUY":
        if risk_level == "high" or near_high:
            rationale = (
                "Price near 52w high compresses margin of safety — initiate small, add on pullback"
                if near_high else
                "Elite fundamentals offset by elevated risk — size conservatively and build on confirmation"
            )
            return {"type": "starter", "range": "2–4%", "rationale": rationale}
        return {
            "type": "core", "range": "5–10%",
            "rationale": "Strong fundamentals with manageable risk support a full core allocation",
        }

    if decision == "BUY":
        if at_high:
            return {
                "type": "starter", "range": "1–2%",
                "rationale": "Solid quality case but price near 52w high — size small and wait for better entry",
            }
        return {
            "type": "starter", "range": "1–3%",
            "rationale": "Solid quality case warrants initial exposure; scale on fundamental improvement",
        }

    return {"type": "none", "range": "0%",
            "rationale": "Insufficient quality conviction for capital deployment at this time"}


_SYSTEM_PROMPT = """You are a SENIOR INVESTMENT ANALYST interpreting pre-scored stock opportunities.

================================================================
YOUR ROLE — READ CAREFULLY
================================================================

A deterministic scoring engine has ALREADY made the investment decision.
Your job is to ENRICH it — NOT to change it.

You will be given:
  - The FINAL DECISION (already made — you MUST return it unchanged)
  - Quality score and all fired signals
  - Full fundamental and market data

You must output structured JSON enrichment covering:
  1. thesis_type      — what kind of investment is this?
  2. reason           — concise trade-off anchored in Tier 1 fundamentals
  3. key_signals      — up to 5 Tier 1 signals only (no duplicates)
  4. risk_breakdown   — structured 3-axis risk model
  5. entry_triggers   — why is this interesting NOW?
  6. notes            — compliance guardrail confirmation

================================================================
THESIS CLASSIFICATION — MANDATORY, NEVER SKIP
================================================================

Classify the stock into EXACTLY ONE of:

  quality_compounder
    → Stable high margins + strong ROE + consistent FCF + moderate growth
    → Long-term hold, durable moat
    → Example: MSFT, AAPL, V

  high_growth_speculative
    → High revenue/earnings growth (>25%) + unstable or thin margins
    → ROE inconsistent or low + high volatility
    → High reward but sustainability unproven
    → Example: CELH, early-stage SaaS, biotech pre-profitability

  value_play
    → Low forward P/E relative to peers + depressed price
    → Margins stable but growth subdued
    → Catalyst needed: re-rating, buyback, activist
    → Example: cheap industrial, unloved consumer staple

  turnaround
    → Previously weak fundamentals showing recovery signals
    → Earnings growth accelerating from a low base
    → High uncertainty on sustainability
    → Example: post-restructuring retailer, recovering airline

  cyclical
    → Earnings driven by macro/commodity cycle
    → Margins expand/contract with cycle
    → Timing-sensitive entry
    → Example: steel, energy, semiconductors at cycle trough

RULES:
  - High growth (>25%) + high volatility + thin/unstable margins → high_growth_speculative
  - Strong margins + ROE + FCF → quality_compounder
  - NEVER say "no thesis" — every stock has a classification
  - When in doubt between two types, pick the one that best describes the PRIMARY risk

================================================================
SIGNAL HIERARCHY — STRICTLY ENFORCED
================================================================

Tier 1 — HARD SIGNALS (anchor confidence and reason):
  revenue growth | earnings growth | profit margin | operating margin
  return on equity | free cash flow yield | forward P/E

Tier 2 — SOFT SIGNALS (context only, never drive confidence):
  news sentiment | daily price movement | short-term volatility

RULES:
  - key_signals must contain ONLY Tier 1 signals
  - Maximum 5 key_signals — no duplicates
  - If Tier 1 signals are strong → confidence must be moderate or high
  - Tier 2 signals only appear in risk_breakdown.sentiment_risk

================================================================
NEWS CLASSIFICATION — REQUIRED
================================================================

  structural  → earnings impairment | lost major contract | guidance cut |
                balance-sheet blow-up | fraud
                → CAN raise fundamental_risk to high
                → CAN lower confidence to low

  temporary   → share price fell | analyst price-target cut | sentiment dip |
                macro fear | short-term volume
                → CAN raise sentiment_risk to high
                → CANNOT lower confidence below moderate if Tier 1 is strong
                → CANNOT change decision

  none        → no material news
                → no impact

================================================================
STRUCTURED RISK MODEL — 3 AXES
================================================================

Output:
{
  "risk_breakdown": {
    "volatility_risk":   "low | medium | high",   // annualised vol vs peers
    "fundamental_risk":  "low | medium | high",   // quality of Tier 1 signals
    "sentiment_risk":    "low | medium | high"    // news + market reaction
  }
}

Rules:
  fundamental_risk = low   if 3+ strong Tier 1 signals fire
  fundamental_risk = medium if 1-2 Tier 1 signals fire or one is weak
  fundamental_risk = high  if structural news risk OR Tier 1 is sparse
  volatility_risk  = high  if annualised vol > 45%
  sentiment_risk   = high  if structural news; medium if temporary; low if none

================================================================
ENTRY TRIGGERS — WHY NOW?
================================================================

List 1–3 concrete reasons this is interesting at the CURRENT price:

Examples:
  - "Valuation compression: forward P/E 19x vs 5-year average 40x"
  - "Price at 12% of 52-week range — sentiment-driven dip, fundamentals intact"
  - "Earnings growth accelerating while price declined — multiple compression"
  - "Analyst consensus buy with +100% upside — market skepticism creates entry"

RULES FOR ENTRY TRIGGERS:
  - NEVER cite high 52w range position (>60% of range) as a positive entry trigger
  - High range position = reduced margin of safety, NOT an opportunity
  - Only cite 52w range as positive if price is in the LOWER 40% of the range
  - If price is near 52w high, cite it as a RISK FACTOR instead

NEVER use vague triggers like "looks interesting" or "good entry point".

================================================================
CONFIDENCE RULES
================================================================

  high     → 3+ Tier 1 signals strong; no structural risk
  moderate → Tier 1 case holds but 1-2 caveats OR temporary news headwind
  low      → Tier 1 sparse OR structural risk present

================================================================
HARD GUARDRAILS
================================================================

  ✗ NEVER return a different decision than the one given
  ✗ NEVER return IGNORE if decision is STRONG_BUY or BUY
  ✗ NEVER say "no thesis" — always classify
  ✗ NEVER use sentiment as primary driver of confidence
  ✗ NEVER repeat the same signal twice in key_signals
  ✓ ALWAYS anchor reason in Tier 1 fundamentals
  ✓ ALWAYS explain the quality vs risk trade-off
  ✓ ALWAYS classify thesis_type

================================================================
OUTPUT FORMAT — JSON ONLY, NO MARKDOWN
================================================================

{
  "decision":      "<MUST MATCH EXACTLY>",
  "thesis_type":   "quality_compounder | high_growth_speculative | value_play | turnaround | cyclical",
  "confidence":    "high | moderate | low",
  "entry_quality": "strong | moderate | weak",
  "time_horizon_bias": "short_term | long_term",
  "news_impact":   "structural | temporary | none",
  "reason":        "one sentence: explicit trade-off anchored in Tier 1 fundamentals",
  "key_signals":   ["max 5, Tier 1 only, no duplicates"],
  "risk_breakdown": {
    "volatility_risk":  "low | medium | high",
    "fundamental_risk": "low | medium | high",
    "sentiment_risk":   "low | medium | high"
  },
  "entry_triggers": ["why now — 1 to 3 concrete reasons"],
  "notes": [
    "LLM did not override deterministic score",
    "Short-term signals did not override fundamentals"
  ]
}"""


@lru_cache(maxsize=1)
def _get_decision_llm():
    model    = os.getenv("ALPHA_SCANNER_LLM_MODEL") or os.getenv("PORTFOLIO_LLM_MODEL") or None
    provider = os.getenv("PORTFOLIO_LLM_PROVIDER", "ollama")
    if model:
        try:
            provider = infer_provider(model)
        except ValueError:
            pass
    return get_llm(model_name=provider, model=model)


def _fmt_ratio(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return default


def _build_human_message(
    ticker: str,
    market_data: Dict[str, Any],
    signal_result: Dict[str, Any],
    opportunity_type: str,
    news_sentiment: Optional[Dict[str, Any]],
    deterministic_decision: str,
) -> str:
    quality_score = signal_result.get("quality_score", signal_result["score"])
    quality_tier  = signal_result.get("quality_tier", signal_result["tier"]).upper()
    sign          = "+" if quality_score > 0 else ""
    signals_text  = (
        "\n".join(f"    {s}" for s in signal_result.get("quality_signals", signal_result["signals"]))
        or "    (no signals fired)"
    )

    pe_trailing = market_data.get("pe_ratio")
    pe_forward  = market_data.get("forward_pe")
    pe_t_str    = f"{pe_trailing:.1f}" if pe_trailing is not None else "N/A"
    pe_f_str    = f"{pe_forward:.1f}"  if pe_forward  is not None else "N/A"

    analyst_rating = (market_data.get("analyst_rating") or "none").replace("_", " ")
    analyst_count  = market_data.get("analyst_count", 0) or 0
    analyst_target = market_data.get("analyst_target")
    analyst_str    = f"${analyst_target:.2f}" if analyst_target else "N/A"
    analyst_line   = (
        f"{analyst_rating} ({analyst_count} analysts, target {analyst_str})"
        if analyst_count >= 1 else "N/A (no analyst coverage)"
    )

    news_block = ""
    if news_sentiment:
        news_block = (
            f"News sentiment   : {news_sentiment.get('sentiment','neutral').upper()}"
            f" ({news_sentiment.get('headline_count', 0)} headlines)\n"
            f"News catalyst    : {news_sentiment.get('catalyst','N/A')}\n"
        )

    # 52w position
    high_52  = market_data.get("52w_high", 0.0)
    low_52   = market_data.get("52w_low",  0.0)
    price    = market_data.get("price", 0.0)
    pos_52   = "N/A"
    if high_52 and low_52 and high_52 > low_52:
        pos_52 = f"{(price - low_52) / (high_52 - low_52):.0%} of 52w range"

    score_block = (
        "--------------------------------------------------\n"
        "DETERMINISTIC SCORE (source of truth — do not override)\n"
        "--------------------------------------------------\n"
        f"  Quality score    : {sign}{quality_score}  ({quality_tier})\n"
        f"  DECISION (FINAL) : {deterministic_decision}  ← RETURN THIS EXACTLY\n"
        f"  Internal type    : {opportunity_type}\n"
        f"  Signals fired:\n{signals_text}\n"
        "--------------------------------------------------"
    )

    return (
        f"{score_block}\n\n"
        f"Ticker           : {ticker}\n"
        f"Sector           : {market_data.get('sector','Unknown')}\n"
        f"Price            : ${price:.2f}  ({market_data.get('change_pct', 0.0):+.1f}% today)\n"
        f"52w position     : {pos_52}\n"
        f"Volatility       : {market_data.get('volatility', 0.0):.0%} annualised\n"
        f"Market cap       : ${((market_data.get('market_cap') or 0.0)/1e9):.1f}B\n"
        f"Trailing P/E     : {pe_t_str}\n"
        f"Forward P/E      : {pe_f_str}\n"
        f"Profit margin    : {_fmt_ratio(market_data.get('profit_margins'))}\n"
        f"Operating margin : {_fmt_ratio(market_data.get('operating_margins'))}\n"
        f"Return on equity : {_fmt_ratio(market_data.get('return_on_equity'))}\n"
        f"Debt / equity    : {market_data.get('debt_to_equity','N/A')}\n"
        f"FCF / mkt cap    : {_fmt_ratio(market_data.get('fcf_yield'))}\n"
        f"Revenue growth   : {_fmt_ratio(market_data.get('revenue_growth'))}\n"
        f"Earnings growth  : {_fmt_ratio(market_data.get('earnings_growth'))}\n"
        f"Analyst          : {analyst_line}\n"
        f"{news_block}"
        "\nClassify thesis_type, assess risk, list entry triggers, confirm decision. "
        "JSON only."
    )


def _parse_risk_breakdown(raw: Any) -> Dict[str, str]:
    defaults = {"volatility_risk": "medium", "fundamental_risk": "medium", "sentiment_risk": "medium"}
    if not isinstance(raw, dict):
        return defaults
    out = {}
    for k in ("volatility_risk", "fundamental_risk", "sentiment_risk"):
        v = str(raw.get(k, "medium")).lower()
        out[k] = v if v in _VALID_RISK else "medium"
    return out


def _overall_risk(breakdown: Dict[str, str]) -> str:
    """Derive single risk_level from 3-axis breakdown (worst axis wins)."""
    order = {"high": 2, "medium": 1, "low": 0}
    worst = max(breakdown.values(), key=lambda x: order.get(x, 0))
    return worst


def _parse_llm_response(content: str, deterministic_decision: str) -> Dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text  = parts[1].lstrip("json").strip() if len(parts) > 1 else text

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("[DecisionAgent] JSON parse error: %s | raw: %.200s", exc, content)
        return _fallback_response(deterministic_decision, "LLM response could not be parsed.")

    # ── GUARDRAIL: decision is always deterministic ────────────────────────
    llm_decision = str(parsed.get("decision", "")).upper()
    if llm_decision != deterministic_decision:
        logger.warning(
            "[DecisionAgent] LLM tried to change decision %r → %r — enforcing %r",
            deterministic_decision, llm_decision, deterministic_decision,
        )

    decision = deterministic_decision  # authoritative, always

    thesis_type = str(parsed.get("thesis_type", "high_growth_speculative")).lower()
    if thesis_type not in _VALID_THESIS:
        thesis_type = "high_growth_speculative"

    confidence = str(parsed.get("confidence", "moderate")).lower()
    if confidence not in _VALID_CONFIDENCE:
        confidence = "moderate"

    entry_quality = str(parsed.get("entry_quality", "moderate")).lower()
    if entry_quality not in _VALID_ENTRY_QUALITY:
        entry_quality = "moderate"

    time_horizon = str(parsed.get("time_horizon_bias", "long_term")).lower()
    if time_horizon not in _VALID_TIME_HORIZON:
        time_horizon = "long_term"

    news_impact = str(parsed.get("news_impact", "none")).lower()
    if news_impact not in _VALID_NEWS_IMPACT:
        news_impact = "none"

    reason = str(parsed.get("reason", "")).strip() or "No reason provided."

    key_signals = parsed.get("key_signals", parsed.get("key_supporting_signals", []))
    if not isinstance(key_signals, list):
        key_signals = []
    key_signals = list(dict.fromkeys(key_signals))[:5]  # deduplicate, cap at 5

    risk_breakdown = _parse_risk_breakdown(parsed.get("risk_breakdown", {}))
    risk_level     = _overall_risk(risk_breakdown)

    entry_triggers = parsed.get("entry_triggers", [])
    if not isinstance(entry_triggers, list):
        entry_triggers = []

    notes = parsed.get("notes", [])
    if not isinstance(notes, list):
        notes = [str(notes)] if notes else []
    if "LLM did not override deterministic score" not in " ".join(notes):
        notes.append("LLM did not override deterministic score")

    return {
        "decision":        decision,
        "thesis_type":     thesis_type,
        "confidence":      confidence,
        "entry_quality":   entry_quality,
        "risk_level":      risk_level,
        "risk_breakdown":  risk_breakdown,
        "time_horizon_bias": time_horizon,
        "news_impact":     news_impact,
        "reason":          reason,
        "key_signals":     key_signals,
        "entry_triggers":  entry_triggers,
        "notes":           notes,
    }


def _fallback_response(deterministic_decision: str, reason: str) -> Dict[str, Any]:
    """Safe fallback — always preserves deterministic decision."""
    return {
        "decision":        deterministic_decision,
        "thesis_type":     "high_growth_speculative",
        "confidence":      "low",
        "entry_quality":   "weak",
        "risk_level":      "high",
        "risk_breakdown":  {"volatility_risk": "high", "fundamental_risk": "high", "sentiment_risk": "high"},
        "time_horizon_bias": "long_term",
        "news_impact":     "none",
        "reason":          reason,
        "key_signals":     [],
        "entry_triggers":  [],
        "notes":           ["LLM did not override deterministic score", "Fallback response — LLM unavailable"],
    }


class OpportunityDecisionAgent:
    """
    LLM risk interpreter. The deterministic score drives the decision.
    The LLM adds thesis classification, structured risk, position sizing,
    and entry context — it cannot change the decision tier.
    """

    def run(
        self,
        ticker: str,
        market_data: Dict[str, Any],
        signal_result: Dict[str, Any],
        opportunity_type: str = "quality_watchlist",
        news_sentiment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        quality_score          = signal_result.get("quality_score", signal_result["score"])
        deterministic_decision = _score_to_decision(quality_score)

        # Inject FCF yield into market_data for the prompt (computed from existing fields)
        fcf   = market_data.get("free_cash_flow")
        cap   = market_data.get("market_cap")
        if fcf and cap and cap > 0 and fcf > 0:
            market_data = {**market_data, "fcf_yield": fcf / cap}

        human_msg = _build_human_message(
            ticker, market_data, signal_result, opportunity_type,
            news_sentiment, deterministic_decision,
        )

        messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_msg)]
        try:
            llm      = _get_decision_llm()
            response = None
            for retry in range(3):
                try:
                    response = llm.invoke(messages)
                    break
                except Exception as exc:
                    if "429" in str(exc) and retry < 2:
                        wait = 2.0 * (2 ** retry)
                        logger.warning("[DecisionAgent] 429 for %s — retry %d in %.0fs", ticker, retry + 1, wait)
                        time.sleep(wait)
                    else:
                        raise
            raw    = response.content if hasattr(response, "content") else str(response)
            result = _parse_llm_response(raw, deterministic_decision)
        except Exception as exc:
            logger.warning("[DecisionAgent] LLM call failed for %s: %s", ticker, exc)
            result = _fallback_response(deterministic_decision, f"LLM unavailable: {exc}")

        get_telemetry_logger().log_llm_interaction(human_msg, result.get("reason", ""))

        decision = result["decision"]
        action   = "BUY" if decision in _BUY_DECISIONS else "IGNORE"

        # Position sizing is deterministic — derived from decision + risk_level
        # Compute 52w range position for sizing input
        _h52  = market_data.get("52w_high", 0.0)
        _l52  = market_data.get("52w_low",  0.0)
        _price = market_data.get("price",   0.0)
        _range_pct = (
            (_price - _l52) / (_h52 - _l52)
            if _h52 and _l52 and _h52 > _l52 else 0.5
        )
        position_sizing = _derive_position_sizing(decision, result["risk_level"], _range_pct)

        return {
            "ticker":           ticker,
            "score":            signal_result["score"],
            "quality_score":    quality_score,
            "quality_tier":     signal_result.get("quality_tier"),
            "type":             opportunity_type,
            # Decision
            "decision":         decision,
            "action":           action,   # legacy field for DecisionNode compat
            # LLM enrichment
            "thesis_type":      result["thesis_type"],
            "confidence":       result["confidence"],
            "entry_quality":    result["entry_quality"],
            "risk_level":       result["risk_level"],
            "risk_breakdown":   result["risk_breakdown"],
            "time_horizon_bias": result["time_horizon_bias"],
            "news_impact":      result["news_impact"],
            "reason":           result["reason"],
            "key_signals":      result["key_signals"],
            "entry_triggers":   result["entry_triggers"],
            "notes":            result["notes"],
            # Position sizing (deterministic from decision + risk)
            "position_sizing":  position_sizing,
        }
