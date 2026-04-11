import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple


from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.agents.portfolio.models import NewsArticle, StockDecision, StockInsight
from src.agents.portfolio.state import PortfolioState
from src.agents.portfolio.tools.validation import validate_decision
from src.agents.portfolio.tools.scoring import score_stock
from src.agents.portfolio.tools.rebalance_tools import compute_portfolio_action
from src.agents.portfolio.tools.news_tools import compute_news_score
from src.observability import get_telemetry_logger
from src.llm import get_llm, get_provider

logger = logging.getLogger(__name__)


def _parse_horizon_years(horizon_str: str) -> float:
    """Parse '5 years', '18 months', etc. to a float number of years."""
    m = re.search(r'(\d+(?:\.\d+)?)\s*(year|month)', horizon_str.lower())
    if not m:
        return 1.0
    value = float(m.group(1))
    return value if m.group(2).startswith('year') else value / 12.0


def _horizon_block(horizon_years: float) -> str:
    """Return the investment-horizon awareness section for the system prompt."""
    if horizon_years >= 3.0:
        return (
            "--------------------------------------------------\n"
            "INVESTMENT HORIZON: LONG-TERM\n"
            "--------------------------------------------------\n"
            "The investor's horizon is LONG-TERM (3+ years).\n"
            "Adjust your reasoning accordingly:\n\n"
            "  - Daily price moves are NOISE at this timescale.\n"
            "    Do NOT exit solely because of one red session.\n"
            "    The quantitative score already excludes the daily-change signal.\n"
            "  - Focus on: PE valuation trend (forward vs trailing), earnings\n"
            "    growth, 52-week range position, and long-term thesis integrity.\n"
            "  - HOLD is strongly preferred when fundamentals are intact.\n"
            "  - Only EXIT if: thesis is fundamentally broken, extreme overvaluation\n"
            "    with no earnings growth, or unrealised loss exceeds -35%.\n"
            "--------------------------------------------------"
        )
    if horizon_years >= 1.5:
        return (
            "--------------------------------------------------\n"
            "INVESTMENT HORIZON: MEDIUM-TERM\n"
            "--------------------------------------------------\n"
            "The investor's horizon is MEDIUM-TERM (1-3 years).\n"
            "Balance near-term momentum signals with fundamental indicators.\n"
            "Avoid reacting to a single session's move in isolation.\n"
            "--------------------------------------------------"
        )
    return (
        "--------------------------------------------------\n"
        "INVESTMENT HORIZON: SHORT-TERM\n"
        "--------------------------------------------------\n"
        "The investor's horizon is SHORT-TERM (< 1 year).\n"
        "Daily price momentum and near-term signals carry higher relevance.\n"
        "--------------------------------------------------"
    )


_SYSTEM_PROMPT = """
You are a quantitative portfolio analyst for a moderate-risk investor.

Your task is to analyze a SINGLE equity position using ONLY the provided input data.
Do NOT assume or invent any information.

--------------------------------------------------
INPUT DATA
--------------------------------------------------
You will receive structured data including:
- position details (quantity, avg price, current price, P&L)
- volatility
- fundamentals (e.g., PE, growth)
- sentiment/news (if available)

--------------------------------------------------
OUTPUT FORMAT (STRICT)
--------------------------------------------------
Return ONLY a valid JSON object. No extra text.

{
  "action": "EXIT" | "REDUCE" | "HOLD" | "DOUBLE_DOWN",
  "confidence": "high" | "moderate" | "low",
  "reason": "one concise sentence",
  "allocation_change": "+10%" | "-15%" | "0%" | "-100%"
}

allocation_change must be a signed percentage string.
Rules by action (see ALLOCATION CHANGE RULES below for full guidance):
  EXIT        → always "-100%"
  REDUCE      → negative value between "-5%" and "-50%"
  HOLD        → always "0%"
  DOUBLE_DOWN → positive value between "+5%" and "+20%"

--------------------------------------------------
DECISION GUIDELINES
--------------------------------------------------

EXIT:
- Thesis is fundamentally broken
- Severe downside risk with no recovery path
- Very high volatility + major negative signals + deep loss

REDUCE:
- Meaningful downside risk but position still has long-term merit
- Score is sell/weak-sell AND position weight is significant (>10%)
- Use instead of EXIT when: small loss (<15%), earnings growth intact, or
  high sector concentration (trim to rebalance, not abandon)

HOLD:
- Balanced risk/reward
- No strong positive or negative signals
- Stable fundamentals

DOUBLE_DOWN:
- Temporary dip or undervaluation
- Strong fundamentals (growth, low forward PE relative to trailing PE)
- No major negative sentiment

--------------------------------------------------
CONFIDENCE GUIDELINES
--------------------------------------------------

high:
- Multiple strong signals align

moderate:
- Mixed signals but one side slightly stronger

low:
- Weak or conflicting signals OR insufficient data

--------------------------------------------------
RULES
--------------------------------------------------

- Be conservative (moderate-risk investor)
- Do NOT hallucinate missing data
- If data is insufficient → choose HOLD with low confidence
- Keep reasoning factual and based on input only
- Reason must reference key factors (risk, fundamentals, sentiment)

--------------------------------------------------
QUANTITATIVE SCORE RULES
--------------------------------------------------

You will receive a QUANTITATIVE SCORE block computed deterministically from
the raw data BEFORE you read it.  Use it as your primary anchor:

  strong_buy  (≥3)  → lean toward DOUBLE_DOWN unless sentiment is very negative
  buy         (+2)    → lean toward HOLD or DOUBLE_DOWN
  buy         (+1)    → HOLD only — weak signal, insufficient conviction for DOUBLE_DOWN
  neutral     (0)     → lean toward HOLD
  sell        (-1/-2) → lean toward REDUCE (partial trim) before considering EXIT
  strong_sell (≤-3)  → lean toward EXIT; use REDUCE if long-term thesis intact

DOUBLE_DOWN MINIMUM THRESHOLD:
  DOUBLE_DOWN requires score >= 2.  A score of +1 is a weak buy signal and does
  NOT justify adding to the position.  Use HOLD instead.

--------------------------------------------------
ALLOCATION CHANGE RULES
--------------------------------------------------

Pick allocation_change that reflects the magnitude of conviction:

  EXIT        → "-100%"  (close the full position — no partial)

  REDUCE      → how much to trim:
    score -1 AND gain > -5%                  → "-10%"
    score -1 AND gain in (-15%, -5%]         → "-15%"
    score -2                                  → "-20%"
    score ≤ -3 OR position weight > 20%       → "-30%" to "-50%"

  HOLD        → "0%"  (no change)

  DOUBLE_DOWN → how much to add:
    score +2                                  → "+5%"
    score +3                                  → "+10%"
    score ≥ 4                                 → "+15%"

  Adjust upward/downward if sentiment strongly confirms or contradicts the
  score signal.  Stay within the bounds shown above.

REDUCE vs EXIT guidance:
  - REDUCE when: score is sell AND (gain_pct > -15% OR fwd_pe improving)
  - EXIT   when: score is sell AND (gain_pct < -20% OR thesis broken)

You may deviate from the score tier IF you cite a specific reason from the
sentiment or portfolio context that overrides it.  You MUST mention the score
tier in your reason field.

--------------------------------------------------
QUANTITATIVE SCORE RULES
--------------------------------------------------

You will receive a QUANTITATIVE SCORE block computed deterministically from
the raw data BEFORE you read it.  Use it as your primary anchor:

  strong_buy  (≥3)  → lean toward DOUBLE_DOWN unless sentiment is very negative
  buy         (+2)    → lean toward HOLD or DOUBLE_DOWN
  buy         (+1)    → HOLD only — insufficient conviction for DOUBLE_DOWN
  neutral     (0)     → lean toward HOLD
  sell        (-1/-2) → lean toward REDUCE (partial trim) before considering EXIT
  strong_sell (≤-3)  → lean toward EXIT; use REDUCE if long-term thesis intact

REDUCE vs EXIT guidance:
  - REDUCE when: score is sell AND (gain_pct > -15% OR fwd_pe improving)
  - EXIT   when: score is sell AND (gain_pct < -20% OR thesis broken)

You may deviate from the score tier IF you cite a specific reason from the
sentiment or portfolio context that overrides it.  You MUST mention the score
tier in your reason field.

"""


def _build_system_prompt(
    critic_issues: Optional[List[str]] = None,
    horizon_years: float = 1.0,
) -> str:
    """
    Returns the base system prompt with a horizon-awareness section appended,
    or — on a critic retry — prepends a MANDATORY CORRECTION block so the
    model sees the rejection reason as the very first thing it reads.
    """
    base = _SYSTEM_PROMPT + "\n" + _horizon_block(horizon_years) + "\n"

    if not critic_issues:
        return base

    issues_text = "\n".join(f"  - {issue}" for issue in critic_issues)
    correction_header = (
        "=" * 58 + "\n"
        "\u26a0  MANDATORY CORRECTION — PREVIOUS RESPONSE REJECTED\n"
        "=" * 58 + "\n"
        "Your previous response was rejected by the critic.\n"
        "Fix the issues raised by the critic:\n"
        f"{issues_text}\n\n"
        "Rules that apply to THIS retry:\n"
        "  1. You MUST resolve every issue listed above.\n"
        "  2. A 'low' confidence answer will be rejected again automatically.\n"
        "  3. If data is truly insufficient, output HOLD with 'moderate' confidence\n"
        "     and explain what data is missing in the reason.\n"
        "=" * 58 + "\n\n"
    )
    return correction_header + base.lstrip()


def _format_quant_score(quant_score: Dict[str, Any]) -> str:
    """
    Renders the deterministic score block injected into every human message.
    The model sees the score BEFORE the free-form data so it acts as an anchor.
    """
    score   = quant_score["score"]
    tier    = quant_score["tier"].upper().replace("_", " ")
    sign    = "+" if score > 0 else ""
    lines   = quant_score["breakdown"]
    body    = "\n".join(lines) if lines else "  (no signals fired)"
    note    = (
        "  Note: daily session change excluded — long-term investor horizon\n"
        if quant_score.get("long_term")
        else ""
    )
    return (
        "--------------------------------------------------\n"
        "QUANTITATIVE SCORE (deterministic — computed before LLM)\n"
        "--------------------------------------------------\n"
        f"  Total score : {sign}{score}  \u2192  {tier}\n"
        f"  Signals:\n{body}\n"
        f"{note}"
        "--------------------------------------------------"
    )


def _format_portfolio_context(portfolio_context: Dict[str, Any]) -> str:
    """
    Renders the portfolio-level context block that appears in every human
    message so the LLM can factor portfolio-wide risk into each decision.
    """
    sector_lines = "\n".join(
        f"    {sector:<22} {pct:.1f}%"
        for sector, pct in sorted(
            portfolio_context.get("sector_exposure", {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )
    top_stock    = portfolio_context.get("top_stock", "")
    top_weight   = portfolio_context.get("top_stock_weight", 0.0)
    conc_risk    = portfolio_context.get("concentration_risk", "unknown").upper()
    high_conc    = portfolio_context.get("high_concentration", False)
    n_positions  = portfolio_context.get("total_positions", 0)

    conc_warning = ""
    if high_conc:
        conc_warning = (
            f"\n  \u26a0 HIGHLY CONCENTRATED: {top_stock} makes up {top_weight:.1f}% of the "
            f"portfolio.\n    Be conservative with DOUBLE_DOWN for this ticker or its sector."
        )

    horizon_str  = portfolio_context.get("investment_horizon", "")
    horizon_line = f"  Investment horizon : {horizon_str}\n" if horizon_str else ""

    return (
        "--------------------------------------------------\n"
        "PORTFOLIO CONTEXT (consider this, not just the stock)\n"
        "--------------------------------------------------\n"
        f"  Concentration risk : {conc_risk}\n"
        f"  Total positions    : {n_positions}\n"
        f"  Top holding        : {top_stock} ({top_weight:.1f}% of portfolio)\n"
        f"{horizon_line}"
        f"  Sector exposure    :\n{sector_lines}\n"
        f"{conc_warning}\n"
        "--------------------------------------------------"
    )


def _build_human_message(
    ticker: str,
    insight: StockInsight,
    news: List[NewsArticle],
    gain_pct: float,
    stock_allocation_pct: Optional[float] = None,
    portfolio_context: Optional[Dict[str, Any]] = None,
    quant_score: Optional[Dict[str, Any]] = None,
) -> str:
    pe_trailing = insight.pe_ratio
    pe_forward  = insight.forward_pe
    pe_trailing_str = f"{pe_trailing:.1f}" if pe_trailing is not None else "N/A"
    pe_forward_str  = f"{pe_forward:.1f}"  if pe_forward  is not None else "N/A"

    if stock_allocation_pct is not None:
        conc_flag = " \u26a0 HIGH CONCENTRATION" if stock_allocation_pct > 40 else ""
        weight_line = f"Portfolio weight: {stock_allocation_pct:.1f}%{conc_flag}\n"
    else:
        weight_line = ""

    portfolio_block = (
        "\n" + _format_portfolio_context(portfolio_context) + "\n"
        if portfolio_context
        else ""
    )

    score_block = (
        "\n" + _format_quant_score(quant_score) + "\n"
        if quant_score
        else ""
    )

    return (
        f"{score_block}"
        f"Ticker        : {ticker}\n"
        f"Current price : ${insight.price:.2f}\n"
        f"Average cost  : ${insight.avg_cost:.2f}\n"
        f"Unrealized P&L: {gain_pct:+.1f}%\n"
        f"Daily change  : {insight.change_pct:+.1f}%\n"
        f"Volatility    : {insight.volatility:.0%}  (annualised)\n"
        f"Trailing P/E  : {pe_trailing_str}\n"
        f"Forward P/E   : {pe_forward_str}\n"
        f"52w high/low  : ${insight.week_52_high:.2f} / ${insight.week_52_low:.2f}\n"
        f"{weight_line}"
        f"{portfolio_block}"
        + (
            f"\nRecent news:\n"
            + "\n".join(
                f"  - [{n.sentiment.upper()}] {n.headline}" for n in news
            )
            + "\n\n"
            if news
            else "\n"
        )
        + "What is your recommendation? Respond with JSON only."
    )


@lru_cache(maxsize=1)
def _get_decision_llm():
    """
    Lazy singleton for the decision LLM.
    Override the provider with the PORTFOLIO_LLM_PROVIDER env var.
    Supported values: ollama (default), openai, google.

    StreamingStdOutCallbackHandler is intentionally excluded: with parallel
    execution (ThreadPoolExecutor) multiple threads fire token callbacks
    simultaneously, interleaving tokens on stdout and producing garbled output.
    All response content is logged via telemetry after each call completes.
    """
    model = os.getenv("PORTFOLIO_LLM_PROVIDER", "ollama")
    return get_llm(model_name=model)


def _parse_llm_response(content: str, gain_pct: float) -> Dict[str, Any]:
    """Extract and validate the JSON block from the LLM response."""
    text = content.strip()
    # Strip markdown code fences if the model wraps the output
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    parsed = json.loads(text)
    action = parsed["action"].upper()
    if action not in ("EXIT", "HOLD", "DOUBLE_DOWN", "REDUCE"):
        raise ValueError(f"Unexpected action value: {action!r}")
    allocation_change = str(parsed.get("allocation_change", "0%")).strip()
    return {
        "action": action,
        "confidence": parsed["confidence"].lower(),
        "reason": parsed["reason"],
        "gain_pct": round(gain_pct, 2),
        "allocation_change": allocation_change,
    }


def _apply_action_floor(result: Dict[str, Any], quant_score: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce minimum score thresholds for aggressive actions.

    DOUBLE_DOWN requires score >= 2.  A score of +1 is a weak buy — the LLM
    sometimes promotes it to DOUBLE_DOWN anyway.  Downgrade to HOLD.
    """
    score = quant_score.get("score")
    if score is None:
        return result
    if result["action"] == "DOUBLE_DOWN" and score < 2:
        result["action"] = "HOLD"
        result["allocation_change"] = "0%"
        result["reason"] = (
            f"[Downgraded from DOUBLE_DOWN: score={score} < 2 threshold] "
            + result["reason"]
        )
    return result


# ---------------------------------------------------------------------------
# Score → allocation change lookup tables (deterministic, no LLM involvement)
# ---------------------------------------------------------------------------
_REDUCE_BASE: Dict[int, int] = {
    -1: -15,
    -2: -30,
}  # scores ≤ -3 default to -45

_DD_BASE: Dict[int, int] = {
    2: 10,
}  # scores ≥ 3 default to +20

# Extra trim applied when position weight exceeds these thresholds
_WEIGHT_PENALTY: list = [
    (30.0, -15),  # weight > 30% → extra -15%
    (20.0, -10),  # weight > 20% → extra -10%
    (10.0, -5),   # weight > 10% → extra -5%
]


def _apply_allocation_change(
    result: Dict[str, Any],
    quant_score: Dict[str, Any],
    weight_pct: float,
) -> Dict[str, Any]:
    """
    Compute a fully deterministic allocation_change from the quant score and
    current portfolio weight. The LLM's suggested value is discarded entirely.

    Rules:
      EXIT        → "-100%" always
      HOLD        → "0%" always
      REDUCE      → base from score, plus concentration penalty by weight:
                      score -1 → -15%  |  score -2 → -30%  |  score ≤-3 → -45%
                      weight >30% → -15 extra  |  >20% → -10 extra  |  >10% → -5 extra
                      capped at -70% total
      DOUBLE_DOWN → base from score:
                      score 2 → +10%  |  score ≥3 → +20%
    """
    action = result["action"]

    if action == "EXIT":
        result["allocation_change"] = "-100%"
        return result
    if action == "HOLD":
        result["allocation_change"] = "0%"
        return result

    score = quant_score.get("score", 0)

    if action == "REDUCE":
        base = _REDUCE_BASE.get(score, -45)  # score ≤ -3 → -45
        penalty = 0
        for threshold, extra in _WEIGHT_PENALTY:
            if weight_pct > threshold:
                penalty = extra
                break
        numeric = max(-70, base + penalty)  # cap at -70%

    elif action == "DOUBLE_DOWN":
        numeric = _DD_BASE.get(score, 20)  # score ≥ 3 → +20

    else:
        numeric = 0

    sign = "+" if numeric > 0 else ""
    result["allocation_change"] = f"{sign}{numeric:.0f}%"
    return result


def _apply_score_confidence(result: Dict[str, Any], quant_score: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override LLM-reported confidence with a deterministic rule based on the
    absolute value of the quantitative score.

    LLMs tend to report "high" confidence far too often regardless of signal
    strength.  Anchoring to the score magnitude produces more calibrated output:

      abs(score) == 0       → moderate  (neutral signal, LLM is speculating)
      abs(score) == 1       → moderate  (weak signal — one factor tipped the scale)
      abs(score) >= 2       → high      (multiple signals align)

    The LLM confidence is only preserved when no quant score is available.
    """
    score = quant_score.get("score")
    if score is None:
        return result
    abs_score = abs(score)
    if abs_score >= 2:
        result["confidence"] = "high"
    else:
        result["confidence"] = "moderate"
    return result



class DecisionAgent(BaseAgent):
    """
    Generates a trade recommendation for each portfolio position.

    Uses the framework LLM provider (src/llm/providers.py) selected via the
    PORTFOLIO_LLM_PROVIDER environment variable (default: ollama).
    Raises on LLM failure or unparseable output — no silent fallback.
    """

    def run(self, state: PortfolioState) -> PortfolioState:
        decisions: dict = {}
        telemetry = get_telemetry_logger()
        is_retry = state.critic_retry_count > 0
        stock_allocation: Dict[str, float] = state.risk_metrics.stock_allocation

        # Parse investment horizon once — used for scoring and prompt tuning.
        horizon_str   = state.user_profile.investment_horizon
        horizon_years = _parse_horizon_years(horizon_str)

        # Build the portfolio-level context once — shared across all per-ticker decisions.
        portfolio_context: Dict[str, Any] = {
            "high_concentration":  state.risk_metrics.high_concentration,
            "top_stock":           state.risk_metrics.top_stock,
            "top_stock_weight":    stock_allocation.get(state.risk_metrics.top_stock, 0.0),
            "concentration_risk":  state.risk_metrics.concentration_risk,
            "sector_exposure":     state.sector_allocation,
            "total_positions":     len(state.portfolio),
            "investment_horizon":  horizon_str,
        }

        # ── build per-ticker args list ────────────────────────────────────
        # On retry: carry forward previously approved decisions unchanged.
        # Only re-run tickers the critic explicitly flagged — preserves good
        # decisions and avoids re-introducing noise into already-approved ones.
        carried: Dict[str, StockDecision] = {}
        if is_retry:
            for ticker, entry in state.critic_feedback.per_ticker.items():
                if entry.status == "ok" and ticker in state.decisions:
                    carried[ticker] = state.decisions[ticker]
            flagged = [t for t, e in state.critic_feedback.per_ticker.items() if e.status == "flagged"]
            logger.info(
                "[DecisionAgent] Retry #%d — re-running %d flagged ticker(s): %s | "
                "carrying forward %d approved: %s",
                state.critic_retry_count,
                len(flagged), flagged,
                len(carried), list(carried),
            )

        ticker_args: List[Tuple] = []
        for ticker, insight in state.stock_insights.items():
            if ticker in carried:
                continue  # skip — critic already approved this ticker's decision
            news        = state.news.get(ticker, [])
            gain_pct    = ((insight.price - insight.avg_cost) / insight.avg_cost) * 100
            news_score  = compute_news_score(news)
            quant_score = score_stock(insight, gain_pct, horizon_years=horizon_years, news_score=news_score)
            allocation_pct = stock_allocation.get(ticker)
            entry = state.critic_feedback.per_ticker.get(ticker) if is_retry else None
            critic_issues: Optional[List[str]] = (
                entry.issues or None if entry else None
            )
            ticker_args.append((
                ticker, insight, news, gain_pct, allocation_pct,
                critic_issues or [], portfolio_context, quant_score,
                horizon_years,
            ))

        # ── parallel LLM calls ────────────────────────────────────────────
        # Seed decisions with the carried-forward approved ones.
        decisions: dict = dict(carried)
        quant_scores: dict = {t[0]: t[7] for t in ticker_args}  # ticker → quant_score

        def _call(args: Tuple) -> Tuple[str, StockDecision]:
            tkr, ins, nws, gpct, alloc, issues, pctx, qscore, hyrs = args
            return tkr, self._decide(tkr, ins, nws, gpct, alloc, issues, pctx, qscore, horizon_years=hyrs)

        # Cap concurrency to the provider's declared limit.
        # OllamaProvider.max_concurrency == 1 → sequential calls, preventing 429s.
        # Cloud providers default to 10, preserving full parallelism.
        _provider_name = os.getenv("PORTFOLIO_LLM_PROVIDER", "ollama")
        _max_workers = min(len(ticker_args), get_provider(_provider_name).max_concurrency) or 1

        with ThreadPoolExecutor(max_workers=_max_workers) as pool:
            futures = {pool.submit(_call, args): args[0] for args in ticker_args}
            for future in as_completed(futures):
                ticker, decision = future.result()  # propagates exceptions
                decisions[ticker] = decision

        # ── telemetry + logging (sequential, after all futures done) ──────
        for ticker, decision in decisions.items():
            quant_score    = quant_scores[ticker]
            allocation_pct = stock_allocation.get(ticker)
            alloc_note     = f" weight={allocation_pct:.1f}%" if allocation_pct is not None else ""
            telemetry.log_event(
                "decision_made",
                {
                    "ticker":      ticker,
                    "action":      decision.action,
                    "confidence":  decision.confidence,
                    "gain_pct":    decision.gain_pct,
                    "news_count":  len(state.news.get(ticker, [])),
                    "retry":       is_retry,
                    "quant_score": quant_score["score"],
                    "quant_tier":  quant_score["tier"],
                },
            )
            logger.info(
                "[DecisionAgent] %s: %-13s [%s]  gain: %+.1f%%  score: %+d (%s)%s",
                ticker,
                decision.action,
                decision.confidence.upper(),
                decision.gain_pct,
                quant_score["score"],
                quant_score["tier"],
                alloc_note,
            )

        state.decisions = decisions

        portfolio_action = compute_portfolio_action(
            sector_allocation=state.sector_allocation,
            decisions=decisions,
            risk_metrics=state.risk_metrics,
            portfolio=state.portfolio,
        )
        state.portfolio_action = portfolio_action
        if portfolio_action.rebalance:
            logger.info(
                "[DecisionAgent] Portfolio action: REBALANCE | %s %.1f%% → %s | Priority exits: %s",
                portfolio_action.reduce_sector,
                portfolio_action.current_exposure,
                portfolio_action.target_exposure,
                portfolio_action.priority_exits or "none",
            )
        return state

    def _decide(
        self,
        ticker: str,
        insight: StockInsight,
        news: List[NewsArticle],
        gain_pct: float,
        stock_allocation_pct: Optional[float] = None,
        critic_issues: Optional[List[str]] = None,
        portfolio_context: Optional[Dict[str, Any]] = None,
        quant_score: Optional[Dict[str, Any]] = None,
        horizon_years: float = 1.0,
    ) -> StockDecision:
        telemetry = get_telemetry_logger()
        human_msg = _build_human_message(
            ticker, insight, news, gain_pct, stock_allocation_pct, portfolio_context, quant_score
        )
        # On a critic retry the system prompt leads with the rejection reasons
        # so the model sees them as the highest-priority instruction.
        system_prompt = _build_system_prompt(critic_issues or None, horizon_years=horizon_years)
        llm = _get_decision_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_msg),
        ]

        last_error: Exception = RuntimeError("No attempts made")
        for attempt in range(2):
            # Retry loop for transient Ollama 429s (shouldn't happen with
            # max_concurrency=1 but guards against external concurrent callers).
            for _retry in range(3):
                try:
                    response = llm.invoke(messages)
                    break
                except Exception as exc:
                    if "429" in str(exc) and _retry < 2:
                        _wait = 2.0 * (2 ** _retry)
                        logger.warning(
                            "[DecisionAgent] %s: Ollama 429 — retry %d/2 in %.0fs",
                            ticker, _retry + 1, _wait,
                        )
                        time.sleep(_wait)
                    else:
                        raise
            telemetry.log_llm_interaction(
                prompt=f"[{ticker}] {human_msg}",
                response=response.content,
            )
            try:
                result = _parse_llm_response(response.content, gain_pct)
                result = _apply_action_floor(result, quant_score or {})
                result = _apply_allocation_change(result, quant_score or {}, stock_allocation_pct or 0.0)
                result = _apply_score_confidence(result, quant_score or {})
                valid, val_err = validate_decision(result)
                if not valid:
                    raise ValueError(val_err)
                return StockDecision.model_validate(result)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = exc
                if attempt == 0:
                    # Append the bad response and a correction request so the
                    # model can see what it produced and why it was rejected.
                    messages.append(AIMessage(content=response.content))
                    messages.append(
                        HumanMessage(
                            content=(
                                f"Your previous response was invalid: {exc}\n"
                                "Return ONLY a valid JSON object matching the required schema."
                            )
                        )
                    )

        raise ValueError(
            f"[DecisionAgent] {ticker}: LLM produced invalid output after 2 attempts: {last_error}"
        )
