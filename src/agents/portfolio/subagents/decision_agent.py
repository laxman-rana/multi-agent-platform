import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional


from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.portfolio.state import PortfolioState
from src.agents.portfolio.tools.validation import validate_decision
from src.observability import get_telemetry_logger
from src.llm import get_llm


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
  "action": "EXIT" | "HOLD" | "DOUBLE_DOWN",
  "confidence": "high" | "moderate" | "low",
  "reason": "one concise sentence"
}

--------------------------------------------------
DECISION GUIDELINES
--------------------------------------------------

EXIT:
- Significant downside risk
- High volatility with negative signals
- Weak fundamentals or negative sentiment

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

"""


def _build_human_message(
    ticker: str,
    insight: Dict[str, Any],
    news: List[Dict[str, str]],
    gain_pct: float,
    stock_allocation_pct: Optional[float] = None,
    critic_issues: Optional[List[str]] = None,
) -> str:
    news_lines = "\n".join(
        f"  - [{n.get('sentiment', 'neutral').upper()}] {n['headline']}" for n in news
    ) or "  (no recent news)"

    pe_trailing = insight.get('pe_ratio')
    pe_forward  = insight.get('forward_pe')
    pe_trailing_str = f"{pe_trailing:.1f}" if pe_trailing is not None else "N/A"
    pe_forward_str  = f"{pe_forward:.1f}"  if pe_forward  is not None else "N/A"

    if stock_allocation_pct is not None:
        conc_flag = " \u26a0 HIGH CONCENTRATION" if stock_allocation_pct > 40 else ""
        weight_line = f"Portfolio weight: {stock_allocation_pct:.1f}%{conc_flag}\n"
    else:
        weight_line = ""

    critic_block = ""
    if critic_issues:
        formatted = "\n".join(f"  - {issue}" for issue in critic_issues)
        critic_block = (
            f"\n--------------------------------------------------\n"
            f"PREVIOUS ATTEMPT REJECTED BY CRITIC:\n"
            f"{formatted}\n"
            f"Address ALL of the above issues in your new response.\n"
            f"--------------------------------------------------\n"
        )

    return (
        f"Ticker        : {ticker}\n"
        f"Current price : ${insight['price']:.2f}\n"
        f"Average cost  : ${insight['avg_cost']:.2f}\n"
        f"Unrealized P&L: {gain_pct:+.1f}%\n"
        f"Daily change  : {insight['change_pct']:+.1f}%\n"
        f"Volatility    : {insight['volatility']:.0%}  (annualised)\n"
        f"Trailing P/E  : {pe_trailing_str}\n"
        f"Forward P/E   : {pe_forward_str}\n"
        f"52w high/low  : ${insight.get('52w_high', 0):.2f} / ${insight.get('52w_low', 0):.2f}\n"
        f"{weight_line}"
        f"\nRecent news:\n{news_lines}\n"
        f"{critic_block}\n"
        "What is your recommendation? Respond with JSON only."
    )


@lru_cache(maxsize=1)
def _get_decision_llm():
    """
    Lazy singleton for the decision LLM.
    Override the provider with the PORTFOLIO_LLM_PROVIDER env var.
    Supported values: ollama (default), openai, google.
    """
    model = os.getenv("PORTFOLIO_LLM_PROVIDER", "ollama")
    return get_llm(model_name=model, callbacks=[StreamingStdOutCallbackHandler()])


def _parse_llm_response(content: str, gain_pct: float) -> Dict[str, Any]:
    """Extract and validate the JSON block from the LLM response."""
    text = content.strip()
    # Strip markdown code fences if the model wraps the output
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    parsed = json.loads(text)
    action = parsed["action"].upper()
    if action not in ("EXIT", "HOLD", "DOUBLE_DOWN"):
        raise ValueError(f"Unexpected action value: {action!r}")
    return {
        "action": action,
        "confidence": parsed["confidence"].lower(),
        "reason": parsed["reason"],
        "gain_pct": round(gain_pct, 2),
    }



class DecisionAgent:
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
        stock_allocation: Dict[str, float] = state.risk_metrics.get("stock_allocation", {})

        if is_retry:
            print(
                f"  [DecisionAgent] Retry #{state.critic_retry_count} after critic rejection"
            )

        for ticker, insight in state.stock_insights.items():
            news = state.news.get(ticker, [])
            gain_pct = ((insight["price"] - insight["avg_cost"]) / insight["avg_cost"]) * 100

            allocation_pct = stock_allocation.get(ticker)
            per_ticker_feedback = state.critic_feedback.get("per_ticker", {})
            critic_issues: Optional[List[str]] = (
                per_ticker_feedback.get(ticker, {}).get("issues") or None
                if is_retry
                else None
            )

            decision = self._decide(
                ticker, insight, news, gain_pct, allocation_pct, critic_issues
            )
            decisions[ticker] = decision
            telemetry.log_event(
                "decision_made",
                {
                    "ticker": ticker,
                    "action": decision["action"],
                    "confidence": decision["confidence"],
                    "gain_pct": decision["gain_pct"],
                    "news_count": len(news),
                    "retry": is_retry,
                },
            )
            alloc_note = f" weight={allocation_pct:.1f}%" if allocation_pct is not None else ""
            print(
                f"  [DecisionAgent] {ticker}: {decision['action']:<13} "
                f"[{decision['confidence'].upper()}]  "
                f"gain: {decision['gain_pct']:+.1f}%{alloc_note}"
            )

        state.decisions = decisions
        return state

    def _decide(
        self,
        ticker: str,
        insight: Dict[str, Any],
        news: List[Dict[str, str]],
        gain_pct: float,
        stock_allocation_pct: Optional[float] = None,
        critic_issues: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        telemetry = get_telemetry_logger()
        human_msg = _build_human_message(
            ticker, insight, news, gain_pct, stock_allocation_pct, critic_issues
        )
        llm = _get_decision_llm()
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_msg),
        ]

        last_error: Exception = RuntimeError("No attempts made")
        for attempt in range(2):
            response = llm.invoke(messages)
            telemetry.log_llm_interaction(
                prompt=f"[{ticker}] {human_msg}",
                response=response.content,
            )
            try:
                result = _parse_llm_response(response.content, gain_pct)
                valid, val_err = validate_decision(result)
                if not valid:
                    raise ValueError(val_err)
                return result
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
