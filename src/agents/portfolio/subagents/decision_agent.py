import json
import os
from functools import lru_cache
from typing import Any, Dict, List


from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.portfolio.state import PortfolioState
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
) -> str:
    news_lines = "\n".join(
        f"  - [{n.get('sentiment', 'neutral').upper()}] {n['headline']}" for n in news
    ) or "  (no recent news)"

    pe_trailing = insight.get('pe_ratio', 0)
    pe_forward  = insight.get('forward_pe', 0)
    pe_trailing_str = f"{pe_trailing:.1f}" if pe_trailing else "N/A"
    pe_forward_str  = f"{pe_forward:.1f}"  if pe_forward  else "N/A"

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
        f"\nRecent news:\n{news_lines}\n\n"
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

        for ticker, insight in state.stock_insights.items():
            news = state.news.get(ticker, [])
            gain_pct = ((insight["price"] - insight["avg_cost"]) / insight["avg_cost"]) * 100
            decision = self._decide(ticker, insight, news, gain_pct)
            decisions[ticker] = decision
            telemetry.log_event(
                "decision_made",
                {
                    "ticker": ticker,
                    "action": decision["action"],
                    "confidence": decision["confidence"],
                    "gain_pct": decision["gain_pct"],
                    "news_count": len(news),
                },
            )
            print(
                f"  {ticker}: {decision['action']:<13} "
                f"[{decision['confidence'].upper()}]  "
                f"gain: {decision['gain_pct']:+.1f}%"
            )

        state.decisions = decisions
        return state

    def _decide(
        self,
        ticker: str,
        insight: Dict[str, Any],
        news: List[Dict[str, str]],
        gain_pct: float,
    ) -> Dict[str, Any]:
        telemetry = get_telemetry_logger()
        human_msg = _build_human_message(ticker, insight, news, gain_pct)
        llm = _get_decision_llm()
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_msg),
        ]
        response = llm.invoke(messages)
        result = _parse_llm_response(response.content, gain_pct)
        telemetry.log_llm_interaction(
            prompt=f"[{ticker}] {human_msg}",
            response=response.content,
        )
        return result
