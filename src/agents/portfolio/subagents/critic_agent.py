import json
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.portfolio.state import PortfolioState
from src.llm import get_llm
from src.observability import get_telemetry_logger

logger = logging.getLogger(__name__)


# Flag if more than this fraction of positions are EXIT/REDUCE recommendations
_MAX_EXIT_RATE = 0.5

# Confidence levels ranked lowest → highest
_CONFIDENCE_RANK: Dict[str, int] = {"low": 0, "moderate": 1, "high": 2}

_LLM_CRITIC_SYSTEM = """You are a senior portfolio risk manager reviewing decisions made by another agent.

IMPORTANT: Structural checks (confidence level, loss thresholds, exit rates) are
already validated. Do NOT repeat them. Only flag qualitative issues.

Check:
1. Overreaction to news — is a REDUCE/EXIT driven purely by short-term headlines
   with no fundamental backing?
2. Quant score alignment — does the action contradict the quantitative score
   direction without a stated reason?
3. Risk/reward coherence — does the overall action mix suit a {risk_level}
   investor with a {horizon} horizon?
4. Internal inconsistency — conflicting decisions on correlated tickers with
   no explanation.

If everything looks reasonable, approve it. Do not invent problems.

Respond ONLY with a JSON object. No markdown, no explanation outside the JSON.

{{
  "approved": true | false,
  "issues": [
    {{"ticker": "AAPL", "issue": "..."}}
  ],
  "summary": "One sentence verdict."
}}

Return approved=true and an empty issues list when no problems are found."""


def _run_llm_critique(
    decisions: Dict[str, Any],
    user_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Call the LLM critic and return parsed JSON. Falls back to approval on any error.

    Provider and model are read from PORTFOLIO_CRITIC_LLM_PROVIDER /
    PORTFOLIO_CRITIC_LLM_MODEL env vars, falling back to the decision-agent
    values when not set. Configure both to a different model than the decision
    agent to avoid self-review bias.
    """
    import os
    critic_provider = os.getenv(
        "PORTFOLIO_CRITIC_LLM_PROVIDER",
        os.getenv("PORTFOLIO_LLM_PROVIDER", "ollama"),
    )
    critic_model = os.getenv("PORTFOLIO_CRITIC_LLM_MODEL") or None

    risk_level = user_profile.get("risk_level", "moderate")
    horizon = user_profile.get("investment_horizon", "unknown")

    system_prompt = _LLM_CRITIC_SYSTEM.format(risk_level=risk_level, horizon=horizon)

    lines = []
    for ticker, d in decisions.items():
        lines.append(
            f"{ticker}: action={d.get('action')} confidence={d.get('confidence')} "
            f"gain_pct={d.get('gain_pct', 0):.1f}% "
            f"allocation_change={d.get('allocation_change', 'n/a')} "
            f'reason="{d.get("reason", "")}"'
        )
    human_msg = "Portfolio decisions to review:\n\n" + "\n".join(lines)

    try:
        llm = get_llm(model_name=critic_provider, model=critic_model)
        logger.debug(
            "[CriticAgent] LLM call: provider=%s model=%s",
            critic_provider,
            critic_model or "(env default)",
        )
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_msg)])
        raw = response.content if hasattr(response, "content") else str(response)

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        return json.loads(raw)
    except Exception as exc:
        logger.warning("[CriticAgent] LLM critique failed — skipping LLM pass: %s", exc)
        return {"approved": True, "issues": [], "summary": "LLM critique unavailable."}


class CriticAgent:
    """
    Two-stage validation of DecisionAgent output.

    Stage 1 — Hardcoded rules (fast, deterministic, always runs):
      - Rejects any decision with "low" confidence
      - Flags high exit/reduce rates across the whole portfolio
      - Flags DOUBLE_DOWN on positions already down >20%

    Stage 2 — LLM qualitative critique (only runs when stage 1 passes):
      A single portfolio-level LLM call that checks overreaction to news,
      quant score alignment, risk/reward coherence, and internal inconsistency
      across correlated tickers.

    Sets critic_feedback["approved"] = False when a re-run is required.
    Sets critic_feedback["feedback"] to a human-readable summary of issues
    which DecisionAgent appends to its prompt on the next retry.
    """

    def run(self, state: PortfolioState) -> PortfolioState:
        decisions = state.decisions
        feedback: Dict[str, Any] = {
            "approved": True,
            "warnings": [],
            "per_ticker": {},
            "feedback": "",
        }

        # ------------------------------------------------------------------
        # Stage 1: hardcoded deterministic rules
        # ------------------------------------------------------------------
        exit_count = sum(1 for d in decisions.values() if d["action"] in ("EXIT", "REDUCE"))
        exit_rate = exit_count / len(decisions) if decisions else 0

        if exit_rate > _MAX_EXIT_RATE:
            feedback["warnings"].append(
                f"High exit/reduce rate ({exit_rate:.0%}) across portfolio. "
                f"Verify this aligns with your long-term strategy before acting."
            )

        for ticker, decision in decisions.items():
            issues = []
            conf = _CONFIDENCE_RANK.get(decision.get("confidence", "moderate"), 1)

            if conf == 0:  # low confidence
                issues.append("Low confidence — gather more data before acting.")
                feedback["approved"] = False

            if decision["action"] == "DOUBLE_DOWN" and decision.get("gain_pct", 0) < -20:
                issues.append("Doubling down on a >20% loss is high risk. Verify thesis first.")
                feedback["warnings"].append(
                    f"{ticker}: Risky double-down on large loss flagged by critic."
                )

            feedback["per_ticker"][ticker] = {
                "status": "flagged" if issues else "ok",
                "issues": issues,
            }

        # ------------------------------------------------------------------
        # Stage 2: LLM qualitative critique (only when stage 1 passes)
        # ------------------------------------------------------------------
        if feedback["approved"]:
            llm_result = _run_llm_critique(decisions, state.user_profile)
            llm_approved = llm_result.get("approved", True)
            llm_issues: list = llm_result.get("issues", [])
            llm_summary: str = llm_result.get("summary", "")

            logger.info("[CriticAgent] LLM verdict: approved=%s | %s", llm_approved, llm_summary)

            if not llm_approved:
                feedback["approved"] = False
                for item in llm_issues:
                    ticker = item.get("ticker", "PORTFOLIO")
                    issue_text = item.get("issue", "")
                    if not issue_text:
                        continue
                    entry = feedback["per_ticker"].setdefault(
                        ticker, {"status": "ok", "issues": []}
                    )
                    entry["issues"].append(f"[LLM critic] {issue_text}")
                    entry["status"] = "flagged"

            if llm_summary:
                feedback["warnings"].append(f"LLM critic: {llm_summary}")

        # ------------------------------------------------------------------
        # Build consolidated feedback string for DecisionAgent retry prompt
        # ------------------------------------------------------------------
        if not feedback["approved"]:
            issue_lines = [
                f"{ticker}: {issue}"
                for ticker, data in feedback["per_ticker"].items()
                for issue in data["issues"]
            ]
            feedback["feedback"] = "; ".join(issue_lines)

        flagged = [t for t, v in feedback["per_ticker"].items() if v["status"] == "flagged"]
        get_telemetry_logger().log_event(
            "critic_review",
            {
                "approved": feedback["approved"],
                "warning_count": len(feedback["warnings"]),
                "flagged_tickers": flagged,
                "warnings": feedback["warnings"],
            },
        )

        if not feedback["approved"]:
            logger.info(
                "[CriticAgent] REJECTED — retry will be triggered | Flagged: %s | Feedback: %s",
                flagged,
                feedback["feedback"],
            )
        else:
            logger.info(
                "[CriticAgent] APPROVED | Warnings: %d | Flagged tickers: %d",
                len(feedback["warnings"]),
                len(flagged),
            )

        state.critic_feedback = feedback
        return state
