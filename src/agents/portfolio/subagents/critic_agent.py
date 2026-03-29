import logging
from typing import Any, Dict

from src.agents.portfolio.state import PortfolioState
from src.observability import get_telemetry_logger

logger = logging.getLogger(__name__)


# Flag if more than this fraction of positions are EXIT recommendations
_MAX_EXIT_RATE = 0.5

# Confidence levels ranked lowest → highest
_CONFIDENCE_RANK: Dict[str, int] = {"low": 0, "moderate": 1, "high": 2}


class CriticAgent:
    """
    Validates DecisionAgent output for quality and portfolio-level consistency.

    Checks performed:
    - Rejects any decision with "low" confidence
    - Flags high exit rates across the whole portfolio
    - Flags double-down decisions on positions already down >20%

    Sets critic_feedback["approved"] = False when re-run is required.
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

        # Portfolio-level check: too many exits/reduces at once
        exit_count = sum(1 for d in decisions.values() if d["action"] in ("EXIT", "REDUCE"))
        exit_rate = exit_count / len(decisions) if decisions else 0

        if exit_rate > _MAX_EXIT_RATE:
            feedback["warnings"].append(
                f"High exit/reduce rate ({exit_rate:.0%}) across portfolio. "
                f"Verify this aligns with your long-term strategy before acting."
            )

        # Per-ticker checks
        for ticker, decision in decisions.items():
            issues = []
            conf = _CONFIDENCE_RANK.get(decision.get("confidence", "moderate"), 1)

            if conf == 0:  # low confidence
                issues.append("Low confidence — gather more data before acting.")
                feedback["approved"] = False

            if decision["action"] == "DOUBLE_DOWN" and decision.get("gain_pct", 0) < -20:
                issues.append(
                    "Doubling down on a >20% loss is high risk. Verify thesis first."
                )
                feedback["warnings"].append(
                    f"{ticker}: Risky double-down on large loss flagged by critic."
                )

            feedback["per_ticker"][ticker] = {
                "status": "flagged" if issues else "ok",
                "issues": issues,
            }

        # Build a single feedback string for DecisionAgent's retry prompt
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
