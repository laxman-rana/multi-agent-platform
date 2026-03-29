"""
validation.py
-------------
Reusable structural validation for DecisionAgent output.
Called after every LLM response to catch malformed or out-of-spec decisions
before they propagate downstream.
"""

import re
from typing import Any, Dict, Tuple

_VALID_ACTIONS: frozenset = frozenset({"EXIT", "HOLD", "DOUBLE_DOWN", "REDUCE"})
_VALID_CONFIDENCE: frozenset = frozenset({"high", "moderate", "low"})
_ALLOCATION_CHANGE_RE = re.compile(r"^[+-]?\d+(\.\d+)?%$")


def validate_decision(decision: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate a single-ticker decision dict produced by DecisionAgent.

    Checks:
    - action    : must be EXIT | HOLD | DOUBLE_DOWN  (case-insensitive)
    - confidence: must be high | moderate | low      (case-insensitive)
    - reason    : must be a non-empty string

    Returns:
        (True,  "")         when the decision passes all checks.
        (False, error_msg)  when any check fails.
    """
    action = str(decision.get("action", "")).upper()
    if action not in _VALID_ACTIONS:
        return False, (
            f"Invalid action {action!r}. Expected one of {sorted(_VALID_ACTIONS)}."
        )

    confidence = str(decision.get("confidence", "")).lower()
    if confidence not in _VALID_CONFIDENCE:
        return False, (
            f"Invalid confidence {confidence!r}. "
            f"Expected one of {sorted(_VALID_CONFIDENCE)}."
        )

    reason = str(decision.get("reason", "")).strip()
    if not reason:
        return False, "Missing or empty 'reason' field."

    allocation_change = str(decision.get("allocation_change", "")).strip()
    if not allocation_change:
        return False, "Missing or empty 'allocation_change' field."
    if not _ALLOCATION_CHANGE_RE.match(allocation_change):
        return False, (
            f"Invalid allocation_change {allocation_change!r}. "
            "Expected a signed percentage string like '+10%', '-15%', or '0%'."
        )

    return True, ""


def validate_all_decisions(
    decisions: Dict[str, Dict[str, Any]],
) -> Tuple[bool, Dict[str, str]]:
    """
    Validate every ticker decision in the decisions dict.

    Returns:
        (True, {})                when all decisions pass.
        (False, {ticker: error})  mapping each invalid ticker to its error message.
    """
    errors: Dict[str, str] = {}
    for ticker, decision in decisions.items():
        valid, msg = validate_decision(decision)
        if not valid:
            errors[ticker] = msg
    return len(errors) == 0, errors
