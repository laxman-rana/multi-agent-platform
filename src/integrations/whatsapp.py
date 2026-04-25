from __future__ import annotations

import re
from typing import Any


_IGNORED_TOKENS = {
    "A",
    "AN",
    "AND",
    "ARE",
    "AT",
    "BUY",
    "CAN",
    "FOR",
    "GIVE",
    "HELLO",
    "HI",
    "HOW",
    "I",
    "IS",
    "ME",
    "OF",
    "ON",
    "OR",
    "PLEASE",
    "RECOMMENDATION",
    "RECOMMENDATIONS",
    "SCAN",
    "SEND",
    "SHOULD",
    "STOCK",
    "STOCKS",
    "THE",
    "THESE",
    "THIS",
    "TICKER",
    "TICKERS",
    "TO",
    "TODAY",
    "WHAT",
}

_TICKER_PATTERN = re.compile(r"\$?[A-Za-z][A-Za-z.\-]{0,9}")


def extract_tickers_from_text(message: str) -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()

    for match in _TICKER_PATTERN.findall(message or ""):
        token = match.lstrip("$").strip().upper()
        if not token:
            continue
        if token in _IGNORED_TOKENS:
            continue
        if len(token) == 1 and token not in {"F", "T", "X"}:
            continue
        if token not in seen:
            seen.add(token)
            tickers.append(token)

    return tickers


def format_opportunity_reply(tickers: list[str], opportunities: list[dict[str, Any]]) -> str:
    if not opportunities:
        return (
            f"I scanned {', '.join(tickers)} and did not find any BUY signals right now. "
            "Try again later or send a different set of tickers."
        )

    lines = [f"I scanned {', '.join(tickers)} and found {len(opportunities)} opportunity signal(s):"]

    for opportunity in opportunities[:3]:
        ticker = opportunity.get("ticker", "?")
        confidence = str(opportunity.get("confidence", "unknown")).upper()
        reason = str(opportunity.get("reason", "No reason provided.")).strip()
        lines.append(f"- {ticker}: {confidence} confidence. {reason}")

    if len(opportunities) > 3:
        lines.append(f"Plus {len(opportunities) - 3} more result(s).")

    return "\n".join(lines)
