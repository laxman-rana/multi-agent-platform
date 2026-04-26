from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from src.agents.opportunity.workflow import trigger_scan


def normalize_tickers(tickers: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    for ticker in tickers:
        value = ticker.strip().upper()
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            normalized.append(value)

    if not normalized:
        raise HTTPException(status_code=422, detail="Provide at least one non-empty ticker symbol.")

    return normalized


def normalize_market(market: str | None) -> str:
    value = (market or "US").strip().upper()
    allowed = {"US", "IN", "IN_MID", "IN_SMALL"}
    if value not in allowed:
        raise HTTPException(
            status_code=422,
            detail="Market must be one of: US, IN, IN_MID, IN_SMALL.",
        )
    return value


def run_opportunity_scan(
    tickers: list[str],
    market: str = "US",
) -> tuple[list[str], str, list[dict[str, Any]]]:
    normalized = normalize_tickers(tickers)
    normalized_market = normalize_market(market)

    try:
        opportunities = trigger_scan(tickers=normalized, market=normalized_market)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Opportunity scan failed: {exc}") from exc

    return normalized, normalized_market, opportunities
