from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException

from src.agents.supervisor.workflow import run_full
from src.integrations.company_resolver import resolve_company_names
from src.integrations.whatsapp import extract_tickers_from_text


def _unique(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _build_supervisor_query(message: str, resolved_tickers: list[str]) -> str:
    if not resolved_tickers:
        return message

    ticker_context = ", ".join(resolved_tickers)
    return (
        f"{message}\n\n"
        f"Resolved ticker context: the likely ticker symbols relevant to the user's request are {ticker_context}. "
        "If you use the opportunity worker, pass these symbols in its tickers argument when appropriate."
    )


def _parse_opportunity_worker_result(raw_result: str | None) -> list[dict[str, Any]]:
    if not raw_result:
        return []

    try:
        payload = json.loads(raw_result)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def run_supervisor_query(message: str, model: str | None = None) -> dict[str, Any]:
    user_message = (message or "").strip()
    if not user_message:
        raise HTTPException(status_code=422, detail="Provide a non-empty message.")

    explicit_tickers = extract_tickers_from_text(user_message)
    company_tickers = resolve_company_names(user_message)
    resolved_tickers = _unique(explicit_tickers + company_tickers)
    supervisor_query = _build_supervisor_query(user_message, resolved_tickers)

    try:
        result = run_full(query=supervisor_query, model=model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Supervisor query failed: {exc}") from exc

    opportunities = _parse_opportunity_worker_result(result.worker_results.get("scan_opportunities"))
    return {
        "message": user_message,
        "resolved_tickers": resolved_tickers,
        "reply_text": result.answer,
        "opportunities": opportunities,
        "opportunity_count": len(opportunities),
        "worker_results": result.worker_results,
    }
