from __future__ import annotations

import os
from urllib.parse import parse_qs
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.api.opportunity_service import run_opportunity_scan
from src.api.supervisor_service import run_supervisor_query
from src.integrations.whatsapp import extract_tickers_from_text, format_opportunity_reply


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("API_CORS_ALLOW_ORIGINS", "*").strip()
    if not raw:
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


class OpportunityScanRequest(BaseModel):
    tickers: list[str] = Field(
        ...,
        description="Ticker symbols to scan, for example ['AAPL', 'MSFT', 'NVDA'].",
    )


class OpportunityScanResponse(BaseModel):
    tickers: list[str]
    opportunity_count: int
    opportunities: list[dict[str, Any]]


class AssistantQueryRequest(BaseModel):
    message: str = Field(
        ...,
        description="Natural-language user request, for example 'Should I buy Microsoft today?'",
    )
    model: str | None = Field(
        default=None,
        description="Optional model override for the supervisor layer.",
    )


class AssistantQueryResponse(BaseModel):
    message: str
    resolved_tickers: list[str]
    opportunity_count: int
    opportunities: list[dict[str, Any]]
    reply_text: str
    worker_results: dict[str, str]


class WhatsAppWebhookResponse(BaseModel):
    sender: str | None
    message: str
    resolved_tickers: list[str]
    opportunity_count: int
    opportunities: list[dict[str, Any]]
    reply_text: str


app = FastAPI(
    title="Multi-Agent Platform API",
    version="0.1.0",
    description=(
        "HTTP API for the opportunity scanner agent. "
        "The existing CLI remains unchanged."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "multi-agent-platform",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "opportunity_scan": "/api/v1/opportunity/scan",
        "assistant_query": "/api/v1/assistant/query",
        "whatsapp_webhook": "/webhooks/whatsapp",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/opportunity/scan", response_model=OpportunityScanResponse)
def scan_opportunities(payload: OpportunityScanRequest) -> OpportunityScanResponse:
    tickers, opportunities = run_opportunity_scan(payload.tickers)

    return OpportunityScanResponse(
        tickers=tickers,
        opportunity_count=len(opportunities),
        opportunities=opportunities,
    )


@app.post("/api/v1/assistant/query", response_model=AssistantQueryResponse)
def assistant_query(payload: AssistantQueryRequest) -> AssistantQueryResponse:
    result = run_supervisor_query(message=payload.message, model=payload.model)
    return AssistantQueryResponse(**result)


async def _read_webhook_payload(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "").lower()

    if "application/json" in content_type:
        payload = await request.json()
        if isinstance(payload, dict):
            return payload
        return {}

    raw_body = (await request.body()).decode("utf-8", errors="ignore")
    parsed = parse_qs(raw_body, keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in parsed.items()}


@app.post("/webhooks/whatsapp", response_model=WhatsAppWebhookResponse)
async def whatsapp_webhook(request: Request) -> WhatsAppWebhookResponse:
    payload = await _read_webhook_payload(request)

    message = str(
        payload.get("Body")
        or payload.get("message")
        or payload.get("text")
        or ""
    ).strip()
    sender = payload.get("From") or payload.get("from")

    if not message:
        reply_text = (
            "Send a message like 'Should I buy Microsoft?' or 'scan NVDA MSFT' and I will analyze it."
        )
        return WhatsAppWebhookResponse(
            sender=sender,
            message=message,
            resolved_tickers=[],
            opportunity_count=0,
            opportunities=[],
            reply_text=reply_text,
        )

    result = run_supervisor_query(message=message)
    reply_text = result["reply_text"]
    if not reply_text.strip() and result["resolved_tickers"]:
        reply_text = format_opportunity_reply(result["resolved_tickers"], result["opportunities"])

    return WhatsAppWebhookResponse(
        sender=sender,
        message=message,
        resolved_tickers=result["resolved_tickers"],
        opportunity_count=result["opportunity_count"],
        opportunities=result["opportunities"],
        reply_text=reply_text,
    )
