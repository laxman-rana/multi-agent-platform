from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from urllib.parse import parse_qs
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.api.opportunity_service import run_opportunity_scan
from src.api.supervisor_service import run_supervisor_query
from src.integrations.whatsapp import extract_tickers_from_text, format_opportunity_reply


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("API_CORS_ALLOW_ORIGINS", "*").strip()
    if not raw:
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _client_ip_key(request: Request) -> str:
    """Resolve the best client IP from common reverse-proxy headers."""
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        client_ip = forwarded_for.split(",", 1)[0].strip()
        if client_ip:
            return client_ip

    real_ip = request.headers.get("x-real-ip", "").strip()
    if real_ip:
        return real_ip

    return get_remote_address(request)


def _env_rate_limit(name: str, default: str) -> str:
    raw = os.getenv(name, default).strip()
    return raw or default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return value if value > 0 else default


class TTLCache:
    """Small in-memory TTL cache for single-instance deployments."""

    def __init__(self, max_entries: int = 256) -> None:
        self._max_entries = max_entries
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None

            expires_at, value = entry
            if expires_at <= now:
                self._store.pop(key, None)
                return None

            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        expires_at = time.time() + ttl_seconds
        with self._lock:
            self._store[key] = (expires_at, value)
            self._store.move_to_end(key)
            self._prune_locked(time.time())

    def _prune_locked(self, now: float) -> None:
        expired_keys = [key for key, (expires_at, _) in self._store.items() if expires_at <= now]
        for key in expired_keys:
            self._store.pop(key, None)

        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)


def _hash_cache_key(prefix: str, payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _json_response(payload: BaseModel, cache_status: str | None = None) -> JSONResponse:
    headers = {}
    if cache_status:
        headers["X-Cache"] = cache_status
    return JSONResponse(content=payload.model_dump(), headers=headers)


class OpportunityScanRequest(BaseModel):
    tickers: list[str] = Field(
        ...,
        description="Ticker symbols to scan, for example ['AAPL', 'MSFT', 'NVDA'].",
    )
    market: str = Field(
        default="US",
        description="Market universe. Supported values: US, IN, IN_MID, IN_SMALL.",
    )


class OpportunityScanResponse(BaseModel):
    tickers: list[str]
    market: str
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


_DOCS_ENABLED = _env_flag("API_ENABLE_DOCS", default=True)
_RATE_LIMITS_ENABLED = _env_flag("API_ENABLE_RATE_LIMITS", default=True)
_ASSISTANT_RATE_LIMIT = _env_rate_limit("API_RATE_LIMIT_ASSISTANT_QUERY", "5/minute")
_SCAN_RATE_LIMIT = _env_rate_limit("API_RATE_LIMIT_OPPORTUNITY_SCAN", "10/minute")
_WHATSAPP_RATE_LIMIT = _env_rate_limit("API_RATE_LIMIT_WHATSAPP_WEBHOOK", "10/minute")
_RATE_LIMIT_STRATEGY = os.getenv("API_RATE_LIMIT_STRATEGY", "moving-window").strip() or "moving-window"
_CACHE_ENABLED = _env_flag("API_ENABLE_CACHE", default=True)
_CACHE_MAX_ENTRIES = _env_int("API_CACHE_MAX_ENTRIES", 256)
_CACHE_TTL_ASSISTANT_QUERY = _env_int("API_CACHE_TTL_ASSISTANT_QUERY_SECONDS", 180)
_CACHE_TTL_OPPORTUNITY_SCAN = _env_int("API_CACHE_TTL_OPPORTUNITY_SCAN_SECONDS", 120)

limiter = Limiter(
    key_func=_client_ip_key,
    enabled=_RATE_LIMITS_ENABLED,
    headers_enabled=True,
    strategy=_RATE_LIMIT_STRATEGY,
)
cache = TTLCache(max_entries=_CACHE_MAX_ENTRIES)


app = FastAPI(
    title="Multi-Agent Platform API",
    version="0.1.0",
    description=(
        "HTTP API for the opportunity scanner agent. "
        "The existing CLI remains unchanged."
    ),
    docs_url="/docs" if _DOCS_ENABLED else None,
    redoc_url="/redoc" if _DOCS_ENABLED else None,
    openapi_url="/openapi.json" if _DOCS_ENABLED else None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SlowAPIMiddleware)


@app.get("/")
def root() -> dict[str, str]:
    payload = {
        "service": "multi-agent-platform",
        "status": "ok",
        "health": "/health",
        "opportunity_scan": "/api/v1/opportunity/scan",
        "assistant_query": "/api/v1/assistant/query",
        "whatsapp_webhook": "/webhooks/whatsapp",
    }
    if _DOCS_ENABLED:
        payload["docs"] = "/docs"
    return payload


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/opportunity/scan", response_model=OpportunityScanResponse)
@limiter.limit(_SCAN_RATE_LIMIT)
def scan_opportunities(request: Request, payload: OpportunityScanRequest) -> JSONResponse:
    if _CACHE_ENABLED:
        cache_key = _hash_cache_key(
            "opportunity_scan",
            {"tickers": payload.tickers, "market": payload.market},
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return _json_response(OpportunityScanResponse(**cached), cache_status="HIT")

    tickers, market, opportunities = run_opportunity_scan(payload.tickers, market=payload.market)

    response_payload = OpportunityScanResponse(
        tickers=tickers,
        market=market,
        opportunity_count=len(opportunities),
        opportunities=opportunities,
    )
    if _CACHE_ENABLED:
        cache.set(cache_key, response_payload.model_dump(), _CACHE_TTL_OPPORTUNITY_SCAN)
    return _json_response(response_payload, cache_status="MISS" if _CACHE_ENABLED else None)


@app.post("/api/v1/assistant/query", response_model=AssistantQueryResponse)
@limiter.limit(_ASSISTANT_RATE_LIMIT)
def assistant_query(request: Request, payload: AssistantQueryRequest) -> JSONResponse:
    if _CACHE_ENABLED:
        cache_key = _hash_cache_key(
            "assistant_query",
            {"message": payload.message, "model": payload.model},
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return _json_response(AssistantQueryResponse(**cached), cache_status="HIT")

    result = run_supervisor_query(message=payload.message, model=payload.model)
    response_payload = AssistantQueryResponse(**result)
    if _CACHE_ENABLED:
        cache.set(cache_key, response_payload.model_dump(), _CACHE_TTL_ASSISTANT_QUERY)
    return _json_response(response_payload, cache_status="MISS" if _CACHE_ENABLED else None)


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
@limiter.limit(_WHATSAPP_RATE_LIMIT)
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
