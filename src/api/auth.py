"""JWT-based authentication backed by Supabase SDK.

Flow
----
1.  Friend submits name/email/reason at POST /access/request
2.  You get an email notification; open Supabase dashboard → Table Editor
    and flip status → 'approved' for that row.
3.  A Postgres trigger calls POST /internal/send-key on your API.
4.  Your API generates a secure key, stores its SHA-256 hash in Supabase,
    and emails the plaintext key to the friend via Resend.
5.  Friend calls POST /auth/token {"api_key": "..."}
6.  Your API hashes the supplied key, looks it up in Supabase
    (status must be 'approved'), and issues a short-lived JWT.
7.  Friend uses the JWT as  Authorization: Bearer <token>  on all calls.

Environment variables
---------------------
SUPABASE_URL            Project URL — Supabase dashboard → Settings → API
SUPABASE_SERVICE_KEY    service_role key (same page as above)
API_JWT_SECRET          Long random string for signing JWTs
API_JWT_EXPIRE_MINUTES  Token lifetime in minutes (default: 60)
API_INTERNAL_SECRET     Shared secret for /internal/* routes — must match
                        the value set in Supabase SQL as app.internal_secret
RESEND_API_KEY          From https://resend.com (free: 3 000 emails/month)
RESEND_FROM_EMAIL       Verified sender address in your Resend account
ADMIN_EMAIL             Where YOU receive new-request notifications
"""
from __future__ import annotations

import hashlib
import os
import secrets
import time
from functools import lru_cache
from typing import Annotated

import httpx
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from postgrest.exceptions import APIError
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set. "
            "See .env.example for instructions."
        )
    return value


def _jwt_expire_minutes() -> int:
    try:
        v = int(os.getenv("API_JWT_EXPIRE_MINUTES", "60").strip())
        return v if v > 0 else 60
    except ValueError:
        return 60


ALGORITHM = "HS256"
_TABLE = "api_access_requests"
_KEY_PREFIX = "map_"  # issued keys look like:  map_<random>


# ---------------------------------------------------------------------------
# Supabase client — one instance for the process lifetime
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _db() -> Client:
    """Return a cached Supabase client initialised from env vars."""
    return create_client(
        _require_env("SUPABASE_URL"),
        _require_env("SUPABASE_SERVICE_KEY"),
    )


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _generate_api_key() -> str:
    return f"{_KEY_PREFIX}{secrets.token_urlsafe(32)}"


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _key_hint(raw_key: str) -> str:
    return raw_key[-4:] if len(raw_key) >= 4 else "****"


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------

def create_access_token(subject: str) -> tuple[str, int]:
    """Return (encoded_jwt, expires_at_unix_ts)."""
    now = int(time.time())
    expires_at = now + _jwt_expire_minutes() * 60
    token = jwt.encode(
        {"sub": subject, "iat": now, "exp": expires_at},
        _require_env("API_JWT_SECRET"),
        algorithm=ALGORITHM,
    )
    return token, expires_at


_bearer_scheme = HTTPBearer(auto_error=True)


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(
            token,
            _require_env("API_JWT_SECRET"),
            algorithms=[ALGORITHM],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired — call POST /auth/token to get a fresh one.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_auth(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_scheme)],
) -> dict:
    """FastAPI dependency — inject into any protected route."""
    return _decode_token(credentials.credentials)


# ---------------------------------------------------------------------------
# Internal secret guard (for /internal/* routes called by Supabase triggers)
# ---------------------------------------------------------------------------

def require_internal(request: Request) -> None:
    """FastAPI dependency — validates X-Internal-Secret header."""
    expected = os.getenv("API_INTERNAL_SECRET", "").strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API_INTERNAL_SECRET is not configured.",
        )
    provided = request.headers.get("X-Internal-Secret", "")
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid internal secret.",
        )


# ---------------------------------------------------------------------------
# API-key → JWT exchange  (POST /auth/token)
# ---------------------------------------------------------------------------

def exchange_api_key(raw_key: str) -> tuple[str, int]:
    """Validate a raw API key and return (jwt, expires_at)."""
    try:
        result = (
            _db()
            .table(_TABLE)
            .select("email, api_key_hint")
            .eq("api_key_hash", _hash_key(raw_key))
            .eq("status", "approved")
            .limit(1)
            .execute()
        )
    except APIError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Auth backend error: {exc.message}",
        )

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    row = result.data[0]
    subject = f"user:{row['email']}:hint:{row.get('api_key_hint', '????')}"
    return create_access_token(subject)


# ---------------------------------------------------------------------------
# Access-request workflow  (POST /access/request)
# ---------------------------------------------------------------------------

def create_access_request(name: str, email: str, reason: str) -> dict:
    """Insert a pending access request row. Raises 409 on duplicate email."""
    try:
        result = (
            _db()
            .table(_TABLE)
            .insert({"name": name, "email": email, "reason": reason, "status": "pending"})
            .execute()
        )
        return result.data[0]
    except APIError as exc:
        # Supabase surfaces the Postgres unique-violation as a 409
        if "duplicate" in exc.message.lower() or "unique" in exc.message.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An access request for this email already exists.",
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error: {exc.message}",
        )


def generate_and_store_key(request_id: str) -> str:
    """Generate an API key, store its hash in Supabase, return the plaintext key."""
    raw_key = _generate_api_key()
    try:
        _db().table(_TABLE).update(
            {
                "api_key_hash": _hash_key(raw_key),
                "api_key_hint": _key_hint(raw_key),
                "reviewed_at":  "now()",
            }
        ).eq("id", request_id).execute()
    except APIError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to store key: {exc.message}",
        )
    return raw_key


# ---------------------------------------------------------------------------
# Email via Resend
# ---------------------------------------------------------------------------

def send_new_request_notification(name: str, email: str, reason: str, request_id: str) -> None:
    """Email YOU when a new access request arrives."""
    admin_email = os.getenv("ADMIN_EMAIL", "").strip()
    if not admin_email:
        return
    _send_email(
        to=admin_email,
        subject=f"[API Access] New request from {name}",
        html=f"""
        <h2>New API Access Request</h2>
        <p><b>Name:</b> {name}</p>
        <p><b>Email:</b> {email}</p>
        <p><b>Reason:</b> {reason or '—'}</p>
        <p><b>Request ID:</b> <code>{request_id}</code></p>
        <hr>
        <p>To approve: open your
        <a href="https://supabase.com/dashboard">Supabase dashboard</a>
        → Table Editor → <code>api_access_requests</code>,
        find the row, and change <code>status</code> to <code>approved</code>.
        The key will be generated and emailed to them automatically.</p>
        """,
    )


def send_api_key_email(name: str, email: str, raw_key: str) -> None:
    """Email the approved user their API key with quick-start instructions."""
    _send_email(
        to=email,
        subject="Your Multi-Agent Platform API Key",
        html=f"""
        <h2>Welcome, {name}!</h2>
        <p>Your access request has been approved. Here is your API key:</p>
        <pre style="background:#f4f4f4;padding:12px;border-radius:6px;font-size:15px">{raw_key}</pre>
        <h3>Quick start</h3>
        <p><b>Step 1 — exchange your key for a short-lived JWT (valid 60 min):</b></p>
        <pre style="background:#f4f4f4;padding:12px;border-radius:6px">curl -X POST https://your-app.railway.app/auth/token \\
  -H "Content-Type: application/json" \\
  -d '{{"api_key": "{raw_key}"}}'</pre>
        <p><b>Step 2 — call the API:</b></p>
        <pre style="background:#f4f4f4;padding:12px;border-radius:6px">curl -X POST https://your-app.railway.app/api/v1/assistant/query \\
  -H "Authorization: Bearer &lt;your_token&gt;" \\
  -H "Content-Type: application/json" \\
  -d '{{"message": "Should I buy AAPL?"}}'</pre>
        <p>Keep your key private. Reply to this email if you need it rotated.</p>
        """,
    )


def _send_email(to: str, subject: str, html: str) -> None:
    """Send via Resend. Logs on failure — never crashes the API."""
    api_key   = os.getenv("RESEND_API_KEY", "").strip()
    from_addr = os.getenv("RESEND_FROM_EMAIL", "noreply@yourdomain.com").strip()

    if not api_key:
        print(f"[auth] RESEND_API_KEY not set — skipping email to {to}: {subject}")
        return

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                },
                json={"from": from_addr, "to": [to], "subject": subject, "html": html},
            )
        resp.raise_for_status()
    except Exception as exc:
        print(f"[auth] Email to {to} failed: {exc}")