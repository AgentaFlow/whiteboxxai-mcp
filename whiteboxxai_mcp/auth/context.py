"""Resolves the SessionManager tool code should use for the current transport.

stdio: one process serves one WhiteBoxXAI org, so a single module-level
SessionManager singleton is built from WHITEBOXXAI_MCP_* env config on first
use and reused for the process lifetime.

Streamable HTTP (issue #226): the hosted endpoint can serve many different
WhiteBoxXAI orgs through one process, so there is no server-wide credential
-- each connection supplies its own JWT via the Authorization header. The
ASGI wrapper in asgi.py resolves that header to a per-connection
SessionManager (via `resolve_http_session`, TTL-evicted, keyed on the JWT)
and sets it into `_active_http_session` for the duration of that request;
`get_session()` below checks that first and falls back to the stdio
singleton only when it's unset. asgi.py rejects any request with no/malformed
Authorization header before this module is ever consulted, so there is no
code path where a missing HTTP credential silently falls back to a shared
one -- see asgi.py's module docstring for the full reasoning.
"""

from __future__ import annotations

import hashlib
import time
from contextvars import ContextVar
from typing import Dict, Optional, Tuple

from whiteboxxai_mcp.auth.session import SessionManager
from whiteboxxai_mcp.config import MCPSettings, settings

_stdio_session: SessionManager | None = None

_active_http_session: ContextVar[SessionManager | None] = ContextVar(
    "_active_http_session", default=None
)


def build_session_from_settings(cfg: MCPSettings) -> SessionManager:
    """Construct a SessionManager from MCPSettings (api_key)."""
    return SessionManager(
        base_url=cfg.API_BASE_URL,
        api_key=cfg.API_KEY,
        timeout=cfg.API_TIMEOUT,
    )


def get_stdio_session() -> SessionManager:
    """Return the process-wide SessionManager singleton for stdio transport."""
    global _stdio_session
    if _stdio_session is None:
        _stdio_session = build_session_from_settings(settings)
    return _stdio_session


def get_session() -> SessionManager:
    """Return the SessionManager tool code should use for the current call.

    HTTP mode: returns the per-connection session asgi.py set into
    `_active_http_session` for this request. stdio mode: that contextvar is
    never set, so this falls back to the process-wide singleton.
    """
    http_session = _active_http_session.get()
    if http_session is not None:
        return http_session
    return get_stdio_session()


def reset_stdio_session() -> None:
    """Clear the singleton. Intended for tests, not production use."""
    global _stdio_session
    _stdio_session = None


# --- Per-connection cache for Streamable HTTP (issue #226) -----------------
#
# Keyed on a SHA-256 hash of the raw JWT, not the raw JWT itself. The cached
# SessionManager already holds the raw JWT internally (it needs it to
# authenticate outbound calls), so hashing the key buys nothing against "does
# the JWT live in server memory" -- what it buys is that the JWT never
# appears as a literal dict key in a debugger's variable inspector, a
# crash-dump heap walk, or a cache-hit/miss log line.
#
# Eviction is lazy-on-access plus an opportunistic amortized sweep, not a
# real background asyncio task. This is deliberate: eviction here is
# side-effect-free, because the cache key is always re-derivable from the
# next request's own Authorization header -- a miss just transparently
# rebuilds an equivalent SessionManager (cheap; building one doesn't touch
# the network until first real use). A background task would need its own
# start/cancel lifecycle wired into backend/main.py's lifespan for no
# correctness benefit, only to bound memory slightly sooner.

_HttpCacheEntry = tuple[SessionManager, float]  # (session, last_access_monotonic)

_http_session_cache: dict[str, _HttpCacheEntry] = {}
_last_sweep: float = 0.0


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _sweep_stale_entries(now: float, ttl_seconds: int) -> None:
    global _last_sweep
    stale_keys = [
        key
        for key, (_, last_access) in _http_session_cache.items()
        if now - last_access > ttl_seconds
    ]
    for key in stale_keys:
        del _http_session_cache[key]
    _last_sweep = now


def resolve_http_session(base_url: str, token: str) -> SessionManager:
    """Resolve the per-connection SessionManager for an inbound HTTP JWT.

    Cache hit within TTL: returns the same SessionManager instance (so its
    own internal SDK-client caching in SessionManager.get_client() is
    preserved across calls within one MCP session). Cache miss (new token,
    or the previous entry aged out): builds a new static-JWT SessionManager
    -- HTTP-mode sessions are always static-JWT, never email/password, per
    the documented v1 limitation that a long MCP session needs the client to
    reconnect with a fresh JWT once it expires (there is no refresh path for
    an already-issued JWT the server didn't mint itself).
    """
    now = time.monotonic()
    ttl = settings.HTTP_SESSION_CACHE_TTL_SECONDS
    sweep_interval = settings.HTTP_SESSION_CACHE_SWEEP_INTERVAL_SECONDS

    if now - _last_sweep > sweep_interval:
        _sweep_stale_entries(now, ttl)

    key = _hash_token(token)
    entry = _http_session_cache.get(key)
    if entry is not None:
        session, last_access = entry
        if now - last_access <= ttl:
            _http_session_cache[key] = (session, now)
            return session
        del _http_session_cache[key]

    session = SessionManager(
        base_url=base_url,
        api_key=token,
        timeout=settings.API_TIMEOUT,
    )
    _http_session_cache[key] = (session, now)
    return session


def reset_http_session_cache() -> None:
    """Clear the HTTP session cache. Intended for tests, not production use."""
    global _last_sweep
    _http_session_cache.clear()
    _last_sweep = 0.0
