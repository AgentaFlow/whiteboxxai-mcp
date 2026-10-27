"""Streamable HTTP transport (issue #226): `build_http_app()` for backend/main.py to mount at /mcp.

Unlike stdio, an HTTP `/mcp` endpoint can serve many different WhiteBoxXAI
orgs through one process -- there is no server-wide credential. Every
request must carry its own `Authorization: Bearer <jwt>` header; the app
returned here rejects any request without one, at the ASGI layer, before an
MCP protocol session is ever created -- there is no fallback to a shared or
stdio-configured credential in HTTP mode. See auth/context.py for the
per-connection SessionManager cache this resolves the header into.

This module must not import `backend.core.config` (or anything else from
`backend`): `mcp_server` is a standalone, independently-installable
distribution with its own stdio-only consumers who don't have `backend`
installed. `build_http_app()` takes `base_url` as a parameter instead --
backend/main.py is the one caller that supplies it explicitly.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Optional

from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from whiteboxxai_mcp.auth.context import _active_http_session, resolve_http_session
from whiteboxxai_mcp.config import MCPSettings
from whiteboxxai_mcp.server import build_server

_BEARER_PREFIX = "bearer "


def _extract_bearer_token(scope: Scope) -> str | None:
    """Read the raw JWT out of an ASGI http scope's Authorization header.

    Returns None for a missing header, a non-Bearer scheme, or an empty
    token -- all of which the caller must treat as "reject", not "fall back
    to something else".
    """
    headers = dict(scope.get("headers") or [])
    raw = headers.get(b"authorization")
    if not raw:
        return None
    try:
        text = raw.decode("latin-1")
    except UnicodeDecodeError:
        return None
    if not text.lower().startswith(_BEARER_PREFIX):
        return None
    token = text[len(_BEARER_PREFIX) :].strip()
    return token or None


class _AuthGateApp:
    """Wraps FastMCP's Streamable HTTP ASGI app with a per-request auth boundary.

    Rejecting here (before delegating to the inner mcp app) rather than
    inside get_session() on the first tool call means a caller with no/bad
    credential never gets an MCP protocol session, an Mcp-Session-Id, etc.
    spun up for it -- a clean 401 on the very first request instead of a
    tool-call-shaped error later. This mirrors the shape of mcp's own
    optional RequireAuthMiddleware (streamable_http_app() wraps the same
    route in it when FastMCP is given a token_verifier) -- we don't use that
    machinery ourselves, since WhiteBoxXAI JWTs aren't OAuth2 resource-server
    tokens, but a transport-level rejection is the right shape to copy.

    No JWT signature/shape validation happens here beyond "is there a
    Bearer-prefixed non-empty string" -- real verification is already the
    backend's job on each tool's own outbound SDK call.
    """

    def __init__(self, inner_app: ASGIApp, fastmcp: FastMCP, base_url: str) -> None:
        self._inner_app = inner_app
        self.fastmcp = fastmcp
        self._base_url = base_url

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._inner_app(scope, receive, send)
            return

        token = _extract_bearer_token(scope)
        if token is None:
            response = JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "detail": (
                        "Missing or malformed Authorization header. The "
                        "WhiteBoxXAI MCP HTTP transport requires "
                        "'Authorization: Bearer <jwt>' on every request -- "
                        "there is no shared or default credential."
                    ),
                    "status_code": 401,
                },
            )
            await response(scope, receive, send)
            return

        session = resolve_http_session(self._base_url, token)
        reset_token = _active_http_session.set(session)
        try:
            await self._inner_app(scope, receive, send)
        finally:
            _active_http_session.reset(reset_token)


_cached_mcp: FastMCP | None = None


def _get_or_build_mcp(cfg: MCPSettings) -> FastMCP:
    """Reuse one FastMCP (and its 197 registered tools) across repeated
    build_http_app() calls, rebuilding only its StreamableHTTPSessionManager.

    build_server() -- FastMCP construction plus registering every tool
    across all 18 domain modules -- measured at ~0.6s, almost entirely
    `@mcp.tool(...)`'s per-tool JSON-schema generation. build_http_app() is
    called once per real backend.main.lifespan() cycle (see backend/main.py's
    _MCPAppProxy docstring for why that can't be a module-level singleton
    built once) -- in production that's once per process lifetime, but
    tests enter/exit lifespan() once per test function via the ordinary
    `client` fixture, reused across hundreds of tests in one process. Paying
    0.6s on every one of those would add minutes to the suite for a cost
    that has nothing to do with what any of those tests are actually
    checking. FastMCP.streamable_http_app() lazily builds `_session_manager`
    only when it's None and otherwise reuses whatever's already there
    (confirmed by reading its source, not assumed) -- resetting it here
    forces a fresh, never-yet-`run()` StreamableHTTPSessionManager each call
    while the expensive, side-effect-free tool registry is built exactly
    once.
    """
    global _cached_mcp
    if _cached_mcp is None:
        _cached_mcp = build_server(cfg, streamable_http_path="/")
    else:
        _cached_mcp._session_manager = None
    return _cached_mcp


def build_http_app(base_url: str | None = None) -> _AuthGateApp:
    """Build the ASGI app for backend/main.py to mount at /mcp.

    `base_url` should be the backend's own address (e.g.
    `backend.core.config.settings.API_URL`) -- each MCP tool call round-trips
    back into the same backend process over loopback HTTP via the SDK
    client, exactly like a stdio or browser client would, so the same
    auth/RLS/tenant-isolation code path is exercised either way. Falls back
    to MCPSettings().API_BASE_URL only when not supplied, for standalone/test
    use outside of backend/main.py.
    """
    cfg = MCPSettings()
    resolved_base_url = base_url or cfg.API_BASE_URL

    mcp = _get_or_build_mcp(cfg)
    inner_app = mcp.streamable_http_app()
    return _AuthGateApp(inner_app, mcp, resolved_base_url)


@contextlib.asynccontextmanager
async def mcp_lifespan(http_app: _AuthGateApp) -> AsyncIterator[None]:
    """Drive the mounted app's own lifespan from backend/main.py's lifespan().

    Starlette's Router dispatches an ASGI "lifespan" scope straight to the
    top-level app's own lifespan context manager and never into a mounted
    sub-app's routes -- so `app.mount("/mcp", build_http_app())` alone would
    never run the Starlette app streamable_http_app() returns internally
    build with `lifespan=lambda app: self.session_manager.run()`. Without
    that ever running, StreamableHTTPSessionManager's task group is never
    initialized, and the very first request to /mcp raises RuntimeError
    ("Task group is not initialized..."). This replicates that same
    `session_manager.run()` call, just driven explicitly from the outer
    app's lifespan instead of relying on Mount() to forward it (it doesn't).
    """
    async with http_app.fastmcp.session_manager.run():
        yield
