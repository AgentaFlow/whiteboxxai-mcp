"""Configuration for the WhiteBoxXAI MCP server.

Mirrors the pydantic-settings shape of backend/core/config.py: uppercase
fields, .env support, and a module-level singleton.
"""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPSettings(BaseSettings):
    """Settings loaded from WHITEBOXXAI_MCP_* environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="WHITEBOXXAI_MCP_",
        case_sensitive=True,
        extra="allow",
    )

    # WhiteBoxXAI API connection
    API_BASE_URL: str = "https://api.whiteboxxai.com"
    API_TIMEOUT: int = 30

    # Auth (stdio/single-tenant mode -- see auth/session.py). Provide:
    # API_KEY (a scoped, revocable key from
    # POST /api/v1/api-keys, never needs refreshing and doesn't require
    # storing a password).
    API_KEY: str | None = None

    # Streamable HTTP transport only (see asgi.py, auth/context.py) -- how
    # long a per-connection SessionManager stays cached after its last use,
    # and how often the cache opportunistically prunes stale entries. TTL
    # defaults to the backend's own ACCESS_TOKEN_EXPIRE_MINUTES (30 min):
    # long enough to cover think-time between tool calls in one sitting,
    # short enough that an abandoned connection's entry doesn't linger.
    # Purely a memory/latency tuning knob, never a correctness concern -- a
    # cache miss just transparently rebuilds an equivalent SessionManager
    # from the same inbound Authorization header on the very next request.
    HTTP_SESSION_CACHE_TTL_SECONDS: int = 30 * 60
    HTTP_SESSION_CACHE_SWEEP_INTERVAL_SECONDS: int = 5 * 60

    # Logging / diagnostics (verbose mode, request/response inspection).
    VERBOSE: bool = False
    LOG_LEVEL: str = "INFO"


settings = MCPSettings()
