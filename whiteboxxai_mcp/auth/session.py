"""Session management: WhiteBoxXAI API Key usage.

The backend supports long-lived, individually revocable API keys
(`POST /api/v1/api-keys`). Use WHITEBOXXAI_MCP_API_KEY for access.
The key doesn't expire on a fixed schedule the way a JWT does, only on
explicit revocation (at which point the backend rejects it with a 401,
which surfaces as a normal request error).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from whiteboxxai import WhiteBoxXAI

from whiteboxxai_mcp.client_factory import build_client

logger = logging.getLogger(__name__)


class SessionManager:
    """Owns one WhiteBoxXAI credential + SDK client.

    Construct with:
    - `api_key`: a scoped, revocable key from POST /api/v1/api-keys.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: int = 30,
    ):
        if not api_key:
            raise ValueError("SessionManager requires `api_key`")

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

        self._client: WhiteBoxXAI | None = None
        self._lock = asyncio.Lock()

    async def get_client(self) -> WhiteBoxXAI:
        """Return a WhiteBoxXAI SDK client with a valid (non-expired) credential."""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = build_client(
                        self._api_key, self._base_url, self._timeout
                    )
                    logger.info(
                        "WhiteBoxXAI session authenticated for %s", self._base_url
                    )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying SDK client's HTTP connections, if any were opened."""
        if self._client is not None:
            await self._client.aclose()
