"""Maps WhiteBoxXAI SDK exceptions to clear, structured MCP tool errors.

Any exception raised inside a FastMCP tool function is automatically
converted by the `mcp` SDK into a structured
`CallToolResult(isError=True, content=[TextContent(text=str(exc))])` --
see `mcp.server.lowlevel.server.Server.call_tool`'s `except Exception`
branch. So mapping here just means raising a clear, typed, descriptive
exception; no manual result-content construction is needed.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from whiteboxxai.exceptions import (
    APIError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
)

from whiteboxxai_mcp.auth.session import SessionManager

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BackendValidationError(Exception):
    """The WhiteBoxXAI backend rejected the request as invalid (HTTP 422)."""


class BackendAuthError(Exception):
    """Could not authenticate to the WhiteBoxXAI backend."""


class BackendQuotaExceededError(Exception):
    """The organization's tier/quota limit was exceeded (HTTP 429). Not a bug."""


class BackendUnavailableError(Exception):
    """The WhiteBoxXAI backend (or a required background worker) is unreachable."""


async def call_with_session(
    session: SessionManager,
    make_call: Callable[[Any], Awaitable[T]],
) -> T:
    """Run an SDK call through `session`.

    `make_call` takes the WhiteBoxXAI SDK client and returns the
    awaitable resource-method call, e.g.:

        await call_with_session(session, lambda client: client.models.aget(model_id))
    """
    client = await session.get_client()
    try:
        return await make_call(client)
    except AuthenticationError as exc:
        raise BackendAuthError(
            "Authentication to WhiteBoxXAI failed. Ensure WHITEBOXXAI_MCP_API_KEY is correct "
            "and has not been revoked."
        ) from exc
    except ValidationError as exc:
        raise BackendValidationError(
            f"WhiteBoxXAI rejected the request: {exc}"
        ) from exc
    except RateLimitError as exc:
        raise BackendQuotaExceededError(
            "WhiteBoxXAI API quota/tier limit exceeded (HTTP 429). This is a "
            f"plan limit, not an error -- wait and retry, or upgrade the plan. ({exc})"
        ) from exc
    except APIError as exc:
        raise BackendUnavailableError(
            f"WhiteBoxXAI backend request failed: {exc}"
        ) from exc
