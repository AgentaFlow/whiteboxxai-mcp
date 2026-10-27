"""Tests for call_with_session's SDK-exception -> MCP-error mapping."""

from unittest.mock import AsyncMock

import pytest
from whiteboxxai.exceptions import (
    APIError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
)

from whiteboxxai_mcp.errors import (
    BackendAuthError,
    BackendQuotaExceededError,
    BackendUnavailableError,
    BackendValidationError,
    call_with_session,
)


class FakeSession:
    """Minimal stand-in for SessionManager, tracking refresh calls."""

    def __init__(self, client):
        self.client = client
        self.refresh_calls = 0

    async def get_client(self):
        return self.client


@pytest.mark.asyncio
async def test_validation_error_maps_to_backend_validation_error():
    session = FakeSession(client=object())

    async def make_call(client):
        raise ValidationError("bad field")

    with pytest.raises(BackendValidationError):
        await call_with_session(session, make_call)


@pytest.mark.asyncio
async def test_rate_limit_error_maps_to_quota_exceeded():
    session = FakeSession(client=object())

    async def make_call(client):
        raise RateLimitError("too many requests")

    with pytest.raises(BackendQuotaExceededError):
        await call_with_session(session, make_call)


@pytest.mark.asyncio
async def test_generic_api_error_maps_to_backend_unavailable():
    session = FakeSession(client=object())

    async def make_call(client):
        raise APIError("connection refused")

    with pytest.raises(BackendUnavailableError):
        await call_with_session(session, make_call)


@pytest.mark.asyncio
async def test_auth_error_raises_backend_auth_error():
    session = FakeSession(client="client-v1")

    async def make_call(client):
        raise AuthenticationError("revoked")

    with pytest.raises(BackendAuthError):
        await call_with_session(session, make_call)


@pytest.mark.asyncio
async def test_successful_call():
    session = FakeSession(client="client-v1")

    async def make_call(client):
        return {"value": client}

    result = await call_with_session(session, make_call)

    assert result == {"value": "client-v1"}
