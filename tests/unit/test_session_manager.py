"""Tests for SessionManager.

No live WhiteBoxXAI backend is used.
"""

import asyncio

import pytest

from whiteboxxai_mcp.auth.session import SessionManager


class TestConstruction:
    def test_requires_api_key(self):
        with pytest.raises(ValueError):
            SessionManager(base_url="https://api.test.com")

    def test_accepts_api_key(self):
        session = SessionManager(
            base_url="https://api.test.com", api_key="wbx_live_123"
        )
        assert session._api_key == "wbx_live_123"


class TestLoginAndRefresh:
    @pytest.mark.asyncio
    async def test_get_client_builds_client(self):
        session = SessionManager(
            base_url="https://api.test.com", api_key="wbx_live_123"
        )
        client = await session.get_client()
        assert client.config.api_key == "wbx_live_123"

    @pytest.mark.asyncio
    async def test_concurrent_calls_only_build_once(self):
        session = SessionManager(
            base_url="https://api.test.com", api_key="wbx_live_123"
        )
        results = await asyncio.gather(
            session.get_client(), session.get_client(), session.get_client()
        )
        # Verify the exact same instance is returned for all calls
        assert all(c is results[0] for c in results)
