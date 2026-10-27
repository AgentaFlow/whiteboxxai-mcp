"""Behavior tests for whiteboxxai_mcp.tools.risk_register (issue #227).

Brings the M1 AI Risk Register domain up to the same behavioral-delegation
bar as the M2/#137 domains (see test_tools_explanations.py, the template
this file follows).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.tools import risk_register


@pytest.fixture
def mcp_with_fake_session(monkeypatch):
    mcp = FastMCP(name="test")
    risk_register.register(mcp)

    fake_client = MagicMock()
    fake_client.risk_register.alist = AsyncMock(
        return_value={"items": [{"id": "r1"}], "total": 1, "skip": 0, "limit": 50}
    )
    fake_client.risk_register.aget = AsyncMock(
        return_value={"id": "r1", "status": "identified", "severity": "medium"}
    )
    fake_client.risk_register.aportfolio = AsyncMock(
        return_value={"heat_map": {}, "top_risks": []}
    )

    class FakeSession:
        """Minimal stand-in for SessionManager (mirrors test_error_mapping.py)."""

        async def get_client(self, force_refresh: bool = False):
            return fake_client

    monkeypatch.setattr(risk_register, "get_session", lambda: FakeSession())
    return mcp, fake_client


@pytest.mark.asyncio
async def test_list_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_risk_register_list")

    result = await tool.run({})

    fake_client.risk_register.alist.assert_awaited_once_with(
        status=None, severity=None, category=None, model_id=None, limit=50, skip=0
    )
    assert result == {"items": [{"id": "r1"}], "total": 1, "skip": 0, "limit": 50}


@pytest.mark.asyncio
async def test_list_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_risk_register_list")

    await tool.run(
        {
            "status": "assessed",
            "severity": "high",
            "category": "bias",
            "model_id": "m1",
            "limit": 10,
            "skip": 5,
        }
    )

    fake_client.risk_register.alist.assert_awaited_once_with(
        status="assessed",
        severity="high",
        category="bias",
        model_id="m1",
        limit=10,
        skip=5,
    )


@pytest.mark.asyncio
async def test_get_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_risk_register_get")

    result = await tool.run({"entry_id": "r1"})

    fake_client.risk_register.aget.assert_awaited_once_with("r1")
    assert result == {"id": "r1", "status": "identified", "severity": "medium"}


@pytest.mark.asyncio
async def test_portfolio_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_risk_register_portfolio")

    result = await tool.run({})

    fake_client.risk_register.aportfolio.assert_awaited_once_with()
    assert result == {"heat_map": {}, "top_risks": []}
