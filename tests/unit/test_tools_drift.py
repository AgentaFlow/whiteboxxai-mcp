"""Behavior tests for whiteboxxai_mcp.tools.drift (issue #227).

Brings the M1 drift-detection domain up to the same behavioral-delegation
bar as the M2/#137 domains (see test_tools_explanations.py, the template
this file follows).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.tools import drift


@pytest.fixture
def mcp_with_fake_session(monkeypatch):
    mcp = FastMCP(name="test")
    drift.register(mcp)

    fake_client = MagicMock()
    fake_client.drift.adetect = AsyncMock(
        return_value={"model_id": "m1", "drift_detected": False, "features": {}}
    )
    fake_client.drift.acreate_report = AsyncMock(
        return_value={"report_id": "r1", "model_id": "m1", "drift_detected": False}
    )
    fake_client.drift.aget_reports = AsyncMock(return_value=[{"report_id": "r1"}])
    fake_client.drift.aget_report = AsyncMock(
        return_value={"report_id": "r1", "model_id": "m1", "features": {}}
    )
    fake_client.drift.aget_trend = AsyncMock(
        return_value={"model_id": "m1", "trend": []}
    )

    class FakeSession:
        """Minimal stand-in for SessionManager (mirrors test_error_mapping.py)."""

        async def get_client(self, force_refresh: bool = False):
            return fake_client

    monkeypatch.setattr(drift, "get_session", lambda: FakeSession())
    return mcp, fake_client


@pytest.mark.asyncio
async def test_detect_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_drift_detect")

    result = await tool.run({"model_id": "m1"})

    fake_client.drift.adetect.assert_awaited_once_with(
        "m1", window_size=1000, feature_names=None
    )
    assert result == {"model_id": "m1", "drift_detected": False, "features": {}}


@pytest.mark.asyncio
async def test_detect_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_drift_detect")

    await tool.run(
        {"model_id": "m1", "window_size": 500, "feature_names": ["age", "income"]}
    )

    fake_client.drift.adetect.assert_awaited_once_with(
        "m1", window_size=500, feature_names=["age", "income"]
    )


@pytest.mark.asyncio
async def test_create_report_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_drift_create_report")

    result = await tool.run({"model_id": "m1"})

    fake_client.drift.acreate_report.assert_awaited_once_with("m1", window_size=1000)
    assert result == {"report_id": "r1", "model_id": "m1", "drift_detected": False}


@pytest.mark.asyncio
async def test_create_report_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_drift_create_report")

    await tool.run({"model_id": "m1", "window_size": 2000})

    fake_client.drift.acreate_report.assert_awaited_once_with("m1", window_size=2000)


@pytest.mark.asyncio
async def test_list_reports_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_drift_list_reports")

    result = await tool.run({"model_id": "m1"})

    fake_client.drift.aget_reports.assert_awaited_once_with("m1", limit=10, skip=0)
    assert result == [{"report_id": "r1"}]


@pytest.mark.asyncio
async def test_list_reports_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_drift_list_reports")

    await tool.run({"model_id": "m1", "limit": 5, "skip": 20})

    fake_client.drift.aget_reports.assert_awaited_once_with("m1", limit=5, skip=20)


@pytest.mark.asyncio
async def test_get_report_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_drift_get_report")

    result = await tool.run({"model_id": "m1", "report_id": "r1"})

    fake_client.drift.aget_report.assert_awaited_once_with("m1", "r1")
    assert result == {"report_id": "r1", "model_id": "m1", "features": {}}


@pytest.mark.asyncio
async def test_trend_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_drift_trend")

    result = await tool.run({"model_id": "m1"})

    fake_client.drift.aget_trend.assert_awaited_once_with("m1", days=7)
    assert result == {"model_id": "m1", "trend": []}


@pytest.mark.asyncio
async def test_trend_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_drift_trend")

    await tool.run({"model_id": "m1", "days": 30})

    fake_client.drift.aget_trend.assert_awaited_once_with("m1", days=30)
