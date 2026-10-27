"""Behavior tests for whiteboxxai_mcp.tools.alert_management (issue #224).

Only the 13 endpoints registered as MCP tools are covered here --
escalation-policy CRUD and bulk_action are SDK-only (see
tests/sdk/test_resources.py's TestAlertManagementResource) and
intentionally have no tool to test.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.tools import alert_management


@pytest.fixture
def mcp_with_fake_session(monkeypatch):
    mcp = FastMCP(name="test")
    alert_management.register(mcp)

    fake_client = MagicMock()
    fake_client.alert_management.aget_history = AsyncMock(
        return_value={"alert_id": "a1"}
    )
    fake_client.alert_management.aget_analytics = AsyncMock(
        return_value={"total_alerts": 0}
    )
    fake_client.alert_management.aget_trend = AsyncMock(
        return_value={"data_points": []}
    )
    fake_client.alert_management.aget_velocity = AsyncMock(
        return_value={"current_hour_count": 0}
    )
    fake_client.alert_management.aget_grouping = AsyncMock(return_value={"groups": []})
    fake_client.alert_management.adetect_fatigue = AsyncMock(
        return_value={"is_fatigued": False}
    )
    fake_client.alert_management.aacknowledge = AsyncMock(
        return_value={"alert_id": "a1", "status": "ACKNOWLEDGED"}
    )
    fake_client.alert_management.aresolve = AsyncMock(
        return_value={"alert_id": "a1", "status": "RESOLVED"}
    )
    fake_client.alert_management.asnooze = AsyncMock(
        return_value={"alert_id": "a1", "status": "SNOOZED"}
    )
    fake_client.alert_management.acreate_comment = AsyncMock(return_value={"id": "c1"})
    fake_client.alert_management.aget_comments = AsyncMock(return_value=[{"id": "c1"}])
    fake_client.alert_management.aprocess_pending_escalations = AsyncMock(
        return_value={"processed_count": 0}
    )
    fake_client.alert_management.aunsnooze_expired_alerts = AsyncMock(
        return_value={"reactivated_count": 0}
    )

    class FakeSession:
        async def get_client(self, force_refresh: bool = False):
            return fake_client

    monkeypatch.setattr(alert_management, "get_session", lambda: FakeSession())
    return mcp, fake_client


@pytest.mark.asyncio
async def test_get_history_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_management_get_history")

    result = await tool.run({"alert_id": "a1"})

    fake_client.alert_management.aget_history.assert_awaited_once_with("a1")
    assert result == {"alert_id": "a1"}


@pytest.mark.asyncio
async def test_get_analytics_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_management_get_analytics")

    await tool.run({})

    fake_client.alert_management.aget_analytics.assert_awaited_once_with(
        start_time=None, end_time=None
    )


@pytest.mark.asyncio
async def test_get_trend_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_management_get_trend")

    await tool.run({"start": "2026-01-01", "end": "2026-01-02"})

    fake_client.alert_management.aget_trend.assert_awaited_once_with(
        "2026-01-01",
        "2026-01-02",
        interval="hour",
        group_by=None,
        filters=None,
    )


@pytest.mark.asyncio
async def test_get_velocity_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_management_get_velocity")

    await tool.run({})

    fake_client.alert_management.aget_velocity.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_get_grouping_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_management_get_grouping")

    await tool.run({"group_by": "severity"})

    fake_client.alert_management.aget_grouping.assert_awaited_once_with("severity")


@pytest.mark.asyncio
async def test_detect_fatigue_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_management_detect_fatigue")

    await tool.run({})

    fake_client.alert_management.adetect_fatigue.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_acknowledge_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_management_acknowledge")

    await tool.run({"alert_id": "a1", "notes": "investigating"})

    fake_client.alert_management.aacknowledge.assert_awaited_once_with(
        "a1", notes="investigating"
    )


@pytest.mark.asyncio
async def test_resolve_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_management_resolve")

    await tool.run({"alert_id": "a1"})

    fake_client.alert_management.aresolve.assert_awaited_once_with(
        "a1", resolution_notes=None
    )


@pytest.mark.asyncio
async def test_snooze_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_management_snooze")

    await tool.run({"alert_id": "a1", "snooze_minutes": 30})

    fake_client.alert_management.asnooze.assert_awaited_once_with("a1", 30)


@pytest.mark.asyncio
async def test_create_comment_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_management_create_comment")

    await tool.run({"alert_id": "a1", "comment": "looking into it"})

    fake_client.alert_management.acreate_comment.assert_awaited_once_with(
        "a1", "looking into it", comment_type="note"
    )


@pytest.mark.asyncio
async def test_get_comments_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_management_get_comments")

    await tool.run({"alert_id": "a1"})

    fake_client.alert_management.aget_comments.assert_awaited_once_with("a1")


@pytest.mark.asyncio
async def test_process_pending_escalations_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool(
        "whitebox_alert_management_process_pending_escalations"
    )

    await tool.run({})

    fake_client.alert_management.aprocess_pending_escalations.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_unsnooze_expired_alerts_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool(
        "whitebox_alert_management_unsnooze_expired_alerts"
    )

    await tool.run({})

    fake_client.alert_management.aunsnooze_expired_alerts.assert_awaited_once_with()


def test_escalation_and_bulk_action_are_not_registered_as_tools():
    """Scope decision (issue #224): escalation-policy CRUD and bulk-action
    are admin-console-shaped and stay SDK-only."""
    mcp = FastMCP(name="test")
    alert_management.register(mcp)
    names = {t.name for t in mcp._tool_manager.list_tools()}
    excluded = {
        "whitebox_alert_management_create_escalation_policy",
        "whitebox_alert_management_list_escalation_policies",
        "whitebox_alert_management_get_escalation_policy",
        "whitebox_alert_management_update_escalation_policy",
        "whitebox_alert_management_delete_escalation_policy",
        "whitebox_alert_management_bulk_action",
    }
    assert not (excluded & names)
