"""Behavior tests for whiteboxxai_mcp.tools.alerts (issue #141)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.tools import alerts


@pytest.fixture
def mcp_with_fake_session(monkeypatch):
    mcp = FastMCP(name="test")
    alerts.register(mcp)

    fake_client = MagicMock()
    fake_client.alerts.acreate = AsyncMock(return_value={"id": "r1"})
    fake_client.alerts.alist = AsyncMock(return_value=[{"id": "r1"}])
    fake_client.alerts.aget_rule = AsyncMock(return_value={"id": "r1"})
    fake_client.alerts.aupdate_rule = AsyncMock(return_value={"id": "r1"})
    fake_client.alerts.adelete_rule = AsyncMock(return_value={"deleted": True})
    fake_client.alerts.aevaluate_rule = AsyncMock(return_value={"triggered": False})
    fake_client.alerts.alist_instances = AsyncMock(return_value=[{"id": "a1"}])
    fake_client.alerts.aget_instance = AsyncMock(return_value={"id": "a1"})
    fake_client.alerts.aacknowledge = AsyncMock(return_value={"id": "a1"})
    fake_client.alerts.aresolve = AsyncMock(return_value={"id": "a1"})
    fake_client.alerts.asnooze = AsyncMock(return_value={"id": "a1"})
    fake_client.alerts.astatistics = AsyncMock(return_value={"total_alerts": 0})

    class FakeSession:
        async def get_client(self, force_refresh: bool = False):
            return fake_client

    monkeypatch.setattr(alerts, "get_session", lambda: FakeSession())
    return mcp, fake_client


@pytest.mark.asyncio
async def test_create_rule_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alerts_create_rule")

    conditions = [{"metric_name": "accuracy", "operator": "lt", "threshold": 0.8}]
    result = await tool.run(
        {
            "name": "a1",
            "alert_type": "drift",
            "severity": "high",
            "conditions": conditions,
        }
    )

    fake_client.alerts.acreate.assert_awaited_once_with(
        "a1",
        "drift",
        "high",
        conditions,
        model_id=None,
        description=None,
        logic_operator="AND",
        notification_channels=None,
        throttle_minutes=60,
        cooldown_minutes=0,
        is_active=True,
    )
    assert result == {"id": "r1"}


@pytest.mark.asyncio
async def test_list_rules_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alerts_list_rules")

    await tool.run({"model_id": "m1"})

    fake_client.alerts.alist.assert_awaited_once_with(
        model_id="m1", alert_type=None, is_active=None, skip=0, limit=100
    )


@pytest.mark.asyncio
async def test_get_rule_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alerts_get_rule")

    await tool.run({"rule_id": "r1"})

    fake_client.alerts.aget_rule.assert_awaited_once_with("r1")


@pytest.mark.asyncio
async def test_update_rule_only_sends_provided_fields(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alerts_update_rule")

    await tool.run({"rule_id": "r1", "is_active": False})

    fake_client.alerts.aupdate_rule.assert_awaited_once_with("r1", is_active=False)


@pytest.mark.asyncio
async def test_delete_rule_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alerts_delete_rule")

    await tool.run({"rule_id": "r1"})

    fake_client.alerts.adelete_rule.assert_awaited_once_with("r1")


@pytest.mark.asyncio
async def test_evaluate_rule_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alerts_evaluate_rule")

    await tool.run({"rule_id": "r1", "metric_values": {"accuracy": 0.5}})

    fake_client.alerts.aevaluate_rule.assert_awaited_once_with(
        "r1",
        metric_values={"accuracy": 0.5},
        model_id=None,
        historical_values=None,
        metadata=None,
    )


@pytest.mark.asyncio
async def test_list_instances_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alerts_list_instances")

    await tool.run({"severity": "high"})

    fake_client.alerts.alist_instances.assert_awaited_once_with(
        rule_id=None,
        model_id=None,
        alert_type=None,
        severity="high",
        status=None,
        hours=None,
        skip=0,
        limit=100,
    )


@pytest.mark.asyncio
async def test_get_instance_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alerts_get_instance")

    await tool.run({"alert_id": "a1"})

    fake_client.alerts.aget_instance.assert_awaited_once_with("a1")


@pytest.mark.asyncio
async def test_acknowledge_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alerts_acknowledge")

    await tool.run({"alert_id": "a1", "user_id": "u1"})

    fake_client.alerts.aacknowledge.assert_awaited_once_with(
        "a1", user_id="u1", notes=None
    )


@pytest.mark.asyncio
async def test_resolve_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alerts_resolve")

    await tool.run({"alert_id": "a1", "user_id": "u1"})

    fake_client.alerts.aresolve.assert_awaited_once_with(
        "a1", user_id="u1", resolution_notes=None
    )


@pytest.mark.asyncio
async def test_snooze_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alerts_snooze")

    await tool.run({"alert_id": "a1", "snooze_minutes": 30})

    fake_client.alerts.asnooze.assert_awaited_once_with("a1", 30)


@pytest.mark.asyncio
async def test_statistics_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alerts_statistics")

    await tool.run({"hours": 48})

    fake_client.alerts.astatistics.assert_awaited_once_with(hours=48)


_TRIAGE_ALERT = {
    "id": "a1",
    "rule_id": "r1",
    "model_id": "m1",
    "alert_type": "drift",
    "severity": "high",
    "status": "active",
    "title": "PSI Breach",
    "message": "PSI above 0.25",
}


@pytest.mark.asyncio
async def test_get_triage_context_combines_alert_and_drift_reports(
    mcp_with_fake_session,
):
    mcp, fake_client = mcp_with_fake_session
    fake_client.alerts.aget_instance = AsyncMock(return_value=_TRIAGE_ALERT)
    fake_client.drift.aget_reports = AsyncMock(
        return_value=[
            {
                "id": "dr1",
                "model_id": "m1",
                "drift_score": 0.35,
                "drift_detected": True,
                "metrics": {
                    "features_drifted": 1,
                    "total_features": 4,
                    "feature_drifts": [],
                },
            }
        ]
    )
    tool = mcp._tool_manager.get_tool("whitebox_alerts_get_triage_context")

    result = await tool.run({"alert_id": "a1"})

    fake_client.alerts.aget_instance.assert_awaited_once_with("a1")
    fake_client.drift.aget_reports.assert_awaited_once_with("m1", limit=1, skip=0)
    assert result["alert"]["id"] == "a1"
    assert result["model_id"] == "m1"
    assert len(result["latest_drift_reports"]) == 1
    assert result["drift_context_error"] is None
    assert "1 of 4 features drifted" in result["plain_language_verdict"]


@pytest.mark.asyncio
async def test_get_triage_context_reports_drift_lookup_failure(
    mcp_with_fake_session,
):
    mcp, fake_client = mcp_with_fake_session
    fake_client.alerts.aget_instance = AsyncMock(return_value=_TRIAGE_ALERT)
    fake_client.drift.aget_reports = AsyncMock(side_effect=RuntimeError("timeout"))
    tool = mcp._tool_manager.get_tool("whitebox_alerts_get_triage_context")

    result = await tool.run({"alert_id": "a1"})

    assert result["latest_drift_reports"] == []
    assert result["drift_context_error"] == "RuntimeError: timeout"
    assert "Drift context unavailable" in result["plain_language_verdict"]
    assert "No persisted drift reports" not in result["plain_language_verdict"]
