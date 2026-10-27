"""Behavior tests for whiteboxxai_mcp.tools.alert_intelligence (issue #224)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.tools import alert_intelligence


@pytest.fixture
def mcp_with_fake_session(monkeypatch):
    mcp = FastMCP(name="test")
    alert_intelligence.register(mcp)

    fake_client = MagicMock()
    fake_client.alert_intelligence.adetect_correlations = AsyncMock(
        return_value={"alert_id": "a1"}
    )
    fake_client.alert_intelligence.aget_correlation_clusters = AsyncMock(
        return_value={"num_clusters": 0}
    )
    fake_client.alert_intelligence.aanalyze_root_cause = AsyncMock(
        return_value={"alert_id": "a1"}
    )
    fake_client.alert_intelligence.averify_root_cause = AsyncMock(
        return_value={"id": 1, "verified": True}
    )
    fake_client.alert_intelligence.aget_similar_incidents = AsyncMock(
        return_value=[{"id": 1}]
    )
    fake_client.alert_intelligence.acalculate_priority = AsyncMock(
        return_value={"priority_score": 80.0}
    )
    fake_client.alert_intelligence.aadjust_priority = AsyncMock(
        return_value={"priority_score": 90.0}
    )
    fake_client.alert_intelligence.aget_priority_distribution = AsyncMock(
        return_value={"critical_count": 1}
    )
    fake_client.alert_intelligence.aget_priority_queue = AsyncMock(return_value=[])
    fake_client.alert_intelligence.adetect_patterns = AsyncMock(
        return_value={"total_patterns": 0}
    )
    fake_client.alert_intelligence.aget_pattern = AsyncMock(return_value={"id": 5})
    fake_client.alert_intelligence.adeactivate_pattern = AsyncMock(
        return_value={"id": 5, "is_active": False}
    )
    fake_client.alert_intelligence.aapply_fatigue_reduction = AsyncMock(
        return_value={"alerts_throttled": 0}
    )
    fake_client.alert_intelligence.aget_throttle_recommendations = AsyncMock(
        return_value=[]
    )
    fake_client.alert_intelligence.aml_group_alerts = AsyncMock(
        return_value={"clustering_method": "kmeans"}
    )
    fake_client.alert_intelligence.atemporal_group_alerts = AsyncMock(
        return_value={"total_alerts": 0}
    )
    fake_client.alert_intelligence.apredict_trends = AsyncMock(
        return_value={
            "predicted_alert_count": 12,
            "trend_direction": "increasing",
            "sample_size": 168,
        }
    )
    fake_client.alert_intelligence.adetect_seasonality = AsyncMock(
        return_value={"pattern_type": "daily"}
    )
    fake_client.alert_intelligence.aenhanced_trend_analysis = AsyncMock(
        return_value={"trend_summary": "stable"}
    )
    fake_client.alert_intelligence.aforecast_volume = AsyncMock(
        return_value={"forecast_period": "24 hours"}
    )

    class FakeSession:
        async def get_client(self, force_refresh: bool = False):
            return fake_client

    monkeypatch.setattr(alert_intelligence, "get_session", lambda: FakeSession())
    return mcp, fake_client


@pytest.mark.asyncio
async def test_detect_correlations_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_intelligence_detect_correlations")

    result = await tool.run({"alert_id": "a1"})

    fake_client.alert_intelligence.adetect_correlations.assert_awaited_once_with(
        "a1", look_back_hours=24, min_score=0.5
    )
    assert result == {"alert_id": "a1"}


@pytest.mark.asyncio
async def test_get_correlation_clusters_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool(
        "whitebox_alert_intelligence_get_correlation_clusters"
    )

    await tool.run({"min_score": 0.9})

    fake_client.alert_intelligence.aget_correlation_clusters.assert_awaited_once_with(
        min_cluster_size=3, min_score=0.9, hours_back=24
    )


@pytest.mark.asyncio
async def test_analyze_root_cause_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_intelligence_analyze_root_cause")

    await tool.run({"alert_id": "a1", "analyze_history": False})

    fake_client.alert_intelligence.aanalyze_root_cause.assert_awaited_once_with(
        "a1", analyze_history=False
    )


@pytest.mark.asyncio
async def test_verify_root_cause_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_intelligence_verify_root_cause")

    await tool.run({"root_cause_id": 1, "verified": True, "notes": "confirmed"})

    fake_client.alert_intelligence.averify_root_cause.assert_awaited_once_with(
        1, True, notes="confirmed"
    )


@pytest.mark.asyncio
async def test_get_similar_incidents_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool(
        "whitebox_alert_intelligence_get_similar_incidents"
    )

    await tool.run({"cause_type": "drift"})

    fake_client.alert_intelligence.aget_similar_incidents.assert_awaited_once_with(
        "drift", limit=10
    )


@pytest.mark.asyncio
async def test_calculate_priority_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_intelligence_calculate_priority")

    await tool.run({"alert_id": "a1"})

    fake_client.alert_intelligence.acalculate_priority.assert_awaited_once_with(
        "a1",
        business_impact_weight=0.3,
        urgency_weight=0.3,
        criticality_weight=0.25,
        history_weight=0.15,
    )


@pytest.mark.asyncio
async def test_adjust_priority_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_intelligence_adjust_priority")

    await tool.run(
        {
            "alert_id": "a1",
            "new_priority_score": 95.0,
            "new_priority_level": "critical",
            "adjustment_reason": "customer escalation",
        }
    )

    fake_client.alert_intelligence.aadjust_priority.assert_awaited_once_with(
        "a1", 95.0, "critical", "customer escalation"
    )


@pytest.mark.asyncio
async def test_get_priority_distribution_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool(
        "whitebox_alert_intelligence_get_priority_distribution"
    )

    await tool.run({})

    fake_client.alert_intelligence.aget_priority_distribution.assert_awaited_once_with(
        hours_back=24
    )


@pytest.mark.asyncio
async def test_get_priority_queue_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_intelligence_get_priority_queue")

    await tool.run({"limit": 5})

    fake_client.alert_intelligence.aget_priority_queue.assert_awaited_once_with(limit=5)


@pytest.mark.asyncio
async def test_detect_patterns_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_intelligence_detect_patterns")

    await tool.run({})

    fake_client.alert_intelligence.adetect_patterns.assert_awaited_once_with(
        hours_back=168
    )


@pytest.mark.asyncio
async def test_get_pattern_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_intelligence_get_pattern")

    await tool.run({"pattern_id": 5})

    fake_client.alert_intelligence.aget_pattern.assert_awaited_once_with(5)


@pytest.mark.asyncio
async def test_deactivate_pattern_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_intelligence_deactivate_pattern")

    await tool.run({"pattern_id": 5})

    fake_client.alert_intelligence.adeactivate_pattern.assert_awaited_once_with(5)


@pytest.mark.asyncio
async def test_apply_fatigue_reduction_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool(
        "whitebox_alert_intelligence_apply_fatigue_reduction"
    )

    await tool.run({"auto_snooze_threshold": 20})

    fake_client.alert_intelligence.aapply_fatigue_reduction.assert_awaited_once_with(
        enable_auto_throttling=True,
        enable_intelligent_grouping=True,
        enable_auto_snooze=True,
        similarity_threshold=0.8,
        grouping_window_minutes=60,
        auto_snooze_threshold=20,
    )


@pytest.mark.asyncio
async def test_get_throttle_recommendations_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool(
        "whitebox_alert_intelligence_get_throttle_recommendations"
    )

    await tool.run({})

    fake_client.alert_intelligence.aget_throttle_recommendations.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_ml_grouping_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_intelligence_ml_grouping")

    await tool.run({"clustering_method": "dbscan"})

    fake_client.alert_intelligence.aml_group_alerts.assert_awaited_once_with(
        clustering_method="dbscan", num_clusters=None
    )


@pytest.mark.asyncio
async def test_temporal_grouping_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_intelligence_temporal_grouping")

    await tool.run({})

    fake_client.alert_intelligence.atemporal_group_alerts.assert_awaited_once_with(
        time_window_minutes=60, similarity_threshold=0.7
    )


@pytest.mark.asyncio
async def test_predict_trends_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_intelligence_predict_trends")

    result = await tool.run({})

    fake_client.alert_intelligence.apredict_trends.assert_awaited_once_with(
        time_horizon="next_day"
    )
    assert result == {
        "predicted_alert_count": 12,
        "trend_direction": "increasing",
        "sample_size": 168,
    }


@pytest.mark.asyncio
async def test_detect_seasonality_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_intelligence_detect_seasonality")

    await tool.run({})

    fake_client.alert_intelligence.adetect_seasonality.assert_awaited_once_with(
        days_back=30
    )


@pytest.mark.asyncio
async def test_enhanced_trend_analysis_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool(
        "whitebox_alert_intelligence_enhanced_trend_analysis"
    )

    await tool.run({})

    fake_client.alert_intelligence.aenhanced_trend_analysis.assert_awaited_once_with(
        metric="alert_count", hours_back=168
    )


@pytest.mark.asyncio
async def test_forecast_volume_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_alert_intelligence_forecast_volume")

    await tool.run({})

    fake_client.alert_intelligence.aforecast_volume.assert_awaited_once_with(
        forecast_hours=24
    )
