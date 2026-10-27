"""Behavior tests for whiteboxxai_mcp.tools.llm_monitoring (issue #141)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.tools import llm_monitoring


@pytest.fixture
def mcp_with_fake_session(monkeypatch):
    mcp = FastMCP(name="test")
    llm_monitoring.register(mcp)

    fake_client = MagicMock()
    fake_client.llm.alog_call = AsyncMock(return_value={"id": "l1"})
    fake_client.llm.alog_calls_batch = AsyncMock(return_value={"count": 2})
    fake_client.llm.aget_log = AsyncMock(return_value={"id": "l1"})
    fake_client.llm.aquery_logs = AsyncMock(return_value={"logs": [], "total": 0})
    fake_client.llm.asession_logs = AsyncMock(return_value=[{"id": "l1"}])
    fake_client.llm.aget_stats = AsyncMock(return_value={"total_tokens": 100})
    fake_client.llm.aget_recent = AsyncMock(return_value=[{"id": "l1"}])
    fake_client.llm.acost_breakdown = AsyncMock(return_value={"total_cost": 1.0})
    fake_client.llm.aperformance = AsyncMock(return_value={"avg_latency_ms": 100.0})
    fake_client.llm.atrends_tokens = AsyncMock(return_value=[{"date": "2026-08-14"}])
    fake_client.llm.atrends_costs = AsyncMock(return_value=[{"date": "2026-08-14"}])
    fake_client.llm.ausage_stats = AsyncMock(return_value=[{"provider": "openai"}])
    fake_client.llm.acost_threshold_alert = AsyncMock(return_value={"exceeded": False})
    fake_client.llm.alatency_threshold_alert = AsyncMock(
        return_value={"exceeded": False}
    )
    fake_client.llm.aerror_rate_alert = AsyncMock(return_value={"exceeded": False})
    fake_client.llm.acleanup_logs = AsyncMock(return_value={"deleted_count": 0})

    class FakeSession:
        async def get_client(self, force_refresh: bool = False):
            return fake_client

    monkeypatch.setattr(llm_monitoring, "get_session", lambda: FakeSession())
    return mcp, fake_client


@pytest.mark.asyncio
async def test_log_call_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_monitoring_log_call")

    result = await tool.run(
        {
            "provider": "openai",
            "model_name": "gpt-4o",
            "prompt": "hi",
            "latency_ms": 120.0,
            "prompt_version_id": "pv1",
        }
    )

    fake_client.llm.alog_call.assert_awaited_once_with(
        provider="openai",
        model_name="gpt-4o",
        prompt="hi",
        latency_ms=120.0,
        model_id=None,
        completion=None,
        prompt_tokens=None,
        completion_tokens=None,
        finish_reason=None,
        temperature=None,
        max_tokens=None,
        status=None,
        error_message=None,
        user_id=None,
        session_id=None,
        request_id=None,
        environment=None,
        prompt_version_id="pv1",
    )
    assert result == {"id": "l1"}


@pytest.mark.asyncio
async def test_log_calls_batch_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_monitoring_log_calls_batch")

    logs = [
        {
            "provider": "openai",
            "model_name": "gpt-4o",
            "prompt": "hi",
            "latency_ms": 1.0,
        }
    ]
    await tool.run({"logs": logs})

    fake_client.llm.alog_calls_batch.assert_awaited_once_with(logs)


@pytest.mark.asyncio
async def test_get_log_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_monitoring_get_log")

    await tool.run({"log_id": "l1"})

    fake_client.llm.aget_log.assert_awaited_once_with("l1")


@pytest.mark.asyncio
async def test_query_logs_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_monitoring_query_logs")

    await tool.run({"model_id": "m1", "limit": 25})

    fake_client.llm.aquery_logs.assert_awaited_once_with(
        start_time=None,
        end_time=None,
        model_id="m1",
        provider=None,
        model_name=None,
        status=None,
        user_id=None,
        session_id=None,
        environment=None,
        min_cost=None,
        max_cost=None,
        min_latency_ms=None,
        max_latency_ms=None,
        skip=0,
        limit=25,
    )


@pytest.mark.asyncio
async def test_session_logs_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_monitoring_session_logs")

    await tool.run({"session_id": "s1"})

    fake_client.llm.asession_logs.assert_awaited_once_with("s1")


@pytest.mark.asyncio
async def test_usage_stats_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_monitoring_usage_stats")

    await tool.run({"model_id": "m1"})

    fake_client.llm.aget_stats.assert_awaited_once_with(
        model_id="m1", provider=None, environment=None, start_time=None, end_time=None
    )


@pytest.mark.asyncio
async def test_recent_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_monitoring_recent")

    await tool.run({"limit": 10})

    fake_client.llm.aget_recent.assert_awaited_once_with(
        limit=10, model_id=None, provider=None
    )


@pytest.mark.asyncio
async def test_cost_breakdown_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_monitoring_cost_breakdown")

    await tool.run({})

    fake_client.llm.acost_breakdown.assert_awaited_once_with(
        start_time=None, end_time=None, include_user=False, include_environment=False
    )


@pytest.mark.asyncio
async def test_performance_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_monitoring_performance")

    await tool.run({"model_id": "m1"})

    fake_client.llm.aperformance.assert_awaited_once_with(
        start_time=None, end_time=None, model_id="m1", provider=None
    )


@pytest.mark.asyncio
async def test_trends_tokens_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_monitoring_trends_tokens")

    await tool.run({"granularity": "week"})

    fake_client.llm.atrends_tokens.assert_awaited_once_with(
        start_time=None, end_time=None, granularity="week", model_id=None
    )


@pytest.mark.asyncio
async def test_trends_costs_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_monitoring_trends_costs")

    await tool.run({})

    fake_client.llm.atrends_costs.assert_awaited_once_with(
        start_time=None, end_time=None, granularity="day", provider=None
    )


@pytest.mark.asyncio
async def test_usage_stats_detailed_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_monitoring_usage_stats_detailed")

    await tool.run(
        {"start_time": "2026-08-01T00:00:00Z", "end_time": "2026-08-14T00:00:00Z"}
    )

    fake_client.llm.ausage_stats.assert_awaited_once_with(
        start_time="2026-08-01T00:00:00Z",
        end_time="2026-08-14T00:00:00Z",
        granularity="day",
        provider=None,
        model_name=None,
        model_id=None,
        user_id=None,
        environment=None,
    )


@pytest.mark.asyncio
async def test_cost_threshold_alert_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_monitoring_cost_threshold_alert")

    await tool.run({"threshold": 50.0})

    fake_client.llm.acost_threshold_alert.assert_awaited_once_with(
        period_minutes=60, threshold=50.0
    )


@pytest.mark.asyncio
async def test_latency_threshold_alert_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_monitoring_latency_threshold_alert")

    await tool.run({})

    fake_client.llm.alatency_threshold_alert.assert_awaited_once_with(
        period_minutes=60, threshold_ms=5000.0
    )


@pytest.mark.asyncio
async def test_error_rate_alert_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_monitoring_error_rate_alert")

    await tool.run({})

    fake_client.llm.aerror_rate_alert.assert_awaited_once_with(
        period_minutes=60, threshold_percent=10.0
    )


@pytest.mark.asyncio
async def test_cleanup_logs_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_monitoring_cleanup_logs")

    await tool.run({"days": 7})

    fake_client.llm.acleanup_logs.assert_awaited_once_with(days=7)
