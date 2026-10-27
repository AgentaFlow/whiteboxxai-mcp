"""Behavior tests for whiteboxxai_mcp.tools.predictions (issue #227).

Brings the M1 prediction-logging domain up to the same behavioral-delegation
bar as the M2/#137 domains (see test_tools_explanations.py, the template
this file follows).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.tools import predictions


@pytest.fixture
def mcp_with_fake_session(monkeypatch):
    mcp = FastMCP(name="test")
    predictions.register(mcp)

    fake_client = MagicMock()
    fake_client.predictions.alog = AsyncMock(
        return_value={"prediction_id": "p1", "model_id": "m1"}
    )
    fake_client.predictions.alog_batch = AsyncMock(
        return_value={"total": 2, "logged": 2, "failed": 0, "errors": []}
    )
    fake_client.predictions.aget = AsyncMock(
        return_value={"prediction_id": "p1", "model_id": "m1"}
    )
    fake_client.predictions.aquery = AsyncMock(return_value=[{"prediction_id": "p1"}])
    fake_client.predictions.aget_stats = AsyncMock(
        return_value={"model_id": "m1", "count": 100, "avg_latency_ms": 12.5}
    )
    fake_client.predictions.aget_recent = AsyncMock(
        return_value=[{"prediction_id": "p1"}]
    )

    class FakeSession:
        """Minimal stand-in for SessionManager (mirrors test_error_mapping.py)."""

        async def get_client(self, force_refresh: bool = False):
            return fake_client

    monkeypatch.setattr(predictions, "get_session", lambda: FakeSession())
    return mcp, fake_client


@pytest.mark.asyncio
async def test_log_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_predictions_log")

    result = await tool.run(
        {
            "model_id": "m1",
            "input_data": {"age": 42},
            "output_data": {"approved": True},
        }
    )

    fake_client.predictions.alog.assert_awaited_once_with(
        model_id="m1",
        input_data={"age": 42},
        output_data={"approved": True},
        prediction_id=None,
        latency_ms=None,
        metadata=None,
    )
    assert result == {"prediction_id": "p1", "model_id": "m1"}


@pytest.mark.asyncio
async def test_log_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_predictions_log")

    await tool.run(
        {
            "model_id": "m1",
            "input_data": {"age": 42},
            "output_data": {"approved": True},
            "prediction_id": "custom-p1",
            "latency_ms": 15.2,
            "metadata": {"source": "batch-job"},
        }
    )

    fake_client.predictions.alog.assert_awaited_once_with(
        model_id="m1",
        input_data={"age": 42},
        output_data={"approved": True},
        prediction_id="custom-p1",
        latency_ms=15.2,
        metadata={"source": "batch-job"},
    )


@pytest.mark.asyncio
async def test_log_batch_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_predictions_log_batch")

    predictions_payload = [
        {"input_data": {"age": 42}, "output_data": {"approved": True}},
        {"input_data": {"age": 30}, "output_data": {"approved": False}},
    ]
    result = await tool.run({"model_id": "m1", "predictions": predictions_payload})

    fake_client.predictions.alog_batch.assert_awaited_once_with(
        "m1", predictions_payload
    )
    assert result == {"total": 2, "logged": 2, "failed": 0, "errors": []}


@pytest.mark.asyncio
async def test_get_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_predictions_get")

    result = await tool.run({"prediction_id": "p1"})

    fake_client.predictions.aget.assert_awaited_once_with("p1")
    assert result == {"prediction_id": "p1", "model_id": "m1"}


@pytest.mark.asyncio
async def test_query_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_predictions_query")

    result = await tool.run({"model_id": "m1"})

    fake_client.predictions.aquery.assert_awaited_once_with(
        model_id="m1", start_time=None, end_time=None, limit=100, offset=0
    )
    assert result == [{"prediction_id": "p1"}]


@pytest.mark.asyncio
async def test_query_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_predictions_query")

    await tool.run(
        {
            "model_id": "m1",
            "start_time": "2026-08-01T00:00:00Z",
            "end_time": "2026-08-24T00:00:00Z",
            "limit": 25,
            "offset": 50,
        }
    )

    fake_client.predictions.aquery.assert_awaited_once_with(
        model_id="m1",
        start_time="2026-08-01T00:00:00Z",
        end_time="2026-08-24T00:00:00Z",
        limit=25,
        offset=50,
    )


@pytest.mark.asyncio
async def test_stats_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_predictions_stats")

    result = await tool.run({"model_id": "m1"})

    fake_client.predictions.aget_stats.assert_awaited_once_with(
        "m1", start_time=None, end_time=None
    )
    assert result == {"model_id": "m1", "count": 100, "avg_latency_ms": 12.5}


@pytest.mark.asyncio
async def test_stats_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_predictions_stats")

    await tool.run(
        {
            "model_id": "m1",
            "start_time": "2026-08-01T00:00:00Z",
            "end_time": "2026-08-24T00:00:00Z",
        }
    )

    fake_client.predictions.aget_stats.assert_awaited_once_with(
        "m1",
        start_time="2026-08-01T00:00:00Z",
        end_time="2026-08-24T00:00:00Z",
    )


@pytest.mark.asyncio
async def test_recent_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_predictions_recent")

    result = await tool.run({"model_id": "m1"})

    fake_client.predictions.aget_recent.assert_awaited_once_with("m1", limit=10)
    assert result == [{"prediction_id": "p1"}]


@pytest.mark.asyncio
async def test_recent_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_predictions_recent")

    await tool.run({"model_id": "m1", "limit": 5})

    fake_client.predictions.aget_recent.assert_awaited_once_with("m1", limit=5)
