"""Behavior tests for whiteboxxai_mcp.tools.metrics (issue #141)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.tools import metrics


@pytest.fixture
def mcp_with_fake_session(monkeypatch):
    mcp = FastMCP(name="test")
    metrics.register(mcp)

    fake_client = MagicMock()
    fake_client.metrics.acreate = AsyncMock(return_value={"id": "m1"})
    fake_client.metrics.acreate_batch = AsyncMock(return_value=[{"id": "m1"}])
    fake_client.metrics.acalculate_classification = AsyncMock(
        return_value={"accuracy": 0.9}
    )
    fake_client.metrics.acalculate_regression = AsyncMock(return_value={"mae": 0.1})
    fake_client.metrics.aget_model_metrics = AsyncMock(return_value=[{"id": "m1"}])
    fake_client.metrics.alatest = AsyncMock(return_value={"id": "m1"})
    fake_client.metrics.atimeseries = AsyncMock(return_value={"data_points": []})
    fake_client.metrics.aaggregate = AsyncMock(return_value=[{"period": "daily"}])
    fake_client.metrics.atrend = AsyncMock(return_value={"direction": "stable"})
    fake_client.metrics.arolling = AsyncMock(return_value={"mean": 0.9})
    fake_client.metrics.asummary = AsyncMock(return_value={"total_metrics": 1})
    fake_client.metrics.adelete = AsyncMock(return_value={"deleted": True})

    class FakeSession:
        async def get_client(self, force_refresh: bool = False):
            return fake_client

    monkeypatch.setattr(metrics, "get_session", lambda: FakeSession())
    return mcp, fake_client


@pytest.mark.asyncio
async def test_create_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_metrics_create")

    result = await tool.run(
        {
            "model_id": "m1",
            "metric_type": "accuracy",
            "metric_name": "accuracy",
            "metric_value": 0.9,
        }
    )

    fake_client.metrics.acreate.assert_awaited_once_with(
        "m1", "accuracy", "accuracy", 0.9, metric_metadata=None, sample_size=None
    )
    assert result == {"id": "m1"}


@pytest.mark.asyncio
async def test_create_batch_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_metrics_create_batch")

    metrics_payload = [
        {"metric_type": "accuracy", "metric_name": "acc", "metric_value": 0.9}
    ]
    await tool.run({"model_id": "m1", "metrics": metrics_payload})

    fake_client.metrics.acreate_batch.assert_awaited_once_with("m1", metrics_payload)


@pytest.mark.asyncio
async def test_calculate_classification_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_metrics_calculate_classification")

    await tool.run({"y_true": [1, 0], "y_pred": [1, 1]})

    fake_client.metrics.acalculate_classification.assert_awaited_once_with(
        [1, 0], [1, 1], y_pred_proba=None, average="binary", pos_label=1
    )


@pytest.mark.asyncio
async def test_calculate_regression_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_metrics_calculate_regression")

    await tool.run({"y_true": [1.0, 2.0], "y_pred": [1.1, 1.9]})

    fake_client.metrics.acalculate_regression.assert_awaited_once_with(
        [1.0, 2.0], [1.1, 1.9]
    )


@pytest.mark.asyncio
async def test_get_model_metrics_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_metrics_get_model_metrics")

    await tool.run({"model_id": "m1"})

    fake_client.metrics.aget_model_metrics.assert_awaited_once_with(
        "m1", metric_type=None, start_date=None, end_date=None, skip=0, limit=100
    )


@pytest.mark.asyncio
async def test_latest_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_metrics_latest")

    await tool.run({"model_id": "m1", "metric_type": "accuracy"})

    fake_client.metrics.alatest.assert_awaited_once_with("m1", "accuracy")


@pytest.mark.asyncio
async def test_timeseries_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_metrics_timeseries")

    await tool.run(
        {
            "model_id": "m1",
            "metric_type": "accuracy",
            "start_date": "2026-08-01",
            "end_date": "2026-08-14",
        }
    )

    fake_client.metrics.atimeseries.assert_awaited_once_with(
        "m1", "accuracy", "2026-08-01", "2026-08-14"
    )


@pytest.mark.asyncio
async def test_aggregate_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_metrics_aggregate")

    await tool.run({"model_id": "m1", "period": "daily"})

    fake_client.metrics.aaggregate.assert_awaited_once_with(
        "m1",
        "daily",
        metric_type=None,
        start_date=None,
        end_date=None,
        force_recompute=False,
    )


@pytest.mark.asyncio
async def test_trend_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_metrics_trend")

    await tool.run({"model_id": "m1", "metric_type": "accuracy"})

    fake_client.metrics.atrend.assert_awaited_once_with(
        "m1", "accuracy", lookback_days=30
    )


@pytest.mark.asyncio
async def test_rolling_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_metrics_rolling")

    await tool.run({"model_id": "m1", "metric_type": "accuracy"})

    fake_client.metrics.arolling.assert_awaited_once_with(
        "m1", "accuracy", window_days=7
    )


@pytest.mark.asyncio
async def test_summary_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_metrics_summary")

    await tool.run({"model_id": "m1"})

    fake_client.metrics.asummary.assert_awaited_once_with("m1", days=30)


@pytest.mark.asyncio
async def test_delete_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_metrics_delete")

    await tool.run({"model_id": "m1"})

    fake_client.metrics.adelete.assert_awaited_once_with("m1")
