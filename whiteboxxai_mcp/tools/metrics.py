"""MCP tools for model performance metrics (backend/api/v1/metrics.py)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.auth.context import get_session
from whiteboxxai_mcp.errors import call_with_session


def register(mcp: FastMCP) -> None:
    """Register all metrics tools on `mcp`."""

    @mcp.tool(name="whitebox_metrics_create")
    async def whitebox_metrics_create(
        model_id: str,
        metric_type: str,
        metric_name: str,
        metric_value: float,
        metric_metadata: dict[str, Any] | None = None,
        sample_size: int | None = None,
    ) -> dict[str, Any]:
        """Create a new metric record for a model."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.metrics.acreate(
                model_id,
                metric_type,
                metric_name,
                metric_value,
                metric_metadata=metric_metadata,
                sample_size=sample_size,
            ),
        )

    @mcp.tool(name="whitebox_metrics_create_batch")
    async def whitebox_metrics_create_batch(
        model_id: str, metrics: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Create multiple metrics at once (max 100). Each dict in `metrics`
        needs metric_type/metric_name/metric_value."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.metrics.acreate_batch(model_id, metrics)
        )

    @mcp.tool(name="whitebox_metrics_calculate_classification")
    async def whitebox_metrics_calculate_classification(
        y_true: list[Any],
        y_pred: list[Any],
        y_pred_proba: list[float] | None = None,
        average: str = "binary",
        pos_label: int = 1,
    ) -> dict[str, float]:
        """Calculate classification metrics (accuracy/precision/recall/F1/etc.)
        from predictions, without persisting anything."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.metrics.acalculate_classification(
                y_true,
                y_pred,
                y_pred_proba=y_pred_proba,
                average=average,
                pos_label=pos_label,
            ),
        )

    @mcp.tool(name="whitebox_metrics_calculate_regression")
    async def whitebox_metrics_calculate_regression(
        y_true: list[float], y_pred: list[float]
    ) -> dict[str, float]:
        """Calculate regression metrics (MAE/MSE/RMSE/R2/etc.) from
        predictions, without persisting anything."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.metrics.acalculate_regression(y_true, y_pred)
        )

    @mcp.tool(name="whitebox_metrics_get_model_metrics")
    async def whitebox_metrics_get_model_metrics(
        model_id: str,
        metric_type: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get metrics recorded for a model, with optional type/date filters."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.metrics.aget_model_metrics(
                model_id,
                metric_type=metric_type,
                start_date=start_date,
                end_date=end_date,
                skip=skip,
                limit=limit,
            ),
        )

    @mcp.tool(name="whitebox_metrics_latest")
    async def whitebox_metrics_latest(
        model_id: str, metric_type: str
    ) -> dict[str, Any]:
        """Get the most recent metric of a specific type for a model."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.metrics.alatest(model_id, metric_type)
        )

    @mcp.tool(name="whitebox_metrics_timeseries")
    async def whitebox_metrics_timeseries(
        model_id: str, metric_type: str, start_date: str, end_date: str
    ) -> dict[str, Any]:
        """Get time series data for a metric over a required date range."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.metrics.atimeseries(
                model_id, metric_type, start_date, end_date
            ),
        )

    @mcp.tool(name="whitebox_metrics_aggregate")
    async def whitebox_metrics_aggregate(
        model_id: str,
        period: str,
        metric_type: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        force_recompute: bool = False,
    ) -> list[dict[str, Any]]:
        """Get metrics aggregated over `period` (e.g. "daily", "weekly").
        Defaults to the last 30 days if no date range is given."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.metrics.aaggregate(
                model_id,
                period,
                metric_type=metric_type,
                start_date=start_date,
                end_date=end_date,
                force_recompute=force_recompute,
            ),
        )

    @mcp.tool(name="whitebox_metrics_trend")
    async def whitebox_metrics_trend(
        model_id: str, metric_type: str, lookback_days: int = 30
    ) -> dict[str, Any]:
        """Get trend statistics for a metric over `lookback_days`."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.metrics.atrend(
                model_id, metric_type, lookback_days=lookback_days
            ),
        )

    @mcp.tool(name="whitebox_metrics_rolling")
    async def whitebox_metrics_rolling(
        model_id: str, metric_type: str, window_days: int = 7
    ) -> dict[str, Any]:
        """Get rolling-window statistics for a metric over `window_days`."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.metrics.arolling(
                model_id, metric_type, window_days=window_days
            ),
        )

    @mcp.tool(name="whitebox_metrics_summary")
    async def whitebox_metrics_summary(model_id: str, days: int = 30) -> dict[str, Any]:
        """Get a summary of all metrics for a model over the last `days`."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.metrics.asummary(model_id, days=days)
        )

    @mcp.tool(name="whitebox_metrics_delete")
    async def whitebox_metrics_delete(model_id: str) -> dict[str, Any]:
        """Delete all metrics for a model."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.metrics.adelete(model_id)
        )
