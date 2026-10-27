"""MCP tools for WhiteBoxXAI prediction logging (backend/api/v1/predictions.py)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.auth.context import get_session
from whiteboxxai_mcp.errors import call_with_session


def register(mcp: FastMCP) -> None:
    """Register all prediction-logging tools on `mcp`."""

    @mcp.tool(name="whitebox_predictions_log")
    async def whitebox_predictions_log(
        model_id: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        prediction_id: str | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Log a single model prediction for monitoring.

        Args:
            model_id: Registered model's ID
            input_data: Input features as a JSON object
            output_data: The model's prediction/output as a JSON object
            prediction_id: Optional external prediction ID (generated if omitted)
            latency_ms: Prediction latency in milliseconds
            metadata: Additional free-form metadata
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.predictions.alog(
                model_id=model_id,
                input_data=input_data,
                output_data=output_data,
                prediction_id=prediction_id,
                latency_ms=latency_ms,
                metadata=metadata,
            ),
        )

    @mcp.tool(name="whitebox_predictions_log_batch")
    async def whitebox_predictions_log_batch(
        model_id: str, predictions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Log up to 1000 predictions in a single batch call.

        Args:
            model_id: Registered model's ID
            predictions: List of objects, each with "input_data" and
                "output_data" keys (and optionally "prediction_id",
                "latency_ms", "metadata")

        Returns:
            Summary: {total, logged, failed, errors}
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.predictions.alog_batch(model_id, predictions),
        )

    @mcp.tool(name="whitebox_predictions_get")
    async def whitebox_predictions_get(prediction_id: str) -> dict[str, Any]:
        """Get a single logged prediction by its prediction_id."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.predictions.aget(prediction_id)
        )

    @mcp.tool(name="whitebox_predictions_query")
    async def whitebox_predictions_query(
        model_id: str,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query logged predictions for a model, optionally within a time range.

        Args:
            model_id: Registered model's ID (required)
            start_time: ISO-8601 timestamp; only predictions after this time
            end_time: ISO-8601 timestamp; only predictions before this time
            limit: Maximum results to return (1-1000)
            offset: Number of results to skip
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.predictions.aquery(
                model_id=model_id,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                offset=offset,
            ),
        )

    @mcp.tool(name="whitebox_predictions_stats")
    async def whitebox_predictions_stats(
        model_id: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
        """Get prediction volume/latency statistics for a model.

        Defaults to the trailing 24 hours if no time range is given.
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.predictions.aget_stats(
                model_id, start_time=start_time, end_time=end_time
            ),
        )

    @mcp.tool(name="whitebox_predictions_recent")
    async def whitebox_predictions_recent(
        model_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get the most recent predictions for a model (max 100)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.predictions.aget_recent(model_id, limit=limit),
        )
