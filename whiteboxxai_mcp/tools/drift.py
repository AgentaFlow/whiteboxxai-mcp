"""MCP tools for WhiteBoxXAI data drift detection (backend/api/v1/drift.py).

Concept-drift and per-model drift-config endpoints (backend/api/v1/
concept_drift.py, drift_config.py) are not yet wrapped -- tracked for a
later milestone alongside the rest of the non-M1 capability domains.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.auth.context import get_session
from whiteboxxai_mcp.errors import call_with_session
from whiteboxxai_mcp.ui_resources import ALERT_TRIAGE_CARD_URI, ui_tool_meta


def register(mcp: FastMCP) -> None:
    """Register all drift-detection tools on `mcp`."""

    @mcp.tool(name="whitebox_drift_detect")
    async def whitebox_drift_detect(
        model_id: str,
        window_size: int = 1000,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run an ad-hoc data drift check for a model (not persisted as a report).

        Compares the most recent `window_size` predictions against the
        model's baseline data using statistical tests (KS, chi-squared,
        Jensen-Shannon, PSI) per feature. This can take several seconds for
        large window sizes since it computes synchronously.

        Args:
            model_id: Registered model's ID
            window_size: Number of recent predictions to analyze (100-10000)
            feature_names: Optional subset of features to analyze (all if omitted)
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.drift.adetect(
                model_id, window_size=window_size, feature_names=feature_names
            ),
        )

    @mcp.tool(name="whitebox_drift_create_report")
    async def whitebox_drift_create_report(
        model_id: str, window_size: int = 1000
    ) -> dict[str, Any]:
        """Run drift analysis and persist the result as a drift report.

        Use this (rather than whitebox_drift_detect) when the result should
        be retrievable later via whitebox_drift_get_report / _list_reports.
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.drift.acreate_report(
                model_id, window_size=window_size
            ),
        )

    @mcp.tool(name="whitebox_drift_list_reports")
    async def whitebox_drift_list_reports(
        model_id: str, limit: int = 10, skip: int = 0
    ) -> list[dict[str, Any]]:
        """List drift reports for a model, most recent first."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.drift.aget_reports(model_id, limit=limit, skip=skip),
        )

    @mcp.tool(
        name="whitebox_drift_get_report",
        meta=ui_tool_meta(ALERT_TRIAGE_CARD_URI),
    )
    async def whitebox_drift_get_report(
        model_id: str, report_id: str
    ) -> dict[str, Any]:
        """Get a specific drift report, including per-feature statistics."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.drift.aget_report(model_id, report_id)
        )

    @mcp.tool(name="whitebox_drift_trend")
    async def whitebox_drift_trend(model_id: str, days: int = 7) -> dict[str, Any]:
        """Get a model's drift trend over time (1-90 days)."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.drift.aget_trend(model_id, days=days)
        )
