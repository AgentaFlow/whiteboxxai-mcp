"""MCP tools for alert history, analytics, lifecycle, and comments
(backend/api/v1/alert_management.py; issue #224).

The deferred sibling of alerts.py's core rule/instance CRUD (see that
module's precedent, issue #141). Only 13 of the router's 19 endpoints are
exposed here -- escalation-policy CRUD (create/list/get/update/delete,
5 endpoints) and bulk_alert_action (1 endpoint) are admin-console-shaped,
not agent-shaped, per issue #224's scoping decision. They remain
available programmatically via AlertManagementResource in
sdk/whiteboxxai/resources.py.

acknowledge/resolve/snooze here hit a different backend URL
(/alert-management/alerts/{id}/...) than alerts.py's same-named tools
(/alerts/instances/{id}/...) -- they are not duplicates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.auth.context import get_session
from whiteboxxai_mcp.errors import call_with_session


def register(mcp: FastMCP) -> None:
    """Register the agent-facing alert-management tools on `mcp`."""

    # ==================== Alert History ====================

    @mcp.tool(name="whitebox_alert_management_get_history")
    async def whitebox_alert_management_get_history(alert_id: str) -> dict[str, Any]:
        """Get the full action timeline (triggered/acknowledged/snoozed/
        resolved/escalated/commented) for an alert."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.alert_management.aget_history(alert_id)
        )

    # ==================== Alert Analytics ====================

    @mcp.tool(name="whitebox_alert_management_get_analytics")
    async def whitebox_alert_management_get_analytics(
        start_time: str | None = None, end_time: str | None = None
    ) -> dict[str, Any]:
        """Get comprehensive alert analytics (counts, MTTA/MTTR,
        distributions, escalation metrics) for the caller's organization.
        `start_time`/`end_time` are ISO 8601 timestamps."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_management.aget_analytics(
                start_time=start_time, end_time=end_time
            ),
        )

    @mcp.tool(name="whitebox_alert_management_get_trend")
    async def whitebox_alert_management_get_trend(
        start: str,
        end: str,
        interval: str = "hour",
        group_by: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get alert volume as a time series between `start` and `end`
        (ISO 8601 timestamps), bucketed by `interval` (hour/day/week)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_management.aget_trend(
                start, end, interval=interval, group_by=group_by, filters=filters
            ),
        )

    @mcp.tool(name="whitebox_alert_management_get_velocity")
    async def whitebox_alert_management_get_velocity() -> dict[str, Any]:
        """Get alert generation rate trends (hour-over-hour, day-over-day,
        rolling averages) for the caller's organization."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.alert_management.aget_velocity()
        )

    @mcp.tool(name="whitebox_alert_management_get_grouping")
    async def whitebox_alert_management_get_grouping(
        group_by: str,
    ) -> dict[str, Any]:
        """Group active alerts by a dimension: "type", "severity", "model",
        or "rule" -- useful for spotting hotspots."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_management.aget_grouping(group_by),
        )

    @mcp.tool(name="whitebox_alert_management_detect_fatigue")
    async def whitebox_alert_management_detect_fatigue() -> dict[str, Any]:
        """Detect alert-fatigue indicators and get recommendations for
        reducing alert noise."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.alert_management.adetect_fatigue()
        )

    # ==================== Alert Lifecycle Management ====================

    @mcp.tool(name="whitebox_alert_management_acknowledge")
    async def whitebox_alert_management_acknowledge(
        alert_id: str, notes: str | None = None
    ) -> dict[str, Any]:
        """Acknowledge an alert (hits /alert-management/alerts/{id}/acknowledge;
        distinct from whitebox_alerts_acknowledge)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_management.aacknowledge(alert_id, notes=notes),
        )

    @mcp.tool(name="whitebox_alert_management_resolve")
    async def whitebox_alert_management_resolve(
        alert_id: str, resolution_notes: str | None = None
    ) -> dict[str, Any]:
        """Resolve an alert (hits /alert-management/alerts/{id}/resolve;
        distinct from whitebox_alerts_resolve)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_management.aresolve(
                alert_id, resolution_notes=resolution_notes
            ),
        )

    @mcp.tool(name="whitebox_alert_management_snooze")
    async def whitebox_alert_management_snooze(
        alert_id: str, snooze_minutes: int
    ) -> dict[str, Any]:
        """Snooze an alert for `snooze_minutes` (max 1 week; hits
        /alert-management/alerts/{id}/snooze, distinct from
        whitebox_alerts_snooze)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_management.asnooze(alert_id, snooze_minutes),
        )

    # ==================== Alert Comments ====================

    @mcp.tool(name="whitebox_alert_management_create_comment")
    async def whitebox_alert_management_create_comment(
        alert_id: str, comment: str, comment_type: str = "note"
    ) -> dict[str, Any]:
        """Add a comment to an alert (comment_type: note/action/resolution)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_management.acreate_comment(
                alert_id, comment, comment_type=comment_type
            ),
        )

    @mcp.tool(name="whitebox_alert_management_get_comments")
    async def whitebox_alert_management_get_comments(
        alert_id: str,
    ) -> list[dict[str, Any]]:
        """List all comments on an alert."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.alert_management.aget_comments(alert_id)
        )

    # ==================== Maintenance ====================

    @mcp.tool(name="whitebox_alert_management_process_pending_escalations")
    async def whitebox_alert_management_process_pending_escalations() -> dict[str, Any]:
        """Manually trigger processing of pending escalations (normally a
        scheduled background task)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_management.aprocess_pending_escalations(),
        )

    @mcp.tool(name="whitebox_alert_management_unsnooze_expired_alerts")
    async def whitebox_alert_management_unsnooze_expired_alerts() -> dict[str, Any]:
        """Manually trigger reactivation of alerts whose snooze period has
        expired (normally a scheduled background task)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_management.aunsnooze_expired_alerts(),
        )
