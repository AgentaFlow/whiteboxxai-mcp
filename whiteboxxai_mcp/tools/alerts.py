"""MCP tools for alert rule configuration and management
(backend/api/v1/alerts.py). Only the core rule/instance CRUD surface is
covered here -- the larger ML-driven alert_intelligence.py (correlation,
root-cause analysis, pattern prediction) and alert_management.py
(escalation policies, bulk actions, comments) routers have their own tool
modules, `alert_intelligence.py` and `alert_management.py` (issue #224).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.auth.context import get_session
from whiteboxxai_mcp.errors import call_with_session
from whiteboxxai_mcp.ui_resources import (
    ALERT_TRIAGE_CARD_URI,
    build_triage_verdict,
    ui_tool_meta,
)


def register(mcp: FastMCP) -> None:
    """Register all alert-management tools on `mcp`."""

    @mcp.tool(name="whitebox_alerts_create_rule")
    async def whitebox_alerts_create_rule(
        name: str,
        alert_type: str,
        severity: str,
        conditions: list[dict[str, Any]],
        model_id: str | None = None,
        description: str | None = None,
        logic_operator: str = "AND",
        notification_channels: list[dict[str, Any]] | None = None,
        throttle_minutes: int = 60,
        cooldown_minutes: int = 0,
        is_active: bool = True,
    ) -> dict[str, Any]:
        """Create an alert rule with threshold or anomaly-based conditions.

        conditions: list of {metric_name, operator (gt/gte/lt/lte/eq/ne),
        threshold, window_minutes?, aggregation?}, combined by logic_operator.
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alerts.acreate(
                name,
                alert_type,
                severity,
                conditions,
                model_id=model_id,
                description=description,
                logic_operator=logic_operator,
                notification_channels=notification_channels,
                throttle_minutes=throttle_minutes,
                cooldown_minutes=cooldown_minutes,
                is_active=is_active,
            ),
        )

    @mcp.tool(name="whitebox_alerts_list_rules")
    async def whitebox_alerts_list_rules(
        model_id: str | None = None,
        alert_type: str | None = None,
        is_active: bool | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List alert rules for the caller's organization."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alerts.alist(
                model_id=model_id,
                alert_type=alert_type,
                is_active=is_active,
                skip=skip,
                limit=limit,
            ),
        )

    @mcp.tool(name="whitebox_alerts_get_rule")
    async def whitebox_alerts_get_rule(rule_id: str) -> dict[str, Any]:
        """Get a specific alert rule by ID."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.alerts.aget_rule(rule_id)
        )

    @mcp.tool(name="whitebox_alerts_update_rule")
    async def whitebox_alerts_update_rule(
        rule_id: str,
        name: str | None = None,
        description: str | None = None,
        severity: str | None = None,
        is_active: bool | None = None,
        conditions: list[dict[str, Any]] | None = None,
        throttle_minutes: int | None = None,
        cooldown_minutes: int | None = None,
    ) -> dict[str, Any]:
        """Update an alert rule. Only non-null arguments are changed."""
        session = get_session()
        kwargs: dict[str, Any] = {}
        if name is not None:
            kwargs["name"] = name
        if description is not None:
            kwargs["description"] = description
        if severity is not None:
            kwargs["severity"] = severity
        if is_active is not None:
            kwargs["is_active"] = is_active
        if conditions is not None:
            kwargs["conditions"] = conditions
        if throttle_minutes is not None:
            kwargs["throttle_minutes"] = throttle_minutes
        if cooldown_minutes is not None:
            kwargs["cooldown_minutes"] = cooldown_minutes
        return await call_with_session(
            session, lambda client: client.alerts.aupdate_rule(rule_id, **kwargs)
        )

    @mcp.tool(name="whitebox_alerts_delete_rule")
    async def whitebox_alerts_delete_rule(rule_id: str) -> dict[str, Any]:
        """Delete an alert rule."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.alerts.adelete_rule(rule_id)
        )

    @mcp.tool(name="whitebox_alerts_evaluate_rule")
    async def whitebox_alerts_evaluate_rule(
        rule_id: str,
        metric_values: dict[str, Any] | None = None,
        model_id: str | None = None,
        historical_values: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Test whether an alert rule would trigger given metric values,
        without creating a real alert instance. Useful for validating a
        rule before activation."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alerts.aevaluate_rule(
                rule_id,
                metric_values=metric_values,
                model_id=model_id,
                historical_values=historical_values,
                metadata=metadata,
            ),
        )

    @mcp.tool(name="whitebox_alerts_list_instances")
    async def whitebox_alerts_list_instances(
        rule_id: str | None = None,
        model_id: str | None = None,
        alert_type: str | None = None,
        severity: str | None = None,
        status: str | None = None,
        hours: int | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List triggered alert instances with optional filters."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alerts.alist_instances(
                rule_id=rule_id,
                model_id=model_id,
                alert_type=alert_type,
                severity=severity,
                status=status,
                hours=hours,
                skip=skip,
                limit=limit,
            ),
        )

    @mcp.tool(
        name="whitebox_alerts_get_instance",
        meta=ui_tool_meta(ALERT_TRIAGE_CARD_URI),
    )
    async def whitebox_alerts_get_instance(alert_id: str) -> dict[str, Any]:
        """Get a specific triggered alert instance by ID."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.alerts.aget_instance(alert_id)
        )

    @mcp.tool(
        name="whitebox_alerts_get_triage_context",
        meta=ui_tool_meta(ALERT_TRIAGE_CARD_URI),
    )
    async def whitebox_alerts_get_triage_context(
        alert_id: str,
        drift_report_limit: int = 1,
    ) -> dict[str, Any]:
        """Get a triggered alert instance together with its model's latest
        drift reports and a plain-language triage verdict.

        Args:
            alert_id: Triggered alert instance ID
            drift_report_limit: Number of recent drift reports to include when
                the alert is associated with a model (default 1)
        """
        session = get_session()

        async def _fetch_triage(client: Any) -> dict[str, Any]:
            alert = await client.alerts.aget_instance(alert_id)
            model_id = alert.get("model_id") if isinstance(alert, dict) else None
            drift_reports: list[dict[str, Any]] = []
            drift_error: str | None = None
            if model_id:
                # The alert itself loaded, so a drift failure shouldn't fail the
                # whole triage view -- but it must be reported as a failure.
                # Swallowing it into an empty list would read as "no drift
                # reports exist", which is a claim nobody checked.
                try:
                    reports = await client.drift.aget_reports(
                        model_id, limit=drift_report_limit, skip=0
                    )
                    if isinstance(reports, list):
                        drift_reports = reports
                except Exception as exc:
                    drift_error = f"{type(exc).__name__}: {exc}"
            alert_dict = alert if isinstance(alert, dict) else {"id": alert_id}
            return {
                "alert": alert_dict,
                "model_id": model_id,
                "latest_drift_reports": drift_reports,
                "drift_context_error": drift_error,
                "plain_language_verdict": build_triage_verdict(
                    alert_dict, drift_reports, drift_error=drift_error
                ),
            }

        return await call_with_session(session, _fetch_triage)

    @mcp.tool(name="whitebox_alerts_acknowledge")
    async def whitebox_alerts_acknowledge(
        alert_id: str, user_id: str, notes: str | None = None
    ) -> dict[str, Any]:
        """Acknowledge an active alert instance."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alerts.aacknowledge(
                alert_id, user_id=user_id, notes=notes
            ),
        )

    @mcp.tool(name="whitebox_alerts_resolve")
    async def whitebox_alerts_resolve(
        alert_id: str, user_id: str, resolution_notes: str | None = None
    ) -> dict[str, Any]:
        """Resolve an alert instance."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alerts.aresolve(
                alert_id, user_id=user_id, resolution_notes=resolution_notes
            ),
        )

    @mcp.tool(name="whitebox_alerts_snooze")
    async def whitebox_alerts_snooze(
        alert_id: str, snooze_minutes: int
    ) -> dict[str, Any]:
        """Snooze an active alert instance for `snooze_minutes` (max 1 week)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alerts.asnooze(alert_id, snooze_minutes),
        )

    @mcp.tool(name="whitebox_alerts_statistics")
    async def whitebox_alerts_statistics(hours: int = 24) -> dict[str, Any]:
        """Get alert statistics for the organization (counts, severity
        distribution, mean time to acknowledge/resolve) over the last `hours`."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.alerts.astatistics(hours=hours)
        )
