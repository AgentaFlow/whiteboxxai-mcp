"""MCP tools for ML-driven alert intelligence
(backend/api/v1/alert_intelligence.py; issue #224).

Covers correlation analysis, root-cause analysis, priority scoring,
pattern detection, fatigue reduction, ML/temporal grouping, and trend
analysis -- the deferred sibling of alerts.py's core rule/instance CRUD
(see that module's precedent, issue #141).

Every tool derives organization scoping from the caller's own
authenticated session (see AlertIntelligenceResource in
sdk/whiteboxxai/resources.py) -- there is no organization_id argument to
pass, matching alert_management.py's tools (issue #295).

`predict_trends`, `detect_seasonality`, and `forecast_volume` compute real
statistics (numpy-based linear trend + hour-of-day seasonality) from the
organization's actual alert history, not a full forecasting model (no
ARIMA/Prophet) -- see each tool's docstring for what's actually computed
(issue #296).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.auth.context import get_session
from whiteboxxai_mcp.errors import call_with_session


def register(mcp: FastMCP) -> None:
    """Register all alert-intelligence tools on `mcp`."""

    # ==================== Alert Correlation ====================

    @mcp.tool(name="whitebox_alert_intelligence_detect_correlations")
    async def whitebox_alert_intelligence_detect_correlations(
        alert_id: str, look_back_hours: int = 24, min_score: float = 0.5
    ) -> dict[str, Any]:
        """Detect correlations for an alert (temporal, model, feature, and
        pattern-based relationships to other alerts)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.adetect_correlations(
                alert_id, look_back_hours=look_back_hours, min_score=min_score
            ),
        )

    @mcp.tool(name="whitebox_alert_intelligence_get_correlation_clusters")
    async def whitebox_alert_intelligence_get_correlation_clusters(
        min_cluster_size: int = 3,
        min_score: float = 0.7,
        hours_back: int = 24,
    ) -> dict[str, Any]:
        """Get clusters of highly correlated alerts -- useful for spotting
        widespread issues or cascading failures."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.aget_correlation_clusters(
                min_cluster_size=min_cluster_size,
                min_score=min_score,
                hours_back=hours_back,
            ),
        )

    # ==================== Root Cause Analysis ====================

    @mcp.tool(name="whitebox_alert_intelligence_analyze_root_cause")
    async def whitebox_alert_intelligence_analyze_root_cause(
        alert_id: str, analyze_history: bool = True
    ) -> dict[str, Any]:
        """Analyze potential root causes (data drift, model degradation,
        feature anomalies, system issues) for an alert."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.aanalyze_root_cause(
                alert_id, analyze_history=analyze_history
            ),
        )

    @mcp.tool(name="whitebox_alert_intelligence_verify_root_cause")
    async def whitebox_alert_intelligence_verify_root_cause(
        root_cause_id: int, verified: bool, notes: str | None = None
    ) -> dict[str, Any]:
        """Mark a previously computed root cause as verified or rejected."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.averify_root_cause(
                root_cause_id, verified, notes=notes
            ),
        )

    @mcp.tool(name="whitebox_alert_intelligence_get_similar_incidents")
    async def whitebox_alert_intelligence_get_similar_incidents(
        cause_type: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Find past alerts sharing the same root-cause type -- useful for
        learning from historical resolutions."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.aget_similar_incidents(
                cause_type, limit=limit
            ),
        )

    # ==================== Alert Priority Ranking ====================

    @mcp.tool(name="whitebox_alert_intelligence_calculate_priority")
    async def whitebox_alert_intelligence_calculate_priority(
        alert_id: str,
        business_impact_weight: float = 0.3,
        urgency_weight: float = 0.3,
        criticality_weight: float = 0.25,
        history_weight: float = 0.15,
    ) -> dict[str, Any]:
        """Compute a weighted priority score (0-100) and level
        (critical/high/medium/low) for an alert."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.acalculate_priority(
                alert_id,
                business_impact_weight=business_impact_weight,
                urgency_weight=urgency_weight,
                criticality_weight=criticality_weight,
                history_weight=history_weight,
            ),
        )

    @mcp.tool(name="whitebox_alert_intelligence_adjust_priority")
    async def whitebox_alert_intelligence_adjust_priority(
        alert_id: str,
        new_priority_score: float,
        new_priority_level: str,
        adjustment_reason: str,
    ) -> dict[str, Any]:
        """Manually override a calculated priority. `adjustment_reason` is
        required justification for the override."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.aadjust_priority(
                alert_id, new_priority_score, new_priority_level, adjustment_reason
            ),
        )

    @mcp.tool(name="whitebox_alert_intelligence_get_priority_distribution")
    async def whitebox_alert_intelligence_get_priority_distribution(
        hours_back: int = 24,
    ) -> dict[str, Any]:
        """Get a breakdown of alerts by priority level."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.aget_priority_distribution(
                hours_back=hours_back
            ),
        )

    @mcp.tool(name="whitebox_alert_intelligence_get_priority_queue")
    async def whitebox_alert_intelligence_get_priority_queue(
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get the prioritized alert queue for triage and assignment."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.aget_priority_queue(limit=limit),
        )

    # ==================== Pattern Detection ====================

    @mcp.tool(name="whitebox_alert_intelligence_detect_patterns")
    async def whitebox_alert_intelligence_detect_patterns(
        hours_back: int = 168,
    ) -> dict[str, Any]:
        """Detect recurring temporal/frequency/correlation/anomaly patterns
        in alerts."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.adetect_patterns(
                hours_back=hours_back
            ),
        )

    @mcp.tool(name="whitebox_alert_intelligence_get_pattern")
    async def whitebox_alert_intelligence_get_pattern(
        pattern_id: int,
    ) -> dict[str, Any]:
        """Get a specific alert pattern's details."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.alert_intelligence.aget_pattern(pattern_id)
        )

    @mcp.tool(name="whitebox_alert_intelligence_deactivate_pattern")
    async def whitebox_alert_intelligence_deactivate_pattern(
        pattern_id: int,
    ) -> dict[str, Any]:
        """Mark a pattern inactive (resolved or no longer relevant)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.adeactivate_pattern(pattern_id),
        )

    # ==================== Fatigue Reduction ====================

    @mcp.tool(name="whitebox_alert_intelligence_apply_fatigue_reduction")
    async def whitebox_alert_intelligence_apply_fatigue_reduction(
        enable_auto_throttling: bool = True,
        enable_intelligent_grouping: bool = True,
        enable_auto_snooze: bool = True,
        similarity_threshold: float = 0.8,
        grouping_window_minutes: int = 60,
        auto_snooze_threshold: int = 10,
    ) -> dict[str, Any]:
        """Apply automated throttling/grouping/snoozing to reduce alert
        noise. Returns a summary of actions taken and recommendations."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.aapply_fatigue_reduction(
                enable_auto_throttling=enable_auto_throttling,
                enable_intelligent_grouping=enable_intelligent_grouping,
                enable_auto_snooze=enable_auto_snooze,
                similarity_threshold=similarity_threshold,
                grouping_window_minutes=grouping_window_minutes,
                auto_snooze_threshold=auto_snooze_threshold,
            ),
        )

    @mcp.tool(name="whitebox_alert_intelligence_get_throttle_recommendations")
    async def whitebox_alert_intelligence_get_throttle_recommendations() -> (
        list[dict[str, Any]]
    ):
        """Get recommended alert-throttling rules to reduce fatigue without
        missing critical issues."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.aget_throttle_recommendations(),
        )

    # ==================== Enhanced Grouping ====================

    @mcp.tool(name="whitebox_alert_intelligence_ml_grouping")
    async def whitebox_alert_intelligence_ml_grouping(
        clustering_method: str = "kmeans",
        num_clusters: int | None = None,
    ) -> dict[str, Any]:
        """Cluster alerts via ML (kmeans/dbscan/hierarchical). Automatically
        determines the optimal number of clusters if not specified."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.aml_group_alerts(
                clustering_method=clustering_method,
                num_clusters=num_clusters,
            ),
        )

    @mcp.tool(name="whitebox_alert_intelligence_temporal_grouping")
    async def whitebox_alert_intelligence_temporal_grouping(
        time_window_minutes: int = 60,
        similarity_threshold: float = 0.7,
    ) -> dict[str, Any]:
        """Group alerts occurring in the same time window with similar
        characteristics."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.atemporal_group_alerts(
                time_window_minutes=time_window_minutes,
                similarity_threshold=similarity_threshold,
            ),
        )

    # ==================== Enhanced Trend Analysis ====================

    @mcp.tool(name="whitebox_alert_intelligence_predict_trends")
    async def whitebox_alert_intelligence_predict_trends(
        time_horizon: str = "next_day",
    ) -> dict[str, Any]:
        """Forecast alert volume/severity/type for `time_horizon`
        (next_hour/next_day/next_week).

        Uses a least-squares linear trend (numpy.polyfit) over the
        organization's real historical hourly alert counts, not a full
        time-series forecasting model (no ARIMA/Prophet). Returns
        trend_direction="insufficient_data" and zeroed values when fewer
        than 24 real hourly data points exist; sample_size in the response
        always discloses exactly how much real history backed the estimate.
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.apredict_trends(
                time_horizon=time_horizon
            ),
        )

    @mcp.tool(name="whitebox_alert_intelligence_detect_seasonality")
    async def whitebox_alert_intelligence_detect_seasonality(
        days_back: int = 30,
    ) -> dict[str, Any]:
        """Detect hour-of-day seasonality in alert volume.

        Real detection from the organization's actual alert history (24
        hourly buckets summed across the requested window). Day-of-week/
        week-of-month cycles are not computed, so pattern_type is always
        "hourly" (or "insufficient_data" below a 24-data-point floor),
        never a fabricated "daily"/"weekly" label.
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.adetect_seasonality(
                days_back=days_back
            ),
        )

    @mcp.tool(name="whitebox_alert_intelligence_enhanced_trend_analysis")
    async def whitebox_alert_intelligence_enhanced_trend_analysis(
        metric: str = "alert_count", hours_back: int = 168
    ) -> dict[str, Any]:
        """Combine real alert-volume time-series data with a real
        hour-of-day seasonality estimate, a numpy.polyfit trend prediction
        (with an R²-based confidence score), and real z-score anomaly
        detection over the same window -- all computed from the
        organization's actual alert history. Not a full forecasting model
        (no ARIMA/Prophet)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.aenhanced_trend_analysis(
                metric=metric, hours_back=hours_back
            ),
        )

    @mcp.tool(name="whitebox_alert_intelligence_forecast_volume")
    async def whitebox_alert_intelligence_forecast_volume(
        forecast_hours: int = 24,
    ) -> dict[str, Any]:
        """Forecast future alert volume with confidence intervals.

        Uses a least-squares linear trend (numpy.polyfit) over real
        historical hourly alert counts, shaped by the organization's real
        hour-of-day seasonality. NOT ARIMA/Prophet. Forward-looking
        anomaly_predictions are intentionally always empty -- predicting
        future anomalies needs its own model, out of scope here; anomaly
        detection over *observed* data is available via
        whitebox_alert_intelligence_enhanced_trend_analysis instead. Falls
        back to an honestly-labeled zeroed forecast
        (seasonal_pattern.type == "insufficient_data") when fewer than 24
        hourly data points exist.
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.alert_intelligence.aforecast_volume(
                forecast_hours=forecast_hours
            ),
        )
