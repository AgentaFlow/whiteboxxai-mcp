"""MCP tools for WhiteBoxXAI bias/fairness auditing (backend/api/v1/fairness.py)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.auth.context import get_session
from whiteboxxai_mcp.errors import call_with_session


def register(mcp: FastMCP) -> None:
    """Register all bias/fairness tools on `mcp`."""

    @mcp.tool(name="whitebox_fairness_audit")
    async def whitebox_fairness_audit(
        model_id: str,
        sensitive_attributes: list[str],
        y_true: list[int],
        y_pred: list[int],
        group_data: dict[str, list[Any]],
        y_prob: list[float] | None = None,
        metrics_to_compute: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run a bias/fairness audit on model predictions across demographic groups.

        Unlike most WhiteBoxXAI tools, this is not a data-source query: the
        caller must already have the ground-truth labels, predicted labels,
        and per-attribute group assignments in hand (e.g. pulled via
        whitebox_predictions_query and joined with demographic data), all
        aligned to the same sample order.

        Args:
            model_id: Registered model's ID
            sensitive_attributes: Protected attributes to analyze (race,
                gender, age, ethnicity, religion, disability,
                sexual_orientation, national_origin, marital_status, or custom)
            y_true: Ground-truth labels, same length/order as y_pred and group_data values
            y_pred: Predicted labels
            group_data: Per-attribute group assignments, e.g.
                {"gender": ["M", "F", "M", ...]}, one array per attribute in
                sensitive_attributes, same length/order as y_true/y_pred
            y_prob: Optional predicted probabilities (enables calibration metrics)
            metrics_to_compute: Optional subset of fairness metrics to compute
                (demographic_parity, equal_opportunity, equalized_odds,
                disparate_impact, predictive_parity, calibration,
                false_positive_rate_parity, false_negative_rate_parity,
                statistical_parity_difference, average_odds_difference) --
                computes all of them if omitted

        Returns:
            Full audit results: per-metric pass/fail against fairness
            thresholds, per-group comparisons, an overall fairness score,
            and mitigation recommendations.
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.fairness.aaudit(
                model_id=model_id,
                sensitive_attributes=sensitive_attributes,
                y_true=y_true,
                y_pred=y_pred,
                group_data=group_data,
                y_prob=y_prob,
                metrics_to_compute=metrics_to_compute,
            ),
        )

    @mcp.tool(name="whitebox_fairness_get_audit")
    async def whitebox_fairness_get_audit(audit_id: str) -> dict[str, Any]:
        """Get full bias audit results by audit_id."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.fairness.aget_audit(audit_id)
        )

    @mcp.tool(name="whitebox_fairness_list_audits")
    async def whitebox_fairness_list_audits(
        model_id: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """List bias audit summaries, optionally filtered to one model."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.fairness.alist_audits(
                model_id=model_id, limit=limit, offset=offset
            ),
        )

    @mcp.tool(name="whitebox_fairness_bias_history")
    async def whitebox_fairness_bias_history(
        model_id: str, days: int = 30
    ) -> dict[str, Any]:
        """Get a model's historical bias/fairness trend (1-365 days)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.fairness.aget_bias_history(model_id, days=days),
        )

    @mcp.tool(name="whitebox_fairness_metric_history")
    async def whitebox_fairness_metric_history(
        model_id: str, metric_type: str, days: int = 30
    ) -> dict[str, Any]:
        """Get history for one specific fairness metric on a model.

        Args:
            model_id: Registered model's ID
            metric_type: One of demographic_parity, equal_opportunity,
                equalized_odds, disparate_impact, predictive_parity,
                calibration, false_positive_rate_parity,
                false_negative_rate_parity, statistical_parity_difference,
                average_odds_difference
            days: Lookback window in days (1-365)
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.fairness.aget_metric_history(
                model_id, metric_type, days=days
            ),
        )

    @mcp.tool(name="whitebox_fairness_latest_audit")
    async def whitebox_fairness_latest_audit(model_id: str) -> dict[str, Any]:
        """Get the most recent bias audit for a model."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.fairness.aget_latest_audit(model_id)
        )
