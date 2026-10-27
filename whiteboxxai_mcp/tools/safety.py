"""MCP tools for content safety analysis (backend/api/v1/safety.py):
toxicity/PII/harmful-content detection and threshold management.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.auth.context import get_session
from whiteboxxai_mcp.errors import call_with_session


def register(mcp: FastMCP) -> None:
    """Register all content-safety tools on `mcp`."""

    @mcp.tool(name="whitebox_safety_analyze")
    async def whitebox_safety_analyze(
        content: str,
        content_type: str = "text",
        check_toxicity: bool = True,
        check_pii: bool = True,
        check_harmful: bool = True,
        language: str = "en",
        llm_log_id: str | None = None,
        model_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        environment: str | None = None,
    ) -> dict[str, Any]:
        """Analyze content for toxicity, PII, and harmful content.

        Returns full safety analysis: toxicity_scores, pii_results,
        harmful_content, safety_status, flagged_categories, etc.
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.safety.aanalyze(
                content=content,
                content_type=content_type,
                check_toxicity=check_toxicity,
                check_pii=check_pii,
                check_harmful=check_harmful,
                language=language,
                llm_log_id=llm_log_id,
                model_id=model_id,
                user_id=user_id,
                session_id=session_id,
                environment=environment,
            ),
        )

    @mcp.tool(name="whitebox_safety_analyze_batch")
    async def whitebox_safety_analyze_batch(
        contents: list[str],
        content_type: str = "text",
        check_toxicity: bool = True,
        check_pii: bool = True,
        check_harmful: bool = True,
        language: str = "en",
        model_id: str | None = None,
        user_id: str | None = None,
        environment: str | None = None,
    ) -> list[dict[str, Any]]:
        """Analyze multiple content items in one request (max 100). Items
        that fail analysis are silently skipped, not raised."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.safety.aanalyze_batch(
                contents=contents,
                content_type=content_type,
                check_toxicity=check_toxicity,
                check_pii=check_pii,
                check_harmful=check_harmful,
                language=language,
                model_id=model_id,
                user_id=user_id,
                environment=environment,
            ),
        )

    @mcp.tool(name="whitebox_safety_get_score")
    async def whitebox_safety_get_score(score_id: str) -> dict[str, Any]:
        """Get a specific safety score by ID."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.safety.aget_score(score_id)
        )

    @mcp.tool(name="whitebox_safety_get_scores")
    async def whitebox_safety_get_scores(
        start_time: str | None = None,
        end_time: str | None = None,
        model_id: str | None = None,
        safety_status: str | None = None,
        contains_pii: bool | None = None,
        environment: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query safety analysis scores with filters."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.safety.aget_scores(
                start_time=start_time,
                end_time=end_time,
                model_id=model_id,
                safety_status=safety_status,
                contains_pii=contains_pii,
                environment=environment,
                limit=limit,
                offset=offset,
            ),
        )

    @mcp.tool(name="whitebox_safety_stats")
    async def whitebox_safety_stats(
        start_time: str | None = None,
        end_time: str | None = None,
        model_id: str | None = None,
        environment: str | None = None,
    ) -> dict[str, Any]:
        """Get aggregated safety statistics (per-category toxicity averages,
        flagged categories, PII types)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.safety.aget_stats(
                start_time=start_time,
                end_time=end_time,
                model_id=model_id,
                environment=environment,
            ),
        )

    @mcp.tool(name="whitebox_safety_trends")
    async def whitebox_safety_trends(
        start_time: str | None = None,
        end_time: str | None = None,
        granularity: str = "day",
        model_id: str | None = None,
        environment: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get content-safety trend over time (granularity: hour/day/week)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.safety.atrends(
                start_time=start_time,
                end_time=end_time,
                granularity=granularity,
                model_id=model_id,
                environment=environment,
            ),
        )

    @mcp.tool(name="whitebox_safety_create_threshold")
    async def whitebox_safety_create_threshold(
        name: str,
        model_id: str | None = None,
        description: str | None = None,
        toxicity_threshold: float = 0.7,
        severe_toxicity_threshold: float = 0.5,
        identity_attack_threshold: float = 0.7,
        insult_threshold: float = 0.7,
        profanity_threshold: float = 0.7,
        threat_threshold: float = 0.5,
        sexually_explicit_threshold: float = 0.7,
        block_pii: bool = False,
        allowed_pii_types: list[str] | None = None,
        max_pii_count: int | None = None,
        action_on_violation: str = "warning",
        send_alerts: bool = True,
    ) -> dict[str, Any]:
        """Create a safety threshold configuration controlling when content
        is flagged as unsafe. action_on_violation: "warning", "block", or "alert"."""
        session = get_session()
        kwargs: dict[str, Any] = {
            "model_id": model_id,
            "description": description,
            "toxicity_threshold": toxicity_threshold,
            "severe_toxicity_threshold": severe_toxicity_threshold,
            "identity_attack_threshold": identity_attack_threshold,
            "insult_threshold": insult_threshold,
            "profanity_threshold": profanity_threshold,
            "threat_threshold": threat_threshold,
            "sexually_explicit_threshold": sexually_explicit_threshold,
            "block_pii": block_pii,
            "allowed_pii_types": allowed_pii_types,
            "max_pii_count": max_pii_count,
            "action_on_violation": action_on_violation,
            "send_alerts": send_alerts,
        }
        return await call_with_session(
            session, lambda client: client.safety.acreate_threshold(name, **kwargs)
        )

    @mcp.tool(name="whitebox_safety_list_thresholds")
    async def whitebox_safety_list_thresholds(
        model_id: str | None = None, is_active: bool | None = None
    ) -> list[dict[str, Any]]:
        """List safety thresholds for the caller's organization."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.safety.alist_thresholds(
                model_id=model_id, is_active=is_active
            ),
        )

    @mcp.tool(name="whitebox_safety_get_threshold")
    async def whitebox_safety_get_threshold(threshold_id: str) -> dict[str, Any]:
        """Get a specific safety threshold by ID."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.safety.aget_threshold(threshold_id)
        )

    @mcp.tool(name="whitebox_safety_update_threshold")
    async def whitebox_safety_update_threshold(
        threshold_id: str,
        name: str | None = None,
        description: str | None = None,
        is_active: bool | None = None,
        toxicity_threshold: float | None = None,
        block_pii: bool | None = None,
        action_on_violation: str | None = None,
        send_alerts: bool | None = None,
    ) -> dict[str, Any]:
        """Update a safety threshold configuration. Only non-null arguments
        are changed."""
        session = get_session()
        kwargs: dict[str, Any] = {}
        if name is not None:
            kwargs["name"] = name
        if description is not None:
            kwargs["description"] = description
        if is_active is not None:
            kwargs["is_active"] = is_active
        if toxicity_threshold is not None:
            kwargs["toxicity_threshold"] = toxicity_threshold
        if block_pii is not None:
            kwargs["block_pii"] = block_pii
        if action_on_violation is not None:
            kwargs["action_on_violation"] = action_on_violation
        if send_alerts is not None:
            kwargs["send_alerts"] = send_alerts
        return await call_with_session(
            session,
            lambda client: client.safety.aupdate_threshold(threshold_id, **kwargs),
        )

    @mcp.tool(name="whitebox_safety_delete_threshold")
    async def whitebox_safety_delete_threshold(threshold_id: str) -> dict[str, Any]:
        """Delete a safety threshold configuration."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.safety.adelete_threshold(threshold_id)
        )
