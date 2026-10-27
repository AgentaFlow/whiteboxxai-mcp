"""MCP tools for LLM call monitoring/logging (backend/api/v1/llm_monitoring.py):
token/cost/latency tracking for LLM requests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.auth.context import get_session
from whiteboxxai_mcp.errors import call_with_session


def register(mcp: FastMCP) -> None:
    """Register all LLM-monitoring tools on `mcp`."""

    @mcp.tool(name="whitebox_llm_monitoring_log_call")
    async def whitebox_llm_monitoring_log_call(
        provider: str,
        model_name: str,
        prompt: str,
        latency_ms: float,
        model_id: str | None = None,
        completion: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        finish_reason: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        status: str | None = None,
        error_message: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        environment: str | None = None,
        prompt_version_id: str | None = None,
    ) -> dict[str, Any]:
        """Log a single LLM call with real token/cost/latency data."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm.alog_call(
                provider=provider,
                model_name=model_name,
                prompt=prompt,
                latency_ms=latency_ms,
                model_id=model_id,
                completion=completion,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                temperature=temperature,
                max_tokens=max_tokens,
                status=status,
                error_message=error_message,
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                environment=environment,
                prompt_version_id=prompt_version_id,
            ),
        )

    @mcp.tool(name="whitebox_llm_monitoring_log_calls_batch")
    async def whitebox_llm_monitoring_log_calls_batch(
        logs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Log multiple LLM calls in one request (max 1000). Each dict is
        shaped like whitebox_llm_monitoring_log_call's arguments."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.llm.alog_calls_batch(logs)
        )

    @mcp.tool(name="whitebox_llm_monitoring_get_log")
    async def whitebox_llm_monitoring_get_log(log_id: str) -> dict[str, Any]:
        """Get a single LLM log by ID."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.llm.aget_log(log_id)
        )

    @mcp.tool(name="whitebox_llm_monitoring_query_logs")
    async def whitebox_llm_monitoring_query_logs(
        start_time: str | None = None,
        end_time: str | None = None,
        model_id: str | None = None,
        provider: str | None = None,
        model_name: str | None = None,
        status: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        environment: str | None = None,
        min_cost: float | None = None,
        max_cost: float | None = None,
        min_latency_ms: float | None = None,
        max_latency_ms: float | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Query LLM logs with filters and pagination. Returns {logs, total, skip, limit}.

        Passing a model_id owned by the caller's organization is the only
        fully tenant-scoped way to call this today (results are otherwise
        platform-wide).
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm.aquery_logs(
                start_time=start_time,
                end_time=end_time,
                model_id=model_id,
                provider=provider,
                model_name=model_name,
                status=status,
                user_id=user_id,
                session_id=session_id,
                environment=environment,
                min_cost=min_cost,
                max_cost=max_cost,
                min_latency_ms=min_latency_ms,
                max_latency_ms=max_latency_ms,
                skip=skip,
                limit=limit,
            ),
        )

    @mcp.tool(name="whitebox_llm_monitoring_session_logs")
    async def whitebox_llm_monitoring_session_logs(
        session_id: str,
    ) -> list[dict[str, Any]]:
        """Get all LLM logs for a specific session."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.llm.asession_logs(session_id)
        )

    @mcp.tool(name="whitebox_llm_monitoring_usage_stats")
    async def whitebox_llm_monitoring_usage_stats(
        model_id: str | None = None,
        provider: str | None = None,
        environment: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
        """Get aggregated LLM usage statistics (tokens, cost, latency, by_model)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm.aget_stats(
                model_id=model_id,
                provider=provider,
                environment=environment,
                start_time=start_time,
                end_time=end_time,
            ),
        )

    @mcp.tool(name="whitebox_llm_monitoring_recent")
    async def whitebox_llm_monitoring_recent(
        limit: int = 100,
        model_id: str | None = None,
        provider: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get recent LLM logs (requires a model_id owned by the caller's org)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm.aget_recent(
                limit=limit, model_id=model_id, provider=provider
            ),
        )

    @mcp.tool(name="whitebox_llm_monitoring_cost_breakdown")
    async def whitebox_llm_monitoring_cost_breakdown(
        start_time: str | None = None,
        end_time: str | None = None,
        include_user: bool = False,
        include_environment: bool = False,
    ) -> dict[str, Any]:
        """Get detailed cost breakdown (platform-wide aggregate)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm.acost_breakdown(
                start_time=start_time,
                end_time=end_time,
                include_user=include_user,
                include_environment=include_environment,
            ),
        )

    @mcp.tool(name="whitebox_llm_monitoring_performance")
    async def whitebox_llm_monitoring_performance(
        start_time: str | None = None,
        end_time: str | None = None,
        model_id: str | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
        """Get LLM latency/throughput performance metrics."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm.aperformance(
                start_time=start_time,
                end_time=end_time,
                model_id=model_id,
                provider=provider,
            ),
        )

    @mcp.tool(name="whitebox_llm_monitoring_trends_tokens")
    async def whitebox_llm_monitoring_trends_tokens(
        start_time: str | None = None,
        end_time: str | None = None,
        granularity: str = "day",
        model_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get token usage trend over time (granularity: hour/day/week)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm.atrends_tokens(
                start_time=start_time,
                end_time=end_time,
                granularity=granularity,
                model_id=model_id,
            ),
        )

    @mcp.tool(name="whitebox_llm_monitoring_trends_costs")
    async def whitebox_llm_monitoring_trends_costs(
        start_time: str | None = None,
        end_time: str | None = None,
        granularity: str = "day",
        provider: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get cost trend over time (platform-wide; granularity: hour/day/week)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm.atrends_costs(
                start_time=start_time,
                end_time=end_time,
                granularity=granularity,
                provider=provider,
            ),
        )

    @mcp.tool(name="whitebox_llm_monitoring_usage_stats_detailed")
    async def whitebox_llm_monitoring_usage_stats_detailed(
        start_time: str,
        end_time: str,
        granularity: str = "day",
        provider: str | None = None,
        model_name: str | None = None,
        model_id: str | None = None,
        user_id: str | None = None,
        environment: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get detailed, grouped usage statistics (granularity: hour/day/week/month)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm.ausage_stats(
                start_time=start_time,
                end_time=end_time,
                granularity=granularity,
                provider=provider,
                model_name=model_name,
                model_id=model_id,
                user_id=user_id,
                environment=environment,
            ),
        )

    @mcp.tool(name="whitebox_llm_monitoring_cost_threshold_alert")
    async def whitebox_llm_monitoring_cost_threshold_alert(
        period_minutes: int = 60, threshold: float = 100.0
    ) -> dict[str, Any]:
        """Check whether costs over the recent period exceed a threshold."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm.acost_threshold_alert(
                period_minutes=period_minutes, threshold=threshold
            ),
        )

    @mcp.tool(name="whitebox_llm_monitoring_latency_threshold_alert")
    async def whitebox_llm_monitoring_latency_threshold_alert(
        period_minutes: int = 60, threshold_ms: float = 5000.0
    ) -> dict[str, Any]:
        """Check whether average latency over the recent period exceeds a threshold."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm.alatency_threshold_alert(
                period_minutes=period_minutes, threshold_ms=threshold_ms
            ),
        )

    @mcp.tool(name="whitebox_llm_monitoring_error_rate_alert")
    async def whitebox_llm_monitoring_error_rate_alert(
        period_minutes: int = 60, threshold_percent: float = 10.0
    ) -> dict[str, Any]:
        """Check whether the error rate over the recent period exceeds a threshold."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm.aerror_rate_alert(
                period_minutes=period_minutes, threshold_percent=threshold_percent
            ),
        )

    @mcp.tool(name="whitebox_llm_monitoring_cleanup_logs")
    async def whitebox_llm_monitoring_cleanup_logs(days: int = 30) -> dict[str, Any]:
        """Delete LLM logs older than `days`. Admin-only on the backend."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.llm.acleanup_logs(days=days)
        )
