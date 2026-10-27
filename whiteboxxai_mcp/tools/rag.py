"""MCP tools for RAG pipeline monitoring (backend/api/v1/rag.py):
retrieval logging, evaluations, and quality metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.auth.context import get_session
from whiteboxxai_mcp.errors import call_with_session


def register(mcp: FastMCP) -> None:
    """Register all RAG-monitoring tools on `mcp`."""

    @mcp.tool(name="whitebox_rag_log_retrieval")
    async def whitebox_rag_log_retrieval(
        query: str,
        results: list[dict[str, Any]],
        top_k: int,
        retrieval_method: str,
        query_type: str = "semantic",
        retrieval_params: dict[str, Any] | None = None,
        retrieval_latency_ms: float | None = None,
        answer: str | None = None,
        answer_latency_ms: float | None = None,
        ground_truth_ids: list[str] | None = None,
        llm_log_id: str | None = None,
        model_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        environment: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Log a retrieval operation, with real quality metrics computed
        server-side when ground_truth_ids is supplied."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.rag.alog_retrieval(
                query=query,
                results=results,
                top_k=top_k,
                retrieval_method=retrieval_method,
                query_type=query_type,
                retrieval_params=retrieval_params,
                retrieval_latency_ms=retrieval_latency_ms,
                answer=answer,
                answer_latency_ms=answer_latency_ms,
                ground_truth_ids=ground_truth_ids,
                llm_log_id=llm_log_id,
                model_id=model_id,
                user_id=user_id,
                session_id=session_id,
                environment=environment,
                metadata=metadata,
            ),
        )

    @mcp.tool(name="whitebox_rag_get_retrieval")
    async def whitebox_rag_get_retrieval(log_id: str) -> dict[str, Any]:
        """Get a specific retrieval log by ID."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.rag.aget_retrieval(log_id)
        )

    @mcp.tool(name="whitebox_rag_list_retrievals")
    async def whitebox_rag_list_retrievals(
        model_id: str,
        start_time: str | None = None,
        end_time: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        retrieval_method: str | None = None,
        environment: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query retrieval logs for a model. model_id is required (owned by
        the caller's organization) until organization-wide RAG log scoping ships."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.rag.alist_retrievals(
                model_id,
                start_time=start_time,
                end_time=end_time,
                user_id=user_id,
                session_id=session_id,
                retrieval_method=retrieval_method,
                environment=environment,
                limit=limit,
                offset=offset,
            ),
        )

    @mcp.tool(name="whitebox_rag_stats")
    async def whitebox_rag_stats(
        model_id: str | None = None,
        environment: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
        """Get aggregated RAG statistics (precision/recall/MRR/NDCG/relevance/latency)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.rag.aget_stats(
                model_id=model_id,
                environment=environment,
                start_time=start_time,
                end_time=end_time,
            ),
        )

    @mcp.tool(name="whitebox_rag_trends")
    async def whitebox_rag_trends(
        start_time: str | None = None,
        end_time: str | None = None,
        granularity: str = "day",
        model_id: str | None = None,
        environment: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get RAG quality metrics trend over time (granularity: hour/day/week)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.rag.atrends(
                start_time=start_time,
                end_time=end_time,
                granularity=granularity,
                model_id=model_id,
                environment=environment,
            ),
        )

    @mcp.tool(name="whitebox_rag_create_evaluation")
    async def whitebox_rag_create_evaluation(
        name: str,
        queries: list[str],
        description: str | None = None,
        evaluation_type: str = "manual",
        model_id: str | None = None,
        system_version: str | None = None,
        ground_truth: dict[str, list[str]] | None = None,
        environment: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a RAG evaluation, aggregating already-logged retrievals for
        each query. Queries with no matching logged retrieval are reported
        in detailed_results.queries_without_data, not silently scored as zero."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.rag.acreate_evaluation(
                name=name,
                queries=queries,
                description=description,
                evaluation_type=evaluation_type,
                model_id=model_id,
                system_version=system_version,
                ground_truth=ground_truth,
                environment=environment,
                metadata=metadata,
            ),
        )

    @mcp.tool(name="whitebox_rag_get_evaluation")
    async def whitebox_rag_get_evaluation(evaluation_id: str) -> dict[str, Any]:
        """Get a specific RAG evaluation by ID."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.rag.aget_evaluation(evaluation_id)
        )

    @mcp.tool(name="whitebox_rag_list_evaluations")
    async def whitebox_rag_list_evaluations(
        model_id: str | None = None,
        evaluation_type: str | None = None,
        environment: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List RAG evaluations with optional filters."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.rag.alist_evaluations(
                model_id=model_id,
                evaluation_type=evaluation_type,
                environment=environment,
                limit=limit,
                offset=offset,
            ),
        )

    @mcp.tool(name="whitebox_rag_metrics_precision")
    async def whitebox_rag_metrics_precision(
        start_time: str | None = None,
        end_time: str | None = None,
        model_id: str | None = None,
        environment: str | None = None,
        k_values: list[int] | None = None,
    ) -> dict[str, float | None]:
        """Get Precision@K metrics for the given K values (default [1, 3, 5, 10]).
        A K with no scoreable logged retrievals returns null, not a
        fabricated 0.0."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.rag.ametrics_precision(
                start_time=start_time,
                end_time=end_time,
                model_id=model_id,
                environment=environment,
                k_values=k_values,
            ),
        )

    @mcp.tool(name="whitebox_rag_metrics_relevance")
    async def whitebox_rag_metrics_relevance(
        start_time: str | None = None,
        end_time: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, float]:
        """Get average relevance metrics (context/answer/faithfulness)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.rag.ametrics_relevance(
                start_time=start_time, end_time=end_time, model_id=model_id
            ),
        )
