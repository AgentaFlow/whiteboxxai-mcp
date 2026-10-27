"""Behavior tests for whiteboxxai_mcp.tools.rag (issue #141)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.tools import rag


@pytest.fixture
def mcp_with_fake_session(monkeypatch):
    mcp = FastMCP(name="test")
    rag.register(mcp)

    fake_client = MagicMock()
    fake_client.rag.alog_retrieval = AsyncMock(return_value={"id": "r1"})
    fake_client.rag.aget_retrieval = AsyncMock(return_value={"id": "r1"})
    fake_client.rag.alist_retrievals = AsyncMock(return_value=[{"id": "r1"}])
    fake_client.rag.aget_stats = AsyncMock(return_value={"total_retrievals": 1})
    fake_client.rag.atrends = AsyncMock(return_value=[{"date": "2026-08-14"}])
    fake_client.rag.acreate_evaluation = AsyncMock(return_value={"id": "ev1"})
    fake_client.rag.aget_evaluation = AsyncMock(return_value={"id": "ev1"})
    fake_client.rag.alist_evaluations = AsyncMock(return_value=[{"id": "ev1"}])
    fake_client.rag.ametrics_precision = AsyncMock(return_value={"precision@1": 0.5})
    fake_client.rag.ametrics_relevance = AsyncMock(
        return_value={"context_relevance": 0.8}
    )

    class FakeSession:
        async def get_client(self, force_refresh: bool = False):
            return fake_client

    monkeypatch.setattr(rag, "get_session", lambda: FakeSession())
    return mcp, fake_client


@pytest.mark.asyncio
async def test_log_retrieval_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_rag_log_retrieval")

    results = [{"document_id": "d1", "rank": 1, "score": 0.9}]
    result = await tool.run(
        {
            "query": "what is x",
            "results": results,
            "top_k": 5,
            "retrieval_method": "vector",
        }
    )

    fake_client.rag.alog_retrieval.assert_awaited_once_with(
        query="what is x",
        results=results,
        top_k=5,
        retrieval_method="vector",
        query_type="semantic",
        retrieval_params=None,
        retrieval_latency_ms=None,
        answer=None,
        answer_latency_ms=None,
        ground_truth_ids=None,
        llm_log_id=None,
        model_id=None,
        user_id=None,
        session_id=None,
        environment=None,
        metadata=None,
    )
    assert result == {"id": "r1"}


@pytest.mark.asyncio
async def test_get_retrieval_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_rag_get_retrieval")

    await tool.run({"log_id": "r1"})

    fake_client.rag.aget_retrieval.assert_awaited_once_with("r1")


@pytest.mark.asyncio
async def test_list_retrievals_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_rag_list_retrievals")

    await tool.run({"model_id": "m1", "limit": 10})

    fake_client.rag.alist_retrievals.assert_awaited_once_with(
        "m1",
        start_time=None,
        end_time=None,
        user_id=None,
        session_id=None,
        retrieval_method=None,
        environment=None,
        limit=10,
        offset=0,
    )


@pytest.mark.asyncio
async def test_stats_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_rag_stats")

    await tool.run({"model_id": "m1"})

    fake_client.rag.aget_stats.assert_awaited_once_with(
        model_id="m1", environment=None, start_time=None, end_time=None
    )


@pytest.mark.asyncio
async def test_trends_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_rag_trends")

    await tool.run({"granularity": "week"})

    fake_client.rag.atrends.assert_awaited_once_with(
        start_time=None,
        end_time=None,
        granularity="week",
        model_id=None,
        environment=None,
    )


@pytest.mark.asyncio
async def test_create_evaluation_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_rag_create_evaluation")

    await tool.run({"name": "eval1", "queries": ["q1", "q2"]})

    fake_client.rag.acreate_evaluation.assert_awaited_once_with(
        name="eval1",
        queries=["q1", "q2"],
        description=None,
        evaluation_type="manual",
        model_id=None,
        system_version=None,
        ground_truth=None,
        environment=None,
        metadata=None,
    )


@pytest.mark.asyncio
async def test_get_evaluation_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_rag_get_evaluation")

    await tool.run({"evaluation_id": "ev1"})

    fake_client.rag.aget_evaluation.assert_awaited_once_with("ev1")


@pytest.mark.asyncio
async def test_list_evaluations_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_rag_list_evaluations")

    await tool.run({"model_id": "m1"})

    fake_client.rag.alist_evaluations.assert_awaited_once_with(
        model_id="m1", evaluation_type=None, environment=None, limit=100, offset=0
    )


@pytest.mark.asyncio
async def test_metrics_precision_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_rag_metrics_precision")

    await tool.run({"model_id": "m1"})

    fake_client.rag.ametrics_precision.assert_awaited_once_with(
        start_time=None, end_time=None, model_id="m1", environment=None, k_values=None
    )


@pytest.mark.asyncio
async def test_metrics_relevance_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_rag_metrics_relevance")

    await tool.run({"model_id": "m1"})

    fake_client.rag.ametrics_relevance.assert_awaited_once_with(
        start_time=None, end_time=None, model_id="m1"
    )
