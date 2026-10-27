"""Behavior tests for whiteboxxai_mcp.tools.safety (issue #141)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.tools import safety


@pytest.fixture
def mcp_with_fake_session(monkeypatch):
    mcp = FastMCP(name="test")
    safety.register(mcp)

    fake_client = MagicMock()
    fake_client.safety.aanalyze = AsyncMock(return_value={"id": "s1"})
    fake_client.safety.aanalyze_batch = AsyncMock(return_value=[{"id": "s1"}])
    fake_client.safety.aget_score = AsyncMock(return_value={"id": "s1"})
    fake_client.safety.aget_scores = AsyncMock(return_value=[{"id": "s1"}])
    fake_client.safety.aget_stats = AsyncMock(return_value={"total_analyses": 1})
    fake_client.safety.atrends = AsyncMock(return_value=[{"date": "2026-08-14"}])
    fake_client.safety.acreate_threshold = AsyncMock(return_value={"id": "t1"})
    fake_client.safety.alist_thresholds = AsyncMock(return_value=[{"id": "t1"}])
    fake_client.safety.aget_threshold = AsyncMock(return_value={"id": "t1"})
    fake_client.safety.aupdate_threshold = AsyncMock(return_value={"id": "t1"})
    fake_client.safety.adelete_threshold = AsyncMock(return_value={"deleted": True})

    class FakeSession:
        async def get_client(self, force_refresh: bool = False):
            return fake_client

    monkeypatch.setattr(safety, "get_session", lambda: FakeSession())
    return mcp, fake_client


@pytest.mark.asyncio
async def test_analyze_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_safety_analyze")

    result = await tool.run({"content": "hello"})

    fake_client.safety.aanalyze.assert_awaited_once_with(
        content="hello",
        content_type="text",
        check_toxicity=True,
        check_pii=True,
        check_harmful=True,
        language="en",
        llm_log_id=None,
        model_id=None,
        user_id=None,
        session_id=None,
        environment=None,
    )
    assert result == {"id": "s1"}


@pytest.mark.asyncio
async def test_analyze_batch_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_safety_analyze_batch")

    await tool.run({"contents": ["a", "b"]})

    fake_client.safety.aanalyze_batch.assert_awaited_once_with(
        contents=["a", "b"],
        content_type="text",
        check_toxicity=True,
        check_pii=True,
        check_harmful=True,
        language="en",
        model_id=None,
        user_id=None,
        environment=None,
    )


@pytest.mark.asyncio
async def test_get_score_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_safety_get_score")

    await tool.run({"score_id": "s1"})

    fake_client.safety.aget_score.assert_awaited_once_with("s1")


@pytest.mark.asyncio
async def test_get_scores_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_safety_get_scores")

    await tool.run({"model_id": "m1", "limit": 10})

    fake_client.safety.aget_scores.assert_awaited_once_with(
        start_time=None,
        end_time=None,
        model_id="m1",
        safety_status=None,
        contains_pii=None,
        environment=None,
        limit=10,
        offset=0,
    )


@pytest.mark.asyncio
async def test_stats_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_safety_stats")

    await tool.run({"model_id": "m1"})

    fake_client.safety.aget_stats.assert_awaited_once_with(
        start_time=None, end_time=None, model_id="m1", environment=None
    )


@pytest.mark.asyncio
async def test_trends_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_safety_trends")

    await tool.run({"granularity": "week"})

    fake_client.safety.atrends.assert_awaited_once_with(
        start_time=None,
        end_time=None,
        granularity="week",
        model_id=None,
        environment=None,
    )


@pytest.mark.asyncio
async def test_create_threshold_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_safety_create_threshold")

    await tool.run({"name": "strict"})

    fake_client.safety.acreate_threshold.assert_awaited_once_with(
        "strict",
        model_id=None,
        description=None,
        toxicity_threshold=0.7,
        severe_toxicity_threshold=0.5,
        identity_attack_threshold=0.7,
        insult_threshold=0.7,
        profanity_threshold=0.7,
        threat_threshold=0.5,
        sexually_explicit_threshold=0.7,
        block_pii=False,
        allowed_pii_types=None,
        max_pii_count=None,
        action_on_violation="warning",
        send_alerts=True,
    )


@pytest.mark.asyncio
async def test_list_thresholds_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_safety_list_thresholds")

    await tool.run({})

    fake_client.safety.alist_thresholds.assert_awaited_once_with(
        model_id=None, is_active=None
    )


@pytest.mark.asyncio
async def test_get_threshold_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_safety_get_threshold")

    await tool.run({"threshold_id": "t1"})

    fake_client.safety.aget_threshold.assert_awaited_once_with("t1")


@pytest.mark.asyncio
async def test_update_threshold_only_sends_provided_fields(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_safety_update_threshold")

    await tool.run({"threshold_id": "t1", "is_active": False})

    fake_client.safety.aupdate_threshold.assert_awaited_once_with("t1", is_active=False)


@pytest.mark.asyncio
async def test_delete_threshold_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_safety_delete_threshold")

    await tool.run({"threshold_id": "t1"})

    fake_client.safety.adelete_threshold.assert_awaited_once_with("t1")
