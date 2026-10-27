"""Behavior tests for whiteboxxai_mcp.tools.llm_xai (issue #141)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.tools import llm_xai


@pytest.fixture
def mcp_with_fake_session(monkeypatch):
    mcp = FastMCP(name="test")
    llm_xai.register(mcp)

    fake_client = MagicMock()
    fake_client.llm_xai.aattention = AsyncMock(return_value={"id": "e1"})
    fake_client.llm_xai.atoken_importance = AsyncMock(return_value={"id": "e2"})
    fake_client.llm_xai.aprompt_sensitivity = AsyncMock(return_value={"id": "e3"})
    fake_client.llm_xai.acounterfactuals = AsyncMock(return_value={"id": "e4"})
    fake_client.llm_xai.adebug_prompt = AsyncMock(return_value={"analysis_id": "a1"})
    fake_client.llm_xai.aget_explanation = AsyncMock(return_value={"id": "e1"})
    fake_client.llm_xai.aget_prompt_analysis = AsyncMock(return_value={"id": "a1"})
    fake_client.llm_xai.alist_explanations = AsyncMock(return_value=[{"id": "e1"}])
    fake_client.llm_xai.alist_prompt_analyses = AsyncMock(return_value=[{"id": "a1"}])
    fake_client.llm_xai.astats = AsyncMock(return_value={"total_explanations": 1})
    fake_client.llm_xai.avisualize_attention = AsyncMock(
        return_value={"type": "attention_heatmap"}
    )
    fake_client.llm_xai.avisualize_token_importance = AsyncMock(
        return_value={"type": "token_importance_bar"}
    )
    fake_client.llm_xai.abatch_analyze = AsyncMock(return_value={"analyses": {}})

    class FakeSession:
        async def get_client(self, force_refresh: bool = False):
            return fake_client

    monkeypatch.setattr(llm_xai, "get_session", lambda: FakeSession())
    return mcp, fake_client


@pytest.mark.asyncio
async def test_attention_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_xai_attention")

    result = await tool.run({"prompt": "hello", "attention_weights": [[0.1, 0.9]]})

    fake_client.llm_xai.aattention.assert_awaited_once_with(
        "hello",
        completion=None,
        model_id=None,
        layers=None,
        aggregate_heads=True,
        attention_weights=[[0.1, 0.9]],
        top_k=10,
        llm_log_id=None,
    )
    assert result == {"id": "e1"}


@pytest.mark.asyncio
async def test_token_importance_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_xai_token_importance")

    await tool.run({"prompt": "hello", "completion": "world"})

    fake_client.llm_xai.atoken_importance.assert_awaited_once_with(
        "hello",
        "world",
        model_id=None,
        method="perturbation",
        top_k=10,
        llm_log_id=None,
    )


@pytest.mark.asyncio
async def test_prompt_sensitivity_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_xai_prompt_sensitivity")

    await tool.run({"prompt": "hello", "baseline_output": "world"})

    fake_client.llm_xai.aprompt_sensitivity.assert_awaited_once_with(
        "hello",
        "world",
        model_id=None,
        num_perturbations=10,
        perturbation_type="mixed",
        llm_log_id=None,
    )


@pytest.mark.asyncio
async def test_counterfactuals_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_xai_counterfactuals")

    await tool.run({"prompt": "hello", "original_output": "world"})

    fake_client.llm_xai.acounterfactuals.assert_awaited_once_with(
        "hello",
        "world",
        target_change="any",
        model_id=None,
        max_edits=3,
        num_candidates=10,
        search_strategy="greedy",
        llm_log_id=None,
    )


@pytest.mark.asyncio
async def test_debug_prompt_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_xai_debug_prompt")

    await tool.run({"prompt": "hello", "prompt_version_id": "pv1"})

    fake_client.llm_xai.adebug_prompt.assert_awaited_once_with(
        "hello",
        model_id=None,
        check_clarity=True,
        check_specificity=True,
        check_completeness=True,
        check_consistency=True,
        suggest_improvements=True,
        llm_log_id=None,
        prompt_version_id="pv1",
    )


@pytest.mark.asyncio
async def test_get_explanation_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_xai_get_explanation")

    result = await tool.run({"explanation_id": "e1"})

    fake_client.llm_xai.aget_explanation.assert_awaited_once_with("e1")
    assert result == {"id": "e1"}


@pytest.mark.asyncio
async def test_get_prompt_analysis_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_xai_get_prompt_analysis")

    await tool.run({"analysis_id": "a1"})

    fake_client.llm_xai.aget_prompt_analysis.assert_awaited_once_with("a1")


@pytest.mark.asyncio
async def test_list_explanations_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_xai_list_explanations")

    await tool.run({"model_id": "m1"})

    fake_client.llm_xai.alist_explanations.assert_awaited_once_with(
        llm_log_id=None, explanation_type=None, model_id="m1", limit=50
    )


@pytest.mark.asyncio
async def test_list_prompt_analyses_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_xai_list_prompt_analyses")

    await tool.run({"model_id": "m1"})

    fake_client.llm_xai.alist_prompt_analyses.assert_awaited_once_with(
        prompt=None, model_id="m1", min_quality_score=None, limit=50
    )


@pytest.mark.asyncio
async def test_stats_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_xai_stats")

    await tool.run({"model_id": "m1", "days": 30})

    fake_client.llm_xai.astats.assert_awaited_once_with(model_id="m1", days=30)


@pytest.mark.asyncio
async def test_visualize_attention_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_xai_visualize_attention")

    await tool.run({"explanation_id": "e1"})

    fake_client.llm_xai.avisualize_attention.assert_awaited_once_with("e1")


@pytest.mark.asyncio
async def test_visualize_token_importance_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_xai_visualize_token_importance")

    await tool.run({"explanation_id": "e1", "top_k": 5})

    fake_client.llm_xai.avisualize_token_importance.assert_awaited_once_with(
        "e1", top_k=5
    )


@pytest.mark.asyncio
async def test_batch_analyze_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_llm_xai_batch_analyze")

    await tool.run({"llm_log_id": "l1"})

    fake_client.llm_xai.abatch_analyze.assert_awaited_once_with(
        "l1",
        include_attention=True,
        include_token_importance=True,
        include_sensitivity=False,
        include_prompt_debug=True,
    )
