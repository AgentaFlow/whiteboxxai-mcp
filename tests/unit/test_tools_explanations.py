"""Behavior tests for whiteboxxai_mcp.tools.explanations (issue #137, extended
by issue #225).

This was the first behavioral-delegation test in the project -- nothing
before it asserted a tool actually delegates to the right SDK method with
the right arguments (everything else had only protocol/schema-level
coverage via test_server_registration.py). Issue #141 retrofitted the same
pattern onto the M2 domains; issue #227 retrofitted it onto the remaining
M1 domains (test_tools_models.py, test_tools_predictions.py,
test_tools_drift.py, test_tools_fairness.py, test_tools_risk_register.py,
test_tools_governance.py), so every domain now has this coverage.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.tools import explanations


@pytest.fixture
def mcp_with_fake_session(monkeypatch):
    mcp = FastMCP(name="test")
    explanations.register(mcp)

    fake_client = MagicMock()
    fake_client.explanations.agenerate_async = AsyncMock(
        return_value={"explanation_id": "e1", "status": "pending"}
    )
    fake_client.explanations.aget = AsyncMock(
        return_value={"id": "e1", "status": "completed"}
    )
    fake_client.explanations.agenerate = AsyncMock(
        return_value={"id": "e1", "status": "completed", "feature_weights": {}}
    )
    fake_client.explanations.agenerate_bulk = AsyncMock(
        return_value={"explanation_ids": ["e1", "e2"], "total": 2, "message": "ok"}
    )
    fake_client.explanations.alist_by_model = AsyncMock(
        return_value={"explanations": [], "total": 0, "limit": 50, "offset": 0}
    )
    fake_client.explanations.aget_by_prediction = AsyncMock(
        return_value={"id": "e1", "prediction_id": "p1"}
    )
    fake_client.explanations.aget_stats = AsyncMock(
        return_value={"model_id": "m1", "total_explanations": 0}
    )
    fake_client.explanations.aset_config = AsyncMock(
        return_value={"id": 1, "model_id": "m1", "enabled": True}
    )
    fake_client.explanations.aget_config = AsyncMock(
        return_value={"id": 1, "model_id": "m1", "enabled": True}
    )
    fake_client.explanations.adelete_config = AsyncMock(
        return_value={"status": "success"}
    )
    fake_client.explanations.acompare = AsyncMock(
        return_value={
            "comparison_type": "feature_importance",
            "num_explanations": 2,
            "comparison_results": {},
        }
    )
    fake_client.explanations.avisualize = AsyncMock(
        return_value={
            "explanation_id": "e1",
            "plot_type": "waterfall",
            "data": {},
            "generated_at": "2026-08-21T00:00:00Z",
        }
    )
    fake_client.explanations.avisualize_multi = AsyncMock(
        return_value={
            "explanation_id": "e1,e2",
            "plot_type": "summary",
            "data": {},
            "generated_at": "2026-08-21T00:00:00Z",
        }
    )

    class FakeSession:
        """Minimal stand-in for SessionManager (mirrors test_error_mapping.py)."""

        async def get_client(self, force_refresh: bool = False):
            return fake_client

    monkeypatch.setattr(explanations, "get_session", lambda: FakeSession())
    return mcp, fake_client


@pytest.mark.asyncio
async def test_generate_async_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_explanations_generate_async")

    result = await tool.run({"model_id": "m1", "instance": {"age": 42}})

    fake_client.explanations.agenerate_async.assert_awaited_once_with(
        "m1",
        {"age": 42},
        method=None,
        prediction_id=None,
        num_features=10,
        num_samples=5000,
        total_cfs=3,
        desired_class="opposite",
        features_to_vary="all",
    )
    assert result == {"explanation_id": "e1", "status": "pending"}


@pytest.mark.asyncio
async def test_generate_async_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_explanations_generate_async")

    await tool.run(
        {
            "model_id": "m1",
            "instance": {"age": 42},
            "method": "lime",
            "prediction_id": "p1",
            "num_features": 5,
            "num_samples": 1000,
        }
    )

    fake_client.explanations.agenerate_async.assert_awaited_once_with(
        "m1",
        {"age": 42},
        method="lime",
        prediction_id="p1",
        num_features=5,
        num_samples=1000,
        total_cfs=3,
        desired_class="opposite",
        features_to_vary="all",
    )


@pytest.mark.asyncio
async def test_get_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_explanations_get")

    result = await tool.run({"explanation_id": "e1"})

    fake_client.explanations.aget.assert_awaited_once_with("e1")
    assert result == {"id": "e1", "status": "completed"}


@pytest.mark.asyncio
async def test_get_redacts_pii_and_attaches_plain_language_summary(
    mcp_with_fake_session,
):
    mcp, fake_client = mcp_with_fake_session
    fake_client.explanations.aget = AsyncMock(
        return_value={
            "id": "e1",
            "method": "shap_kernel",
            "status": "completed",
            "base_value": 0.25,
            "model_output": 0.75,
            "feature_weights": {"income": 0.4, "credit_score": 0.1},
            "feature_values": {
                "income": 85000,
                "contact_email": "customer@example.com",
            },
        }
    )
    tool = mcp._tool_manager.get_tool("whitebox_explanations_get")

    result = await tool.run({"explanation_id": "e1"})

    assert result["feature_values"]["contact_email"] == "[REDACTED:EMAIL_ADDRESS]"
    assert "SHAP_KERNEL explanation completed" in result["plain_language_summary"]


@pytest.mark.asyncio
async def test_generate_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_explanations_generate")

    result = await tool.run({"model_id": "m1", "instance": {"age": 42}})

    fake_client.explanations.agenerate.assert_awaited_once_with(
        "m1",
        {"age": 42},
        method=None,
        prediction_id=None,
        num_features=10,
        num_samples=5000,
        use_cache=True,
        total_cfs=3,
        desired_class="opposite",
        features_to_vary="all",
    )
    assert result == {"id": "e1", "status": "completed", "feature_weights": {}}


@pytest.mark.asyncio
async def test_generate_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_explanations_generate")

    await tool.run(
        {
            "model_id": "m1",
            "instance": {"age": 42},
            "method": "lime",
            "prediction_id": "p1",
            "num_features": 5,
            "num_samples": 1000,
            "use_cache": False,
        }
    )

    fake_client.explanations.agenerate.assert_awaited_once_with(
        "m1",
        {"age": 42},
        method="lime",
        prediction_id="p1",
        num_features=5,
        num_samples=1000,
        use_cache=False,
        total_cfs=3,
        desired_class="opposite",
        features_to_vary="all",
    )


@pytest.mark.asyncio
async def test_generate_bulk_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_explanations_generate_bulk")

    result = await tool.run({"model_id": "m1", "instances": [{"age": 42}, {"age": 30}]})

    fake_client.explanations.agenerate_bulk.assert_awaited_once_with(
        "m1",
        [{"age": 42}, {"age": 30}],
        method=None,
        num_features=10,
        num_samples=5000,
        parallel=True,
        total_cfs=3,
        desired_class="opposite",
        features_to_vary="all",
    )
    assert result == {"explanation_ids": ["e1", "e2"], "total": 2, "message": "ok"}


@pytest.mark.asyncio
async def test_list_by_model_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_explanations_list_by_model")

    result = await tool.run({"model_id": "m1"})

    fake_client.explanations.alist_by_model.assert_awaited_once_with(
        "m1", method=None, status=None, limit=50, offset=0
    )
    assert result == {"explanations": [], "total": 0, "limit": 50, "offset": 0}


@pytest.mark.asyncio
async def test_get_by_prediction_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_explanations_get_by_prediction")

    result = await tool.run({"prediction_id": "p1"})

    fake_client.explanations.aget_by_prediction.assert_awaited_once_with("p1")
    assert result == {"id": "e1", "prediction_id": "p1"}


@pytest.mark.asyncio
async def test_stats_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_explanations_stats")

    result = await tool.run({"model_id": "m1"})

    fake_client.explanations.aget_stats.assert_awaited_once_with("m1")
    assert result == {"model_id": "m1", "total_explanations": 0}


@pytest.mark.asyncio
async def test_set_config_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_explanations_set_config")

    result = await tool.run({"model_id": "m1"})

    fake_client.explanations.aset_config.assert_awaited_once_with(
        "m1",
        enabled=True,
        default_method="shap_kernel",
        auto_explain=False,
        cache_enabled=True,
        cache_ttl_hours=24,
        lime_config=None,
        shap_config=None,
        visualization_config=None,
    )
    assert result == {"id": 1, "model_id": "m1", "enabled": True}


@pytest.mark.asyncio
async def test_get_config_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_explanations_get_config")

    result = await tool.run({"model_id": "m1"})

    fake_client.explanations.aget_config.assert_awaited_once_with("m1")
    assert result == {"id": 1, "model_id": "m1", "enabled": True}


@pytest.mark.asyncio
async def test_delete_config_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_explanations_delete_config")

    result = await tool.run({"model_id": "m1"})

    fake_client.explanations.adelete_config.assert_awaited_once_with("m1")
    assert result == {"status": "success"}


@pytest.mark.asyncio
async def test_compare_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_explanations_compare")

    result = await tool.run({"explanation_ids": ["e1", "e2"]})

    fake_client.explanations.acompare.assert_awaited_once_with(
        ["e1", "e2"], comparison_type="feature_importance"
    )
    assert result == {
        "comparison_type": "feature_importance",
        "num_explanations": 2,
        "comparison_results": {},
    }


@pytest.mark.asyncio
async def test_visualize_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_explanations_visualize")

    result = await tool.run({"explanation_id": "e1", "plot_type": "waterfall"})

    fake_client.explanations.avisualize.assert_awaited_once_with(
        "e1", "waterfall", max_features=10, config=None
    )
    assert result["plot_type"] == "waterfall"


@pytest.mark.asyncio
async def test_visualize_multi_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_explanations_visualize_multi")

    result = await tool.run({"explanation_ids": ["e1", "e2"], "plot_type": "summary"})

    fake_client.explanations.avisualize_multi.assert_awaited_once_with(
        ["e1", "e2"],
        "summary",
        feature_index=None,
        interaction_index=None,
        max_features=20,
        highlight_index=None,
    )
    assert result["plot_type"] == "summary"
