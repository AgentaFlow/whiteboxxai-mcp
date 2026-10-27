"""Behavior tests for whiteboxxai_mcp.tools.fairness (issue #227).

Brings the M1 bias/fairness domain up to the same behavioral-delegation
bar as the M2/#137 domains (see test_tools_explanations.py, the template
this file follows).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.tools import fairness


@pytest.fixture
def mcp_with_fake_session(monkeypatch):
    mcp = FastMCP(name="test")
    fairness.register(mcp)

    fake_client = MagicMock()
    fake_client.fairness.aaudit = AsyncMock(
        return_value={"audit_id": "a1", "model_id": "m1", "overall_score": 0.9}
    )
    fake_client.fairness.aget_audit = AsyncMock(
        return_value={"audit_id": "a1", "model_id": "m1"}
    )
    fake_client.fairness.alist_audits = AsyncMock(return_value=[{"audit_id": "a1"}])
    fake_client.fairness.aget_bias_history = AsyncMock(
        return_value={"model_id": "m1", "history": []}
    )
    fake_client.fairness.aget_metric_history = AsyncMock(
        return_value={
            "model_id": "m1",
            "metric_type": "demographic_parity",
            "history": [],
        }
    )
    fake_client.fairness.aget_latest_audit = AsyncMock(
        return_value={"audit_id": "a1", "model_id": "m1"}
    )

    class FakeSession:
        """Minimal stand-in for SessionManager (mirrors test_error_mapping.py)."""

        async def get_client(self, force_refresh: bool = False):
            return fake_client

    monkeypatch.setattr(fairness, "get_session", lambda: FakeSession())
    return mcp, fake_client


@pytest.mark.asyncio
async def test_audit_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_fairness_audit")

    result = await tool.run(
        {
            "model_id": "m1",
            "sensitive_attributes": ["gender"],
            "y_true": [1, 0, 1],
            "y_pred": [1, 0, 0],
            "group_data": {"gender": ["M", "F", "M"]},
        }
    )

    fake_client.fairness.aaudit.assert_awaited_once_with(
        model_id="m1",
        sensitive_attributes=["gender"],
        y_true=[1, 0, 1],
        y_pred=[1, 0, 0],
        group_data={"gender": ["M", "F", "M"]},
        y_prob=None,
        metrics_to_compute=None,
    )
    assert result == {"audit_id": "a1", "model_id": "m1", "overall_score": 0.9}


@pytest.mark.asyncio
async def test_audit_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_fairness_audit")

    await tool.run(
        {
            "model_id": "m1",
            "sensitive_attributes": ["gender"],
            "y_true": [1, 0, 1],
            "y_pred": [1, 0, 0],
            "group_data": {"gender": ["M", "F", "M"]},
            "y_prob": [0.9, 0.2, 0.4],
            "metrics_to_compute": ["demographic_parity", "equal_opportunity"],
        }
    )

    fake_client.fairness.aaudit.assert_awaited_once_with(
        model_id="m1",
        sensitive_attributes=["gender"],
        y_true=[1, 0, 1],
        y_pred=[1, 0, 0],
        group_data={"gender": ["M", "F", "M"]},
        y_prob=[0.9, 0.2, 0.4],
        metrics_to_compute=["demographic_parity", "equal_opportunity"],
    )


@pytest.mark.asyncio
async def test_get_audit_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_fairness_get_audit")

    result = await tool.run({"audit_id": "a1"})

    fake_client.fairness.aget_audit.assert_awaited_once_with("a1")
    assert result == {"audit_id": "a1", "model_id": "m1"}


@pytest.mark.asyncio
async def test_list_audits_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_fairness_list_audits")

    result = await tool.run({})

    fake_client.fairness.alist_audits.assert_awaited_once_with(
        model_id=None, limit=50, offset=0
    )
    assert result == [{"audit_id": "a1"}]


@pytest.mark.asyncio
async def test_list_audits_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_fairness_list_audits")

    await tool.run({"model_id": "m1", "limit": 10, "offset": 20})

    fake_client.fairness.alist_audits.assert_awaited_once_with(
        model_id="m1", limit=10, offset=20
    )


@pytest.mark.asyncio
async def test_bias_history_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_fairness_bias_history")

    result = await tool.run({"model_id": "m1"})

    fake_client.fairness.aget_bias_history.assert_awaited_once_with("m1", days=30)
    assert result == {"model_id": "m1", "history": []}


@pytest.mark.asyncio
async def test_bias_history_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_fairness_bias_history")

    await tool.run({"model_id": "m1", "days": 90})

    fake_client.fairness.aget_bias_history.assert_awaited_once_with("m1", days=90)


@pytest.mark.asyncio
async def test_metric_history_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_fairness_metric_history")

    result = await tool.run({"model_id": "m1", "metric_type": "demographic_parity"})

    fake_client.fairness.aget_metric_history.assert_awaited_once_with(
        "m1", "demographic_parity", days=30
    )
    assert result == {
        "model_id": "m1",
        "metric_type": "demographic_parity",
        "history": [],
    }


@pytest.mark.asyncio
async def test_metric_history_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_fairness_metric_history")

    await tool.run({"model_id": "m1", "metric_type": "demographic_parity", "days": 60})

    fake_client.fairness.aget_metric_history.assert_awaited_once_with(
        "m1", "demographic_parity", days=60
    )


@pytest.mark.asyncio
async def test_latest_audit_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_fairness_latest_audit")

    result = await tool.run({"model_id": "m1"})

    fake_client.fairness.aget_latest_audit.assert_awaited_once_with("m1")
    assert result == {"audit_id": "a1", "model_id": "m1"}
