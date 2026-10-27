"""Behavior tests for whiteboxxai_mcp.tools.models (issue #227).

Brings the M1 model-registry domain up to the same behavioral-delegation
bar as the M2/#137 domains (see test_tools_explanations.py, the template
this file follows).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.tools import models


@pytest.fixture
def mcp_with_fake_session(monkeypatch):
    mcp = FastMCP(name="test")
    models.register(mcp)

    fake_client = MagicMock()
    fake_client.models.aregister = AsyncMock(
        return_value={"model_id": "m1", "name": "my-model", "status": "ACTIVE"}
    )
    fake_client.models.aget = AsyncMock(
        return_value={"model_id": "m1", "name": "my-model", "status": "ACTIVE"}
    )
    fake_client.models.alist = AsyncMock(return_value=[{"model_id": "m1"}])
    fake_client.models.aupdate = AsyncMock(
        return_value={"model_id": "m1", "description": "updated"}
    )
    fake_client.models.aupdate_status = AsyncMock(
        return_value={"model_id": "m1", "status": "ARCHIVED"}
    )
    fake_client.models.aget_versions = AsyncMock(
        return_value=[{"model_id": "m1", "version": "1.0.0"}]
    )
    fake_client.models.aget_latest = AsyncMock(
        return_value={"model_id": "m1", "version": "2.0.0"}
    )
    fake_client.models.aupdate_baseline = AsyncMock(
        return_value={"model_id": "m1", "baseline_metrics": {"accuracy": 0.95}}
    )
    fake_client.models.aarchive = AsyncMock(
        return_value={"model_id": "m1", "status": "ARCHIVED"}
    )
    fake_client.models.arestore = AsyncMock(
        return_value={"model_id": "m1", "status": "ACTIVE"}
    )
    fake_client.models.adelete = AsyncMock(return_value={"status": "success"})
    fake_client.models.acompare_versions = AsyncMock(
        return_value={
            "version1_id": "v1",
            "version2_id": "v2",
            "has_changes": True,
            "breaking_changes": False,
            "change_summary": "Retrained baseline",
        }
    )
    fake_client.models.aversion_changelog = AsyncMock(
        return_value={
            "version1_id": "v1",
            "version2_id": "v2",
            "changelog": "## Changes\n- Retrained",
        }
    )
    fake_client.models.alist_git_versions = AsyncMock(
        return_value=[{"id": "v1", "commit_hash": "abc1234"}]
    )
    fake_client.models.aarchitecture_changes_timeline = AsyncMock(
        return_value={
            "model_id": "m1",
            "total_versions": 2,
            "total_changes": 1,
            "changes": [{"summary": "Updated weights"}],
        }
    )

    class FakeSession:
        """Minimal stand-in for SessionManager (mirrors test_error_mapping.py)."""

        async def get_client(self, force_refresh: bool = False):
            return fake_client

    monkeypatch.setattr(models, "get_session", lambda: FakeSession())
    return mcp, fake_client


@pytest.mark.asyncio
async def test_register_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_register")

    result = await tool.run(
        {
            "name": "my-model",
            "model_type": "classification",
            "features": ["age", "income"],
            "baseline_metrics": {"accuracy": 0.94},
        }
    )

    fake_client.models.aregister.assert_awaited_once_with(
        name="my-model",
        model_type="classification",
        framework=None,
        version=None,
        features=["age", "income"],
        baseline_metrics={"accuracy": 0.94},
    )
    assert result == {"model_id": "m1", "name": "my-model", "status": "ACTIVE"}


@pytest.mark.asyncio
async def test_register_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_register")

    await tool.run(
        {
            "name": "my-model",
            "model_type": "classification",
            "features": ["age", "income"],
            "baseline_metrics": {"accuracy": 0.94},
            "framework": "sklearn",
            "version": "1.0.0",
            "description": "A model",
            "target_variable": "approved",
            "fairness_attributes": ["gender"],
            "tags": ["prod"],
        }
    )

    fake_client.models.aregister.assert_awaited_once_with(
        name="my-model",
        model_type="classification",
        framework="sklearn",
        version="1.0.0",
        features=["age", "income"],
        baseline_metrics={"accuracy": 0.94},
        description="A model",
        target_variable="approved",
        fairness_attributes=["gender"],
        tags=["prod"],
    )


@pytest.mark.asyncio
async def test_get_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_get")

    result = await tool.run({"model_id": "m1"})

    fake_client.models.aget.assert_awaited_once_with("m1")
    assert result == {"model_id": "m1", "name": "my-model", "status": "ACTIVE"}


@pytest.mark.asyncio
async def test_list_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_list")

    result = await tool.run({})

    fake_client.models.alist.assert_awaited_once_with(
        skip=0,
        limit=100,
        status_filter=None,
        model_type=None,
        tags=None,
        search=None,
    )
    assert result == [{"model_id": "m1"}]


@pytest.mark.asyncio
async def test_list_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_list")

    await tool.run(
        {
            "skip": 10,
            "limit": 5,
            "status_filter": "ACTIVE",
            "model_type": "classification",
            "search": "fraud",
            "tags": "prod,critical",
        }
    )

    fake_client.models.alist.assert_awaited_once_with(
        skip=10,
        limit=5,
        status_filter="ACTIVE",
        model_type="classification",
        tags="prod,critical",
        search="fraud",
    )


@pytest.mark.asyncio
async def test_update_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_update")

    result = await tool.run({"model_id": "m1"})

    fake_client.models.aupdate.assert_awaited_once_with("m1")
    assert result == {"model_id": "m1", "description": "updated"}


@pytest.mark.asyncio
async def test_update_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_update")

    await tool.run(
        {
            "model_id": "m1",
            "description": "new description",
            "target_variable": "approved",
            "fairness_attributes": ["gender", "race"],
            "tags": ["prod"],
        }
    )

    fake_client.models.aupdate.assert_awaited_once_with(
        "m1",
        description="new description",
        target_variable="approved",
        fairness_attributes=["gender", "race"],
        tags=["prod"],
    )


@pytest.mark.asyncio
async def test_update_status_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_update_status")

    result = await tool.run({"model_id": "m1", "new_status": "ARCHIVED"})

    fake_client.models.aupdate_status.assert_awaited_once_with("m1", "ARCHIVED")
    assert result == {"model_id": "m1", "status": "ARCHIVED"}


@pytest.mark.asyncio
async def test_versions_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_versions")

    result = await tool.run({"model_name": "my-model"})

    fake_client.models.aget_versions.assert_awaited_once_with(
        "my-model", skip=0, limit=100
    )
    assert result == [{"model_id": "m1", "version": "1.0.0"}]


@pytest.mark.asyncio
async def test_versions_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_versions")

    await tool.run({"model_name": "my-model", "skip": 5, "limit": 10})

    fake_client.models.aget_versions.assert_awaited_once_with(
        "my-model", skip=5, limit=10
    )


@pytest.mark.asyncio
async def test_latest_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_latest")

    result = await tool.run({"model_name": "my-model"})

    fake_client.models.aget_latest.assert_awaited_once_with("my-model")
    assert result == {"model_id": "m1", "version": "2.0.0"}


@pytest.mark.asyncio
async def test_update_baseline_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_update_baseline")

    result = await tool.run({"model_id": "m1", "baseline_metrics": {"accuracy": 0.95}})

    fake_client.models.aupdate_baseline.assert_awaited_once_with(
        "m1",
        {"accuracy": 0.95},
        baseline_data_hash=None,
        baseline_data_count=None,
    )
    assert result == {"model_id": "m1", "baseline_metrics": {"accuracy": 0.95}}


@pytest.mark.asyncio
async def test_update_baseline_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_update_baseline")

    await tool.run(
        {
            "model_id": "m1",
            "baseline_metrics": {"accuracy": 0.95},
            "baseline_data_hash": "abc123",
            "baseline_data_count": 5000,
        }
    )

    fake_client.models.aupdate_baseline.assert_awaited_once_with(
        "m1",
        {"accuracy": 0.95},
        baseline_data_hash="abc123",
        baseline_data_count=5000,
    )


@pytest.mark.asyncio
async def test_archive_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_archive")

    result = await tool.run({"model_id": "m1"})

    fake_client.models.aarchive.assert_awaited_once_with("m1")
    assert result == {"model_id": "m1", "status": "ARCHIVED"}


@pytest.mark.asyncio
async def test_restore_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_restore")

    result = await tool.run({"model_id": "m1"})

    fake_client.models.arestore.assert_awaited_once_with("m1")
    assert result == {"model_id": "m1", "status": "ACTIVE"}


@pytest.mark.asyncio
async def test_delete_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_delete")

    result = await tool.run({"model_id": "m1"})

    fake_client.models.adelete.assert_awaited_once_with("m1")
    assert result == {"status": "success"}


@pytest.mark.asyncio
async def test_compare_versions_delegates_to_sdk_and_enriches_verdict(
    mcp_with_fake_session,
):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_compare_versions")

    result = await tool.run({"version1_id": "v1", "version2_id": "v2"})

    fake_client.models.acompare_versions.assert_awaited_once_with("v1", "v2")
    assert result["version1_id"] == "v1"
    assert "plain_language_verdict" in result
    assert "[NON-BREAKING ARCHITECTURE DIFF]" in result["plain_language_verdict"]


@pytest.mark.asyncio
async def test_version_changelog_delegates_to_sdk_and_enriches_verdict(
    mcp_with_fake_session,
):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_version_changelog")

    result = await tool.run({"version1_id": "v1", "version2_id": "v2"})

    fake_client.models.aversion_changelog.assert_awaited_once_with("v1", "v2")
    assert result["changelog"] == "## Changes\n- Retrained"
    assert "plain_language_verdict" in result
    assert "Markdown changelog generated" in result["plain_language_verdict"]


@pytest.mark.asyncio
async def test_list_git_versions_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_list_git_versions")

    result = await tool.run({"model_id": "m1", "limit": 25})

    fake_client.models.alist_git_versions.assert_awaited_once_with("m1", limit=25)
    assert result == [{"id": "v1", "commit_hash": "abc1234"}]


@pytest.mark.asyncio
async def test_architecture_timeline_delegates_to_sdk_and_enriches_verdict(
    mcp_with_fake_session,
):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_models_architecture_timeline")

    result = await tool.run({"model_id": "m1"})

    fake_client.models.aarchitecture_changes_timeline.assert_awaited_once_with("m1")
    assert result["total_changes"] == 1
    assert "plain_language_verdict" in result
    assert "Architecture changes timeline" in result["plain_language_verdict"]
