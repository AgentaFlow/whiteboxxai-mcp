"""Behavior tests for whiteboxxai_mcp.tools.agent_workflows (issue #141)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.tools import agent_workflows


@pytest.fixture
def mcp_with_fake_session(monkeypatch):
    mcp = FastMCP(name="test")
    agent_workflows.register(mcp)

    fake_client = MagicMock()
    fake_client.agent_workflows.acreate_and_start = AsyncMock(return_value={"id": "w1"})
    fake_client.agent_workflows.astart = AsyncMock(return_value={"id": "w1"})
    fake_client.agent_workflows.acomplete = AsyncMock(return_value={"id": "w1"})
    fake_client.agent_workflows.alist = AsyncMock(
        return_value={"workflows": [], "total": 0}
    )
    fake_client.agent_workflows.aget = AsyncMock(return_value={"id": "w1"})
    fake_client.agent_workflows.adelete = AsyncMock(return_value={"deleted": True})
    fake_client.agent_workflows.aregister_agent = AsyncMock(return_value={"id": "ag1"})
    fake_client.agent_workflows.alist_agents = AsyncMock(return_value=[{"id": "ag1"}])
    fake_client.agent_workflows.acreate_execution = AsyncMock(return_value={"id": "e1"})
    fake_client.agent_workflows.alog_interaction = AsyncMock(return_value={"id": "i1"})
    fake_client.agent_workflows.alist_interactions = AsyncMock(
        return_value=[{"id": "i1"}]
    )
    fake_client.agent_workflows.acreate_task = AsyncMock(return_value={"id": "t1"})
    fake_client.agent_workflows.aupdate_task_status = AsyncMock(
        return_value={"id": "t1"}
    )
    fake_client.agent_workflows.alist_tasks = AsyncMock(return_value=[{"id": "t1"}])
    fake_client.agent_workflows.aanalytics = AsyncMock(return_value={"total_tokens": 0})
    fake_client.agent_workflows.acost_breakdown = AsyncMock(
        return_value={"total_cost": 0}
    )
    fake_client.agent_workflows.abottlenecks = AsyncMock(
        return_value={"slowest_agents": []}
    )
    fake_client.agent_workflows.atimeline = AsyncMock(return_value={"events": []})

    class FakeSession:
        async def get_client(self, force_refresh: bool = False):
            return fake_client

    monkeypatch.setattr(agent_workflows, "get_session", lambda: FakeSession())
    return mcp, fake_client


@pytest.mark.asyncio
async def test_create_and_start_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_agent_workflows_create_and_start")

    result = await tool.run({"name": "wf1", "framework": "crewai"})

    fake_client.agent_workflows.acreate_and_start.assert_awaited_once_with(
        "wf1", "crewai", meta_data=None, tags=None
    )
    assert result == {"id": "w1"}


@pytest.mark.asyncio
async def test_start_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_agent_workflows_start")

    await tool.run({"workflow_id": "w1"})

    fake_client.agent_workflows.astart.assert_awaited_once_with("w1", inputs=None)


@pytest.mark.asyncio
async def test_complete_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_agent_workflows_complete")

    await tool.run({"workflow_id": "w1", "status": "completed"})

    fake_client.agent_workflows.acomplete.assert_awaited_once_with(
        "w1", "completed", outputs=None, error_message=None, trigger_analytics=True
    )


@pytest.mark.asyncio
async def test_list_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_agent_workflows_list")

    await tool.run({"framework": "crewai"})

    fake_client.agent_workflows.alist.assert_awaited_once_with(
        status=None, framework="crewai", skip=0, limit=100
    )


@pytest.mark.asyncio
async def test_get_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_agent_workflows_get")

    await tool.run({"workflow_id": "w1"})

    fake_client.agent_workflows.aget.assert_awaited_once_with("w1")


@pytest.mark.asyncio
async def test_delete_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_agent_workflows_delete")

    await tool.run({"workflow_id": "w1"})

    fake_client.agent_workflows.adelete.assert_awaited_once_with("w1")


@pytest.mark.asyncio
async def test_register_agent_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_agent_workflows_register_agent")

    await tool.run({"workflow_id": "w1", "name": "Researcher"})

    fake_client.agent_workflows.aregister_agent.assert_awaited_once_with(
        "w1",
        "Researcher",
        role=None,
        agent_type=None,
        model_name=None,
        llm_provider=None,
        llm_config=None,
        system_prompt=None,
        goal=None,
        backstory=None,
        tools=None,
        capabilities=None,
        meta_data=None,
    )


@pytest.mark.asyncio
async def test_list_agents_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_agent_workflows_list_agents")

    await tool.run({"workflow_id": "w1"})

    fake_client.agent_workflows.alist_agents.assert_awaited_once_with("w1")


@pytest.mark.asyncio
async def test_create_execution_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_agent_workflows_create_execution")

    await tool.run({"workflow_id": "w1", "agent_id": "ag1"})

    fake_client.agent_workflows.acreate_execution.assert_awaited_once_with(
        "w1", "ag1", inputs=None, parent_execution_id=None, execution_order=None
    )


@pytest.mark.asyncio
async def test_log_interaction_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_agent_workflows_log_interaction")

    await tool.run({"workflow_id": "w1", "interaction_type": "delegation"})

    fake_client.agent_workflows.alog_interaction.assert_awaited_once_with(
        "w1",
        "delegation",
        from_agent_id=None,
        to_agent_id=None,
        from_execution_id=None,
        to_execution_id=None,
        message=None,
        meta_data=None,
    )


@pytest.mark.asyncio
async def test_list_interactions_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_agent_workflows_list_interactions")

    await tool.run({"workflow_id": "w1"})

    fake_client.agent_workflows.alist_interactions.assert_awaited_once_with("w1")


@pytest.mark.asyncio
async def test_create_task_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_agent_workflows_create_task")

    await tool.run({"workflow_id": "w1", "task_name": "Research"})

    fake_client.agent_workflows.acreate_task.assert_awaited_once_with(
        "w1",
        "Research",
        description=None,
        task_type=None,
        expected_output=None,
        agent_id=None,
        parent_task_id=None,
        priority=0,
        input_data=None,
        context=None,
    )


@pytest.mark.asyncio
async def test_update_task_status_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_agent_workflows_update_task_status")

    await tool.run({"task_id": "t1", "status": "completed"})

    fake_client.agent_workflows.aupdate_task_status.assert_awaited_once_with(
        "t1", "completed", output_data=None, error_message=None
    )


@pytest.mark.asyncio
async def test_list_tasks_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_agent_workflows_list_tasks")

    await tool.run({"workflow_id": "w1"})

    fake_client.agent_workflows.alist_tasks.assert_awaited_once_with("w1")


@pytest.mark.asyncio
async def test_analytics_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_agent_workflows_analytics")

    await tool.run({"workflow_id": "w1"})

    fake_client.agent_workflows.aanalytics.assert_awaited_once_with("w1")


@pytest.mark.asyncio
async def test_cost_breakdown_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_agent_workflows_cost_breakdown")

    await tool.run({"workflow_id": "w1"})

    fake_client.agent_workflows.acost_breakdown.assert_awaited_once_with("w1")


@pytest.mark.asyncio
async def test_bottlenecks_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_agent_workflows_bottlenecks")

    await tool.run({"workflow_id": "w1"})

    fake_client.agent_workflows.abottlenecks.assert_awaited_once_with("w1")


@pytest.mark.asyncio
async def test_timeline_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_agent_workflows_timeline")

    await tool.run({"workflow_id": "w1"})

    fake_client.agent_workflows.atimeline.assert_awaited_once_with("w1")
