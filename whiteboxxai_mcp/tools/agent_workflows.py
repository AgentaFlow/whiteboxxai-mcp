"""MCP tools for multi-agent workflow tracking (backend/api/v1/agent_workflows.py):
CrewAI/LangChain/AutoGen/n8n workflow, agent, execution, interaction, and
task tracking, plus cost/bottleneck/timeline analytics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.auth.context import get_session
from whiteboxxai_mcp.errors import call_with_session


def register(mcp: FastMCP) -> None:
    """Register all multi-agent-workflow tools on `mcp`."""

    @mcp.tool(name="whitebox_agent_workflows_create_and_start")
    async def whitebox_agent_workflows_create_and_start(
        name: str,
        framework: str,
        meta_data: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new multi-agent workflow.

        framework: "crewai", "langchain", "autogen", "n8n", or "custom".
        Returns the created workflow (including its id) for subsequent
        agent/task registration.
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.agent_workflows.acreate_and_start(
                name, framework, meta_data=meta_data, tags=tags
            ),
        )

    @mcp.tool(name="whitebox_agent_workflows_start")
    async def whitebox_agent_workflows_start(
        workflow_id: str, inputs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Mark a workflow as running. Call after registering agents/tasks
        and before executing."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.agent_workflows.astart(workflow_id, inputs=inputs),
        )

    @mcp.tool(name="whitebox_agent_workflows_complete")
    async def whitebox_agent_workflows_complete(
        workflow_id: str,
        status: str,
        outputs: dict[str, Any] | None = None,
        error_message: str | None = None,
        trigger_analytics: bool = True,
    ) -> dict[str, Any]:
        """Mark a workflow completed/failed/cancelled and optionally trigger
        async analytics (metrics, cost, bottlenecks, timeline)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.agent_workflows.acomplete(
                workflow_id,
                status,
                outputs=outputs,
                error_message=error_message,
                trigger_analytics=trigger_analytics,
            ),
        )

    @mcp.tool(name="whitebox_agent_workflows_list")
    async def whitebox_agent_workflows_list(
        status: str | None = None,
        framework: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """List workflows for the organization. Returns {workflows, total, skip, limit}."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.agent_workflows.alist(
                status=status, framework=framework, skip=skip, limit=limit
            ),
        )

    @mcp.tool(name="whitebox_agent_workflows_get")
    async def whitebox_agent_workflows_get(workflow_id: str) -> dict[str, Any]:
        """Get a workflow by ID."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.agent_workflows.aget(workflow_id)
        )

    @mcp.tool(name="whitebox_agent_workflows_delete")
    async def whitebox_agent_workflows_delete(workflow_id: str) -> dict[str, Any]:
        """Delete a workflow and all related agents/executions/tasks/interactions."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.agent_workflows.adelete(workflow_id)
        )

    @mcp.tool(name="whitebox_agent_workflows_register_agent")
    async def whitebox_agent_workflows_register_agent(
        workflow_id: str,
        name: str,
        role: str | None = None,
        agent_type: str | None = None,
        model_name: str | None = None,
        llm_provider: str | None = None,
        llm_config: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        goal: str | None = None,
        backstory: str | None = None,
        tools: list[str] | None = None,
        capabilities: dict[str, Any] | None = None,
        meta_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register an agent in a workflow. Call once per agent before
        starting execution."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.agent_workflows.aregister_agent(
                workflow_id,
                name,
                role=role,
                agent_type=agent_type,
                model_name=model_name,
                llm_provider=llm_provider,
                llm_config=llm_config,
                system_prompt=system_prompt,
                goal=goal,
                backstory=backstory,
                tools=tools,
                capabilities=capabilities,
                meta_data=meta_data,
            ),
        )

    @mcp.tool(name="whitebox_agent_workflows_list_agents")
    async def whitebox_agent_workflows_list_agents(
        workflow_id: str,
    ) -> list[dict[str, Any]]:
        """List all agents registered in a workflow."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.agent_workflows.alist_agents(workflow_id)
        )

    @mcp.tool(name="whitebox_agent_workflows_create_execution")
    async def whitebox_agent_workflows_create_execution(
        workflow_id: str,
        agent_id: str,
        inputs: dict[str, Any] | None = None,
        parent_execution_id: str | None = None,
        execution_order: int | None = None,
    ) -> dict[str, Any]:
        """Log that an agent started executing. Set parent_execution_id for
        hierarchical/delegated executions."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.agent_workflows.acreate_execution(
                workflow_id,
                agent_id,
                inputs=inputs,
                parent_execution_id=parent_execution_id,
                execution_order=execution_order,
            ),
        )

    @mcp.tool(name="whitebox_agent_workflows_log_interaction")
    async def whitebox_agent_workflows_log_interaction(
        workflow_id: str,
        interaction_type: str,
        from_agent_id: str | None = None,
        to_agent_id: str | None = None,
        from_execution_id: str | None = None,
        to_execution_id: str | None = None,
        message: str | None = None,
        meta_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Log agent-to-agent communication. interaction_type: handoff,
        delegation, query, feedback, or broadcast."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.agent_workflows.alog_interaction(
                workflow_id,
                interaction_type,
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                from_execution_id=from_execution_id,
                to_execution_id=to_execution_id,
                message=message,
                meta_data=meta_data,
            ),
        )

    @mcp.tool(name="whitebox_agent_workflows_list_interactions")
    async def whitebox_agent_workflows_list_interactions(
        workflow_id: str,
    ) -> list[dict[str, Any]]:
        """List all agent-to-agent interactions in a workflow, chronologically."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.agent_workflows.alist_interactions(workflow_id),
        )

    @mcp.tool(name="whitebox_agent_workflows_create_task")
    async def whitebox_agent_workflows_create_task(
        workflow_id: str,
        task_name: str,
        description: str | None = None,
        task_type: str | None = None,
        expected_output: str | None = None,
        agent_id: str | None = None,
        parent_task_id: str | None = None,
        priority: int = 0,
        input_data: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a workflow task. task_type: research/write/analyze/delegate/review."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.agent_workflows.acreate_task(
                workflow_id,
                task_name,
                description=description,
                task_type=task_type,
                expected_output=expected_output,
                agent_id=agent_id,
                parent_task_id=parent_task_id,
                priority=priority,
                input_data=input_data,
                context=context,
            ),
        )

    @mcp.tool(name="whitebox_agent_workflows_update_task_status")
    async def whitebox_agent_workflows_update_task_status(
        task_id: str,
        status: str,
        output_data: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        """Update a task's status. status: pending/running/completed/failed/skipped."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.agent_workflows.aupdate_task_status(
                task_id, status, output_data=output_data, error_message=error_message
            ),
        )

    @mcp.tool(name="whitebox_agent_workflows_list_tasks")
    async def whitebox_agent_workflows_list_tasks(
        workflow_id: str,
    ) -> list[dict[str, Any]]:
        """List all tasks in a workflow, with status, assignments, and results."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.agent_workflows.alist_tasks(workflow_id)
        )

    @mcp.tool(name="whitebox_agent_workflows_analytics")
    async def whitebox_agent_workflows_analytics(workflow_id: str) -> dict[str, Any]:
        """Get workflow analytics (tokens, costs, execution counts, durations).
        Calculated asynchronously after workflow completion."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.agent_workflows.aanalytics(workflow_id)
        )

    @mcp.tool(name="whitebox_agent_workflows_cost_breakdown")
    async def whitebox_agent_workflows_cost_breakdown(
        workflow_id: str,
    ) -> dict[str, Any]:
        """Get per-agent cost breakdown (tokens, cost, execution count, avg
        duration). This may take up to ~30s: it runs a Celery task the
        backend blocks on server-side before responding."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.agent_workflows.acost_breakdown(workflow_id)
        )

    @mcp.tool(name="whitebox_agent_workflows_bottlenecks")
    async def whitebox_agent_workflows_bottlenecks(workflow_id: str) -> dict[str, Any]:
        """Identify workflow bottlenecks: slowest agents, slowest tasks,
        failing agents. This may take up to ~30s: it runs a Celery task the
        backend blocks on server-side before responding."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.agent_workflows.abottlenecks(workflow_id)
        )

    @mcp.tool(name="whitebox_agent_workflows_timeline")
    async def whitebox_agent_workflows_timeline(workflow_id: str) -> dict[str, Any]:
        """Get a chronological timeline of workflow events (start/end, agent
        executions, task updates, interactions). This may take up to ~30s:
        it runs a Celery task the backend blocks on server-side before
        responding."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.agent_workflows.atimeline(workflow_id)
        )
