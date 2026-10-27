"""MCP tools for Prompt Governance & Versioning (backend/api/v1/prompts.py):
lifecycle management, gated deployments, emergency rollbacks, and prompt drift detection.
"""

from __future__ import annotations

from typing import Any, List, Optional

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.auth.context import get_session
from whiteboxxai_mcp.errors import call_with_session


def register(mcp: FastMCP) -> None:
    """Register all prompt governance & lifecycle tools on `mcp`."""

    @mcp.tool(name="whitebox_prompts_list_templates")
    async def whitebox_prompts_list_templates() -> list[dict[str, Any]]:
        """List all prompt governance templates for the authenticated tenant."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.prompts.alist_templates()
        )

    @mcp.tool(name="whitebox_prompts_get_template")
    async def whitebox_prompts_get_template(template_id: str) -> dict[str, Any]:
        """Get details of a specific prompt template by ID."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.prompts.aget_template(template_id)
        )

    @mcp.tool(name="whitebox_prompts_create_template")
    async def whitebox_prompts_create_template(
        name: str,
        description: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new prompt template container for versioning."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.prompts.acreate_template(
                name=name, description=description, tags=tags
            ),
        )

    @mcp.tool(name="whitebox_prompts_list_versions")
    async def whitebox_prompts_list_versions(
        template_id: str,
    ) -> list[dict[str, Any]]:
        """List all append-only governed versions for a prompt template."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.prompts.alist_versions(template_id)
        )

    @mcp.tool(name="whitebox_prompts_get_version")
    async def whitebox_prompts_get_version(
        template_id: str, version_id: str
    ) -> dict[str, Any]:
        """Get details, lifecycle stage, and verification status of a prompt version."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.prompts.aget_version(template_id, version_id)
        )

    @mcp.tool(name="whitebox_prompts_create_version")
    async def whitebox_prompts_create_version(
        template_id: str,
        user_message_template: str,
        system_message: str | None = None,
        model_name: str | None = None,
        model_id: str | None = None,
        temperature: float | None = None,
        tool_schemas: dict[str, Any] | None = None,
        version_tag: str | None = None,
        commit_hash: str | None = None,
    ) -> dict[str, Any]:
        """Create a new governed prompt version in 'design' lifecycle stage."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.prompts.acreate_version(
                template_id=template_id,
                user_message_template=user_message_template,
                system_message=system_message,
                model_name=model_name,
                model_id=model_id,
                temperature=temperature,
                tool_schemas=tool_schemas,
                version_tag=version_tag,
                commit_hash=commit_hash,
            ),
        )

    @mcp.tool(name="whitebox_prompts_advance_stage")
    async def whitebox_prompts_advance_stage(
        template_id: str,
        version_id: str,
        target_stage: str,
        notes: str | None = None,
        verification_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Advance a prompt version through the lifecycle stages (design -> verify -> deploy -> retired).

        Promotion from verify to deploy requires a verified benchmark run.
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.prompts.aadvance_stage(
                template_id=template_id,
                version_id=version_id,
                target_stage=target_stage,
                notes=notes,
                verification_run_id=verification_run_id,
            ),
        )

    @mcp.tool(name="whitebox_prompts_deploy")
    async def whitebox_prompts_deploy(
        template_id: str,
        version_id: str,
        deployment_notes: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Promote a prompt version to active production deployment.

        Enforces single active deployment per template and requires benchmark verification (unless force=True).
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.prompts.adeploy(
                template_id=template_id,
                version_id=version_id,
                deployment_notes=deployment_notes,
                force=force,
            ),
        )

    @mcp.tool(name="whitebox_prompts_rollback")
    async def whitebox_prompts_rollback(
        template_id: str,
        target_version_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Emergency rollback to a target version or the most recently verified version."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.prompts.arollback(
                template_id=template_id,
                target_version_id=target_version_id,
                reason=reason,
            ),
        )

    @mcp.tool(name="whitebox_prompts_compute_drift")
    async def whitebox_prompts_compute_drift(
        template_id: str,
        version_id: str,
        baseline_dataset_id: str | None = None,
        sample_size: int = 100,
    ) -> dict[str, Any]:
        """Compute prompt drift analysis comparing production completions with baseline expectations."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.prompts.acompute_drift(
                template_id=template_id,
                version_id=version_id,
                baseline_dataset_id=baseline_dataset_id,
                sample_size=sample_size,
            ),
        )

    @mcp.tool(name="whitebox_prompts_list_drift_reports")
    async def whitebox_prompts_list_drift_reports(
        template_id: str,
        version_id: str,
    ) -> list[dict[str, Any]]:
        """List historical prompt drift reports for a prompt version."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.prompts.alist_drift_reports(
                template_id=template_id, version_id=version_id
            ),
        )

    @mcp.tool(name="whitebox_prompts_get_drift_report")
    async def whitebox_prompts_get_drift_report(
        template_id: str,
        version_id: str,
        report_id: str,
    ) -> dict[str, Any]:
        """Get a specific prompt drift report with metrics, divergence, and auditor verdict."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.prompts.aget_drift_report(
                template_id=template_id, version_id=version_id, report_id=report_id
            ),
        )

    @mcp.tool(name="whitebox_prompts_get_incidents")
    async def whitebox_prompts_get_incidents(
        template_id: str,
        version_id: str,
    ) -> dict[str, Any]:
        """Get security incidents and alerts associated with a prompt version."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.prompts.aget_incidents(
                template_id=template_id, version_id=version_id
            ),
        )
