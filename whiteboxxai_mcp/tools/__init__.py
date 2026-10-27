"""Registers all MCP tool domains on a FastMCP server instance."""

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.tools import (
    agent_workflows,
    alert_intelligence,
    alert_management,
    alerts,
    compliance,
    drift,
    explanations,
    fairness,
    governance,
    llm_monitoring,
    llm_xai,
    metrics,
    models,
    predictions,
    prompts,
    rag,
    risk_register,
    safety,
)

_DOMAINS = (
    models,
    predictions,
    prompts,
    drift,
    fairness,
    explanations,
    risk_register,
    governance,
    compliance,
    llm_xai,
    llm_monitoring,
    rag,
    safety,
    alerts,
    alert_intelligence,
    alert_management,
    agent_workflows,
    metrics,
)


def register_all_tools(mcp: FastMCP) -> None:
    """Register every capability domain's tools on `mcp`.

    `alert_management`'s escalation-policy CRUD and bulk-action endpoints
    are deliberately not registered as tools here (admin-console-shaped,
    not agent-shaped) -- they remain reachable via AlertManagementResource
    in the SDK. See issue #224.
    """
    for domain in _DOMAINS:
        domain.register(mcp)
