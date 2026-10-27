"""MCP tools for the AI Risk Register (backend/api/v1/risk_register.py).

Read-only: the risk register is normally populated via the dashboard or
auto-drafted server-side from bias/drift/governance-review events, not
written to by SDK/MCP callers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.auth.context import get_session
from whiteboxxai_mcp.errors import call_with_session


def register(mcp: FastMCP) -> None:
    """Register all AI Risk Register tools on `mcp`."""

    @mcp.tool(name="whitebox_risk_register_list")
    async def whitebox_risk_register_list(
        status: str | None = None,
        severity: str | None = None,
        category: str | None = None,
        model_id: str | None = None,
        limit: int = 50,
        skip: int = 0,
    ) -> dict[str, Any]:
        """List AI risk register entries, optionally filtered.

        Args:
            status: One of identified, assessed, mitigation_planned, mitigated, closed
            severity: One of low, medium, high, critical
            category: Risk category (org-specific taxonomy, e.g. bias, drift, security)
            model_id: Restrict to risks linked to one registered model
            limit: Page size
            skip: Page offset

        Returns:
            {"items": [...], "total": int, "skip": int, "limit": int} --
            each item includes `iso42001_clause_refs`, the ISO 42001 clause
            IDs (e.g. "ISO42001-8.3") this risk maps to.
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.risk_register.alist(
                status=status,
                severity=severity,
                category=category,
                model_id=model_id,
                limit=limit,
                skip=skip,
            ),
        )

    @mcp.tool(name="whitebox_risk_register_get")
    async def whitebox_risk_register_get(entry_id: str) -> dict[str, Any]:
        """Get a single AI risk register entry by ID."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.risk_register.aget(entry_id)
        )

    @mcp.tool(name="whitebox_risk_register_portfolio")
    async def whitebox_risk_register_portfolio() -> dict[str, Any]:
        """Get the likelihood x impact heat map and top open risks across
        the organization's risk register.
        """
        session = get_session()
        return await call_with_session(
            session, lambda client: client.risk_register.aportfolio()
        )
