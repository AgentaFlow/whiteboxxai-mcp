"""MCP tools for ISO/IEC 42001 Readiness and Control Catalog (issue #856).

Wraps `backend/api/v1/iso42001.py` (`/api/v1/compliance/iso42001/score` and
`/api/v1/compliance/iso42001/gaps`) and `backend/api/v1/control_catalog.py`
(`/api/v1/compliance/frameworks` and `/api/v1/compliance/frameworks/{framework}/controls`).

Scope invariant: WhiteBoxXAI's compliance scope is strictly the five supported
frameworks (ISO/IEC 42001, GDPR, CCPA, EU AI Act, NIST AI RMF). When ISO/IEC 42001
controls have not yet been assessed (`assessed: False`), the plain-language
verdict explicitly states "Not yet assessed" rather than reporting 0% as failed
controls.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.auth.context import get_session
from whiteboxxai_mcp.errors import call_with_session
from whiteboxxai_mcp.ui_resources import (
    ISO42001_STATUS_CARD_URI,
    build_iso42001_verdict,
    ui_tool_meta,
)


async def _fetch_iso42001_status(
    client: Any,
    clause_filter: str | None = None,
    severity_filter: str | None = None,
) -> dict[str, Any]:
    """Fetch score, gaps, and ISO 42001-linked risk entries in a single helper."""
    score_raw = await client.compliance.aiso42001_score()
    gaps_raw = await client.compliance.aiso42001_gaps(
        clause_filter=clause_filter,
        severity_filter=severity_filter,
    )
    score_dict = score_raw if isinstance(score_raw, dict) else {}
    gaps_dict = (
        gaps_raw
        if isinstance(gaps_raw, dict)
        else {"gaps": gaps_raw if isinstance(gaps_raw, list) else []}
    )

    linked_risks: list[dict[str, Any]] = []
    try:
        risk_page = await client.risk_register.alist(limit=50)
        items = (
            risk_page.get("items", [])
            if isinstance(risk_page, dict)
            else (risk_page if isinstance(risk_page, list) else [])
        )
        for item in items:
            if isinstance(item, dict) and item.get("iso42001_clause_refs"):
                linked_risks.append(item)
    except Exception:
        linked_risks = []

    verdict = build_iso42001_verdict(score_dict, gaps_dict)
    return {
        "score": score_dict,
        "gaps": gaps_dict,
        "linked_risks": linked_risks,
        "plain_language_verdict": verdict,
    }


def register(mcp: FastMCP) -> None:
    """Register all Compliance & ISO/IEC 42001 tools on `mcp`."""

    @mcp.tool(
        name="whitebox_compliance_iso42001_score",
        meta=ui_tool_meta(ISO42001_STATUS_CARD_URI),
    )
    async def whitebox_compliance_iso42001_score() -> dict[str, Any]:
        """Get the organization's ISO/IEC 42001 readiness score and clause breakdown.

        When `assessed` is False in the response, no control assessments have
        been recorded yet and numeric scores are 0.0 because of unassessed state,
        not control failure.
        """
        session = get_session()
        score_data = await call_with_session(
            session, lambda client: client.compliance.aiso42001_score()
        )
        if isinstance(score_data, dict):
            enriched = dict(score_data)
            enriched["plain_language_verdict"] = build_iso42001_verdict(enriched)
            return enriched
        return score_data

    @mcp.tool(
        name="whitebox_compliance_iso42001_gaps",
        meta=ui_tool_meta(ISO42001_STATUS_CARD_URI),
    )
    async def whitebox_compliance_iso42001_gaps(
        clause_filter: str | None = None,
        severity_filter: str | None = None,
    ) -> dict[str, Any]:
        """Get the organization's open ISO/IEC 42001 compliance gaps.

        Args:
            clause_filter: Optional ISO/IEC 42001 clause prefix (e.g. '4', '8')
            severity_filter: Optional gap severity filter (e.g. 'high', 'critical')
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.compliance.aiso42001_gaps(
                clause_filter=clause_filter,
                severity_filter=severity_filter,
            ),
        )

    @mcp.tool(
        name="whitebox_compliance_iso42001_status",
        meta=ui_tool_meta(ISO42001_STATUS_CARD_URI),
    )
    async def whitebox_compliance_iso42001_status(
        clause_filter: str | None = None,
        severity_filter: str | None = None,
    ) -> dict[str, Any]:
        """Get a composite ISO/IEC 42001 readiness scorecard, open gap matrix,
        and linked AI risk register entries in a single call for the
        `ui://whitebox/iso42001-status-card` widget.
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: _fetch_iso42001_status(
                client,
                clause_filter=clause_filter,
                severity_filter=severity_filter,
            ),
        )

    @mcp.tool(name="whitebox_compliance_list_frameworks")
    async def whitebox_compliance_list_frameworks() -> list[dict[str, Any]]:
        """List all supported compliance frameworks in the Control Catalog
        (ISO/IEC 42001, GDPR, CCPA, EU AI Act, NIST AI RMF) with control counts.
        """
        session = get_session()
        return await call_with_session(
            session, lambda client: client.compliance.alist_frameworks()
        )

    @mcp.tool(name="whitebox_compliance_get_framework_controls")
    async def whitebox_compliance_get_framework_controls(
        framework: str,
    ) -> dict[str, Any]:
        """Get the full control catalog for one compliance framework.

        Args:
            framework: Framework slug (e.g. 'iso_42001', 'eu_ai_act', 'nist_ai_rmf',
                'gdpr', 'ccpa')
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.compliance.aget_framework_controls(framework),
        )
