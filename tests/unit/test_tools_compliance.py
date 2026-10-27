"""Behavior tests for `whiteboxxai_mcp.tools.compliance` (issue #856)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.tools import compliance


@pytest.fixture
def mcp_with_fake_session(monkeypatch):
    mcp = FastMCP(name="test")
    compliance.register(mcp)

    fake_client = MagicMock()
    fake_client.compliance.aiso42001_score = AsyncMock(
        return_value={
            "overall_score": 78.5,
            "assessed": True,
            "implementation_status": {
                "implemented": 22,
                "partial": 5,
                "not_implemented": 3,
            },
            "clause_scores": {"4": 85.0, "8": 72.0},
        }
    )
    fake_client.compliance.aiso42001_gaps = AsyncMock(
        return_value={
            "gaps": [{"control_id": "ISO42001-8.3", "severity": "high"}],
            "total_gaps": 1,
        }
    )
    fake_client.compliance.alist_frameworks = AsyncMock(
        return_value=[{"framework": "iso_42001", "total_controls": 38}]
    )
    fake_client.compliance.aget_framework_controls = AsyncMock(
        return_value={"framework": "iso_42001", "controls": [{"id": "A.6.2.6"}]}
    )
    fake_client.risk_register.alist = AsyncMock(
        return_value={
            "items": [
                {
                    "id": "r-1",
                    "title": "Credit drift risk",
                    "iso42001_clause_refs": ["ISO42001-8.3"],
                },
                {
                    "id": "r-2",
                    "title": "Unmapped operational note",
                    "iso42001_clause_refs": [],
                },
            ],
            "total": 2,
        }
    )

    class FakeSession:
        async def get_client(self, force_refresh: bool = False):
            return fake_client

    monkeypatch.setattr(compliance, "get_session", lambda: FakeSession())
    return mcp, fake_client


@pytest.mark.asyncio
async def test_iso42001_score_enriches_plain_language_verdict(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_compliance_iso42001_score")

    result = await tool.run({})

    fake_client.compliance.aiso42001_score.assert_awaited_once_with()
    assert result["overall_score"] == 78.5
    assert "78.5%" in result["plain_language_verdict"]
    assert "22 implemented" in result["plain_language_verdict"]


@pytest.mark.asyncio
async def test_iso42001_score_handles_unassessed_honestly(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    fake_client.compliance.aiso42001_score = AsyncMock(
        return_value={
            "overall_score": 0.0,
            "assessed": False,
            "implementation_status": {
                "implemented": 0,
                "partial": 0,
                "not_implemented": 0,
            },
            "clause_scores": {},
        }
    )
    tool = mcp._tool_manager.get_tool("whitebox_compliance_iso42001_score")

    result = await tool.run({})

    assert "Not yet assessed" in result["plain_language_verdict"]


@pytest.mark.asyncio
async def test_iso42001_gaps_passes_filters(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_compliance_iso42001_gaps")

    result = await tool.run(
        {
            "clause_filter": "8",
            "severity_filter": "high",
        }
    )

    fake_client.compliance.aiso42001_gaps.assert_awaited_once_with(
        clause_filter="8",
        severity_filter="high",
    )
    assert result["total_gaps"] == 1


@pytest.mark.asyncio
async def test_iso42001_status_combines_score_gaps_and_linked_risks(
    mcp_with_fake_session,
):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_compliance_iso42001_status")

    result = await tool.run({"clause_filter": "8"})

    assert result["score"]["overall_score"] == 78.5
    assert result["gaps"]["total_gaps"] == 1
    assert len(result["linked_risks"]) == 1
    assert result["linked_risks"][0]["id"] == "r-1"
    assert "78.5%" in result["plain_language_verdict"]
    assert "Open compliance gaps: 1." in result["plain_language_verdict"]


@pytest.mark.asyncio
async def test_list_frameworks_and_get_framework_controls(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    list_tool = mcp._tool_manager.get_tool("whitebox_compliance_list_frameworks")
    controls_tool = mcp._tool_manager.get_tool(
        "whitebox_compliance_get_framework_controls"
    )

    frameworks = await list_tool.run({})
    controls = await controls_tool.run({"framework": "iso_42001"})

    assert frameworks == [{"framework": "iso_42001", "total_controls": 38}]
    assert controls["framework"] == "iso_42001"
    fake_client.compliance.aget_framework_controls.assert_awaited_once_with("iso_42001")
