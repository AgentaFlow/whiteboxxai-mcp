"""MCP tools for governance review boards and RACI (backend/api/v1/review_boards.py).

Supports board/request/RACI inspection plus review voting from the dossier
card (`whitebox_governance_submit_decision`, issues #856 and #860).

A vote has to be attributable to a person who saw the evidence, which takes
three separate checks, none of them made here:
- The credential: the backend's `require_human_session` refuses organization
  API keys (`wbx_live_...`), so the vote is recorded under the reviewer's own
  JWT (in practice the remote Streamable HTTP `/mcp` transport).
- The action: the vote tool is app-only (`_meta.ui.visibility: ["app"]`), so
  the host leaves it out of the agent's tool list and only a click in the
  dossier card can call it. Without this, an agent holding a reviewer's JWT
  could cast the vote itself.
- The presence and the evidence: the backend requires a fresh authenticator
  code with every vote and refuses it (409) if the request changed since the
  dossier was shown. This server only forwards both and relays the backend's
  answer; it keeps no copy of the check to drift out of step.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from whiteboxxai.exceptions import APIError, RateLimitError

from whiteboxxai_mcp.auth.context import get_session
from whiteboxxai_mcp.errors import call_with_session
from whiteboxxai_mcp.ui_resources import (
    GOVERNANCE_DOSSIER_CARD_URI,
    enrich_review_dossier,
    ui_tool_meta,
)


async def _submit_vote(
    client: Any,
    request_id: str,
    decision: str,
    evidence_snapshot_hash: str,
    totp_code: str,
    justification: str | None = None,
    conditions: str | None = None,
) -> dict[str, Any]:
    """Forward the vote and surface the backend's refusal as a tool error.

    The backend answers a stale dossier (409), a missing or wrong authenticator
    code (400), an account without 2FA (403) and a lockout (429) with a message
    written for the reviewer. Left to `call_with_session` those would become
    "backend request failed" -- or, for the lockout, a "plan limit" notice -- so
    they are turned into tool errors here, which the card shows as they are.
    """
    try:
        return await client.governance.asubmit_decision(
            request_id,
            decision=decision,
            justification=justification,
            conditions=conditions,
            evidence_snapshot_hash=evidence_snapshot_hash,
            totp_code=totp_code,
        )
    except RateLimitError as exc:
        raise ToolError(str(exc)) from exc
    except APIError as exc:
        # ServerError and connection failures are APIErrors too; leave those to
        # call_with_session, which words them as "backend unavailable".
        status = getattr(exc, "status_code", None)
        if status is None or status >= 500:
            raise
        response = getattr(exc, "response", None)
        detail = response.get("detail") if isinstance(response, dict) else None
        raise ToolError(detail if isinstance(detail, str) else str(exc)) from exc


def register(mcp: FastMCP) -> None:
    """Register all governance review board / RACI tools on `mcp`."""

    @mcp.tool(name="whitebox_governance_list_boards")
    async def whitebox_governance_list_boards() -> list[dict[str, Any]]:
        """List governance review boards for the organization."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.governance.alist_boards()
        )

    @mcp.tool(
        name="whitebox_governance_list_review_requests",
        meta=ui_tool_meta(GOVERNANCE_DOSSIER_CARD_URI),
    )
    async def whitebox_governance_list_review_requests(
        board_id: str | None = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        """List governance review requests, optionally filtered.

        Args:
            board_id: Restrict to one review board
            status: One of draft, submitted, under_review, approved, denied, withdrawn
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.governance.alist_review_requests(
                board_id=board_id, status=status
            ),
        )

    @mcp.tool(
        name="whitebox_governance_get_review_request",
        meta=ui_tool_meta(GOVERNANCE_DOSSIER_CARD_URI),
    )
    async def whitebox_governance_get_review_request(
        request_id: str,
    ) -> dict[str, Any]:
        """Get a specific governance review request dossier (board, submitter,
        decisions, comments, vote tally, and SHA-256 evidence snapshot hash).
        """
        session = get_session()
        result = await call_with_session(
            session,
            lambda client: client.governance.aget_review_request(request_id),
        )
        return enrich_review_dossier(result)

    @mcp.tool(
        name="whitebox_governance_submit_decision",
        meta=ui_tool_meta(GOVERNANCE_DOSSIER_CARD_URI, visibility=["app"]),
    )
    async def whitebox_governance_submit_decision(
        request_id: str,
        decision: str,
        evidence_snapshot_hash: str,
        totp_code: str,
        justification: str | None = None,
        conditions: str | None = None,
    ) -> dict[str, Any]:
        """Record a board member's vote on a governance review request.

        App-only: called by the dossier card's vote buttons, never by the agent.
        Needs the reviewer's own JWT (the backend refuses organization API keys),
        two-factor authentication enabled on their account, and a current
        authenticator code. Refused if the request changed since the dossier was
        shown.

        Args:
            request_id: UUID of the governance review request
            decision: One of 'approve', 'deny', 'abstain', 'request_changes'
            evidence_snapshot_hash: `evidence_snapshot_hash` from the
                `whitebox_governance_get_review_request` result the reviewer saw
            totp_code: The reviewer's current authenticator code, or a backup code
            justification: Optional rationale, recorded with the vote
            conditions: Optional conditions of approval
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: _submit_vote(
                client,
                request_id=request_id,
                decision=decision,
                evidence_snapshot_hash=evidence_snapshot_hash,
                totp_code=totp_code,
                justification=justification,
                conditions=conditions,
            ),
        )

    @mcp.tool(name="whitebox_governance_raci_grid")
    async def whitebox_governance_raci_grid(
        board_id: str | None = None,
    ) -> dict[str, Any]:
        """Get the RACI grid: review requests and their Responsible /
        Accountable / Consulted / Informed assignments (ISO 42001 Clause
        5.3, Annex A.3), optionally scoped to one board.
        """
        session = get_session()
        return await call_with_session(
            session, lambda client: client.governance.araci_grid(board_id=board_id)
        )
