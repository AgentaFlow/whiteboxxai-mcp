"""Behavior tests for whiteboxxai_mcp.tools.governance (issue #227).

Brings the M1 governance review board / RACI domain up to the same
behavioral-delegation bar as the M2/#137 domains (see
test_tools_explanations.py, the template this file follows).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from whiteboxxai.exceptions import APIError, RateLimitError, ServerError

from whiteboxxai_mcp.errors import BackendUnavailableError
from whiteboxxai_mcp.tools import governance


@pytest.fixture
def mcp_with_fake_session(monkeypatch):
    mcp = FastMCP(name="test")
    governance.register(mcp)

    fake_client = MagicMock()
    fake_client.governance.alist_boards = AsyncMock(
        return_value=[{"board_id": "b1", "name": "AI Ethics Board"}]
    )
    fake_client.governance.alist_review_requests = AsyncMock(
        return_value=[{"request_id": "rr1", "status": "under_review"}]
    )
    # ReviewRequestDetail field names (backend/schemas/governance_review_board.py).
    fake_client.governance.aget_review_request = AsyncMock(
        return_value={
            "id": "rr1",
            "board_id": "b1",
            "model_id": "m1",
            "title": "Deploy Credit Scorer",
            "description": "Promote v4 to production",
            "request_type": "model_deployment",
            "status": "under_review",
            "priority": "high",
            "attachments": [],
            "decisions": [],
            "evidence_snapshot_hash": "cd" * 32,
        }
    )
    fake_client.governance.araci_grid = AsyncMock(
        return_value={"board_id": "b1", "grid": []}
    )
    fake_client.governance.asubmit_decision = AsyncMock(
        return_value={"id": "dec1", "decision": "approve"}
    )

    class FakeSession:
        """Minimal stand-in for SessionManager (mirrors test_error_mapping.py)."""

        async def get_client(self, force_refresh: bool = False):
            return fake_client

    monkeypatch.setattr(governance, "get_session", lambda: FakeSession())
    return mcp, fake_client


@pytest.mark.asyncio
async def test_list_boards_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_governance_list_boards")

    result = await tool.run({})

    fake_client.governance.alist_boards.assert_awaited_once_with()
    assert result == [{"board_id": "b1", "name": "AI Ethics Board"}]


@pytest.mark.asyncio
async def test_list_review_requests_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_governance_list_review_requests")

    result = await tool.run({})

    fake_client.governance.alist_review_requests.assert_awaited_once_with(
        board_id=None, status=None
    )
    assert result == [{"request_id": "rr1", "status": "under_review"}]


@pytest.mark.asyncio
async def test_list_review_requests_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_governance_list_review_requests")

    await tool.run({"board_id": "b1", "status": "approved"})

    fake_client.governance.alist_review_requests.assert_awaited_once_with(
        board_id="b1", status="approved"
    )


@pytest.mark.asyncio
async def test_get_review_request_delegates_and_enriches_dossier(
    mcp_with_fake_session,
):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_governance_get_review_request")

    result = await tool.run({"request_id": "rr1"})

    fake_client.governance.aget_review_request.assert_awaited_once_with("rr1")
    assert result["id"] == "rr1"
    assert result["evidence_snapshot_hash"] == "cd" * 32
    assert "Deploy Credit Scorer" in result["plain_language_summary"]


_VOTE = {
    "request_id": "rr1",
    "decision": "approve",
    "evidence_snapshot_hash": "cd" * 32,
    "totp_code": "123456",
}


@pytest.mark.asyncio
async def test_submit_decision_forwards_the_vote_the_hash_and_the_code(
    mcp_with_fake_session,
):
    mcp, fake_client = mcp_with_fake_session
    submit_tool = mcp._tool_manager.get_tool("whitebox_governance_submit_decision")

    result = await submit_tool.run({**_VOTE, "justification": "Looks solid"})

    # The backend is the one place the hash and the code are checked: nothing is
    # re-fetched or compared here.
    fake_client.governance.aget_review_request.assert_not_awaited()
    fake_client.governance.asubmit_decision.assert_awaited_once_with(
        "rr1",
        decision="approve",
        justification="Looks solid",
        conditions=None,
        evidence_snapshot_hash="cd" * 32,
        totp_code="123456",
    )
    assert result == {"id": "dec1", "decision": "approve"}


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["evidence_snapshot_hash", "totp_code"])
async def test_submit_decision_requires_the_hash_and_the_code(
    mcp_with_fake_session, missing
):
    mcp, fake_client = mcp_with_fake_session
    submit_tool = mcp._tool_manager.get_tool("whitebox_governance_submit_decision")

    with pytest.raises(ToolError):
        await submit_tool.run({k: v for k, v in _VOTE.items() if k != missing})

    fake_client.governance.asubmit_decision.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, detail",
    [
        (409, "This request changed after you opened it. Reload it first."),
        (400, "That code was not accepted."),
        (403, "Voting and sign-off need two-factor authentication."),
    ],
)
async def test_a_backend_refusal_reaches_the_card_in_the_backends_words(
    mcp_with_fake_session, status, detail
):
    mcp, fake_client = mcp_with_fake_session
    fake_client.governance.asubmit_decision = AsyncMock(
        side_effect=APIError(
            f"API error ({status}): {detail}",
            status_code=status,
            response={"detail": detail},
        )
    )
    submit_tool = mcp._tool_manager.get_tool("whitebox_governance_submit_decision")

    # A tool error (isError: true), carrying the message the reviewer needs --
    # not "backend request failed", and not a result the card would render as a
    # recorded vote.
    with pytest.raises(ToolError, match=detail):
        await submit_tool.run(_VOTE)


@pytest.mark.asyncio
async def test_a_two_factor_lockout_is_not_reported_as_a_plan_limit(
    mcp_with_fake_session,
):
    mcp, fake_client = mcp_with_fake_session
    fake_client.governance.asubmit_decision = AsyncMock(
        side_effect=RateLimitError(
            "Rate limit exceeded. Please retry later. "
            "(Too many incorrect codes. Try again later.)",
            retry_after=840,
        )
    )
    submit_tool = mcp._tool_manager.get_tool("whitebox_governance_submit_decision")

    with pytest.raises(ToolError, match="Too many incorrect codes") as refused:
        await submit_tool.run(_VOTE)
    assert "plan" not in str(refused.value).lower()


@pytest.mark.asyncio
async def test_a_backend_outage_is_still_reported_as_an_outage(
    mcp_with_fake_session,
):
    mcp, fake_client = mcp_with_fake_session
    fake_client.governance.asubmit_decision = AsyncMock(
        side_effect=ServerError("Server error (503): down", status_code=503)
    )
    submit_tool = mcp._tool_manager.get_tool("whitebox_governance_submit_decision")

    with pytest.raises(ToolError, match="backend request failed") as outage:
        await submit_tool.run(_VOTE)
    assert isinstance(outage.value.__cause__, BackendUnavailableError)


def test_submit_decision_is_app_only(mcp_with_fake_session):
    # The host keeps app-only tools out of the agent's tool list, so a vote
    # needs a click in the dossier card, not just a reviewer's JWT.
    mcp, _ = mcp_with_fake_session
    submit_tool = mcp._tool_manager.get_tool("whitebox_governance_submit_decision")
    assert submit_tool.meta["ui"]["visibility"] == ["app"]


@pytest.mark.asyncio
async def test_raci_grid_delegates_to_sdk(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_governance_raci_grid")

    result = await tool.run({})

    fake_client.governance.araci_grid.assert_awaited_once_with(board_id=None)
    assert result == {"board_id": "b1", "grid": []}


@pytest.mark.asyncio
async def test_raci_grid_passes_through_optional_args(mcp_with_fake_session):
    mcp, fake_client = mcp_with_fake_session
    tool = mcp._tool_manager.get_tool("whitebox_governance_raci_grid")

    await tool.run({"board_id": "b1"})

    fake_client.governance.araci_grid.assert_awaited_once_with(board_id="b1")
