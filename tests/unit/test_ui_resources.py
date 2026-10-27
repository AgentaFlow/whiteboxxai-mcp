"""Unit tests for MCP Apps (`ui://` resources, `_meta.ui`, PII masking, and
plain-language audit verdicts) in `whiteboxxai_mcp.ui_resources` (issue #856).
"""

from __future__ import annotations

import pytest

from whiteboxxai_mcp.config import MCPSettings
from whiteboxxai_mcp.server import build_server
from whiteboxxai_mcp.ui_resources import (
    ALERT_TRIAGE_CARD_URI,
    EXPLANATION_CARD_URI,
    GOVERNANCE_DOSSIER_CARD_URI,
    ISO42001_STATUS_CARD_URI,
    MCP_APP_MIME_TYPE,
    MODEL_VERSION_DIFF_CARD_URI,
    STRICT_CSP,
    build_explanation_summary,
    build_iso42001_verdict,
    build_model_diff_verdict,
    build_triage_verdict,
    enrich_model_diff_payload,
    enrich_review_dossier,
    redact_pii_text,
    sanitize_explanation_payload,
    ui_tool_meta,
)


def test_ui_tool_meta_shape():
    assert ui_tool_meta(EXPLANATION_CARD_URI) == {
        "ui": {"resourceUri": EXPLANATION_CARD_URI}
    }
    assert ui_tool_meta(GOVERNANCE_DOSSIER_CARD_URI, visibility=["app"]) == {
        "ui": {"resourceUri": GOVERNANCE_DOSSIER_CARD_URI, "visibility": ["app"]}
    }


def test_redact_pii_text_masks_email_ssn_phone_card_and_ip():
    raw = (
        "Applicant jane.doe@example.com, SSN 123-45-6789, phone 415-555-0199, "
        "card 4111-1111-1111-1111, IP 192.168.1.42"
    )
    masked = redact_pii_text(raw)
    assert "jane.doe@example.com" not in masked
    assert "123-45-6789" not in masked
    assert "415-555-0199" not in masked
    assert "4111-1111-1111-1111" not in masked
    assert "192.168.1.42" not in masked
    assert "[REDACTED:EMAIL_ADDRESS]" in masked
    assert "[REDACTED:US_SSN]" in masked
    assert "[REDACTED:PHONE_NUMBER]" in masked
    assert "[REDACTED:CREDIT_CARD]" in masked
    assert "[REDACTED:IP_ADDRESS]" in masked


def test_sanitize_explanation_payload_masks_feature_values_and_builds_summary():
    payload = {
        "id": "exp-1",
        "method": "shap_tree",
        "status": "completed",
        "base_value": 0.35,
        "model_output": 0.82,
        "feature_weights": {
            "debt_to_income": 0.31,
            "credit_score": -0.18,
            "age": 0.04,
        },
        "feature_values": {
            "debt_to_income": 0.48,
            "applicant_email": "borrower@bank.com",
            "notes": ["SSN 123-45-6789"],
        },
    }
    sanitized = sanitize_explanation_payload(payload)
    assert sanitized["feature_values"]["applicant_email"] == "[REDACTED:EMAIL_ADDRESS]"
    assert sanitized["feature_values"]["notes"] == ["SSN [REDACTED:US_SSN]"]
    assert sanitized["feature_values"]["debt_to_income"] == 0.48
    assert "plain_language_summary" in sanitized
    summary = sanitized["plain_language_summary"]
    assert "SHAP_TREE explanation completed" in summary
    assert "output 0.8200" in summary
    assert "baseline 0.3500" in summary
    assert "debt_to_income (+0.3100)" in summary
    assert "credit_score (-0.1800)" in summary


def test_sanitize_explanation_payload_scrubs_dice_counterfactuals_in_metadata():
    # DiCE puts whole counterfactual rows (raw feature values) in `metadata`.
    payload = {
        "id": "e2",
        "method": "dice",
        "feature_weights": {},
        "metadata": {
            "counterfactuals": [{"email": "jane@example.com", "income": 51000}]
        },
    }
    sanitized = sanitize_explanation_payload(payload)
    cf = sanitized["metadata"]["counterfactuals"][0]
    assert cf["email"] == "[REDACTED:EMAIL_ADDRESS]"
    assert cf["income"] == 51000
    assert payload["metadata"]["counterfactuals"][0]["email"] == "jane@example.com"


def test_sanitize_explanation_payload_preserves_minimal_dict_unchanged():
    minimal = {"id": "e1", "status": "completed"}
    assert sanitize_explanation_payload(minimal) == minimal
    assert build_explanation_summary(minimal) is None


# Shapes below follow the backend response schemas: AlertResponse
# (backend/schemas/alert.py), DriftReportResponse (backend/schemas/drift.py) and
# ReviewRequestDetail / VoteStatus (backend/schemas/governance_review_board.py).
# tests/unit/test_mcp_ui_payload_shapes.py builds them from the real pydantic
# models; these keep the MCP package's own suite self-contained.
_ALERT = {
    "id": "a1",
    "rule_id": "r1",
    "model_id": "m1",
    "alert_type": "drift",
    "severity": "critical",
    "status": "active",
    "title": "Credit Drift Threshold",
    "message": "PSI above 0.25",
}

_DRIFT_REPORT = {
    "id": "dr1",
    "model_id": "m1",
    "drift_score": 0.412,
    "drift_detected": True,
    "metrics": {
        "overall_drift_score": 0.412,
        "severity": "high",
        "features_drifted": 2,
        "total_features": 9,
        "drift_percentage": 22.2,
        "feature_drifts": [
            {
                "feature_name": "income",
                "drift_score": 0.31,
                "p_value": 0.001,
                "test_statistic": 0.31,
                "test_method": "ks",
                "severity": "high",
            }
        ],
    },
}


def test_build_triage_verdict_reads_drift_report_schema():
    verdict = build_triage_verdict(_ALERT, [_DRIFT_REPORT])
    assert verdict.startswith("[CRITICAL | ACTIVE] Credit Drift Threshold.")
    assert "Data drift detected in latest report" in verdict
    assert "2 of 9 features drifted, score 0.4120" in verdict
    # Per-feature severity is the report's own severity copied onto every
    # feature, so the verdict must not name individual features from it.
    assert "income" not in verdict


def test_build_triage_verdict_distinguishes_none_found_from_lookup_failure():
    assert "No persisted drift reports found" in build_triage_verdict(_ALERT, [])

    failed = build_triage_verdict(_ALERT, [], drift_error="APIError: 503")
    assert "Drift context unavailable: APIError: 503" in failed
    assert "No persisted drift reports" not in failed

    no_model = build_triage_verdict({**_ALERT, "model_id": None}, [])
    assert "not tied to a model" in no_model


_REVIEW_REQUEST = {
    "id": "rr-1",
    "board_id": "b-1",
    "model_id": "m-1",
    "title": "Promote Credit Model v3",
    "description": "Passed fairness and drift checks",
    "request_type": "model_deployment",
    "priority": "high",
    "status": "under_review",
    "related_resources": [],
    "due_date": None,
    # Computed by the backend (backend/services/governance_vote_integrity.py);
    # this package no longer has a formula of its own to drift from it.
    "evidence_snapshot_hash": "ab" * 32,
    "attachments": [
        {"id": "att-1", "file_name": "fairness.pdf", "file_type": "pdf"},
    ],
    "decisions": [
        {
            "id": "d-1",
            "board_member_id": "bm-1",
            "decision": "approve",
            "justification": "Meets SR 11-7 bar",
            "conditions": None,
        },
    ],
    "vote_status": {
        "total_members": 5,
        "votes_cast": 2,
        "approve_votes": 2,
        "deny_votes": 0,
        "abstain_votes": 0,
        "request_changes_votes": 0,
        "approval_threshold": 3,
        "requires_unanimous": False,
        "quorum_required": 3,
        "quorum_met": False,
        "threshold_met": False,
        "can_finalize": False,
        "next_action": "Awaiting 1 more vote for quorum",
    },
}


def test_enrich_dossier_reports_the_real_vote_tally():
    enriched = enrich_review_dossier(_REVIEW_REQUEST)
    summary = enriched["plain_language_summary"]
    assert (
        "Votes: 2 approve (threshold 3), 0 deny; 2 of 5 members have voted." in summary
    )
    assert "Next: Awaiting 1 more vote for quorum." in summary
    # The backend's hash is passed through untouched, never recomputed here.
    assert enriched["evidence_snapshot_hash"] == "ab" * 32
    assert "Snapshot check (SHA-256): abababababab..." in summary


def test_enrich_dossier_does_not_invent_a_hash_the_backend_did_not_send():
    # No hash means the card offers no vote buttons; a guessed one would let a
    # vote through against evidence nobody checked.
    req = {k: v for k, v in _REVIEW_REQUEST.items() if k != "evidence_snapshot_hash"}
    enriched = enrich_review_dossier(req)
    assert "evidence_snapshot_hash" not in enriched
    assert "Snapshot check" not in enriched["plain_language_summary"]


def test_enrich_dossier_prints_no_tally_without_vote_status():
    # A missing count must not be shown as 0 -- that would be a tally nobody cast.
    req = {k: v for k, v in _REVIEW_REQUEST.items() if k != "vote_status"}
    assert "Votes" not in enrich_review_dossier(req)["plain_language_summary"]
    partial = {**req, "vote_status": {"next_action": "Submit for review"}}
    assert "Votes" not in enrich_review_dossier(partial)["plain_language_summary"]


def test_build_iso42001_verdict_handles_unassessed_and_assessed():
    unassessed = build_iso42001_verdict({"overall_score": 0.0, "assessed": False})
    assert "Not yet assessed" in unassessed

    assessed = build_iso42001_verdict(
        {
            "overall_score": 84.2,
            "assessed": True,
            "implementation_status": {
                "implemented": 28,
                "partial": 6,
                "not_implemented": 4,
            },
        },
        {"total_gaps": 4},
    )
    assert "84.2%" in assessed
    assert "28 implemented, 6 partial, 4 not implemented" in assessed
    assert "Open compliance gaps: 4." in assessed


def test_build_model_diff_verdict_and_enrich_payload():
    no_change = {
        "version1_id": "v1000000-00",
        "version2_id": "v2000000-00",
        "has_changes": False,
    }
    verdict = build_model_diff_verdict(no_change)
    assert "No architectural, parameter, or hyperparameter changes detected" in verdict

    breaking = {
        "version1_id": "v1000000-00",
        "version2_id": "v2000000-00",
        "has_changes": True,
        "breaking_changes": True,
        "change_summary": "Removed classification head and added dense layer",
        "parameter_delta": 150000,
        "parameter_change_pct": 12.5,
        "layers_added": ["dense_2"],
        "layers_removed": ["head_old"],
        "layers_modified": ["backbone"],
        "input_shape_changed": True,
        "framework_changed": True,
    }
    breaking_verdict = build_model_diff_verdict(breaking)
    assert "[BREAKING ARCHITECTURE CHANGE]" in breaking_verdict
    assert "parameters +150000 (+12.50%)" in breaking_verdict
    assert "layers +1/-1/~1" in breaking_verdict
    assert "tensor I/O shape changed" in breaking_verdict
    assert "framework changed" in breaking_verdict

    timeline = {
        "model_id": "m-1",
        "total_versions": 4,
        "total_changes": 2,
        "changes": [{"summary": "update"}],
        "skipped": [{"reason": "not stored"}],
    }
    timeline_verdict = build_model_diff_verdict(timeline)
    assert (
        "Architecture changes timeline: 2 change events recorded across 4 versions"
        in timeline_verdict
    )
    assert "(1 comparisons skipped)" in timeline_verdict

    enriched = enrich_model_diff_payload(breaking)
    assert enriched["plain_language_verdict"] == breaking_verdict


@pytest.mark.asyncio
async def test_server_registers_ui_resources_with_strict_csp_and_no_secrets():
    mcp = build_server(MCPSettings())
    resources = await mcp.list_resources()
    by_uri = {str(r.uri): r for r in resources}

    for uri in (
        EXPLANATION_CARD_URI,
        ALERT_TRIAGE_CARD_URI,
        GOVERNANCE_DOSSIER_CARD_URI,
        ISO42001_STATUS_CARD_URI,
        MODEL_VERSION_DIFF_CARD_URI,
    ):
        assert uri in by_uri, f"missing UI resource {uri}"
        assert by_uri[uri].mimeType == MCP_APP_MIME_TYPE

        contents = await mcp.read_resource(uri)
        assert len(contents) == 1
        html = contents[0].content
        assert STRICT_CSP in html
        assert "wbx_live_" not in html
        assert "WHITEBOXXAI_MCP_API_KEY" not in html
        assert "ui/notifications/tool-result" in html
        # MCP Apps handshake: the host sends nothing until the view has sent
        # ui/notifications/initialized after ui/initialize resolves.
        assert '"ui/notifications/initialized"' in html
        assert 'WBX_UI_PROTOCOL_VERSION = "2026-01-26"' in html
        assert "__WBX_VERSION__" not in html
        assert "event.source !== window.parent" in html

    dossier = (await mcp.read_resource(GOVERNANCE_DOSSIER_CARD_URI))[0].content
    # Every vote needs a fresh authenticator code, sent as totp_code and cleared
    # as soon as it has been submitted; the vote buttons start disabled.
    assert 'codeInput.autocomplete = "one-time-code"' in dossier
    assert "totp_code: codeInput.value.trim()" in dossier
    assert 'codeInput.value = "";' in dossier
    assert "btn.disabled = true;" in dossier
    assert "expected_evidence_snapshot_hash" not in dossier


@pytest.mark.asyncio
async def test_ui_linked_tools_expose_meta_resource_uri():
    mcp = build_server(MCPSettings())
    tools = {t.name: t for t in await mcp.list_tools()}

    expected_links = {
        "whitebox_explanations_generate": EXPLANATION_CARD_URI,
        "whitebox_explanations_get": EXPLANATION_CARD_URI,
        "whitebox_explanations_get_by_prediction": EXPLANATION_CARD_URI,
        "whitebox_alerts_get_instance": ALERT_TRIAGE_CARD_URI,
        "whitebox_alerts_get_triage_context": ALERT_TRIAGE_CARD_URI,
        "whitebox_drift_get_report": ALERT_TRIAGE_CARD_URI,
        "whitebox_governance_list_review_requests": GOVERNANCE_DOSSIER_CARD_URI,
        "whitebox_governance_get_review_request": GOVERNANCE_DOSSIER_CARD_URI,
        "whitebox_governance_submit_decision": GOVERNANCE_DOSSIER_CARD_URI,
        "whitebox_compliance_iso42001_score": ISO42001_STATUS_CARD_URI,
        "whitebox_compliance_iso42001_gaps": ISO42001_STATUS_CARD_URI,
        "whitebox_compliance_iso42001_status": ISO42001_STATUS_CARD_URI,
        "whitebox_models_compare_versions": MODEL_VERSION_DIFF_CARD_URI,
        "whitebox_models_version_changelog": MODEL_VERSION_DIFF_CARD_URI,
        "whitebox_models_architecture_timeline": MODEL_VERSION_DIFF_CARD_URI,
    }

    app_only = {"whitebox_governance_submit_decision"}
    for tool_name, expected_uri in expected_links.items():
        dumped = tools[tool_name].model_dump(by_alias=True, exclude_none=True)
        expected_ui: dict = {"resourceUri": expected_uri}
        if tool_name in app_only:
            expected_ui["visibility"] = ["app"]
        assert dumped.get("_meta") == {"ui": expected_ui}
