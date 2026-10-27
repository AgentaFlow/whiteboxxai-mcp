# Changelog

All notable changes to `whiteboxxai-mcp` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), with
the MCP-specific breaking/additive/fix rules in WhiteBoxXAI's
MCP/SDK versioning and support policy.

## [Unreleased]

## [0.3.1] - 2026-10-27

### Added
- Added support for dedicated `X-WhiteBox-API-Key` header on hosted Streamable HTTP `/mcp/` transport, enabling publishing on Smithery without OAuth conflicts (issue #930).

### Changed
- Moved to the standalone `AgentaFlow/whiteboxxai-mcp` repository; the package is now
  licensed MIT and published to PyPI and the MCP Registry from CI.

## [0.3.0] - 2026-10-16

### Added
- MCP Apps (`ui://` resources and `_meta.ui.resourceUri` tool metadata) support for
  inline interactive cards in MCP Apps-capable clients (issue #856):
  - `ui://whitebox/explanation-card` linked to `whitebox_explanations_generate`,
    `whitebox_explanations_get`, and `whitebox_explanations_get_by_prediction`.
  - `ui://whitebox/alert-triage-card` linked to `whitebox_alerts_get_instance`,
    `whitebox_alerts_get_triage_context`, and `whitebox_drift_get_report`.
  - `ui://whitebox/governance-dossier-card` linked to
    `whitebox_governance_list_review_requests`,
    `whitebox_governance_get_review_request`, and
    `whitebox_governance_submit_decision`.
  - `ui://whitebox/iso42001-status-card` linked to
    `whitebox_compliance_iso42001_score`, `whitebox_compliance_iso42001_gaps`,
    and `whitebox_compliance_iso42001_status`.
  - `ui://whitebox/model-version-diff-card` linked to
    `whitebox_models_compare_versions`, `whitebox_models_version_changelog`,
    and `whitebox_models_architecture_timeline`.
- Model version comparison and architecture diff tools (`whiteboxxai_mcp.tools.models`):
  `whitebox_models_compare_versions`, `whitebox_models_version_changelog`,
  `whitebox_models_list_git_versions`, and `whitebox_models_architecture_timeline`
  with deterministic plain-language architecture diff verdicts.
- `whitebox_alerts_get_triage_context` composite read tool combining alert instance
  details, latest model drift reports, and a plain-language triage verdict.
- `whitebox_governance_get_review_request` read tool returning a governance review
  dossier with a canonical SHA-256 `evidence_snapshot_hash` and plain-language summary,
  plus `whitebox_governance_submit_decision`. The vote tool is app-only
  (`_meta.ui.visibility: ["app"]`) so only the dossier card's buttons can call it.
  It requires `evidence_snapshot_hash` and `totp_code` (the reviewer's current
  authenticator or backup code) and forwards both: the backend refuses the vote if
  the request changed (409) or the code is missing, wrong or reused, and refuses
  organization API keys (`require_human_session`). The backend's refusal reaches the
  card as a tool error in its own words. The dossier's `evidence_snapshot_hash` is
  computed by the backend and written to the audit trail with the vote (issue #860);
  this package no longer has a hash formula of its own.
- Compliance & ISO/IEC 42001 tool domain (`whiteboxxai_mcp.tools.compliance`):
  `whitebox_compliance_iso42001_score`, `whitebox_compliance_iso42001_gaps`,
  `whitebox_compliance_iso42001_status`, `whitebox_compliance_list_frameworks`, and
  `whitebox_compliance_get_framework_controls`, with explicit unassessed-state
  reporting (`assessed: False`).
- Prompt governance tool domain (`whiteboxxai_mcp.tools.prompts`, 13 tools):
  `whitebox_prompts_list_templates`, `whitebox_prompts_get_template`,
  `whitebox_prompts_create_template`, `whitebox_prompts_list_versions`,
  `whitebox_prompts_get_version`, `whitebox_prompts_create_version`,
  `whitebox_prompts_advance_stage`, `whitebox_prompts_deploy`,
  `whitebox_prompts_rollback`, `whitebox_prompts_compute_drift`,
  `whitebox_prompts_list_drift_reports`, `whitebox_prompts_get_drift_report`,
  and `whitebox_prompts_get_incidents`.
- DiCE counterfactual parameters (`total_cfs`, `desired_class`, `features_to_vary`) on
  `whitebox_explanations_generate`, `whitebox_explanations_generate_bulk`, and
  `whitebox_explanations_generate_async`.
- Deterministic PII redaction (`[REDACTED:<TYPE>]`) on explanation `feature_values`
  and `metadata` (DiCE counterfactuals), and plain-language driver summaries on
  completed explanations.

### Changed
- Requires `whitebox-xai-sdk>=1.3.0` (the new compliance, model-diff and
  governance tools call SDK methods added in 1.3.0).

## [0.2.0] - 2026-08-24

### Added
- 92 new MCP tools across 7 domains (128 total, up from 36): `llm_xai` (13),
  `llm_monitoring` (16), `rag` (10), `safety` (11), `alerts` (12),
  `agent_workflows` (18), `metrics` (12). See `.workitems/141.md`.
- `alert_intelligence` + `alert_management` tool coverage (issue #224).
- Explanations remainder: sync/bulk/compare/visualize/config-CRUD tools (issue #225).
- HTTP/Streamable transport — the server can now be mounted at `/mcp` on the hosted
  backend for remote clients, in addition to the original stdio transport
  (issue #226). HTTP-mode sessions are always static-JWT (no refresh path).

### Fixed
- `AlertsResource.create()`/`list()` targeted the wrong backend path
  (`/api/v1/alerts` instead of `/api/v1/alerts/rules`).
- Several `model_id`/`prediction_id` path and schema fields incorrectly typed `int`
  against real UUID columns (explanations and LLM-XAI endpoints), which 422'd on any
  real ID.

### Note
- This release retroactively covers the M2 tool-domain work (#141), which shipped
  under version 0.1.0 without a version bump — the omission the versioning
  policy exists to prevent going forward.

## [0.1.0] - 2026-08-04

### Added
- Initial release: 36 MCP tools covering models, predictions, drift, bias/fairness,
  AI Risk Register / governance queries, and the async explanation-generation pair.
- Three auth modes: `WHITEBOXXAI_MCP_API_KEY` (recommended, scoped/revocable),
  `WHITEBOXXAI_MCP_EMAIL`/`WHITEBOXXAI_MCP_PASSWORD` (auto-refreshing), and
  `WHITEBOXXAI_MCP_JWT` (static, pre-issued).

<!--
This repo doesn't tag package releases, so entries above link to their originating
work items / PRs instead of the version-compare links Keep a Changelog suggests.
-->
