"""MCP Apps (`ui://` resources and `_meta.ui` helpers) for WhiteBoxXAI (issue #856).

Provides self-contained, sandboxed HTML/SVG widgets served under the `ui://`
scheme (`text/html;profile=mcp-app`) alongside deterministic PII scrubbing,
plain-language audit verdicts, and cryptographic evidence-snapshot hashing.

Security & governance invariants:
- Strict Content-Security-Policy (`default-src 'none'; style-src 'unsafe-inline';
  script-src 'unsafe-inline'; img-src data:;`) blocks all external network
  connections, remote scripts, and data exfiltration from the host iframe.
- Zero credentials, tokens, or internal URLs are embedded in the HTML.
- Every interactive widget action (snooze, refresh, vote) dispatches a JSON-RPC
  `tools/call` over `window.postMessage` to the MCP host, so every action
  traverses the caller's authenticated session, RBAC, and tenant isolation
  boundary.
- The card bridge follows the MCP Apps handshake (spec 2026-01-26): the host
  sends nothing to a view until the view has completed `ui/initialize` and sent
  `ui/notifications/initialized`, so skipping that step leaves every card
  empty in a spec-compliant host.
- Governance votes are app-only (`_meta.ui.visibility: ["app"]`): the host hides
  `whitebox_governance_submit_decision` from the agent, so a vote needs a person
  to click a button in the dossier card -- a valid human JWT alone proves the
  credential, not the action. The backend additionally refuses organization
  API keys for votes (`require_human_session`).
"""

from __future__ import annotations

import re
from typing import Any

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp import __version__

MCP_APP_MIME_TYPE = "text/html;profile=mcp-app"

EXPLANATION_CARD_URI = "ui://whitebox/explanation-card"
ALERT_TRIAGE_CARD_URI = "ui://whitebox/alert-triage-card"
GOVERNANCE_DOSSIER_CARD_URI = "ui://whitebox/governance-dossier-card"
ISO42001_STATUS_CARD_URI = "ui://whitebox/iso42001-status-card"
MODEL_VERSION_DIFF_CARD_URI = "ui://whitebox/model-version-diff-card"

STRICT_CSP = (
    "default-src 'none'; "
    "style-src 'unsafe-inline'; "
    "script-src 'unsafe-inline'; "
    "img-src data:;"
)

# Regex patterns aligned with backend/safety/pii_detector.py, kept standalone
# here so `whiteboxxai-mcp` remains independently installable without `backend`.
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "EMAIL_ADDRESS",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "US_SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "CREDIT_CARD",
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    ),
    (
        "PHONE_NUMBER",
        re.compile(
            r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b"
        ),
    ),
    (
        "IP_ADDRESS",
        re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
    ),
)


def ui_tool_meta(
    resource_uri: str, visibility: list[str] | None = None
) -> dict[str, Any]:
    """Build the MCP Apps `_meta` dictionary linking a tool to its `ui://` resource.

    `visibility` follows the MCP Apps spec: omitted means the default
    `["model", "app"]`; `["app"]` hides the tool from the agent's tool list so
    only a card can call it.
    """
    ui: dict[str, Any] = {"resourceUri": resource_uri}
    if visibility is not None:
        ui["visibility"] = list(visibility)
    return {"ui": ui}


def redact_pii_text(text: str) -> str:
    """Replace common PII patterns in `text` with `[REDACTED:<TYPE>]`."""
    redacted = text
    for label, pattern in _PII_PATTERNS:
        redacted = pattern.sub(f"[REDACTED:{label}]", redacted)
    return redacted


def redact_pii_value(value: Any) -> Any:
    """Recursively mask PII in string values inside dicts, lists, or scalars."""
    if isinstance(value, str):
        return redact_pii_text(value)
    if isinstance(value, dict):
        return {k: redact_pii_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_pii_value(item) for item in value]
    return value


def build_explanation_summary(payload: dict[str, Any]) -> str | None:
    """Build a deterministic plain-language audit summary from real explanation weights.

    Returns None when the explanation has no non-empty `feature_weights` (e.g.
    while status is still `pending`, or for minimal stub payloads), so we never
    fabricate or guess an explanation narrative.
    """
    weights = payload.get("feature_weights")
    if not isinstance(weights, dict) or not weights:
        return None

    numeric_items: list[tuple[str, float]] = []
    for k, v in weights.items():
        if isinstance(v, (int, float)):
            numeric_items.append((str(k), float(v)))

    if not numeric_items:
        return None

    numeric_items.sort(key=lambda item: abs(item[1]), reverse=True)
    top_drivers = numeric_items[:3]
    driver_parts = [f"{name} ({weight:+.4f})" for name, weight in top_drivers]

    method = str(payload.get("method") or "XAI").upper()
    base_val = payload.get("base_value")
    output_val = payload.get("model_output")
    if output_val is None:
        output_val = payload.get("local_prediction")

    header_bits = [f"{method} explanation completed"]
    if isinstance(output_val, (int, float)):
        header_bits.append(f"output {float(output_val):.4f}")
    if isinstance(base_val, (int, float)):
        header_bits.append(f"baseline {float(base_val):.4f}")

    return f"{', '.join(header_bits)}. Top feature drivers: {', '.join(driver_parts)}."


def sanitize_explanation_payload(payload: Any) -> Any:
    """Mask PII in `feature_values` and `metadata`, and attach
    `plain_language_summary` when weights exist.

    `metadata` is scrubbed too because DiCE stores its counterfactuals there,
    and each counterfactual is a full row of raw feature values.

    Preserves the exact input dictionary shape when there is nothing to scrub
    and no non-empty `feature_weights`.
    """
    if not isinstance(payload, dict):
        return payload

    has_feature_values = "feature_values" in payload and isinstance(
        payload["feature_values"], dict
    )
    has_metadata = isinstance(payload.get("metadata"), (dict, list))
    summary = build_explanation_summary(payload)

    if not has_feature_values and not has_metadata and summary is None:
        return payload

    sanitized = dict(payload)
    if has_feature_values:
        sanitized["feature_values"] = redact_pii_value(payload["feature_values"])
    if has_metadata:
        sanitized["metadata"] = redact_pii_value(payload["metadata"])
    if summary is not None and "plain_language_summary" not in sanitized:
        sanitized["plain_language_summary"] = summary
    return sanitized


def build_triage_verdict(
    alert: dict[str, Any],
    drift_reports: list[dict[str, Any]],
    drift_error: str | None = None,
) -> str:
    """Build a deterministic plain-language triage verdict for an alert + drift context.

    Reads the backend's `DriftReportResponse` shape (backend/schemas/drift.py):
    top-level `drift_detected`/`drift_score`, per-feature results under
    `metrics.feature_drifts[]`. `drift_error` set means the drift lookup
    failed -- that is reported as "unavailable", never as "no drift", because
    a failed lookup is not evidence that nothing drifted.
    """
    severity = str(alert.get("severity") or "unknown").upper()
    status = str(alert.get("status") or "active").upper()
    title = str(
        alert.get("title") or alert.get("message") or alert.get("alert_type") or "Alert"
    )
    prefix = f"[{severity} | {status}] {title}."

    if drift_error:
        return f"{prefix} Drift context unavailable: {drift_error}"

    if not drift_reports:
        if not alert.get("model_id"):
            return f"{prefix} This alert is not tied to a model, so no drift context applies."
        return f"{prefix} No persisted drift reports found for this model."

    latest = drift_reports[0] if isinstance(drift_reports[0], dict) else {}
    drift_detected = bool(latest.get("drift_detected"))
    drift_score = latest.get("drift_score")
    raw_metrics = latest.get("metrics")
    metrics: dict[str, Any] = raw_metrics if isinstance(raw_metrics, dict) else {}

    # Deliberately no per-feature "drifted on X, Y" clause: the response's
    # FeatureDrift.severity is the *report's* severity copied onto every
    # feature (backend/api/v1/drift.py), and the per-feature drift boolean
    # isn't exposed -- naming features from it would name all of them.
    features_drifted = metrics.get("features_drifted")
    total_features = metrics.get("total_features")
    if drift_detected:
        drift_clause = "Data drift detected in latest report"
    else:
        drift_clause = "No drift detected in latest report"

    if isinstance(features_drifted, int) and isinstance(total_features, int):
        drift_clause += f" ({features_drifted} of {total_features} features drifted"
        if isinstance(drift_score, (int, float)):
            drift_clause += f", score {float(drift_score):.4f}"
        drift_clause += ")"
    elif isinstance(drift_score, (int, float)):
        drift_clause += f" (score {float(drift_score):.4f})"

    return f"{prefix} {drift_clause}."


def enrich_review_dossier(review_request: Any) -> Any:
    """Attach `plain_language_summary` to a review dossier.

    `evidence_snapshot_hash` comes from the backend (issue #860): it is computed
    from the stored request next to the row lock a vote takes, so there is one
    formula and it lives where the vote is checked. A backend that does not
    return one yields no hash here, and the dossier card then offers no vote
    buttons rather than voting against a guess.
    """
    if not isinstance(review_request, dict):
        return review_request

    enriched = dict(review_request)
    snapshot_hash = enriched.get("evidence_snapshot_hash")
    snapshot_clause = (
        f" Snapshot check (SHA-256): {snapshot_hash[:12]}..."
        if isinstance(snapshot_hash, str) and snapshot_hash
        else ""
    )

    status = str(enriched.get("status") or "unknown").replace("_", " ").upper()
    priority = str(enriched.get("priority") or "unknown").upper()
    title = str(enriched.get("title") or "Governance Review Request")

    # VoteStatus (backend/schemas/governance_review_board.py). Read only the
    # fields that are actually present -- defaulting a missing count to 0
    # would print a tally nobody cast.
    vote_clause = ""
    vote_status = enriched.get("vote_status")
    if isinstance(vote_status, dict):
        approve = vote_status.get("approve_votes")
        deny = vote_status.get("deny_votes")
        threshold = vote_status.get("approval_threshold")
        cast = vote_status.get("votes_cast")
        total = vote_status.get("total_members")
        if all(isinstance(v, int) for v in (approve, deny, threshold)):
            vote_clause = (
                f" Votes: {approve} approve (threshold {threshold}), {deny} deny"
            )
            if isinstance(cast, int) and isinstance(total, int):
                vote_clause += f"; {cast} of {total} members have voted"
            vote_clause += "."
        next_action = vote_status.get("next_action")
        if isinstance(next_action, str) and next_action:
            vote_clause += f" Next: {next_action}."

    enriched[
        "plain_language_summary"
    ] = f"[{status} | Priority: {priority}] {title}.{vote_clause}{snapshot_clause}"
    return enriched


def build_iso42001_verdict(
    score_data: dict[str, Any], gaps_data: dict[str, Any] | None = None
) -> str:
    """Build an honest plain-language audit verdict for ISO/IEC 42001 readiness.

    When `assessed` is False, the backend returns 0.0 across all clauses because
    no control assessments have been recorded yet. We explicitly report that the
    organization is unassessed rather than presenting 0% as failed controls.
    """
    if not isinstance(score_data, dict):
        return "ISO/IEC 42001 readiness data unavailable."

    assessed = score_data.get("assessed")
    if assessed is False:
        return (
            "ISO/IEC 42001 status: Not yet assessed. No control implementation "
            "assessments have been recorded for this organization yet."
        )

    overall = score_data.get("overall_score")
    score_str = (
        f"{float(overall):.1f}%" if isinstance(overall, (int, float)) else "unknown"
    )
    impl = score_data.get("implementation_status") or {}
    implemented = impl.get("implemented", 0) if isinstance(impl, dict) else 0
    partial = impl.get("partial", 0) if isinstance(impl, dict) else 0
    not_impl = impl.get("not_implemented", 0) if isinstance(impl, dict) else 0

    gap_clause = ""
    if isinstance(gaps_data, dict):
        total_gaps = gaps_data.get("total_gaps")
        if total_gaps is None and isinstance(gaps_data.get("gaps"), list):
            total_gaps = len(gaps_data["gaps"])
        if isinstance(total_gaps, int):
            gap_clause = f" Open compliance gaps: {total_gaps}."

    return (
        f"ISO/IEC 42001 readiness score: {score_str} "
        f"({implemented} implemented, {partial} partial, "
        f"{not_impl} not implemented).{gap_clause}"
    )


def build_model_diff_verdict(diff_data: dict[str, Any]) -> str:
    """Build a deterministic plain-language audit verdict for model version comparisons."""
    if not isinstance(diff_data, dict):
        return "Model version comparison data unavailable."

    # Case 1: Architecture timeline response
    if "total_changes" in diff_data or "changes" in diff_data:
        total_changes = diff_data.get("total_changes", 0)
        total_versions = diff_data.get("total_versions", 0)
        skipped = len(diff_data.get("skipped") or [])
        skip_clause = f" ({skipped} comparisons skipped)" if skipped else ""
        return (
            f"Architecture changes timeline: {total_changes} change events recorded "
            f"across {total_versions} versions{skip_clause}."
        )

    # Case 2: Changelog response
    if "changelog" in diff_data and "version1_id" in diff_data:
        v1 = str(diff_data.get("version1_id") or "")[:8]
        v2 = str(diff_data.get("version2_id") or "")[:8]
        return f"Markdown changelog generated for version {v1} -> {v2}."

    # Case 3: Pairwise comparison
    v1 = str(diff_data.get("version1_id") or "")[:8]
    v2 = str(diff_data.get("version2_id") or "")[:8]
    has_changes = diff_data.get("has_changes")

    if has_changes is False:
        return (
            f"No architectural, parameter, or hyperparameter changes detected "
            f"between version {v1} and version {v2}."
        )

    breaking = diff_data.get("breaking_changes", False)
    prefix = (
        "[BREAKING ARCHITECTURE CHANGE]"
        if breaking
        else "[NON-BREAKING ARCHITECTURE DIFF]"
    )
    summary = diff_data.get("change_summary") or "Model differences detected."

    clauses: list[str] = []
    param_delta = diff_data.get("parameter_delta")
    pct = diff_data.get("parameter_change_pct")
    if isinstance(param_delta, (int, float)) and param_delta != 0:
        pct_str = f" ({float(pct):+.2f}%)" if isinstance(pct, (int, float)) else ""
        clauses.append(f"parameters {int(param_delta):+d}{pct_str}")

    layers_added = len(diff_data.get("layers_added") or [])
    layers_removed = len(diff_data.get("layers_removed") or [])
    layers_modified = len(diff_data.get("layers_modified") or [])
    if layers_added or layers_removed or layers_modified:
        clauses.append(f"layers +{layers_added}/-{layers_removed}/~{layers_modified}")

    if diff_data.get("input_shape_changed") or diff_data.get("output_shape_changed"):
        clauses.append("tensor I/O shape changed")

    if diff_data.get("framework_changed"):
        clauses.append("framework changed")

    details = f" ({', '.join(clauses)})" if clauses else ""
    return f"{prefix} {summary}{details}."


def enrich_model_diff_payload(payload: Any) -> Any:
    """Attach `plain_language_verdict` to model diff responses."""
    if not isinstance(payload, dict):
        return payload

    enriched = dict(payload)
    if "plain_language_verdict" not in enriched:
        enriched["plain_language_verdict"] = build_model_diff_verdict(enriched)
    return enriched


_SHARED_STYLES_AND_BRIDGE = """
  <style>
    :root {
      color-scheme: light dark;
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --pos: #22c55e;
      --neg: #ef4444;
      --warn: #f59e0b;
    }
    @media (prefers-color-scheme: light) {
      :root {
        --bg: #f8fafc;
        --card: #ffffff;
        --border: #e2e8f0;
        --text: #0f172a;
        --muted: #475569;
        --accent: #0284c7;
        --pos: #16a34a;
        --neg: #dc2626;
        --warn: #d97706;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 12px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      font-size: 13px;
      line-height: 1.45;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
    }
    .title {
      font-size: 15px;
      font-weight: 600;
      margin: 0;
    }
    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      border: 1px solid var(--border);
    }
    .verdict {
      background: rgba(56, 189, 248, 0.1);
      border-left: 3px solid var(--accent);
      padding: 8px 10px;
      border-radius: 4px;
      margin-bottom: 12px;
      font-size: 12.5px;
    }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }
    .meta-item {
      padding: 6px 8px;
      border: 1px solid var(--border);
      border-radius: 6px;
    }
    .meta-label {
      font-size: 11px;
      color: var(--muted);
    }
    .meta-val {
      font-weight: 600;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      word-break: break-all;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid var(--border);
    }
    button {
      background: var(--card);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    input[type="text"] {
      flex: 1;
      min-width: 160px;
      padding: 5px 8px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--text);
      font-size: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      margin-top: 8px;
    }
    th, td {
      text-align: left;
      padding: 5px 6px;
      border-bottom: 1px solid var(--border);
    }
    th { color: var(--muted); font-weight: 500; }
    .status-msg {
      margin-top: 8px;
      font-size: 11.5px;
      color: var(--muted);
    }
  </style>
  <script>
    function escapeHtml(val) {
      if (val === null || val === undefined) return "";
      return String(val)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    // MCP Apps view-side bridge (spec 2026-01-26). The host sends nothing --
    // not even the tool result this card exists to show -- until the view has
    // completed ui/initialize AND sent ui/notifications/initialized.
    const WBX_UI_PROTOCOL_VERSION = "2026-01-26";
    let _rpcId = 1;
    const _pending = {};

    function setStatus(text) {
      const statusEl = document.getElementById("wbx-status");
      if (statusEl) statusEl.textContent = text;
    }

    function rpcRequest(method, params) {
      const id = _rpcId++;
      return new Promise(function(resolve, reject) {
        _pending[id] = { resolve: resolve, reject: reject };
        window.parent.postMessage(
          { jsonrpc: "2.0", id: id, method: method, params: params || {} },
          "*"
        );
      });
    }

    function rpcNotify(method, params) {
      window.parent.postMessage(
        { jsonrpc: "2.0", method: method, params: params || {} },
        "*"
      );
    }

    function toolResultText(result) {
      if (!result || !Array.isArray(result.content)) return "";
      return result.content
        .filter(function(b) { return b && b.type === "text"; })
        .map(function(b) { return b.text; })
        .join(" ");
    }

    function payloadFromToolResult(result) {
      if (!result || typeof result !== "object") return null;
      let data = result.structuredContent;
      if (data === undefined || data === null) {
        const text = toolResultText(result);
        if (text) {
          try { data = JSON.parse(text); } catch (_) { data = null; }
        }
      }
      // FastMCP wraps a non-object return value (e.g. a list) as {"result": [...]}.
      if (data && typeof data === "object" && !Array.isArray(data)
          && Object.keys(data).length === 1 && Array.isArray(data.result)) {
        data = data.result;
      }
      return data;
    }

    function renderToolResult(result) {
      if (result && result.isError) {
        setStatus("Tool error: " + (toolResultText(result) || "unknown error"));
        return;
      }
      const data = payloadFromToolResult(result);
      if (data && typeof window.renderWhiteBoxWidget === "function") {
        window.renderWhiteBoxWidget(data);
      } else {
        setStatus("The tool returned no structured data to display.");
      }
    }

    // Calls a tool from a card button. opts.render: the tool's result IS this
    // card's data (a refresh). Otherwise the result is only reported in the
    // status line and opts.after runs on success -- a vote or snooze returns a
    // decision or alert object, not the record this card displays, and
    // rendering it would replace the dossier with the wrong ids.
    function callMcpTool(name, args, opts) {
      opts = opts || {};
      setStatus("Calling " + name + "...");
      return rpcRequest("tools/call", { name: name, arguments: args || {} }).then(
        function(result) {
          if (result && result.isError) {
            setStatus("Failed: " + (toolResultText(result) || name));
            return;
          }
          setStatus("Done: " + name);
          if (opts.render) renderToolResult(result);
          if (typeof opts.after === "function") opts.after(result);
        },
        function(err) {
          setStatus("Failed: " + ((err && err.message) || String(err)));
        }
      );
    }

    window.addEventListener("message", function(event) {
      if (event.source !== window.parent) return;
      const msg = event.data;
      if (!msg || typeof msg !== "object" || msg.jsonrpc !== "2.0") return;

      if (msg.method === undefined && msg.id !== undefined) {
        const waiter = _pending[msg.id];
        if (!waiter) return;
        delete _pending[msg.id];
        if (msg.error) waiter.reject(msg.error);
        else waiter.resolve(msg.result);
        return;
      }

      if (msg.method === "ui/notifications/tool-result") {
        renderToolResult(msg.params);
        return;
      }

      if (msg.id !== undefined) {
        // Host-to-view request. Teardown needs only an acknowledgement; anything
        // else is answered with method-not-found so the host isn't left waiting.
        const reply = msg.method === "ui/resource-teardown"
          ? { jsonrpc: "2.0", id: msg.id, result: {} }
          : { jsonrpc: "2.0", id: msg.id, error: { code: -32601, message: "Method not found" } };
        window.parent.postMessage(reply, "*");
      }
    });

    function reportSize() {
      rpcNotify("ui/notifications/size-changed", {
        height: Math.ceil(document.documentElement.scrollHeight)
      });
    }

    function initBridge() {
      rpcRequest("ui/initialize", {
        protocolVersion: WBX_UI_PROTOCOL_VERSION,
        appInfo: { name: "whiteboxxai-mcp", version: "__WBX_VERSION__" },
        appCapabilities: {}
      }).then(
        function() {
          rpcNotify("ui/notifications/initialized", {});
          if (typeof ResizeObserver === "function") {
            new ResizeObserver(reportSize).observe(document.body);
          }
          reportSize();
        },
        function(err) {
          setStatus("The host did not accept the UI handshake: " + ((err && err.message) || String(err)));
        }
      );
    }

    if (document.readyState === "loading") {
      window.addEventListener("DOMContentLoaded", initBridge);
    } else {
      initBridge();
    }
  </script>
""".replace(
    "__WBX_VERSION__", __version__
)


def render_explanation_card_html() -> str:
    """Return the sandboxed HTML/SVG widget for `ui://whitebox/explanation-card`."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="Content-Security-Policy" content="{STRICT_CSP}" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>WhiteBoxXAI Prediction Explanation Card</title>
{_SHARED_STYLES_AND_BRIDGE}
</head>
<body>
  <div class="card" id="wbx-root">
    <div class="header">
      <h1 class="title">Prediction Explanation (SHAP / LIME)</h1>
      <span class="badge" id="exp-status">Waiting for data...</span>
    </div>
    <div class="verdict" id="exp-verdict">
      Awaiting tool result payload from MCP host...
    </div>
    <div class="meta-grid" id="exp-meta"></div>
    <div id="exp-chart"></div>
    <div id="exp-features"></div>
    <div class="actions" id="exp-actions"></div>
    <div class="status-msg" id="wbx-status"></div>
  </div>
  <script>
    window.renderWhiteBoxWidget = function(data) {{
      const status = String(data.status || "completed");
      const method = String(data.method || "shap").toUpperCase();
      document.getElementById("exp-status").textContent = method + " • " + status;

      const verdictEl = document.getElementById("exp-verdict");
      if (data.plain_language_summary) {{
        verdictEl.textContent = data.plain_language_summary;
      }} else if (status === "pending" || status === "processing") {{
        verdictEl.textContent = "Explanation is computing asynchronously on a background worker. Click Refresh Status to poll.";
      }} else {{
        verdictEl.textContent = "Explanation record loaded. Review feature attributions below.";
      }}

      const expId = data.id || data.explanation_id || "—";
      const baseVal = typeof data.base_value === "number" ? data.base_value.toFixed(4) : "—";
      const outVal = typeof data.model_output === "number"
        ? data.model_output.toFixed(4)
        : (typeof data.local_prediction === "number" ? data.local_prediction.toFixed(4) : "—");

      document.getElementById("exp-meta").innerHTML =
        '<div class="meta-item"><div class="meta-label">Explanation ID</div><div class="meta-val">' + escapeHtml(expId) + '</div></div>' +
        '<div class="meta-item"><div class="meta-label">Base Value</div><div class="meta-val">' + escapeHtml(baseVal) + '</div></div>' +
        '<div class="meta-item"><div class="meta-label">Model Output</div><div class="meta-val">' + escapeHtml(outVal) + '</div></div>';

      const weights = data.feature_weights || {{}};
      const entries = Object.entries(weights)
        .filter(function(kv) {{ return typeof kv[1] === "number"; }})
        .sort(function(a, b) {{ return Math.abs(b[1]) - Math.abs(a[1]); }})
        .slice(0, 10);

      const chartEl = document.getElementById("exp-chart");
      if (entries.length > 0) {{
        const maxAbs = Math.max.apply(null, entries.map(function(e) {{ return Math.abs(e[1]); }})) || 1;
        const rowH = 26;
        const svgH = entries.length * rowH + 10;
        let svgRows = "";
        entries.forEach(function(entry, idx) {{
          const name = entry[0];
          const w = entry[1];
          const y = idx * rowH + 6;
          const barW = Math.max(2, Math.round((Math.abs(w) / maxAbs) * 120));
          const x = w >= 0 ? 220 : 220 - barW;
          const fill = w >= 0 ? "var(--pos)" : "var(--neg)";
          svgRows +=
            '<text x="4" y="' + (y + 14) + '" fill="currentColor" font-size="11">' + escapeHtml(name.slice(0, 18)) + '</text>' +
            '<line x1="220" y1="' + y + '" x2="220" y2="' + (y + 20) + '" stroke="var(--border)" />' +
            '<rect x="' + x + '" y="' + (y + 3) + '" width="' + barW + '" height="14" rx="3" fill="' + fill + '" />' +
            '<text x="350" y="' + (y + 14) + '" fill="currentColor" font-size="11" font-family="monospace">' + (w >= 0 ? "+" : "") + w.toFixed(4) + '</text>';
        }});
        chartEl.innerHTML =
          '<svg width="100%" viewBox="0 0 430 ' + svgH + '" role="img" aria-label="Feature attribution bar chart, largest drivers first">' +
          svgRows + '</svg>';
      }} else {{
        chartEl.innerHTML = "";
      }}

      const fvals = data.feature_values || {{}};
      const fkeys = Object.keys(fvals);
      const featEl = document.getElementById("exp-features");
      if (fkeys.length > 0) {{
        let rows = "";
        fkeys.slice(0, 10).forEach(function(k) {{
          rows += "<tr><td>" + escapeHtml(k) + "</td><td>" + escapeHtml(fvals[k]) + "</td></tr>";
        }});
        featEl.innerHTML = '<table><thead><tr><th>Feature</th><th>Value (PII Masked)</th></tr></thead><tbody>' + rows + '</tbody></table>';
      }} else {{
        featEl.innerHTML = "";
      }}

      const actionsEl = document.getElementById("exp-actions");
      actionsEl.innerHTML = "";
      if (expId && expId !== "—") {{
        const pollBtn = document.createElement("button");
        pollBtn.className = "primary";
        pollBtn.textContent = "Refresh Explanation";
        pollBtn.onclick = function() {{
          callMcpTool("whitebox_explanations_get", {{ explanation_id: String(expId) }}, {{ render: true }});
        }};
        actionsEl.appendChild(pollBtn);
      }}
    }};
  </script>
</body>
</html>
"""


def render_alert_triage_card_html() -> str:
    """Return the sandboxed HTML/SVG widget for `ui://whitebox/alert-triage-card`."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="Content-Security-Policy" content="{STRICT_CSP}" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>WhiteBoxXAI Alert &amp; Drift Triage Card</title>
{_SHARED_STYLES_AND_BRIDGE}
</head>
<body>
  <div class="card" id="wbx-root">
    <div class="header">
      <h1 class="title" id="alert-title">Alert &amp; Drift Triage</h1>
      <span class="badge" id="alert-badge">Pending</span>
    </div>
    <div class="verdict" id="alert-verdict">
      Awaiting alert or drift report payload from MCP host...
    </div>
    <div class="meta-grid" id="alert-meta"></div>
    <div id="drift-table"></div>
    <div class="actions" id="alert-actions"></div>
    <div class="status-msg" id="wbx-status"></div>
  </div>
  <script>
    window.renderWhiteBoxWidget = function(data) {{
      // Three tools render here: get_triage_context ({{alert, latest_drift_reports}}),
      // alerts_get_instance (an AlertResponse, which has rule_id) and
      // drift_get_report (a DriftReportResponse). Only a real alert gets the
      // snooze buttons -- a drift report's id is not an alert id.
      const isAlert = Boolean(data.alert) || data.rule_id !== undefined;
      const alertObj = data.alert || (isAlert ? data : {{}});
      const alertId = isAlert ? (alertObj.id || "") : "";
      const modelId = alertObj.model_id || data.model_id || "—";
      const severity = String(alertObj.severity || data.severity || "unknown").toUpperCase();
      const status = isAlert
        ? String(alertObj.status || "unknown").toUpperCase()
        : (data.drift_detected ? "DRIFT DETECTED" : "NO DRIFT");

      document.getElementById("alert-badge").textContent = severity + " • " + status;
      document.getElementById("alert-verdict").textContent =
        data.plain_language_verdict ||
        alertObj.message ||
        (isAlert ? "Alert loaded." : "Drift report loaded. Per-feature test results below.");

      document.getElementById("alert-meta").innerHTML =
        '<div class="meta-item"><div class="meta-label">' + (isAlert ? "Alert ID" : "Drift Report ID") + '</div><div class="meta-val">' + escapeHtml(alertId || data.id || "—") + '</div></div>' +
        '<div class="meta-item"><div class="meta-label">Model ID</div><div class="meta-val">' + escapeHtml(modelId) + '</div></div>' +
        '<div class="meta-item"><div class="meta-label">Status</div><div class="meta-val">' + escapeHtml(status) + '</div></div>';

      // DriftReportResponse (backend/schemas/drift.py): per-feature results
      // live under metrics.feature_drifts[]. There is no per-feature verdict
      // column because the API doesn't expose a per-feature drift flag -- the
      // report-level verdict above is the honest summary.
      const reports = Array.isArray(data.latest_drift_reports)
        ? data.latest_drift_reports
        : (data.metrics ? [data] : []);
      const driftEl = document.getElementById("drift-table");
      const rep = reports.length > 0 ? reports[0] : null;
      const featureDrifts = (rep && rep.metrics && Array.isArray(rep.metrics.feature_drifts))
        ? rep.metrics.feature_drifts
        : [];
      if (featureDrifts.length > 0) {{
        let rows = "";
        featureDrifts.slice(0, 8).forEach(function(f) {{
          const stat = typeof f.test_statistic === "number" ? f.test_statistic.toFixed(4) : "—";
          const pval = typeof f.p_value === "number" ? f.p_value.toFixed(4) : "—";
          rows += "<tr><td>" + escapeHtml(f.feature_name) + "</td><td>" + escapeHtml(f.test_method || "—") + "</td><td>" + escapeHtml(stat) + "</td><td>" + escapeHtml(pval) + "</td></tr>";
        }});
        driftEl.innerHTML = '<table><thead><tr><th>Feature</th><th>Test</th><th>Statistic</th><th>p-value</th></tr></thead><tbody>' + rows + '</tbody></table>';
      }} else {{
        driftEl.innerHTML = "";
      }}

      const actionsEl = document.getElementById("alert-actions");
      actionsEl.innerHTML = "";
      if (alertId) {{
        function refreshTriage() {{
          callMcpTool("whitebox_alerts_get_triage_context", {{ alert_id: String(alertId) }}, {{ render: true }});
        }}

        const snooze1h = document.createElement("button");
        snooze1h.textContent = "Snooze 1h";
        snooze1h.onclick = function() {{
          callMcpTool("whitebox_alerts_snooze", {{ alert_id: String(alertId), snooze_minutes: 60 }}, {{ after: refreshTriage }});
        }};
        actionsEl.appendChild(snooze1h);

        const snooze24h = document.createElement("button");
        snooze24h.textContent = "Snooze 24h";
        snooze24h.onclick = function() {{
          callMcpTool("whitebox_alerts_snooze", {{ alert_id: String(alertId), snooze_minutes: 1440 }}, {{ after: refreshTriage }});
        }};
        actionsEl.appendChild(snooze24h);

        const refreshBtn = document.createElement("button");
        refreshBtn.className = "primary";
        refreshBtn.textContent = "Refresh Triage Context";
        refreshBtn.onclick = refreshTriage;
        actionsEl.appendChild(refreshBtn);
      }}
    }};
  </script>
</body>
</html>
"""


def render_governance_dossier_card_html() -> str:
    """Return the sandboxed HTML widget for `ui://whitebox/governance-dossier-card`."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="Content-Security-Policy" content="{STRICT_CSP}" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>WhiteBoxXAI Governance Review Dossier</title>
{_SHARED_STYLES_AND_BRIDGE}
</head>
<body>
  <div class="card" id="wbx-root">
    <div class="header">
      <h1 class="title" id="gov-title">Governance Review Dossier</h1>
      <span class="badge" id="gov-badge">Loading</span>
    </div>
    <div class="verdict" id="gov-verdict">
      Awaiting governance review dossier from MCP host...
    </div>
    <div class="meta-grid" id="gov-meta"></div>
    <div id="gov-decisions"></div>
    <div class="actions" id="gov-actions"></div>
    <div class="status-msg" id="wbx-status"></div>
  </div>
  <script>
    function openRequest(requestId) {{
      callMcpTool("whitebox_governance_get_review_request", {{ request_id: String(requestId) }}, {{ render: true }});
    }}

    // whitebox_governance_list_review_requests returns a list: show it as a
    // table to pick from. Voting only happens on a single dossier fetched by
    // whitebox_governance_get_review_request, which carries the snapshot hash.
    function renderRequestList(rows) {{
      document.getElementById("gov-title").textContent = "Governance Review Requests";
      document.getElementById("gov-badge").textContent = rows.length + " REQUESTS";
      document.getElementById("gov-verdict").textContent = rows.length
        ? "Open a request to see its full dossier and vote."
        : "No review requests match this filter.";
      document.getElementById("gov-meta").innerHTML = "";
      document.getElementById("gov-actions").innerHTML = "";
      const decEl = document.getElementById("gov-decisions");
      decEl.innerHTML = '<table><thead><tr><th>Title</th><th>Status</th><th>Priority</th><th></th></tr></thead><tbody id="gov-list-body"></tbody></table>';
      const body = document.getElementById("gov-list-body");
      rows.slice(0, 25).forEach(function(r) {{
        const tr = document.createElement("tr");
        tr.innerHTML = "<td>" + escapeHtml(r.title || "—") + "</td><td>" + escapeHtml(r.status || "—") + "</td><td>" + escapeHtml(r.priority || "—") + "</td>";
        const td = document.createElement("td");
        if (r.id) {{
          const btn = document.createElement("button");
          btn.textContent = "Open";
          btn.onclick = function() {{ openRequest(r.id); }};
          td.appendChild(btn);
        }}
        tr.appendChild(td);
        body.appendChild(tr);
      }});
    }}

    window.renderWhiteBoxWidget = function(data) {{
      if (Array.isArray(data)) {{
        renderRequestList(data);
        return;
      }}
      const req = data;
      const reqId = req.id || "";
      const status = String(req.status || "unknown").toUpperCase();
      const priority = String(req.priority || "unknown").toUpperCase();
      const boardName = (req.board && req.board.name) ? req.board.name : (req.board_id || "—");
      const hash = req.evidence_snapshot_hash || "";

      document.getElementById("gov-title").textContent = req.title || "Governance Review Request";
      document.getElementById("gov-badge").textContent = status + " • " + priority;
      document.getElementById("gov-verdict").textContent =
        req.plain_language_summary ||
        req.description ||
        "Governance review request loaded.";

      // VoteStatus (backend/schemas/governance_review_board.py). No tally is
      // shown unless the backend actually returned one.
      const vs = req.vote_status || {{}};
      const tally = (typeof vs.approve_votes === "number" && typeof vs.approval_threshold === "number")
        ? (vs.approve_votes + " approve / " + vs.approval_threshold + " needed, " + (vs.deny_votes || 0) + " deny")
        : "—";

      document.getElementById("gov-meta").innerHTML =
        '<div class="meta-item"><div class="meta-label">Request ID</div><div class="meta-val">' + escapeHtml(reqId || "—") + '</div></div>' +
        '<div class="meta-item"><div class="meta-label">Board</div><div class="meta-val">' + escapeHtml(boardName) + '</div></div>' +
        '<div class="meta-item"><div class="meta-label">Vote Tally</div><div class="meta-val">' + escapeHtml(tally) + '</div></div>' +
        '<div class="meta-item"><div class="meta-label">Snapshot check (SHA-256)</div><div class="meta-val">' + escapeHtml(hash ? hash.slice(0, 16) : "—") + '</div></div>';

      const decisions = Array.isArray(req.decisions) ? req.decisions : [];
      const decEl = document.getElementById("gov-decisions");
      if (decisions.length > 0) {{
        let rows = "";
        decisions.forEach(function(d) {{
          const voter = (d.voter && d.voter.full_name) ? d.voter.full_name : "—";
          rows += "<tr><td>" + escapeHtml(voter) + "</td><td>" + escapeHtml(d.decision) + "</td><td>" + escapeHtml(d.justification || "—") + "</td></tr>";
        }});
        decEl.innerHTML = '<table><thead><tr><th>Voter</th><th>Decision</th><th>Justification</th></tr></thead><tbody>' + rows + '</tbody></table>';
      }} else {{
        decEl.innerHTML = "";
      }}

      const actionsEl = document.getElementById("gov-actions");
      actionsEl.innerHTML = "";
      if (!reqId) return;

      if (hash) {{
        const justInput = document.createElement("input");
        justInput.type = "text";
        justInput.placeholder = "Justification (recorded with your vote)";
        justInput.id = "gov-justification";
        actionsEl.appendChild(justInput);

        // Every vote needs a fresh authenticator code; the backend checks it
        // and each code works once. It is never echoed into the status line,
        // and is cleared as soon as it has been sent.
        const codeInput = document.createElement("input");
        codeInput.type = "text";
        codeInput.id = "gov-code";
        codeInput.inputMode = "numeric";
        codeInput.autocomplete = "one-time-code";
        codeInput.maxLength = 32;
        codeInput.placeholder = "Authenticator code";
        codeInput.setAttribute("aria-label", "Authenticator code");
        actionsEl.appendChild(codeInput);

        const voteButtons = [];
        function syncVoteButtons() {{
          const ready = codeInput.value.trim().length > 0;
          voteButtons.forEach(function(b) {{ b.disabled = !ready; }});
        }}
        codeInput.addEventListener("input", syncVoteButtons);

        function submitVote(decisionType) {{
          const jEl = document.getElementById("gov-justification");
          const payload = {{
            request_id: String(reqId),
            decision: decisionType,
            evidence_snapshot_hash: String(hash),
            totp_code: codeInput.value.trim(),
          }};
          if (jEl && jEl.value) {{
            payload.justification = jEl.value;
          }}
          codeInput.value = "";
          syncVoteButtons();
          callMcpTool("whitebox_governance_submit_decision", payload, {{
            after: function() {{ openRequest(reqId); }}
          }});
        }}

        [["approve", "Approve", "primary"], ["request_changes", "Request Changes", ""], ["deny", "Deny", ""], ["abstain", "Abstain", ""]]
          .forEach(function(spec) {{
            const btn = document.createElement("button");
            if (spec[2]) btn.className = spec[2];
            btn.textContent = spec[1];
            btn.disabled = true;
            btn.onclick = function() {{ submitVote(spec[0]); }};
            voteButtons.push(btn);
            actionsEl.appendChild(btn);
          }});

        setStatus("Voting needs two-factor authentication on your account and a current code. Your vote is recorded under your own signed-in account; organization API keys cannot vote.");
      }}

      const refreshBtn = document.createElement("button");
      refreshBtn.textContent = "Refresh Dossier";
      refreshBtn.onclick = function() {{ openRequest(reqId); }};
      actionsEl.appendChild(refreshBtn);
    }};
  </script>
</body>
</html>
"""


def render_iso42001_status_card_html() -> str:
    """Return the sandboxed HTML widget for `ui://whitebox/iso42001-status-card`."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="Content-Security-Policy" content="{STRICT_CSP}" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>WhiteBoxXAI ISO/IEC 42001 Readiness Card</title>
{_SHARED_STYLES_AND_BRIDGE}
</head>
<body>
  <div class="card" id="wbx-root">
    <div class="header">
      <h1 class="title" id="iso-title">ISO/IEC 42001 Readiness &amp; Gap Matrix</h1>
      <span class="badge" id="iso-badge">Awaiting Data</span>
    </div>
    <div class="verdict" id="iso-verdict">
      Awaiting ISO/IEC 42001 readiness and gap payload from MCP host...
    </div>
    <div class="meta-grid" id="iso-meta"></div>
    <div id="iso-clauses"></div>
    <div id="iso-gaps"></div>
    <div class="actions" id="iso-actions"></div>
    <div class="status-msg" id="wbx-status"></div>
  </div>
  <script>
    window.renderWhiteBoxWidget = function(data) {{
      // Three shapes: the composite status ({{score, gaps}}), the score alone
      // (ReadinessScoreResponse) and the gaps alone ({{gaps, total_gaps}}). With
      // no score object there is no assessment state to report, so the card
      // must not imply one.
      const scoreObj = data.score || (typeof data.overall_score === "number" ? data : null);
      const gapsObj = (data.gaps && !Array.isArray(data.gaps)) ? data.gaps : data;
      const hasScore = scoreObj !== null;
      const assessed = hasScore && scoreObj.assessed === true;
      const overall = hasScore && typeof scoreObj.overall_score === "number" ? scoreObj.overall_score.toFixed(1) + "%" : "—";
      const impl = (hasScore && scoreObj.implementation_status) || {{}};

      document.getElementById("iso-badge").textContent = !hasScore
        ? "GAP LIST"
        : (assessed ? ("READINESS: " + overall) : "NOT YET ASSESSED");
      document.getElementById("iso-verdict").textContent =
        data.plain_language_verdict ||
        (!hasScore
          ? "ISO/IEC 42001 open gaps. Run the status tool for the readiness score."
          : (assessed
            ? ("ISO/IEC 42001 readiness score: " + overall)
            : "ISO/IEC 42001 status: Not yet assessed. No control assessments recorded yet."));

      const totalGaps = typeof gapsObj.total_gaps === "number"
        ? gapsObj.total_gaps
        : (Array.isArray(gapsObj.gaps) ? gapsObj.gaps.length : "—");

      document.getElementById("iso-meta").innerHTML =
        '<div class="meta-item"><div class="meta-label">Assessment State</div><div class="meta-val">' + escapeHtml(!hasScore ? "—" : (assessed ? "Assessed" : "Not Yet Assessed")) + '</div></div>' +
        '<div class="meta-item"><div class="meta-label">Overall Score</div><div class="meta-val">' + escapeHtml(!hasScore ? "—" : (assessed ? overall : "Unassessed")) + '</div></div>' +
        '<div class="meta-item"><div class="meta-label">Implemented Controls</div><div class="meta-val">' + escapeHtml(impl.implemented !== undefined ? impl.implemented : "—") + '</div></div>' +
        '<div class="meta-item"><div class="meta-label">Open Gaps</div><div class="meta-val">' + escapeHtml(totalGaps) + '</div></div>';

      const clauses = (hasScore && scoreObj.clause_scores) || {{}};
      const ckeys = Object.keys(clauses);
      const clausesEl = document.getElementById("iso-clauses");
      if (assessed && ckeys.length > 0) {{
        let rows = "";
        ckeys.forEach(function(clause) {{
          const val = typeof clauses[clause] === "number" ? clauses[clause].toFixed(1) + "%" : clauses[clause];
          rows += "<tr><td>Clause " + escapeHtml(clause) + "</td><td>" + escapeHtml(val) + "</td></tr>";
        }});
        clausesEl.innerHTML = '<table><thead><tr><th>ISO/IEC 42001 Clause</th><th>Readiness Score</th></tr></thead><tbody>' + rows + '</tbody></table>';
      }} else {{
        clausesEl.innerHTML = "";
      }}

      const gapList = Array.isArray(gapsObj.gaps) ? gapsObj.gaps : [];
      const gapsEl = document.getElementById("iso-gaps");
      if (gapList.length > 0) {{
        let rows = "";
        gapList.slice(0, 8).forEach(function(g) {{
          rows += "<tr><td>" + escapeHtml(g.control_id || g.clause || "—") + "</td><td>" + escapeHtml(g.title || g.description || "—") + "</td><td>" + escapeHtml(g.severity || g.status || "—") + "</td></tr>";
        }});
        gapsEl.innerHTML = '<table><thead><tr><th>Control</th><th>Gap Description</th><th>Severity / Status</th></tr></thead><tbody>' + rows + '</tbody></table>';
      }} else {{
        gapsEl.innerHTML = "";
      }}

      const actionsEl = document.getElementById("iso-actions");
      actionsEl.innerHTML = "";
      const refreshBtn = document.createElement("button");
      refreshBtn.className = "primary";
      refreshBtn.textContent = "Refresh ISO 42001 Status";
      refreshBtn.onclick = function() {{
        callMcpTool("whitebox_compliance_iso42001_status", {{}}, {{ render: true }});
      }};
      actionsEl.appendChild(refreshBtn);
    }};
  </script>
</body>
</html>
"""


def render_model_version_diff_card_html() -> str:
    """Return the sandboxed HTML widget for `ui://whitebox/model-version-diff-card`."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="Content-Security-Policy" content="{STRICT_CSP}" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>WhiteBoxXAI Model Version Diff Card</title>
{_SHARED_STYLES_AND_BRIDGE}
</head>
<body>
  <div class="card" id="wbx-root">
    <div class="header">
      <h1 class="title" id="diff-title">Model Version &amp; Architecture Diff</h1>
      <span class="badge" id="diff-badge">Awaiting Data</span>
    </div>
    <div class="verdict" id="diff-verdict">
      Awaiting model version diff payload from MCP host...
    </div>
    <div class="meta-grid" id="diff-meta"></div>
    <div id="diff-changes"></div>
    <div id="diff-layers"></div>
    <div class="actions" id="diff-actions"></div>
    <div class="status-msg" id="wbx-status"></div>
  </div>
  <script>
    window.renderWhiteBoxWidget = function(data) {{
      const v1 = data.version1_id || "";
      const v2 = data.version2_id || "";
      const isTimeline = Array.isArray(data.changes) || typeof data.total_changes === "number";
      const hasChanges = Boolean(data.has_changes);
      const isBreaking = Boolean(data.breaking_changes);

      const badgeEl = document.getElementById("diff-badge");
      if (isTimeline) {{
        badgeEl.textContent = "TIMELINE • " + (data.total_changes || 0) + " CHANGES";
      }} else if (!hasChanges) {{
        badgeEl.textContent = "NO CHANGES DETECTED";
      }} else if (isBreaking) {{
        badgeEl.textContent = "BREAKING CHANGE";
      }} else {{
        badgeEl.textContent = "NON-BREAKING DIFF";
      }}

      document.getElementById("diff-verdict").textContent =
        data.plain_language_verdict ||
        data.change_summary ||
        data.changelog ||
        "Model version comparison record loaded.";

      const metaEl = document.getElementById("diff-meta");
      if (isTimeline) {{
        metaEl.innerHTML =
          '<div class="meta-item"><div class="meta-label">Model ID</div><div class="meta-val">' + escapeHtml(data.model_id || "—") + '</div></div>' +
          '<div class="meta-item"><div class="meta-label">Total Versions</div><div class="meta-val">' + escapeHtml(data.total_versions || "—") + '</div></div>' +
          '<div class="meta-item"><div class="meta-label">Total Changes</div><div class="meta-val">' + escapeHtml(data.total_changes || "0") + '</div></div>' +
          '<div class="meta-item"><div class="meta-label">Skipped Comparisons</div><div class="meta-val">' + escapeHtml((data.skipped && data.skipped.length) || "0") + '</div></div>';
      }} else {{
        const delta = typeof data.parameter_delta === "number" ? (data.parameter_delta > 0 ? "+" : "") + data.parameter_delta.toLocaleString() : "—";
        const pct = typeof data.parameter_change_pct === "number" ? (data.parameter_change_pct > 0 ? "+" : "") + data.parameter_change_pct.toFixed(2) + "%" : "";
        metaEl.innerHTML =
          '<div class="meta-item"><div class="meta-label">Version 1 (Base)</div><div class="meta-val">' + escapeHtml(v1.slice(0, 8) || "—") + '</div></div>' +
          '<div class="meta-item"><div class="meta-label">Version 2 (Target)</div><div class="meta-val">' + escapeHtml(v2.slice(0, 8) || "—") + '</div></div>' +
          '<div class="meta-item"><div class="meta-label">Parameter Delta</div><div class="meta-val">' + escapeHtml(delta + (pct ? " (" + pct + ")" : "")) + '</div></div>' +
          '<div class="meta-item"><div class="meta-label">Tensor I/O &amp; Framework</div><div class="meta-val">' + escapeHtml((data.framework_changed ? "Framework changed" : "Framework unchanged") + (data.input_shape_changed || data.output_shape_changed ? " • Shape diff" : "")) + '</div></div>';
      }}

      const changesEl = document.getElementById("diff-changes");
      if (isTimeline && Array.isArray(data.changes) && data.changes.length > 0) {{
        let rows = "";
        data.changes.slice(0, 10).forEach(function(c) {{
          const commit = c.commit_hash ? c.commit_hash.slice(0, 7) : "—";
          const breakingBadge = c.breaking ? '<span style="color:var(--neg);font-weight:600">[BREAKING]</span> ' : '';
          rows += "<tr><td>" + escapeHtml(commit) + "</td><td>" + breakingBadge + escapeHtml(c.summary || "—") + "</td><td>" + escapeHtml(c.parameter_delta || "0") + "</td></tr>";
        }});
        changesEl.innerHTML = '<table><thead><tr><th>Commit</th><th>Summary</th><th>Param Delta</th></tr></thead><tbody>' + rows + '</tbody></table>';
      }} else if (Array.isArray(data.architecture_changes) && data.architecture_changes.length > 0) {{
        let rows = "";
        data.architecture_changes.slice(0, 10).forEach(function(c) {{
          const oldNew = (c.old_value !== null && c.old_value !== undefined ? escapeHtml(c.old_value) : "—") + " &rarr; " + (c.new_value !== null && c.new_value !== undefined ? escapeHtml(c.new_value) : "—");
          rows += "<tr><td>" + escapeHtml(c.component || "—") + "</td><td>" + escapeHtml(c.change_type || "—") + "</td><td>" + oldNew + "</td><td>" + escapeHtml(c.impact || "—") + "</td></tr>";
        }});
        changesEl.innerHTML = '<table><thead><tr><th>Component</th><th>Type</th><th>Old &rarr; New</th><th>Impact</th></tr></thead><tbody>' + rows + '</tbody></table>';
      }} else if (data.changelog) {{
        changesEl.innerHTML = '<pre style="background:var(--bg);border:1px solid var(--border);padding:8px;border-radius:6px;overflow-x:auto;font-size:12px;white-space:pre-wrap;">' + escapeHtml(data.changelog) + '</pre>';
      }} else {{
        changesEl.innerHTML = "";
      }}

      const layersEl = document.getElementById("diff-layers");
      if (data.layers_added || data.layers_removed || data.layers_modified) {{
        const added = (data.layers_added || []).length;
        const removed = (data.layers_removed || []).length;
        const modified = (data.layers_modified || []).length;
        layersEl.innerHTML = '<div style="font-size:11.5px;color:var(--muted);margin-top:8px;">' +
          'Layer changes: +' + added + ' added, -' + removed + ' removed, ~' + modified + ' modified.' +
          '</div>';
      }} else {{
        layersEl.innerHTML = "";
      }}

      const actionsEl = document.getElementById("diff-actions");
      actionsEl.innerHTML = "";
      if (v1 && v2) {{
        const changelogBtn = document.createElement("button");
        changelogBtn.className = "primary";
        changelogBtn.textContent = "Generate Markdown Changelog";
        changelogBtn.onclick = function() {{
          callMcpTool("whitebox_models_version_changelog", {{ version1_id: String(v1), version2_id: String(v2) }}, {{ render: true }});
        }};
        actionsEl.appendChild(changelogBtn);

        const refreshBtn = document.createElement("button");
        refreshBtn.textContent = "Refresh Comparison";
        refreshBtn.onclick = function() {{
          callMcpTool("whitebox_models_compare_versions", {{ version1_id: String(v1), version2_id: String(v2) }}, {{ render: true }});
        }};
        actionsEl.appendChild(refreshBtn);
      }}
    }};
  </script>
</body>
</html>
"""


def register_ui_resources(mcp: FastMCP) -> None:
    """Register all WhiteBoxXAI MCP Apps (`ui://`) resources on `mcp`."""

    @mcp.resource(
        EXPLANATION_CARD_URI,
        name="whitebox_explanation_card",
        title="WhiteBoxXAI Prediction Explanation Card",
        description=(
            "Interactive SHAP/LIME waterfall visualization card with "
            "plain-language driver verdict and PII-masked feature values."
        ),
        mime_type=MCP_APP_MIME_TYPE,
    )
    def whitebox_explanation_card() -> str:
        return render_explanation_card_html()

    @mcp.resource(
        ALERT_TRIAGE_CARD_URI,
        name="whitebox_alert_triage_card",
        title="WhiteBoxXAI Alert & Drift Triage Card",
        description=(
            "Alert and feature-drift triage card displaying PSI/KS statistics, "
            "plain-language verdict, and inline snooze/refresh actions."
        ),
        mime_type=MCP_APP_MIME_TYPE,
    )
    def whitebox_alert_triage_card() -> str:
        return render_alert_triage_card_html()

    @mcp.resource(
        GOVERNANCE_DOSSIER_CARD_URI,
        name="whitebox_governance_dossier_card",
        title="WhiteBoxXAI Governance Review Dossier Card",
        description=(
            "Governance review dossier: request status, vote tally, decisions "
            "with justifications, a SHA-256 snapshot check, and vote buttons "
            "that record the vote under the signed-in reviewer's own account."
        ),
        mime_type=MCP_APP_MIME_TYPE,
    )
    def whitebox_governance_dossier_card() -> str:
        return render_governance_dossier_card_html()

    @mcp.resource(
        ISO42001_STATUS_CARD_URI,
        name="whitebox_iso42001_status_card",
        title="WhiteBoxXAI ISO/IEC 42001 Readiness & Gap Matrix Card",
        description=(
            "ISO/IEC 42001 readiness scorecard, clause breakdown, and open "
            "compliance gap matrix with honest unassessed-state reporting."
        ),
        mime_type=MCP_APP_MIME_TYPE,
    )
    def whitebox_iso42001_status_card() -> str:
        return render_iso42001_status_card_html()

    @mcp.resource(
        MODEL_VERSION_DIFF_CARD_URI,
        name="whitebox_model_version_diff_card",
        title="WhiteBoxXAI Model Version & Architecture Diff Card",
        description=(
            "Model version comparison card showing architecture changes, "
            "parameter deltas, breaking change detection, and changelog generation."
        ),
        mime_type=MCP_APP_MIME_TYPE,
    )
    def whitebox_model_version_diff_card() -> str:
        return render_model_version_diff_card_html()
