"""Protocol-compliance tests: every tool registers with a valid schema.

No WhiteBoxXAI backend is needed for these -- registration only exercises
FastMCP decorator wiring, not any actual API calls.
"""

from typing import Any

from whiteboxxai_mcp.config import MCPSettings
from whiteboxxai_mcp.server import build_server


def _build() -> Any:
    return build_server(MCPSettings())


def test_server_builds_without_error():
    mcp = _build()
    assert mcp.name == "whiteboxxai"


def test_build_server_accepts_streamable_http_path_override():
    """Issue #226: asgi.py passes streamable_http_path="/" so that once its
    returned app is mounted at /mcp in backend/main.py, the tool is reachable
    at exactly /mcp rather than /mcp/mcp (FastMCP's own internal route
    defaults to "/mcp"). The no-arg stdio call site must keep FastMCP's own
    default untouched.
    """
    mcp_http = build_server(MCPSettings(), streamable_http_path="/")
    assert mcp_http.settings.streamable_http_path == "/"

    mcp_stdio = _build()
    assert mcp_stdio.settings.streamable_http_path == "/mcp"


def test_all_tools_registered_with_unique_names():
    mcp = _build()
    tools = mcp._tool_manager.list_tools()

    assert len(tools) > 0

    names = [t.name for t in tools]
    assert len(names) == len(set(names)), f"duplicate tool names: {names}"

    for name in names:
        assert name.startswith("whitebox_"), f"tool {name!r} missing whitebox_ prefix"


def test_expected_m1_tools_present():
    mcp = _build()
    names = {t.name for t in mcp._tool_manager.list_tools()}

    expected = {
        # models
        "whitebox_models_register",
        "whitebox_models_get",
        "whitebox_models_list",
        "whitebox_models_update",
        "whitebox_models_update_status",
        "whitebox_models_versions",
        "whitebox_models_latest",
        "whitebox_models_update_baseline",
        "whitebox_models_archive",
        "whitebox_models_restore",
        "whitebox_models_delete",
        # predictions
        "whitebox_predictions_log",
        "whitebox_predictions_log_batch",
        "whitebox_predictions_get",
        "whitebox_predictions_query",
        "whitebox_predictions_stats",
        "whitebox_predictions_recent",
        # drift
        "whitebox_drift_detect",
        "whitebox_drift_create_report",
        "whitebox_drift_list_reports",
        "whitebox_drift_get_report",
        "whitebox_drift_trend",
        # fairness
        "whitebox_fairness_audit",
        "whitebox_fairness_get_audit",
        "whitebox_fairness_list_audits",
        "whitebox_fairness_bias_history",
        "whitebox_fairness_metric_history",
        "whitebox_fairness_latest_audit",
    }
    missing = expected - names
    assert not missing, f"missing expected tools: {missing}"


def test_expected_risk_register_and_governance_tools_present():
    """Work item #104 (ISO 42001 alignment): AI Risk Register + governance
    review board / RACI query tools.
    """
    mcp = _build()
    names = {t.name for t in mcp._tool_manager.list_tools()}

    expected = {
        # risk register
        "whitebox_risk_register_list",
        "whitebox_risk_register_get",
        "whitebox_risk_register_portfolio",
        # governance / RACI
        "whitebox_governance_list_boards",
        "whitebox_governance_list_review_requests",
        "whitebox_governance_get_review_request",
        "whitebox_governance_submit_decision",
        "whitebox_governance_raci_grid",
        # compliance / ISO 42001 / Control Catalog (issue #856)
        "whitebox_compliance_iso42001_score",
        "whitebox_compliance_iso42001_gaps",
        "whitebox_compliance_iso42001_status",
        "whitebox_compliance_list_frameworks",
        "whitebox_compliance_get_framework_controls",
    }
    missing = expected - names
    assert not missing, f"missing expected tools: {missing}"


def test_expected_explanations_tools_present():
    """Issue #137: the async two-tool explanation pattern (generate + poll)
    proxying the backend's real Celery-backed SHAP/LIME pipeline. Issue #225
    extends this to the rest of backend/api/v1/explanations.py: sync/bulk
    generate, listing/lookup, stats, config CRUD, compare, and visualize.
    """
    mcp = _build()
    names = {t.name for t in mcp._tool_manager.list_tools()}

    expected = {
        "whitebox_explanations_generate_async",
        "whitebox_explanations_get",
        "whitebox_explanations_generate",
        "whitebox_explanations_generate_bulk",
        "whitebox_explanations_list_by_model",
        "whitebox_explanations_get_by_prediction",
        "whitebox_explanations_stats",
        "whitebox_explanations_set_config",
        "whitebox_explanations_get_config",
        "whitebox_explanations_delete_config",
        "whitebox_explanations_compare",
        "whitebox_explanations_visualize",
        "whitebox_explanations_visualize_multi",
    }
    missing = expected - names
    assert not missing, f"missing expected tools: {missing}"


def test_expected_m2_domain_tools_present():
    """Issue #141: M2 domain coverage -- llm_xai, llm_monitoring, rag,
    safety, alerts, agent_workflows, metrics. Extended by issue #224 to
    cover alert_intelligence and alert_management's deferred M2 slice.
    """
    mcp = _build()
    names = {t.name for t in mcp._tool_manager.list_tools()}

    expected = {
        # llm_xai
        "whitebox_llm_xai_attention",
        "whitebox_llm_xai_token_importance",
        "whitebox_llm_xai_prompt_sensitivity",
        "whitebox_llm_xai_counterfactuals",
        "whitebox_llm_xai_debug_prompt",
        "whitebox_llm_xai_get_explanation",
        "whitebox_llm_xai_get_prompt_analysis",
        "whitebox_llm_xai_list_explanations",
        "whitebox_llm_xai_list_prompt_analyses",
        "whitebox_llm_xai_stats",
        "whitebox_llm_xai_visualize_attention",
        "whitebox_llm_xai_visualize_token_importance",
        "whitebox_llm_xai_batch_analyze",
        # llm_monitoring
        "whitebox_llm_monitoring_log_call",
        "whitebox_llm_monitoring_log_calls_batch",
        "whitebox_llm_monitoring_get_log",
        "whitebox_llm_monitoring_query_logs",
        "whitebox_llm_monitoring_session_logs",
        "whitebox_llm_monitoring_usage_stats",
        "whitebox_llm_monitoring_recent",
        "whitebox_llm_monitoring_cost_breakdown",
        "whitebox_llm_monitoring_performance",
        "whitebox_llm_monitoring_trends_tokens",
        "whitebox_llm_monitoring_trends_costs",
        "whitebox_llm_monitoring_usage_stats_detailed",
        "whitebox_llm_monitoring_cost_threshold_alert",
        "whitebox_llm_monitoring_latency_threshold_alert",
        "whitebox_llm_monitoring_error_rate_alert",
        "whitebox_llm_monitoring_cleanup_logs",
        # rag
        "whitebox_rag_log_retrieval",
        "whitebox_rag_get_retrieval",
        "whitebox_rag_list_retrievals",
        "whitebox_rag_stats",
        "whitebox_rag_trends",
        "whitebox_rag_create_evaluation",
        "whitebox_rag_get_evaluation",
        "whitebox_rag_list_evaluations",
        "whitebox_rag_metrics_precision",
        "whitebox_rag_metrics_relevance",
        # safety
        "whitebox_safety_analyze",
        "whitebox_safety_analyze_batch",
        "whitebox_safety_get_score",
        "whitebox_safety_get_scores",
        "whitebox_safety_stats",
        "whitebox_safety_trends",
        "whitebox_safety_create_threshold",
        "whitebox_safety_list_thresholds",
        "whitebox_safety_get_threshold",
        "whitebox_safety_update_threshold",
        "whitebox_safety_delete_threshold",
        # alerts (core rule/instance CRUD)
        "whitebox_alerts_create_rule",
        "whitebox_alerts_list_rules",
        "whitebox_alerts_get_rule",
        "whitebox_alerts_update_rule",
        "whitebox_alerts_delete_rule",
        "whitebox_alerts_evaluate_rule",
        "whitebox_alerts_list_instances",
        "whitebox_alerts_get_instance",
        "whitebox_alerts_get_triage_context",
        "whitebox_alerts_acknowledge",
        "whitebox_alerts_resolve",
        "whitebox_alerts_snooze",
        "whitebox_alerts_statistics",
        # alert_intelligence (issue #224)
        "whitebox_alert_intelligence_detect_correlations",
        "whitebox_alert_intelligence_get_correlation_clusters",
        "whitebox_alert_intelligence_analyze_root_cause",
        "whitebox_alert_intelligence_verify_root_cause",
        "whitebox_alert_intelligence_get_similar_incidents",
        "whitebox_alert_intelligence_calculate_priority",
        "whitebox_alert_intelligence_adjust_priority",
        "whitebox_alert_intelligence_get_priority_distribution",
        "whitebox_alert_intelligence_get_priority_queue",
        "whitebox_alert_intelligence_detect_patterns",
        "whitebox_alert_intelligence_get_pattern",
        "whitebox_alert_intelligence_deactivate_pattern",
        "whitebox_alert_intelligence_apply_fatigue_reduction",
        "whitebox_alert_intelligence_get_throttle_recommendations",
        "whitebox_alert_intelligence_ml_grouping",
        "whitebox_alert_intelligence_temporal_grouping",
        "whitebox_alert_intelligence_predict_trends",
        "whitebox_alert_intelligence_detect_seasonality",
        "whitebox_alert_intelligence_enhanced_trend_analysis",
        "whitebox_alert_intelligence_forecast_volume",
        # alert_management (issue #224 -- escalation-policy CRUD and
        # bulk_action are deliberately excluded, admin-console-shaped)
        "whitebox_alert_management_get_history",
        "whitebox_alert_management_get_analytics",
        "whitebox_alert_management_get_trend",
        "whitebox_alert_management_get_velocity",
        "whitebox_alert_management_get_grouping",
        "whitebox_alert_management_detect_fatigue",
        "whitebox_alert_management_acknowledge",
        "whitebox_alert_management_resolve",
        "whitebox_alert_management_snooze",
        "whitebox_alert_management_create_comment",
        "whitebox_alert_management_get_comments",
        "whitebox_alert_management_process_pending_escalations",
        "whitebox_alert_management_unsnooze_expired_alerts",
        # agent_workflows
        "whitebox_agent_workflows_create_and_start",
        "whitebox_agent_workflows_start",
        "whitebox_agent_workflows_complete",
        "whitebox_agent_workflows_list",
        "whitebox_agent_workflows_get",
        "whitebox_agent_workflows_delete",
        "whitebox_agent_workflows_register_agent",
        "whitebox_agent_workflows_list_agents",
        "whitebox_agent_workflows_create_execution",
        "whitebox_agent_workflows_log_interaction",
        "whitebox_agent_workflows_list_interactions",
        "whitebox_agent_workflows_create_task",
        "whitebox_agent_workflows_update_task_status",
        "whitebox_agent_workflows_list_tasks",
        "whitebox_agent_workflows_analytics",
        "whitebox_agent_workflows_cost_breakdown",
        "whitebox_agent_workflows_bottlenecks",
        "whitebox_agent_workflows_timeline",
        # metrics
        "whitebox_metrics_create",
        "whitebox_metrics_create_batch",
        "whitebox_metrics_calculate_classification",
        "whitebox_metrics_calculate_regression",
        "whitebox_metrics_get_model_metrics",
        "whitebox_metrics_latest",
        "whitebox_metrics_timeseries",
        "whitebox_metrics_aggregate",
        "whitebox_metrics_trend",
        "whitebox_metrics_rolling",
        "whitebox_metrics_summary",
        "whitebox_metrics_delete",
    }
    missing = expected - names
    assert not missing, f"missing expected tools: {missing}"


def test_every_tool_has_a_description_and_valid_json_schema():
    mcp = _build()
    for tool in mcp._tool_manager.list_tools():
        assert tool.description, f"tool {tool.name!r} has no description (docstring)"
        schema = tool.parameters
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert "properties" in schema


def test_required_tool_arguments_have_no_defaults_baked_into_schema_incorrectly():
    """Sanity check: tools with a required first arg (e.g. model_id) list it as required."""
    mcp = _build()
    by_name = {t.name: t for t in mcp._tool_manager.list_tools()}

    get_model = by_name["whitebox_models_get"]
    assert get_model.parameters["required"] == ["model_id"]

    list_models = by_name["whitebox_models_list"]
    assert (
        "required" not in list_models.parameters
        or list_models.parameters["required"] == []
    )
