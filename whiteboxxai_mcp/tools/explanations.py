"""MCP tools for WhiteBoxXAI explanations (backend/api/v1/explanations.py):
SHAP/LIME generation (sync and async/Celery-backed), bulk generation,
listing/lookup, stats, per-model config CRUD, cross-explanation comparison,
and visualization data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.auth.context import get_session
from whiteboxxai_mcp.errors import call_with_session
from whiteboxxai_mcp.ui_resources import (
    EXPLANATION_CARD_URI,
    sanitize_explanation_payload,
    ui_tool_meta,
)


def register(mcp: FastMCP) -> None:
    """Register explanation tools on `mcp`."""

    @mcp.tool(
        name="whitebox_explanations_generate",
        meta=ui_tool_meta(EXPLANATION_CARD_URI),
    )
    async def whitebox_explanations_generate(
        model_id: str,
        instance: dict[str, Any],
        method: str | None = None,
        prediction_id: str | None = None,
        num_features: int = 10,
        num_samples: int = 5000,
        use_cache: bool = True,
        total_cfs: int = 3,
        desired_class: str = "opposite",
        features_to_vary: str | list[str] = "all",
    ) -> dict[str, Any]:
        """Generate a SHAP/LIME explanation and block until it's computed.

        For a fast model this is fine; for slow models prefer
        whitebox_explanations_generate_async + polling instead, since this
        call has no timeout of its own.

        Args:
            model_id: Registered model's ID
            instance: Feature values for the instance to explain, e.g. {"age": 42, "income": 50000}
            method: "shap" or "lime" (uses the model's configured default if omitted)
            prediction_id: Optional ID of the prediction this explanation is for
            num_features: Number of top features to return (1-100)
            num_samples: Number of samples the explainer draws (100-50000); higher is slower but more stable
            use_cache: Reuse a cached explanation for an identical instance if one exists
        """
        session = get_session()
        result = await call_with_session(
            session,
            lambda client: client.explanations.agenerate(
                model_id,
                instance,
                method=method,
                prediction_id=prediction_id,
                num_features=num_features,
                num_samples=num_samples,
                use_cache=use_cache,
                total_cfs=total_cfs,
                desired_class=desired_class,
                features_to_vary=features_to_vary,
            ),
        )
        return sanitize_explanation_payload(result)

    @mcp.tool(name="whitebox_explanations_generate_bulk")
    async def whitebox_explanations_generate_bulk(
        model_id: str,
        instances: list[dict[str, Any]],
        method: str | None = None,
        num_features: int = 10,
        num_samples: int = 5000,
        parallel: bool = True,
        total_cfs: int = 3,
        desired_class: str = "opposite",
        features_to_vary: str | list[str] = "all",
    ) -> dict[str, Any]:
        """Start explanation generation for multiple instances at once (non-blocking).

        Dispatches one Celery task per instance (max 100 instances per call)
        and returns immediately with one PENDING explanation_id per instance.
        Poll whitebox_explanations_get for each ID, same as
        whitebox_explanations_generate_async.

        Args:
            model_id: Registered model's ID
            instances: Feature values for each instance to explain (max 100)
            method: "shap" or "lime" (uses the model's configured default if omitted)
            num_features: Number of top features to return per explanation (1-100)
            num_samples: Number of samples the explainer draws (100-50000)
            parallel: Retained for backward compatibility; not read by the backend
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.explanations.agenerate_bulk(
                model_id,
                instances,
                method=method,
                num_features=num_features,
                num_samples=num_samples,
                parallel=parallel,
                total_cfs=total_cfs,
                desired_class=desired_class,
                features_to_vary=features_to_vary,
            ),
        )

    @mcp.tool(name="whitebox_explanations_generate_async")
    async def whitebox_explanations_generate_async(
        model_id: str,
        instance: dict[str, Any],
        method: str | None = None,
        prediction_id: str | None = None,
        num_features: int = 10,
        num_samples: int = 5000,
        total_cfs: int = 3,
        desired_class: str = "opposite",
        features_to_vary: str | list[str] = "all",
    ) -> dict[str, Any]:
        """Start real SHAP/LIME explanation generation for a model instance.

        Returns immediately with a PENDING explanation_id -- this does NOT
        block until the explanation is ready. Computation happens on a
        Celery worker and can take several seconds to tens of seconds.
        Call whitebox_explanations_get with the returned explanation_id and
        poll (e.g. every 1-2 seconds) until status is no longer "pending"
        before reading feature_weights.

        Args:
            model_id: Registered model's ID
            instance: Feature values for the instance to explain, e.g. {"age": 42, "income": 50000}
            method: "shap" or "lime" (uses the model's configured default if omitted)
            prediction_id: Optional ID of the prediction this explanation is for
            num_features: Number of top features to return (1-100)
            num_samples: Number of samples the explainer draws (100-50000); higher is slower but more stable
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.explanations.agenerate_async(
                model_id,
                instance,
                method=method,
                prediction_id=prediction_id,
                num_features=num_features,
                num_samples=num_samples,
                total_cfs=total_cfs,
                desired_class=desired_class,
                features_to_vary=features_to_vary,
            ),
        )

    @mcp.tool(
        name="whitebox_explanations_get",
        meta=ui_tool_meta(EXPLANATION_CARD_URI),
    )
    async def whitebox_explanations_get(explanation_id: str) -> dict[str, Any]:
        """Get an explanation by ID, including its status and (once ready) results.

        Poll this after whitebox_explanations_generate_async: while status
        is "pending" the explanation is still computing on a background
        worker and feature_weights will be empty. Poll again until status
        is "completed" (read feature_weights, base_value, score) or
        "failed" (read the error).
        """
        session = get_session()
        result = await call_with_session(
            session, lambda client: client.explanations.aget(explanation_id)
        )
        return sanitize_explanation_payload(result)

    @mcp.tool(name="whitebox_explanations_list_by_model")
    async def whitebox_explanations_list_by_model(
        model_id: str,
        method: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List explanations for a model, optionally filtered by method or status."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.explanations.alist_by_model(
                model_id, method=method, status=status, limit=limit, offset=offset
            ),
        )

    @mcp.tool(
        name="whitebox_explanations_get_by_prediction",
        meta=ui_tool_meta(EXPLANATION_CARD_URI),
    )
    async def whitebox_explanations_get_by_prediction(
        prediction_id: str,
    ) -> dict[str, Any]:
        """Get the explanation associated with a prediction, if one exists."""
        session = get_session()
        result = await call_with_session(
            session,
            lambda client: client.explanations.aget_by_prediction(prediction_id),
        )
        return sanitize_explanation_payload(result)

    @mcp.tool(name="whitebox_explanations_stats")
    async def whitebox_explanations_stats(model_id: str) -> dict[str, Any]:
        """Get aggregated explanation statistics for a model (counts by
        method/status, average computation time)."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.explanations.aget_stats(model_id)
        )

    @mcp.tool(name="whitebox_explanations_set_config")
    async def whitebox_explanations_set_config(
        model_id: str,
        enabled: bool = True,
        default_method: str = "shap_kernel",
        auto_explain: bool = False,
        cache_enabled: bool = True,
        cache_ttl_hours: int = 24,
        lime_config: dict[str, Any] | None = None,
        shap_config: dict[str, Any] | None = None,
        visualization_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create or update a model's explanation configuration (upsert)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.explanations.aset_config(
                model_id,
                enabled=enabled,
                default_method=default_method,
                auto_explain=auto_explain,
                cache_enabled=cache_enabled,
                cache_ttl_hours=cache_ttl_hours,
                lime_config=lime_config,
                shap_config=shap_config,
                visualization_config=visualization_config,
            ),
        )

    @mcp.tool(name="whitebox_explanations_get_config")
    async def whitebox_explanations_get_config(model_id: str) -> dict[str, Any]:
        """Get a model's explanation configuration."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.explanations.aget_config(model_id)
        )

    @mcp.tool(name="whitebox_explanations_delete_config")
    async def whitebox_explanations_delete_config(model_id: str) -> dict[str, Any]:
        """Delete a model's explanation configuration."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.explanations.adelete_config(model_id)
        )

    @mcp.tool(name="whitebox_explanations_compare")
    async def whitebox_explanations_compare(
        explanation_ids: list[str],
        comparison_type: str = "feature_importance",
    ) -> dict[str, Any]:
        """Compare 2-20 explanations against each other.

        comparison_type: "feature_importance", "instance_similarity", or
        "method_agreement".
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.explanations.acompare(
                explanation_ids, comparison_type=comparison_type
            ),
        )

    @mcp.tool(name="whitebox_explanations_visualize")
    async def whitebox_explanations_visualize(
        explanation_id: str,
        plot_type: str,
        max_features: int = 10,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get visualization data for a single explanation.

        plot_type: "waterfall", "force", or "feature_importance".
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.explanations.avisualize(
                explanation_id,
                plot_type,
                max_features=max_features,
                config=config,
            ),
        )

    @mcp.tool(name="whitebox_explanations_visualize_multi")
    async def whitebox_explanations_visualize_multi(
        explanation_ids: list[str],
        plot_type: str,
        feature_index: int | None = None,
        interaction_index: int | None = None,
        max_features: int = 20,
        highlight_index: int | None = None,
    ) -> dict[str, Any]:
        """Get visualization data spanning multiple explanations.

        plot_type: "summary", "decision", or "dependence". feature_index/
        interaction_index are used for dependence plots; highlight_index
        highlights one explanation in decision plots.
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.explanations.avisualize_multi(
                explanation_ids,
                plot_type,
                feature_index=feature_index,
                interaction_index=interaction_index,
                max_features=max_features,
                highlight_index=highlight_index,
            ),
        )
