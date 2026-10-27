"""MCP tools for the WhiteBoxXAI model registry (backend/api/v1/models.py)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.auth.context import get_session
from whiteboxxai_mcp.errors import call_with_session
from whiteboxxai_mcp.ui_resources import (
    MODEL_VERSION_DIFF_CARD_URI,
    enrich_model_diff_payload,
    ui_tool_meta,
)


def _build_register_kwargs(
    description: str | None,
    target_variable: str | None,
    fairness_attributes: list[str] | None,
    tags: list[str] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if description is not None:
        kwargs["description"] = description
    if target_variable is not None:
        kwargs["target_variable"] = target_variable
    if fairness_attributes is not None:
        kwargs["fairness_attributes"] = fairness_attributes
    if tags is not None:
        kwargs["tags"] = tags
    return kwargs


def _build_update_kwargs(
    description: str | None,
    target_variable: str | None,
    fairness_attributes: list[str] | None,
    tags: list[str] | None,
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if description is not None:
        updates["description"] = description
    if target_variable is not None:
        updates["target_variable"] = target_variable
    if fairness_attributes is not None:
        updates["fairness_attributes"] = fairness_attributes
    if tags is not None:
        updates["tags"] = tags
    return updates


def register(mcp: FastMCP) -> None:
    """Register all model-registry tools on `mcp`."""

    @mcp.tool(name="whitebox_models_register")
    async def whitebox_models_register(
        name: str,
        model_type: str,
        features: list[str],
        baseline_metrics: dict[str, Any],
        framework: str | None = None,
        version: str | None = None,
        description: str | None = None,
        target_variable: str | None = None,
        fairness_attributes: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Register a new model in WhiteBoxXAI's model registry.

        Args:
            name: Model name (alphanumeric, spaces, dashes, underscores only)
            model_type: e.g. "classification", "regression"
            features: List of input feature names
            baseline_metrics: Baseline performance metrics, e.g. {"accuracy": 0.94}
            framework: ML framework (sklearn, pytorch, tensorflow, etc.)
            version: Model version (semantic versioning, e.g. "1.0.0")
            description: Human-readable description
            target_variable: Name of the target/label variable
            fairness_attributes: Sensitive attributes to track for bias auditing
            tags: Free-form tags for search/filtering

        Returns:
            The registered model's record, including its generated model_id.
        """
        session = get_session()
        kwargs = _build_register_kwargs(
            description, target_variable, fairness_attributes, tags
        )

        return await call_with_session(
            session,
            lambda client: client.models.aregister(
                name=name,
                model_type=model_type,
                framework=framework,
                version=version,
                features=features,
                baseline_metrics=baseline_metrics,
                **kwargs,
            ),
        )

    @mcp.tool(name="whitebox_models_get")
    async def whitebox_models_get(model_id: str) -> dict[str, Any]:
        """Get a registered model's full record by its model_id."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.models.aget(model_id)
        )

    @mcp.tool(name="whitebox_models_list")
    async def whitebox_models_list(
        skip: int = 0,
        limit: int = 100,
        status_filter: str | None = None,
        model_type: str | None = None,
        search: str | None = None,
        tags: str | None = None,
    ) -> list[dict[str, Any]]:
        """List registered models, optionally filtered by status/type/tags/search text.

        Args:
            skip: Number of results to skip (pagination)
            limit: Maximum results to return (1-1000)
            status_filter: One of ACTIVE, INACTIVE, DEPRECATED, ARCHIVED
            model_type: e.g. "classification", "regression"
            search: Free-text search over name/description
            tags: Comma-separated list of tags to filter by
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.models.alist(
                skip=skip,
                limit=limit,
                status_filter=status_filter,
                model_type=model_type,
                tags=tags,
                search=search,
            ),
        )

    @mcp.tool(name="whitebox_models_update")
    async def whitebox_models_update(
        model_id: str,
        description: str | None = None,
        target_variable: str | None = None,
        fairness_attributes: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Partially update a model's metadata. Only provided fields are changed."""
        session = get_session()
        updates = _build_update_kwargs(
            description, target_variable, fairness_attributes, tags
        )
        return await call_with_session(
            session, lambda client: client.models.aupdate(model_id, **updates)
        )

    @mcp.tool(name="whitebox_models_update_status")
    async def whitebox_models_update_status(
        model_id: str, new_status: str
    ) -> dict[str, Any]:
        """Update a model's status: ACTIVE, INACTIVE, DEPRECATED, or ARCHIVED."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.models.aupdate_status(model_id, new_status)
        )

    @mcp.tool(name="whitebox_models_versions")
    async def whitebox_models_versions(
        model_name: str, skip: int = 0, limit: int = 100
    ) -> list[dict[str, Any]]:
        """List all versions of a model by name, newest first."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.models.aget_versions(
                model_name, skip=skip, limit=limit
            ),
        )

    @mcp.tool(name="whitebox_models_latest")
    async def whitebox_models_latest(model_name: str) -> dict[str, Any]:
        """Get the most recently created version of a model by name."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.models.aget_latest(model_name)
        )

    @mcp.tool(name="whitebox_models_update_baseline")
    async def whitebox_models_update_baseline(
        model_id: str,
        baseline_metrics: dict[str, Any],
        baseline_data_hash: str | None = None,
        baseline_data_count: int | None = None,
    ) -> dict[str, Any]:
        """Refresh a model's baseline metrics/profile, e.g. after retraining.

        This baseline is what drift detection compares recent predictions
        against, so update it whenever the model is retrained or revalidated.
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.models.aupdate_baseline(
                model_id,
                baseline_metrics,
                baseline_data_hash=baseline_data_hash,
                baseline_data_count=baseline_data_count,
            ),
        )

    @mcp.tool(name="whitebox_models_archive")
    async def whitebox_models_archive(model_id: str) -> dict[str, Any]:
        """Archive a model (soft delete; can be restored later)."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.models.aarchive(model_id)
        )

    @mcp.tool(name="whitebox_models_restore")
    async def whitebox_models_restore(model_id: str) -> dict[str, Any]:
        """Restore a previously archived model to active status."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.models.arestore(model_id)
        )

    @mcp.tool(name="whitebox_models_delete")
    async def whitebox_models_delete(model_id: str) -> dict[str, Any]:
        """Permanently delete a model. This is a hard delete and cannot be undone;
        prefer whitebox_models_archive for reversible removal."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.models.adelete(model_id)
        )

    @mcp.tool(
        name="whitebox_models_compare_versions",
        meta=ui_tool_meta(MODEL_VERSION_DIFF_CARD_URI),
    )
    async def whitebox_models_compare_versions(
        version1_id: str, version2_id: str
    ) -> dict[str, Any]:
        """Compare two Git-backed model versions, returning architecture changes, parameter deltas, and breaking change flags."""
        session = get_session()
        raw = await call_with_session(
            session,
            lambda client: client.models.acompare_versions(version1_id, version2_id),
        )
        return enrich_model_diff_payload(raw)

    @mcp.tool(
        name="whitebox_models_version_changelog",
        meta=ui_tool_meta(MODEL_VERSION_DIFF_CARD_URI),
    )
    async def whitebox_models_version_changelog(
        version1_id: str, version2_id: str
    ) -> dict[str, Any]:
        """Generate a markdown changelog between two Git-backed model versions."""
        session = get_session()
        raw = await call_with_session(
            session,
            lambda client: client.models.aversion_changelog(version1_id, version2_id),
        )
        return enrich_model_diff_payload(raw)

    @mcp.tool(name="whitebox_models_list_git_versions")
    async def whitebox_models_list_git_versions(
        model_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """List Git-backed version sources for a model, newest first."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.models.alist_git_versions(model_id, limit=limit),
        )

    @mcp.tool(
        name="whitebox_models_architecture_timeline",
        meta=ui_tool_meta(MODEL_VERSION_DIFF_CARD_URI),
    )
    async def whitebox_models_architecture_timeline(
        model_id: str,
    ) -> dict[str, Any]:
        """Get the architecture changes timeline across all versions of a model."""
        session = get_session()
        raw = await call_with_session(
            session,
            lambda client: client.models.aarchitecture_changes_timeline(model_id),
        )
        return enrich_model_diff_payload(raw)
