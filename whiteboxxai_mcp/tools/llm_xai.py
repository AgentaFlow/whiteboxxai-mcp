"""MCP tools for LLM explainability (backend/api/v1/llm_xai.py):
attention/token-importance/sensitivity/counterfactual analysis and prompt
debugging for LLM calls, distinct from the model-level SHAP/LIME
explanations in tools/explanations.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from whiteboxxai_mcp.auth.context import get_session
from whiteboxxai_mcp.errors import call_with_session


def register(mcp: FastMCP) -> None:
    """Register all LLM-XAI tools on `mcp`."""

    @mcp.tool(name="whitebox_llm_xai_attention")
    async def whitebox_llm_xai_attention(
        prompt: str,
        completion: str | None = None,
        model_id: str | None = None,
        layers: list[int] | None = None,
        aggregate_heads: bool = True,
        attention_weights: list[Any] | None = None,
        top_k: int = 10,
        llm_log_id: str | None = None,
    ) -> dict[str, Any]:
        """Analyze transformer attention patterns for a prompt/completion pair.

        Extracts and aggregates multi-head attention weights to show what
        the model focused on. Requires attention_weights to be supplied
        (this endpoint does not itself run the model).
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm_xai.aattention(
                prompt,
                completion=completion,
                model_id=model_id,
                layers=layers,
                aggregate_heads=aggregate_heads,
                attention_weights=attention_weights,
                top_k=top_k,
                llm_log_id=llm_log_id,
            ),
        )

    @mcp.tool(name="whitebox_llm_xai_token_importance")
    async def whitebox_llm_xai_token_importance(
        prompt: str,
        completion: str,
        model_id: str | None = None,
        method: str = "perturbation",
        top_k: int = 10,
        llm_log_id: str | None = None,
    ) -> dict[str, Any]:
        """Calculate token-level importance scores for a prompt/completion pair.

        method: "gradient", "attention", or "perturbation".
        """
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm_xai.atoken_importance(
                prompt,
                completion,
                model_id=model_id,
                method=method,
                top_k=top_k,
                llm_log_id=llm_log_id,
            ),
        )

    @mcp.tool(name="whitebox_llm_xai_prompt_sensitivity")
    async def whitebox_llm_xai_prompt_sensitivity(
        prompt: str,
        baseline_output: str,
        model_id: str | None = None,
        num_perturbations: int = 10,
        perturbation_type: str = "mixed",
        llm_log_id: str | None = None,
    ) -> dict[str, Any]:
        """Test prompt robustness by perturbing it and measuring output stability."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm_xai.aprompt_sensitivity(
                prompt,
                baseline_output,
                model_id=model_id,
                num_perturbations=num_perturbations,
                perturbation_type=perturbation_type,
                llm_log_id=llm_log_id,
            ),
        )

    @mcp.tool(name="whitebox_llm_xai_counterfactuals")
    async def whitebox_llm_xai_counterfactuals(
        prompt: str,
        original_output: str,
        target_change: str = "any",
        model_id: str | None = None,
        max_edits: int = 3,
        num_candidates: int = 10,
        search_strategy: str = "greedy",
        llm_log_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate minimal prompt edits that change the model's output,
        useful for understanding decision boundaries."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm_xai.acounterfactuals(
                prompt,
                original_output,
                target_change=target_change,
                model_id=model_id,
                max_edits=max_edits,
                num_candidates=num_candidates,
                search_strategy=search_strategy,
                llm_log_id=llm_log_id,
            ),
        )

    @mcp.tool(name="whitebox_llm_xai_debug_prompt")
    async def whitebox_llm_xai_debug_prompt(
        prompt: str,
        model_id: str | None = None,
        check_clarity: bool = True,
        check_specificity: bool = True,
        check_completeness: bool = True,
        check_consistency: bool = True,
        suggest_improvements: bool = True,
        llm_log_id: str | None = None,
        prompt_version_id: str | None = None,
    ) -> dict[str, Any]:
        """Assess prompt quality across multiple dimensions and get
        improvement recommendations."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm_xai.adebug_prompt(
                prompt,
                model_id=model_id,
                check_clarity=check_clarity,
                check_specificity=check_specificity,
                check_completeness=check_completeness,
                check_consistency=check_consistency,
                suggest_improvements=suggest_improvements,
                llm_log_id=llm_log_id,
                prompt_version_id=prompt_version_id,
            ),
        )

    @mcp.tool(name="whitebox_llm_xai_get_explanation")
    async def whitebox_llm_xai_get_explanation(explanation_id: str) -> dict[str, Any]:
        """Get a specific attention/token-importance/sensitivity/counterfactual
        explanation by ID."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.llm_xai.aget_explanation(explanation_id)
        )

    @mcp.tool(name="whitebox_llm_xai_get_prompt_analysis")
    async def whitebox_llm_xai_get_prompt_analysis(
        analysis_id: str,
    ) -> dict[str, Any]:
        """Get a specific prompt debug analysis by ID."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.llm_xai.aget_prompt_analysis(analysis_id)
        )

    @mcp.tool(name="whitebox_llm_xai_list_explanations")
    async def whitebox_llm_xai_list_explanations(
        llm_log_id: str | None = None,
        explanation_type: str | None = None,
        model_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List explanations, optionally filtered by LLM log, type
        (attention_analysis/token_importance/sensitivity_analysis/
        counterfactual_generation), or model."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm_xai.alist_explanations(
                llm_log_id=llm_log_id,
                explanation_type=explanation_type,
                model_id=model_id,
                limit=limit,
            ),
        )

    @mcp.tool(name="whitebox_llm_xai_list_prompt_analyses")
    async def whitebox_llm_xai_list_prompt_analyses(
        prompt: str | None = None,
        model_id: str | None = None,
        min_quality_score: float | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List prompt analyses, optionally filtered by similar prompt text,
        model, or minimum quality score (0-1)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm_xai.alist_prompt_analyses(
                prompt=prompt,
                model_id=model_id,
                min_quality_score=min_quality_score,
                limit=limit,
            ),
        )

    @mcp.tool(name="whitebox_llm_xai_stats")
    async def whitebox_llm_xai_stats(
        model_id: str | None = None, days: int = 7
    ) -> dict[str, Any]:
        """Get aggregated LLM-XAI usage statistics (counts by explanation
        type, average computation time, prompt quality)."""
        session = get_session()
        return await call_with_session(
            session, lambda client: client.llm_xai.astats(model_id=model_id, days=days)
        )

    @mcp.tool(name="whitebox_llm_xai_visualize_attention")
    async def whitebox_llm_xai_visualize_attention(
        explanation_id: str,
    ) -> dict[str, Any]:
        """Get attention-heatmap visualization data for an explanation
        (for rendering with Plotly/D3.js-style tools)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm_xai.avisualize_attention(explanation_id),
        )

    @mcp.tool(name="whitebox_llm_xai_visualize_token_importance")
    async def whitebox_llm_xai_visualize_token_importance(
        explanation_id: str, top_k: int = 20
    ) -> dict[str, Any]:
        """Get token-importance bar-chart visualization data for an explanation."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm_xai.avisualize_token_importance(
                explanation_id, top_k=top_k
            ),
        )

    @mcp.tool(name="whitebox_llm_xai_batch_analyze")
    async def whitebox_llm_xai_batch_analyze(
        llm_log_id: str,
        include_attention: bool = True,
        include_token_importance: bool = True,
        include_sensitivity: bool = False,
        include_prompt_debug: bool = True,
    ) -> dict[str, Any]:
        """Run multiple explanation analyses on a single LLM log in one call
        (more efficient than calling each analysis tool separately)."""
        session = get_session()
        return await call_with_session(
            session,
            lambda client: client.llm_xai.abatch_analyze(
                llm_log_id,
                include_attention=include_attention,
                include_token_importance=include_token_importance,
                include_sensitivity=include_sensitivity,
                include_prompt_debug=include_prompt_debug,
            ),
        )
