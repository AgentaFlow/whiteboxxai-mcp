"""
Semantic Kernel MCP Client Example (issue #227)

Demonstrates loading the WhiteBoxXAI MCP server's tools into Microsoft
Semantic Kernel via `semantic_kernel.connectors.mcp.MCPStdioPlugin`, the
built-in Semantic Kernel<->MCP bridge. The plugin can be registered on a
`Kernel` (see the commented-out section at the bottom) exactly like any
other Semantic Kernel plugin, making every WhiteBoxXAI tool callable by an
LLM through Semantic Kernel's function-calling.

This example invokes tools directly (`await plugin.call_tool(...)`) rather
than driving them through an LLM-orchestrated planner, so it runs
deterministically without needing an LLM API key -- the point being
demonstrated is the MCP<->Semantic Kernel bridge itself, not planning.

Covers, in order:
  1. Connect to the server (spawns `whiteboxxai-mcp` as a subprocess)
  2. List available tools
  3. Call a read tool (whitebox_models_list)
  4. Call a write tool (whitebox_models_register)
  5. The async-explanation poll pattern (see raw_mcp_client.py for why the
     caller, not the tool, is responsible for polling)

Setup:
    pip install "whiteboxxai-mcp[examples]"
    export WHITEBOXXAI_MCP_API_KEY=wbx_...      # see README.md
    python examples/semantic_kernel_mcp_client.py
"""

import asyncio
import json

from semantic_kernel.connectors.mcp import MCPStdioPlugin


def _first_json(result) -> dict:
    """MCPStdioPlugin.call_tool() returns a list of semantic_kernel
    TextContent objects; WhiteBoxXAI's tools always emit exactly one
    JSON-encoded text block for a single-object response."""
    (block,) = result
    return json.loads(block.text)


async def main() -> None:
    plugin = MCPStdioPlugin(
        name="whiteboxxai",
        description="WhiteBoxXAI AI governance and observability tools",
        command="whiteboxxai-mcp",
        args=[],
    )

    async with plugin:
        # 1 & 2. Connect and list tools happen together: entering the
        # `async with` block calls plugin.connect(), which loads every
        # tool as a Semantic Kernel function on `plugin`.
        tool_names = [name for name in dir(plugin) if name.startswith("whitebox_")]
        print(f"Connected. {len(tool_names)} tools available.")

        # 3. Call a read tool.
        models_result = await plugin.call_tool("whitebox_models_list", limit=5)
        print("Existing models (first 5):", models_result[0].text)

        # 4. Call a write tool.
        registered = await plugin.call_tool(
            "whitebox_models_register",
            name="example-fraud-detector",
            model_type="xgboost",
            version="1.0.0",
            features=["transaction_amount", "merchant_category"],
            baseline_metrics={"accuracy": 0.94},
        )
        model = _first_json(registered)
        print(f"Registered model {model['id']} ({model['name']})")

        # 5. Async-explanation poll pattern.
        started = await plugin.call_tool(
            "whitebox_explanations_generate_async",
            model_id=model["id"],
            instance={
                "transaction_amount": 250.0,
                "merchant_category": "electronics",
            },
        )
        explanation = _first_json(started)
        explanation_id = explanation["explanation_id"]
        print(f"Explanation {explanation_id} queued, polling for completion...")

        while True:
            poll_result = await plugin.call_tool(
                "whitebox_explanations_get", explanation_id=explanation_id
            )
            status = _first_json(poll_result)
            if status["status"] != "pending":
                print(f"Explanation finished: status={status['status']}")
                break
            await asyncio.sleep(2)

    # To make these tools callable by an LLM through Semantic Kernel's
    # function-calling instead of calling them directly (needs a real LLM
    # API key/deployment):
    #
    #   from semantic_kernel import Kernel
    #   from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    #
    #   kernel = Kernel()
    #   kernel.add_service(OpenAIChatCompletion(ai_model_id="gpt-4o"))
    #   async with MCPStdioPlugin(
    #       name="whiteboxxai", command="whiteboxxai-mcp", args=[]
    #   ) as plugin:
    #       kernel.add_plugin(plugin)
    #       result = await kernel.invoke_prompt(
    #           "List the 5 most recently registered models."
    #       )


if __name__ == "__main__":
    asyncio.run(main())
