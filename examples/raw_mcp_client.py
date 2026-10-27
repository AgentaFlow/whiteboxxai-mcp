"""
Raw MCP Client Example (issue #227)

Demonstrates driving the WhiteBoxXAI MCP server with nothing but the `mcp`
Python SDK -- no agent framework in between. Useful as a minimal reference
for any MCP-compatible client library, and as a smoke test that the server
is reachable and behaving correctly over stdio.

Covers, in order:
  1. Connect to the server (spawns `whiteboxxai-mcp` as a subprocess)
  2. List available tools
  3. Call a read tool (whitebox_models_list)
  4. Call a write tool (whitebox_models_register)
  5. The async-explanation poll pattern: kick off an async SHAP explanation,
     then poll for it to finish. WhiteBoxXAI intentionally does not make the
     tool itself block-and-poll (an explanation can take longer than a
     typical MCP client's request timeout) -- polling is the *caller's*
     responsibility, which is exactly what this section demonstrates.

Setup:
    pip install whiteboxxai-mcp
    export WHITEBOXXAI_MCP_API_KEY=wbx_...      # see README.md
    python examples/raw_mcp_client.py
"""

import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _first_json(result) -> dict:
    """MCP tools return their result as one or more TextContent blocks;
    WhiteBoxXAI's tools always emit exactly one JSON-encoded block for a
    single-object response."""
    (block,) = (c for c in result.content if hasattr(c, "text"))
    return json.loads(block.text)


async def main() -> None:
    server_params = StdioServerParameters(command="whiteboxxai-mcp", args=[])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. List available tools.
            tools = await session.list_tools()
            print(f"Connected. {len(tools.tools)} tools available.")

            # 2. Call a read tool.
            models = await session.call_tool("whitebox_models_list", {"limit": 5})
            print("Existing models (first 5):", models.content)

            # 3. Call a write tool.
            registered = await session.call_tool(
                "whitebox_models_register",
                {
                    "name": "example-fraud-detector",
                    "model_type": "xgboost",
                    "version": "1.0.0",
                    "features": ["transaction_amount", "merchant_category"],
                    "baseline_metrics": {"accuracy": 0.94},
                },
            )
            model = _first_json(registered)
            print(f"Registered model {model['id']} ({model['name']})")

            # 4. Async-explanation poll pattern.
            started = await session.call_tool(
                "whitebox_explanations_generate_async",
                {
                    "model_id": model["id"],
                    "instance": {
                        "transaction_amount": 250.0,
                        "merchant_category": "electronics",
                    },
                },
            )
            explanation = _first_json(started)
            explanation_id = explanation["explanation_id"]
            print(f"Explanation {explanation_id} queued, polling for completion...")

            while True:
                poll_result = await session.call_tool(
                    "whitebox_explanations_get", {"explanation_id": explanation_id}
                )
                status = _first_json(poll_result)
                if status["status"] != "pending":
                    print(f"Explanation finished: status={status['status']}")
                    break
                await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
