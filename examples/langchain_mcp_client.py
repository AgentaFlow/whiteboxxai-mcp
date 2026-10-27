"""
LangChain MCP Client Example (issue #227)

Demonstrates loading the WhiteBoxXAI MCP server's tools into LangChain via
`langchain-mcp-adapters`, the standard LangChain<->MCP bridge package. Each
tool comes back as a regular `langchain_core.tools.BaseTool`, so it can be
bound onto any LangChain/LangGraph agent (see the commented-out
`create_react_agent` call at the bottom) exactly like a hand-written
`@tool`-decorated function.

This example invokes the tools directly (`await tool.ainvoke(...)`) rather
than driving them through an LLM-orchestrated agent loop, so it runs
deterministically without needing an LLM API key -- the point being
demonstrated is the MCP<->LangChain bridge itself, not agent reasoning.

Covers, in order:
  1. Connect to the server (spawns `whiteboxxai-mcp` as a subprocess) and
     list the tools LangChain sees
  2. Call a read tool (whitebox_models_list)
  3. Call a write tool (whitebox_models_register)
  4. The async-explanation poll pattern (see raw_mcp_client.py for why the
     caller, not the tool, is responsible for polling)

Setup:
    pip install "whiteboxxai-mcp[examples]"
    export WHITEBOXXAI_MCP_API_KEY=wbx_...      # see README.md
    python examples/langchain_mcp_client.py
"""

import asyncio
import json

from langchain_mcp_adapters.client import MultiServerMCPClient


def _first_json(result) -> dict:
    """langchain-mcp-adapters returns a tool call's result as a list of
    content-block dicts (e.g. [{"type": "text", "text": "...", "id": ...}]),
    not a raw string -- WhiteBoxXAI's tools always emit exactly one
    JSON-encoded text block for a single-object response."""
    (block,) = (c for c in result if c.get("type") == "text")
    return json.loads(block["text"])


async def main() -> None:
    client = MultiServerMCPClient(
        {
            "whiteboxxai": {
                "transport": "stdio",
                "command": "whiteboxxai-mcp",
                "args": [],
            }
        }
    )

    # 1. Connect and list tools (as LangChain BaseTool objects).
    tools = await client.get_tools()
    print(f"Connected. {len(tools)} tools available as LangChain tools.")
    tools_by_name = {tool.name: tool for tool in tools}

    # 2. Call a read tool.
    models_result = await tools_by_name["whitebox_models_list"].ainvoke({"limit": 5})
    print("Existing models (first 5):", models_result)

    # 3. Call a write tool.
    registered = await tools_by_name["whitebox_models_register"].ainvoke(
        {
            "name": "example-fraud-detector",
            "model_type": "xgboost",
            "version": "1.0.0",
            "features": ["transaction_amount", "merchant_category"],
            "baseline_metrics": {"accuracy": 0.94},
        }
    )
    model = _first_json(registered)
    print(f"Registered model {model['id']} ({model['name']})")

    # 4. Async-explanation poll pattern.
    started = await tools_by_name["whitebox_explanations_generate_async"].ainvoke(
        {
            "model_id": model["id"],
            "instance": {
                "transaction_amount": 250.0,
                "merchant_category": "electronics",
            },
        }
    )
    explanation = _first_json(started)
    explanation_id = explanation["explanation_id"]
    print(f"Explanation {explanation_id} queued, polling for completion...")

    while True:
        poll_result = await tools_by_name["whitebox_explanations_get"].ainvoke(
            {"explanation_id": explanation_id}
        )
        status = _first_json(poll_result)
        if status["status"] != "pending":
            print(f"Explanation finished: status={status['status']}")
            break
        await asyncio.sleep(2)

    # To drive these tools from an actual LLM-orchestrated agent instead of
    # calling them directly (needs `pip install langgraph` and a real LLM
    # API key):
    #
    #   from langchain_openai import ChatOpenAI
    #   from langgraph.prebuilt import create_react_agent
    #
    #   agent = create_react_agent(ChatOpenAI(model="gpt-4o"), tools)
    #   result = await agent.ainvoke(
    #       {"messages": [("user", "List the 5 most recently registered models.")]}
    #   )


if __name__ == "__main__":
    asyncio.run(main())
