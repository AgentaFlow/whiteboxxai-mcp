# whiteboxxai-mcp

<!-- mcp-name: io.github.AgentaFlow/whiteboxxai-mcp -->

[![PyPI version](https://img.shields.io/pypi/v/whiteboxxai-mcp.svg)](https://pypi.org/project/whiteboxxai-mcp/)
[![Python versions](https://img.shields.io/pypi/pyversions/whiteboxxai-mcp.svg)](https://pypi.org/project/whiteboxxai-mcp/)
[![Test](https://github.com/AgentaFlow/whiteboxxai-mcp/actions/workflows/test.yml/badge.svg)](https://github.com/AgentaFlow/whiteboxxai-mcp/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that exposes
[WhiteBoxXAI](https://whiteboxxai.com) observability and explainability capabilities —
model registry, Git-backed model diffing, prediction logging, drift detection, bias/fairness auditing,
ISO/IEC 42001 readiness and compliance evidence, AI Risk Register / governance review queries, prompt
governance, asynchronous SHAP/LIME explanation generation, LLM explainability (attention/token-importance/sensitivity/counterfactual
analysis and prompt debugging), LLM call monitoring (token/cost/latency tracking), RAG pipeline
monitoring, content safety analysis, alert rules and triage context, alert intelligence,
alert management, multi-agent workflow tracking, and model performance metrics (197 tools across 18 domains) — to any
MCP-compatible client (Claude Desktop, Claude Code, LangChain, custom agent harnesses,
etc.). Supports MCP Apps (`ui://` resources) with sandboxed interactive audit cards,
stdio transport, and Streamable HTTP transport (hosted at `https://api.whiteboxxai.com/mcp/`).

See the [integration docs](https://docs.whiteboxxai.com/integrations/mcp) for full setup,
connection examples, the tool reference, and known v1 limitations. See [`examples/`](./examples)
for runnable LangChain, Semantic Kernel, and raw-MCP-client integrations.

## Install

```bash
pip install whiteboxxai-mcp
```

## Run (stdio — local/self-hosted)

Configure a credential via environment variables, then run the console script.
Stdio transport authenticates via `WHITEBOXXAI_MCP_API_KEY`:

```bash
# An API key generated from the WhiteBoxXAI dashboard's Settings > API Keys tab.
# Never expires on a fixed schedule -- only when explicitly revoked -- and doesn't require storing a password.
export WHITEBOXXAI_MCP_API_BASE_URL="https://api.whiteboxxai.com"
export WHITEBOXXAI_MCP_API_KEY="wbx_live_..."

whiteboxxai-mcp
```

Point your MCP client at this command (e.g. in Claude Desktop's `mcpServers` config,
or Claude Code's `.mcp.json`).

For hosted Streamable HTTP (`https://api.whiteboxxai.com/mcp/`), pass your API key or personal
JWT in the `Authorization: Bearer <token>` header.

## Agent skills

To have a coding agent set this up and call the tools correctly, give it the `whitebox-mcp-server`
skill (and `whitebox-governance-rules`). They are published at
`https://whiteboxxai.com/skills/index.md`. The tool list they reference is pinned in
[`tests/mcp_inspector_snapshot.json`](./tests/mcp_inspector_snapshot.json); a rename or removal
of a tool shows up as a diff there.

## Versioning & support

This package follows [Semantic Versioning](https://semver.org/); see
[`CHANGELOG.md`](./CHANGELOG.md) for release history. Self-hosted stdio usage (above) is
best-effort supported; the hosted `/mcp/` HTTP endpoint is a product surface whose
support/SLA follows the caller's underlying Free/Business Cloud plan.

## Known v1 limitations

- **Auth**: stdio uses `WHITEBOXXAI_MCP_API_KEY` — a scoped, revocable
  credential that isn't tied to a human account's password or session lifecycle.
  Hosted Streamable HTTP (`https://api.whiteboxxai.com/mcp/`) accepts an API key or personal JWT.
- **Alerts**: covers core alert rules and triage context (`whitebox_alerts_*`),
  alert intelligence (`whitebox_alert_intelligence_*`, 20 tools), and alert management
  (`whitebox_alert_management_*`, 13 tools).
- **`whitebox_agent_workflows_cost_breakdown`/`_bottlenecks`/`_timeline`**: these three
  can take up to ~30s — the backend blocks server-side on a Celery task before
  responding, and the SDK client uses an extended (40s) timeout for exactly these calls.
- **Scope**: models, predictions, drift, bias/fairness, AI Risk Register / governance
  queries, prompts, compliance, explanations, LLM explainability, LLM call
  monitoring, RAG monitoring, content safety, alert intelligence, alert management,
  multi-agent workflows, and metrics all ship across 18 domains (197 tools total) over
  both stdio and hosted Streamable HTTP.

## MCP Registry

The server is published to the [official MCP Registry](https://registry.modelcontextprotocol.io/)
as `io.github.AgentaFlow/whiteboxxai-mcp`.

## Development

```bash
pip install -e ".[dev]"
pytest tests/unit

# MCP Inspector tool-list regression check (needs Node.js); after an intentional
# tool change, regenerate the committed snapshot with --write
python tests/check_mcp_inspector_snapshot.py
```

Releases are cut from GitHub releases; see [`RELEASING.md`](./RELEASING.md).

## License

MIT -- see [`LICENSE`](./LICENSE).
