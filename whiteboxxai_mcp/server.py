"""Builds the WhiteBoxXAI MCP server (tool/resource registration)."""

from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from whiteboxxai_mcp.config import MCPSettings
from whiteboxxai_mcp.tools import register_all_tools
from whiteboxxai_mcp.ui_resources import register_ui_resources

# FastMCP auto-enables DNS-rebinding Host-header protection whenever its
# `host` constructor param is left at its own default ("127.0.0.1"), with
# allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"] -- correct for a
# server bound to and reached only via localhost, but backend/main.py mounts
# this behind Azure Front Door under a real public hostname that varies by
# environment, so that allowlist would reject every real request with a 421
# Misdirected Request. Disabled explicitly here rather than chasing an
# environment-specific allowed_hosts list, because the DNS-rebinding threat
# this defends against (a victim's *browser* tricked into sending
# credentialed requests to an attacker-DNS-rebound target resolving to
# localhost/an internal address) doesn't fit this server's actual shape: MCP
# HTTP clients aren't browsers, every request already requires an
# `Authorization: Bearer <jwt>` (asgi.py's _AuthGateApp, no ambient/cookie
# credential to steal), and production traffic is already gated on a valid
# X-Azure-FDID header by FrontDoorHeaderMiddleware before it ever reaches
# this app.
_HTTP_TRANSPORT_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=False
)


def build_server(
    settings: MCPSettings, *, streamable_http_path: str | None = None
) -> FastMCP:
    """Construct a FastMCP instance with all WhiteBoxXAI tools registered.

    `streamable_http_path` is only relevant to the Streamable HTTP transport
    (see asgi.py) -- it is `None` here and left at FastMCP's own default for
    every stdio call site. asgi.py passes "/" so that once its returned app
    is mounted at /mcp in backend/main.py, the tool is reachable at exactly
    /mcp rather than /mcp/mcp (FastMCP.streamable_http_app() registers its
    one internal route at this path, defaulting to "/mcp"). Passing it is
    also what signals HTTP mode here, which is why transport_security is
    only set alongside it -- stdio's FastMCP instance never touches an ASGI
    Host header at all, so the setting would be inert there anyway.
    """
    kwargs: dict = {}
    if streamable_http_path is not None:
        kwargs["streamable_http_path"] = streamable_http_path
        kwargs["transport_security"] = _HTTP_TRANSPORT_SECURITY

    mcp = FastMCP(
        name="whiteboxxai",
        instructions=(
            "Tools for the WhiteBoxXAI AI governance, observability and "
            "explainability platform: model registry, Git-backed model diffing, "
            "prediction logging, data drift detection, bias/fairness auditing, "
            "ISO/IEC 42001 compliance, prompt governance, alert intelligence and "
            "management, multi-agent workflows, and asynchronous SHAP/LIME "
            "explanation generation across 18 domains (197 tools total). All "
            "tool names are prefixed with whitebox_. See each tool's description "
            "for its specific inputs and any known limitations."
        ),
        **kwargs,  # type: ignore[arg-type, unused-ignore]
    )
    register_all_tools(mcp)
    register_ui_resources(mcp)
    return mcp
