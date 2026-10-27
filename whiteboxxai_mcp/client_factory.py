"""Builds WhiteBoxXAI SDK client instances for the MCP server's SessionManager."""

from whiteboxxai import WhiteBoxXAI


def build_client(jwt: str, base_url: str, timeout: int = 30) -> WhiteBoxXAI:
    """Construct a WhiteBoxXAI SDK client authenticated with the given JWT.

    The returned client's underlying httpx clients are lazily built on
    first use (see sdk/whiteboxxai/client.py's sync_client/async_client
    properties) and bake the Authorization header in at that point --
    callers that need to rotate the JWT on an existing client must reset
    `client._sync_client` / `client._async_client` to None after updating
    `client.config.api_key`, not just mutate a header dict.
    """
    return WhiteBoxXAI(api_key=jwt, base_url=base_url, timeout=timeout)
