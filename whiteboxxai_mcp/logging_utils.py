"""Verbose/diagnostic logging setup for the WhiteBoxXAI MCP server.

Enable with WHITEBOXXAI_MCP_VERBOSE=true (or --verbose on the CLI) to log
every outbound WhiteBoxXAI API request/response at DEBUG level -- useful
for debugging tool calls from an AI harness without needing a separate
proxy. httpx's own logger already emits one INFO-level line per request
when its logger is enabled; verbose mode here just turns that on and adds
whiteboxxai_mcp's own DEBUG logs.
"""

import logging


def configure_logging(level: str = "INFO", verbose: bool = False) -> None:
    """Configure logging for the whiteboxxai_mcp package and its HTTP client.

    stdio transport reserves stdout for the MCP JSON-RPC framing, so logs
    must go to stderr -- logging.basicConfig's default stream is stderr,
    which is what we want here.
    """
    effective_level = (
        logging.DEBUG if verbose else getattr(logging, level.upper(), logging.INFO)
    )

    logging.basicConfig(
        level=effective_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    logging.getLogger("whiteboxxai_mcp").setLevel(effective_level)

    if verbose:
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("httpcore").setLevel(logging.INFO)
