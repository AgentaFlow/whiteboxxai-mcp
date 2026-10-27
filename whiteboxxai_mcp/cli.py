"""Console-script entry point: runs the WhiteBoxXAI MCP server over stdio."""

import argparse
import sys

from whiteboxxai_mcp.config import settings
from whiteboxxai_mcp.logging_utils import configure_logging
from whiteboxxai_mcp.server import build_server


def main() -> None:
    """Entry point for the `whiteboxxai-mcp` console script (stdio transport)."""
    parser = argparse.ArgumentParser(
        prog="whiteboxxai-mcp",
        description=(
            "Run the WhiteBoxXAI MCP server over stdio. Configure via "
            "WHITEBOXXAI_MCP_* environment variables (see README.md)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log every outbound WhiteBoxXAI API request/response at DEBUG level.",
    )
    args = parser.parse_args()

    configure_logging(
        level=settings.LOG_LEVEL, verbose=settings.VERBOSE or args.verbose
    )

    if not settings.API_KEY:
        print(
            "whiteboxxai-mcp: no credentials configured. Set: "
            "WHITEBOXXAI_MCP_API_KEY (scoped and revocable, "
            "never needs refreshing). See README.md.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    server = build_server(settings)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
