"""MCP Inspector tool-list regression check (issue #227).

Runs the real `@modelcontextprotocol/inspector` CLI (not a Python
reimplementation) against the actual `whiteboxxai-mcp` stdio entry point,
normalizes its `tools/list` output, and diffs it against the committed
snapshot at tests/mcp_inspector_snapshot.json. This is the only
check in the project that exercises the server through a real, independent
MCP client rather than the `mcp` Python SDK's own `FastMCP`/`ClientSession`
machinery -- it catches protocol-level regressions (a tool that silently
stops registering, a renamed tool, a changed description or input schema)
that unit tests calling `mcp._tool_manager.get_tool(...)` directly cannot,
since those tests share the same in-process object graph as the code under
test.

Usage:
    # CI / local check -- exits non-zero and prints a diff on mismatch
    python tests/check_mcp_inspector_snapshot.py

    # After an intentional tool change -- regenerate the committed snapshot
    python tests/check_mcp_inspector_snapshot.py --write
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from difflib import unified_diff
from pathlib import Path

SNAPSHOT_PATH = Path(__file__).parent / "mcp_inspector_snapshot.json"
DUMMY_API_KEY = "wbx_ci_dummy_never_used_for_real_calls"


def _find_whiteboxxai_mcp() -> str:
    """Locate the whiteboxxai-mcp console-script entry point.

    Prefers whatever's already on PATH (the normal case once `pip install
    -e .` has run, in CI or locally) over guessing a venv layout.
    """
    found = shutil.which("whiteboxxai-mcp")
    if found:
        return found
    raise SystemExit(
        "whiteboxxai-mcp not found on PATH -- install it first: " "pip install -e ."
    )


def _fetch_tools() -> list[dict]:
    server_cmd = _find_whiteboxxai_mcp()
    result = subprocess.run(
        [
            "npx",
            "--yes",
            "@modelcontextprotocol/inspector",
            "--cli",
            server_cmd,
            "-e",
            f"WHITEBOXXAI_MCP_API_KEY={DUMMY_API_KEY}",
            "--method",
            "tools/list",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        shell=(sys.platform == "win32"),
    )
    if result.returncode != 0:
        raise SystemExit(
            "MCP Inspector exited non-zero -- the server likely failed to "
            f"start or answer tools/list.\nstdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"MCP Inspector did not print valid JSON on stdout: {e}\n"
            f"stdout: {result.stdout!r}"
        )
    return sorted(payload["result"]["tools"], key=lambda t: t["name"])


def _render(tools: list[dict]) -> str:
    return json.dumps(tools, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate the committed snapshot instead of diffing against it",
    )
    args = parser.parse_args()

    tools = _fetch_tools()
    rendered = _render(tools)

    if args.write:
        SNAPSHOT_PATH.write_text(rendered, newline="\n")
        print(f"Wrote {len(tools)} tools to {SNAPSHOT_PATH}")
        return 0

    if not SNAPSHOT_PATH.exists():
        raise SystemExit(
            f"{SNAPSHOT_PATH} does not exist -- run with --write to create it."
        )

    expected = SNAPSHOT_PATH.read_text()
    if rendered == expected:
        print(f"OK: {len(tools)} tools match the committed snapshot.")
        return 0

    diff = "\n".join(
        unified_diff(
            expected.splitlines(),
            rendered.splitlines(),
            fromfile="mcp_inspector_snapshot.json (committed)",
            tofile="tools/list (live)",
            lineterm="",
        )
    )
    print(
        "MCP tool-list drift detected. If this change is intentional, "
        "regenerate the snapshot in the same PR:\n"
        "  python tests/check_mcp_inspector_snapshot.py --write\n\n"
        f"{diff}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
